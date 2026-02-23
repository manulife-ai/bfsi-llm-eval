"""Pipeline orchestrator: scrape -> generate -> validate -> assemble -> export."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

from src.scraper.wikipedia import WikipediaScraper
from src.scraper.web import WebScraper
from src.scraper.api import APIScraper
from src.scraper.chunker import Chunk
from src.generator.llm_client import LLMClient
from src.generator.prompt_builder import PromptBuilder
from src.generator.generation_plan import GenerationPlan
from src.dataset.assembler import DatasetAssembler
from src.dataset.exporter import DatasetExporter
from src.dataset.card_generator import CardGenerator

logger = logging.getLogger(__name__)

DEFAULT_GEN_CONFIG = "config/generation_config.yaml"
DEFAULT_SRC_CONFIG = "config/source_config.yaml"
GENERATED_DIR = Path("data/generated")
GENERATED_RECORDS_PATH = GENERATED_DIR / "records.jsonl"
RAW_DIR = Path("data/raw")
FINAL_DIR = Path("data/final")
SCRAPE_ERRORS_PATH = RAW_DIR / "scrape_errors.log"


def load_config(gen_path: str, src_path: str) -> tuple[dict, dict]:
    with open(gen_path) as f:
        gen_cfg = yaml.safe_load(f)
    with open(src_path) as f:
        src_cfg = yaml.safe_load(f)
    return gen_cfg, src_cfg


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# --- Scraping ---


def run_scrape(src_cfg: dict) -> list:
    """Run all enabled scrapers, return list of ScrapedContent."""
    all_content = []
    errors = []

    if src_cfg.get("wikipedia", {}).get("enabled", False):
        try:
            scraper = WikipediaScraper(src_cfg["wikipedia"])
            results = scraper.scrape()
            all_content.extend(results)
            logger.info("Wikipedia: scraped %d articles", len(results))
        except Exception as e:
            msg = f"Wikipedia scraper failed: {e}"
            logger.warning(msg)
            errors.append(msg)

    if src_cfg.get("web_sources", {}).get("enabled", False):
        try:
            scraper = WebScraper(src_cfg["web_sources"])
            results = scraper.scrape()
            all_content.extend(results)
            logger.info("Web: scraped %d pages", len(results))
        except Exception as e:
            msg = f"Web scraper failed: {e}"
            logger.warning(msg)
            errors.append(msg)

    if src_cfg.get("apis", {}).get("enabled", False):
        try:
            scraper = APIScraper(src_cfg["apis"])
            results = scraper.scrape()
            all_content.extend(results)
            logger.info("API: scraped %d items", len(results))
        except Exception as e:
            msg = f"API scraper failed: {e}"
            logger.warning(msg)
            errors.append(msg)

    if errors:
        SCRAPE_ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SCRAPE_ERRORS_PATH, "a") as f:
            for err in errors:
                f.write(err + "\n")

    logger.info("Total scraped content: %d items", len(all_content))
    return all_content


# --- Chunk selection ---


def _select_chunks(
    scraped: list,
    domain: str | None = None,
    seed: int = 42,
) -> list[tuple[str, str, str]]:
    """Return list of (chunk_text, domain, geography) from scraped content."""
    chunks = []
    for sc in scraped:
        if domain and sc.domain != domain:
            continue
        for chunk in sc.chunks:
            chunks.append((chunk.text, sc.domain, sc.geography))
    rng = random.Random(seed)
    rng.shuffle(chunks)
    return chunks


# --- Generation ---


def run_generate(
    gen_cfg: dict,
    scraped: list,
    filter_domain: str | None = None,
    filter_dimension: str | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """Generate eval records from scraped content. Returns list of record dicts."""
    plan = GenerationPlan(gen_cfg)
    targets = plan.compute(filter_domain=filter_domain, filter_dimension=filter_dimension)

    if dry_run:
        total = sum(t.count for t in targets)
        summary = plan.summary()
        logger.info("DRY RUN — would generate %d records", total)
        logger.info("Summary: %s", json.dumps(summary, indent=2))
        return []

    llm_client = LLMClient(gen_cfg["generation_model"])
    builder = PromptBuilder(llm_client, version=gen_cfg.get("dataset", {}).get("version", "1.0.0"))

    # Build chunk pools by domain
    chunk_pool = _select_chunks(scraped, seed=gen_cfg.get("dataset", {}).get("seed", 42))
    chunk_idx = 0

    records: list[dict] = []
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    for target in tqdm(targets, desc="Generating records"):
        # Find chunks matching this target's domain
        domain_chunks = [c for c in chunk_pool if c[1] == target.domain]
        if not domain_chunks:
            domain_chunks = chunk_pool  # fallback to any chunk

        for i in range(target.count):
            chunk_text, chunk_domain, chunk_geo = domain_chunks[i % len(domain_chunks)] if domain_chunks else ("", target.domain, "general")
            try:
                new_records = builder.generate_record(
                    template_id=target.template_id,
                    domain=target.domain,
                    geography=chunk_geo,
                    difficulty=target.difficulty,
                    scraped_chunk=chunk_text,
                )
                records.extend(new_records)
            except Exception as e:
                logger.warning(
                    "Generation failed for %s/%s/%s: %s",
                    target.template_id, target.domain, target.difficulty, e,
                )
                continue

    # Save generated records
    _save_generated_records(records)
    logger.info("Generated %d records total", len(records))
    return records


def _save_generated_records(records: list[dict]) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    with open(GENERATED_RECORDS_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_generated_records() -> list[dict]:
    if not GENERATED_RECORDS_PATH.exists():
        return []
    records = []
    with open(GENERATED_RECORDS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --- Assembly + Export ---


def run_assemble_export(gen_cfg: dict, records: list[dict]) -> dict:
    """Assemble, deduplicate, export, and generate card. Returns stats."""
    output_dir = gen_cfg.get("dataset", {}).get("output_dir", "data/final")
    version = gen_cfg.get("dataset", {}).get("version", "1.0.0")

    assembler = DatasetAssembler(gen_cfg)
    assembled = assembler.assemble(records)
    stats = assembler.stats(assembled)

    exporter = DatasetExporter(output_dir)
    pq_path, jl_path = exporter.export(assembled)

    card_gen = CardGenerator(output_dir)
    card_gen.generate(stats, version=version)

    logger.info("Assembly complete: %d records", stats["total"])
    logger.info("  Parquet: %s", pq_path)
    logger.info("  JSONL: %s", jl_path)
    logger.info("  Stats: %s", json.dumps(stats, indent=2))
    return stats


# --- Incremental mode ---


def _increment_version(version: str) -> str:
    """Increment patch version: 1.0.0 -> 1.0.1."""
    parts = version.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    return ".".join(parts)


def run_incremental(gen_cfg: dict, src_cfg: dict, args) -> None:
    """Run incremental generation: only generate missing records."""
    existing = _load_generated_records()
    existing_ids = {r["id"] for r in existing}
    logger.info("Incremental mode: %d existing records", len(existing))

    # Scrape (with caching, this is fast)
    if not args.no_scrape:
        scraped = run_scrape(src_cfg)
    else:
        scraped = []

    # Compute deficit
    plan = GenerationPlan(gen_cfg)
    total_target = gen_cfg.get("dataset", {}).get("target_total", 2910)
    deficit = max(0, total_target - len(existing))

    if deficit == 0:
        logger.info("No deficit — already have %d records (target %d)", len(existing), total_target)
    else:
        logger.info("Need %d more records (have %d, target %d)", deficit, len(existing), total_target)
        new_records = run_generate(
            gen_cfg, scraped,
            filter_domain=args.domain,
            filter_dimension=args.dimension,
        )
        existing.extend(new_records)

    # Auto-increment patch version
    version = gen_cfg.get("dataset", {}).get("version", "1.0.0")
    gen_cfg.setdefault("dataset", {})["version"] = _increment_version(version)

    run_assemble_export(gen_cfg, existing)


# --- CLI ---


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BFSI LLM Eval Harness — Dataset Generation Pipeline",
    )
    parser.add_argument(
        "--mode", choices=["full_refresh", "incremental"], default="full_refresh",
        help="Pipeline mode (default: full_refresh)",
    )
    parser.add_argument(
        "--config", default=DEFAULT_GEN_CONFIG,
        help=f"Path to generation config YAML (default: {DEFAULT_GEN_CONFIG})",
    )
    parser.add_argument(
        "--source-config", default=DEFAULT_SRC_CONFIG,
        help=f"Path to source config YAML (default: {DEFAULT_SRC_CONFIG})",
    )
    parser.add_argument(
        "--domain", default=None,
        help="Filter generation to a single domain (e.g., banking)",
    )
    parser.add_argument(
        "--dimension", default=None,
        help="Filter generation to a single dimension (e.g., hallucination)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show generation plan without executing",
    )
    parser.add_argument(
        "--no-scrape", action="store_true",
        help="Skip scraping, use cached/existing content",
    )
    parser.add_argument(
        "--version", default=None,
        help="Override dataset version string",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose/debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    gen_cfg, src_cfg = load_config(args.config, args.source_config)

    # Override version if specified
    if args.version:
        gen_cfg.setdefault("dataset", {})["version"] = args.version

    if args.mode == "full_refresh":
        logger.info("=== FULL REFRESH MODE ===")
        _clear_dir(RAW_DIR)
        _clear_dir(GENERATED_DIR)
        _clear_dir(FINAL_DIR)

        # Step 1: Scrape
        if not args.no_scrape:
            scraped = run_scrape(src_cfg)
        else:
            scraped = []
            logger.info("Scraping skipped (--no-scrape)")

        # Step 2: Generate
        records = run_generate(
            gen_cfg, scraped,
            filter_domain=args.domain,
            filter_dimension=args.dimension,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            return

        # Step 3: Assemble + Export
        stats = run_assemble_export(gen_cfg, records)
        logger.info("=== DONE === Total: %d records", stats["total"])

    elif args.mode == "incremental":
        logger.info("=== INCREMENTAL MODE ===")
        run_incremental(gen_cfg, src_cfg, args)
        logger.info("=== DONE (incremental) ===")


if __name__ == "__main__":
    main()
