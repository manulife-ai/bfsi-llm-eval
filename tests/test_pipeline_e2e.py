"""End-to-end pipeline tests with mock LLM (no API credits)."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.pipeline import (
    main,
    run_scrape,
    run_generate,
    run_assemble_export,
    _select_chunks,
    _increment_version,
    _save_generated_records,
    _load_generated_records,
    load_config,
    build_parser,
)
from src.scraper.base import ScrapedContent


# --- Helpers ---

def _fake_scraped(n=5, domain="banking", geography="canada"):
    """Create fake ScrapedContent objects with chunks."""
    results = []
    for i in range(n):
        sc = ScrapedContent(
            source_type="test",
            source_name=f"test-source-{i}",
            domain=domain,
            geography=geography,
            text=f"This is a detailed test passage about {domain} in {geography}. " * 60,
        )
        sc.chunk()
        results.append(sc)
    return results


def _mock_llm_response(difficulty="medium"):
    """Return a valid JSON response that the LLM would produce."""
    rec = {
        "question": "What is the capital adequacy ratio requirement for Canadian banks?",
        "expected_behavior": "Model should cite the correct OSFI guideline.",
        "pass_criteria": {
            "rubric_type": "faithfulness_1_5",
            "min_passing_score": 4,
            "scoring_description": "5=correct; 1=wrong",
            "hard_fail_conditions": [],
            "auto_scoreable": True,
            "scoring_notes": "",
        },
    }
    if difficulty == "hard":
        rec["adversarial_strategy"] = "leading_question_framing"
        rec["expected_failure_modes"] = ["Wrong ratio cited", "Confuses OSFI with SEC rule"]
    return json.dumps(rec)


# --- Unit tests for helpers ---


class TestVersionIncrement:
    def test_patch_increment(self):
        assert _increment_version("1.0.0") == "1.0.1"
        assert _increment_version("1.0.9") == "1.0.10"
        assert _increment_version("2.1.3") == "2.1.4"


class TestChunkSelection:
    def test_filters_by_domain(self):
        scraped = _fake_scraped(3, domain="banking") + _fake_scraped(2, domain="insurance")
        chunks = _select_chunks(scraped, domain="banking")
        assert all(c[1] == "banking" for c in chunks)

    def test_returns_all_when_no_filter(self):
        scraped = _fake_scraped(3, domain="banking") + _fake_scraped(2, domain="insurance")
        chunks = _select_chunks(scraped)
        domains = {c[1] for c in chunks}
        assert "banking" in domains
        assert "insurance" in domains


class TestGeneratedRecordsPersistence:
    def test_save_and_load(self, tmp_path, monkeypatch):
        records_path = tmp_path / "records.jsonl"
        monkeypatch.setattr("src.pipeline.GENERATED_DIR", tmp_path)
        monkeypatch.setattr("src.pipeline.GENERATED_RECORDS_PATH", records_path)
        records = [{"id": "r1", "prompt": "Q1"}, {"id": "r2", "prompt": "Q2"}]
        _save_generated_records(records)
        loaded = _load_generated_records()
        assert len(loaded) == 2
        assert loaded[0]["id"] == "r1"

    def test_load_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.pipeline.GENERATED_RECORDS_PATH", tmp_path / "nope.jsonl")
        assert _load_generated_records() == []


# --- Config loading ---


class TestConfigLoading:
    def test_loads_both_configs(self):
        gen_cfg, src_cfg = load_config(
            "config/generation_config.yaml",
            "config/source_config.yaml",
        )
        assert "generation_model" in gen_cfg
        assert "wikipedia" in src_cfg


# --- Parser ---


class TestParser:
    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.mode == "full_refresh"
        assert args.dry_run is False
        assert args.no_scrape is False

    def test_all_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "--mode", "incremental",
            "--domain", "banking",
            "--dimension", "hallucination",
            "--dry-run",
            "--no-scrape",
            "--version", "2.0.0",
            "-v",
        ])
        assert args.mode == "incremental"
        assert args.domain == "banking"
        assert args.dimension == "hallucination"
        assert args.dry_run is True
        assert args.no_scrape is True
        assert args.version == "2.0.0"
        assert args.verbose is True


# --- E2E with mock LLM ---


class TestPipelineE2E:
    """Full pipeline run with mocked LLM and scrapers."""

    def test_dry_run(self, tmp_path, monkeypatch, capsys):
        """Dry run should print plan without generating."""
        monkeypatch.setattr("src.pipeline.RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr("src.pipeline.GENERATED_DIR", tmp_path / "gen")
        monkeypatch.setattr("src.pipeline.FINAL_DIR", tmp_path / "final")
        monkeypatch.setattr("src.pipeline.SCRAPE_ERRORS_PATH", tmp_path / "raw" / "errors.log")

        main(["--mode", "full_refresh", "--dry-run", "--no-scrape"])
        # Should not create any output files
        assert not (tmp_path / "final" / "dataset.parquet").exists()

    def test_full_pipeline_mock_llm(self, tmp_path, monkeypatch):
        """Full pipeline with mock LLM producing valid records."""
        monkeypatch.setattr("src.pipeline.RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr("src.pipeline.GENERATED_DIR", tmp_path / "gen")
        monkeypatch.setattr("src.pipeline.GENERATED_RECORDS_PATH", tmp_path / "gen" / "records.jsonl")
        monkeypatch.setattr("src.pipeline.FINAL_DIR", tmp_path / "final")
        monkeypatch.setattr("src.pipeline.SCRAPE_ERRORS_PATH", tmp_path / "raw" / "errors.log")
        monkeypatch.setattr("src.dataset.assembler.GENERATED_DIR", tmp_path / "gen")
        monkeypatch.setattr("src.dataset.assembler.VALIDATION_ERRORS_PATH", tmp_path / "gen" / "errors.jsonl")

        # Mock LLM to return valid JSON
        mock_client = MagicMock()
        mock_client.generate.return_value = _mock_llm_response()

        # Mock scraped content
        fake_scraped = []
        for domain in ["banking", "insurance", "investments_wealth", "tax_retirement",
                        "regulatory_compliance", "general_financial_literacy", "financial_history_events"]:
            fake_scraped.extend(_fake_scraped(2, domain=domain))

        with patch("src.pipeline.run_scrape", return_value=fake_scraped), \
             patch("src.pipeline.LLMClient", return_value=mock_client):
            gen_cfg, _ = load_config("config/generation_config.yaml", "config/source_config.yaml")
            gen_cfg["dataset"]["output_dir"] = str(tmp_path / "final")

            records = run_generate(gen_cfg, fake_scraped, dry_run=False)

        # Should have generated records (mock returns 1 record per call)
        assert len(records) > 0

    def test_assemble_export_roundtrip(self, tmp_path, monkeypatch):
        """Assemble + export with synthetic records."""
        monkeypatch.setattr("src.dataset.assembler.GENERATED_DIR", tmp_path)
        monkeypatch.setattr("src.dataset.assembler.VALIDATION_ERRORS_PATH", tmp_path / "errors.jsonl")

        gen_cfg = {
            "dataset": {"seed": 42, "version": "1.0.0", "output_dir": str(tmp_path / "final")},
        }
        records = []
        for i in range(20):
            dim = ["hallucination", "safety", "robustness", "consistency"][i % 4]
            subdim_map = {
                "hallucination": "closed_book_truthfulness",
                "safety": "should_refuse",
                "robustness": "phrasing_variants",
                "consistency": "repeat_stability",
            }
            diff = ["easy", "medium", "hard"][i % 3]
            records.append({
                "id": f"test-{i:04d}",
                "version": "1.0.0",
                "prompt": f"Unique question number {i} about BFSI topics?",
                "dimension": dim,
                "subdimension": subdim_map[dim],
                "source_domain": "banking",
                "geography": "canada",
                "difficulty": diff,
                "language": "en",
                "expected_behavior": "Answer correctly.",
                "pass_criteria": {
                    "rubric_type": "faithfulness_1_5",
                    "min_passing_score": 4,
                    "scoring_description": "5=correct",
                    "hard_fail_conditions": [],
                    "auto_scoreable": True,
                    "scoring_notes": "",
                },
                "prompt_template_id": "H1",
                "reference_context": None,
                "is_adversarial": diff == "hard",
                "adversarial_strategy": "leading_question_framing" if diff == "hard" else None,
                "expected_failure_modes": ["Wrong A", "Wrong B"] if diff == "hard" else [],
                "linked_prompt_ids": [],
            })

        stats = run_assemble_export(gen_cfg, records)
        assert stats["total"] == 20
        assert (tmp_path / "final" / "dataset.parquet").exists()
        assert (tmp_path / "final" / "dataset.jsonl").exists()
        assert (tmp_path / "final" / "README.md").exists()
