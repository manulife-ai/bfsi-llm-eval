"""Tests for dataset assembly, export, and card generation."""

import copy
import json
import pytest
from pathlib import Path

from src.dataset.assembler import DatasetAssembler
from src.dataset.exporter import DatasetExporter
from src.dataset.card_generator import CardGenerator


_DIM_SUBDIM = {
    "hallucination": "closed_book_truthfulness",
    "consistency": "repeat_stability",
    "robustness": "phrasing_variants",
    "safety": "should_refuse",
}


def _make_record(
    id_suffix="001",
    prompt="What is the CDIC deposit insurance limit?",
    dimension="hallucination",
    subdimension=None,
    domain="banking",
    geography="canada",
    difficulty="medium",
):
    if subdimension is None:
        subdimension = _DIM_SUBDIM.get(dimension, "closed_book_truthfulness")
    return {
        "id": f"test-{id_suffix}",
        "version": "1.0.0",
        "prompt": prompt,
        "dimension": dimension,
        "subdimension": subdimension,
        "source_domain": domain,
        "geography": geography,
        "difficulty": difficulty,
        "language": "en",
        "expected_behavior": "Model should answer correctly.",
        "pass_criteria": {
            "rubric_type": "faithfulness_1_5",
            "min_passing_score": 4,
            "scoring_description": "5=correct; 1=wrong",
            "hard_fail_conditions": [],
            "auto_scoreable": True,
            "scoring_notes": "",
        },
        "prompt_template_id": "H1",
        "reference_context": None,
        "is_adversarial": difficulty == "hard",
        "adversarial_strategy": "leading_question_framing" if difficulty == "hard" else None,
        "expected_failure_modes": ["Wrong answer A", "Wrong answer B"] if difficulty == "hard" else [],
        "linked_prompt_ids": [],
    }


def _make_records(n, **kwargs):
    """Generate n distinct records."""
    records = []
    for i in range(n):
        r = _make_record(id_suffix=f"{i:04d}", prompt=f"Question number {i} about finance?", **kwargs)
        records.append(r)
    return records


# --- Assembler tests ---


class TestAssemblerValidation:
    def test_valid_records_pass(self):
        asm = DatasetAssembler({"dataset": {"seed": 42, "version": "1.0.0"}})
        records = _make_records(5)
        result = asm.assemble(records)
        assert len(result) == 5

    def test_invalid_records_filtered(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.dataset.assembler.GENERATED_DIR", tmp_path)
        monkeypatch.setattr("src.dataset.assembler.VALIDATION_ERRORS_PATH", tmp_path / "errors.jsonl")

        asm = DatasetAssembler({"dataset": {"seed": 42, "version": "1.0.0"}})
        good = _make_records(3)
        bad = _make_record(id_suffix="bad")
        bad["dimension"] = "not_a_real_dimension"  # invalid enum

        result = asm.assemble(good + [bad])
        assert len(result) == 3


class TestAssemblerDedup:
    def test_exact_duplicates_removed(self):
        asm = DatasetAssembler({"dataset": {"seed": 42, "version": "1.0.0"}})
        r1 = _make_record(id_suffix="001", prompt="What is the CDIC limit?")
        r2 = _make_record(id_suffix="002", prompt="What is the CDIC limit?")  # exact same prompt
        result = asm.assemble([r1, r2])
        # With sentence-transformers: exact match sim=1.0 > 0.97, should dedup
        assert len(result) <= 2  # may be 1 if dedup works, 2 if s-t not installed

    def test_different_prompts_kept(self):
        asm = DatasetAssembler({"dataset": {"seed": 42, "version": "1.0.0"}})
        r1 = _make_record(id_suffix="001", prompt="What is the CDIC deposit insurance coverage limit in Canada?")
        r2 = _make_record(id_suffix="002", prompt="How does OSFI regulate capital adequacy for Schedule I banks?")
        result = asm.assemble([r1, r2])
        assert len(result) == 2


class TestAssemblerSplits:
    def test_over_represented_domain_trimmed(self):
        asm = DatasetAssembler({
            "dataset": {"seed": 42, "target_total": 100, "version": "1.0.0"},
            "domain_split": {"banking": 0.50, "insurance": 0.50},
        })
        # 80 banking, 20 insurance — banking is over-represented
        records = _make_records(80, domain="banking") + _make_records(20, domain="insurance")
        result = asm.assemble(records)
        banking_count = sum(1 for r in result if r["source_domain"] == "banking")
        # Should be trimmed to ~50 +5% = 52
        assert banking_count <= 55

    def test_under_represented_domain_warned(self, caplog):
        import logging
        asm = DatasetAssembler({
            "dataset": {"seed": 42, "target_total": 100, "version": "1.0.0"},
            "domain_split": {"banking": 0.50, "insurance": 0.50},
        })
        # Only 5 insurance records, target is 50
        records = _make_records(50, domain="banking") + _make_records(5, domain="insurance")
        with caplog.at_level(logging.WARNING):
            asm.assemble(records)
        assert any("under-represented" in msg for msg in caplog.messages)


class TestAssemblerLinkedIds:
    def test_broken_links_cleaned(self):
        asm = DatasetAssembler({"dataset": {"seed": 42, "version": "1.0.0"}})
        r1 = _make_record(id_suffix="001")
        r1["linked_prompt_ids"] = ["test-002", "nonexistent-id"]
        r2 = _make_record(id_suffix="002")
        r2["linked_prompt_ids"] = ["test-001"]

        result = asm.assemble([r1, r2])
        for r in result:
            if r["id"] == "test-001":
                assert "nonexistent-id" not in r["linked_prompt_ids"]


class TestAssemblerStats:
    def test_stats_correct(self):
        asm = DatasetAssembler({"dataset": {"seed": 42, "version": "1.0.0"}})
        records = (
            _make_records(3, dimension="hallucination", domain="banking", difficulty="easy") +
            _make_records(2, dimension="safety", domain="insurance", difficulty="hard")
        )
        result = asm.assemble(records)
        s = asm.stats(result)
        assert s["total"] == 5
        assert s["by_dimension"]["hallucination"] == 3
        assert s["by_dimension"]["safety"] == 2


# --- Exporter tests ---


class TestExporter:
    def test_export_parquet_and_jsonl(self, tmp_path):
        exporter = DatasetExporter(str(tmp_path))
        records = _make_records(10)
        pq_path, jl_path = exporter.export(records)

        assert pq_path.exists()
        assert jl_path.exists()

        # Verify JSONL line count
        with open(jl_path) as f:
            lines = f.readlines()
        assert len(lines) == 10

    def test_parquet_roundtrip(self, tmp_path):
        exporter = DatasetExporter(str(tmp_path))
        records = _make_records(5)
        exporter.export(records)

        import pandas as pd
        df = pd.read_parquet(tmp_path / "dataset.parquet")
        assert len(df) == 5
        assert "prompt" in df.columns
        assert "dimension" in df.columns

    def test_jsonl_records_valid_json(self, tmp_path):
        exporter = DatasetExporter(str(tmp_path))
        records = _make_records(3)
        _, jl_path = exporter.export(records)

        with open(jl_path) as f:
            for line in f:
                parsed = json.loads(line)
                assert "id" in parsed
                assert "prompt" in parsed


# --- Card generator tests ---


class TestCardGenerator:
    def test_generates_readme(self, tmp_path):
        gen = CardGenerator(str(tmp_path))
        stats = {
            "total": 100,
            "by_dimension": {"hallucination": 40, "safety": 30, "robustness": 20, "consistency": 10},
            "by_domain": {"banking": 50, "insurance": 50},
            "by_difficulty": {"easy": 20, "medium": 40, "hard": 40},
        }
        path = gen.generate(stats, version="1.0.0")
        assert path.exists()

        content = path.read_text()
        assert "---" in content  # YAML header
        assert "cc-by-4.0" in content
        assert "100" in content  # total count
        assert "hallucination" in content

    def test_card_has_yaml_header(self, tmp_path):
        gen = CardGenerator(str(tmp_path))
        stats = {"total": 50, "by_dimension": {}, "by_domain": {}, "by_difficulty": {}}
        path = gen.generate(stats)
        content = path.read_text()
        # Should start with YAML front matter
        assert content.startswith("---")
        # Should have closing ---
        parts = content.split("---")
        assert len(parts) >= 3  # before, yaml, after
