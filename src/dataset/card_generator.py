"""Auto-generate HuggingFace dataset card (README.md)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CARD_TEMPLATE = """---
language:
- en
license: cc-by-4.0
tags:
- llm-evaluation
- hallucination
- safety
- BFSI
- finance
- Canada
- USA
task_categories:
- question-answering
- text-classification
pretty_name: "LLM Behavioral Evaluation Dataset — BFSI Edition"
size_categories:
- 1K<n<10K
---

# LLM Behavioral Evaluation Dataset — BFSI Edition

## Dataset Description

A structured evaluation dataset of {total} prompts designed to test LLM behavior across Banking, Financial Services, and Insurance (BFSI) domains. Covers hallucination detection, consistency, robustness, and safety evaluation.

**Version:** {version}
**License:** CC BY 4.0

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total records | {total} |
| Dimensions | 4 (hallucination, consistency, robustness, safety) |
| Domains | 7 BFSI domains |
| Languages | English |
| Difficulty levels | easy, medium, hard |

### Records by Dimension

{dimension_table}

### Records by Domain

{domain_table}

### Records by Difficulty

{difficulty_table}

## Dataset Schema

Each record contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | UUID v4, unique per record |
| `version` | string | Dataset version |
| `prompt` | string | Full prompt text for the model under test |
| `dimension` | string | Eval dimension: hallucination, consistency, robustness, safety |
| `subdimension` | string | Sub-category within dimension |
| `source_domain` | string | BFSI domain |
| `geography` | string | canada, usa, or general |
| `difficulty` | string | easy, medium, or hard |
| `language` | string | Always "en" for v1 |
| `expected_behavior` | string | Description of ideal model response |
| `pass_criteria` | object | Structured scoring rubric |
| `prompt_template_id` | string | Template ID used for generation |
| `reference_context` | string/null | Reference passage for open-book prompts |
| `is_adversarial` | bool | Whether prompt probes a boundary |
| `adversarial_strategy` | string/null | Named adversarial strategy applied |
| `expected_failure_modes` | array | Specific wrong answers a weak model might produce |
| `linked_prompt_ids` | array | IDs of logically linked prompts |

## Intended Use

This dataset is designed for **evaluating LLM behavior** in financial services contexts. It tests:
- **Hallucination**: factual accuracy, faithfulness to sources, uncertainty calibration
- **Consistency**: answer stability across rephrasing and temperature
- **Robustness**: performance across phrasing styles, instruction following
- **Safety**: appropriate refusal, over-refusal avoidance, bias detection, professionalism

## Out of Scope

- Not intended for model training or fine-tuning
- Not a substitute for regulatory compliance testing
- Does not cover non-English languages (v1)

## Generation Methodology

Prompts were generated using a configurable LLM pipeline:
1. Financial content scraped from public sources (Wikipedia, bank websites, regulatory sites)
2. Content chunked and passed to a generation LLM with structured templates
3. Each template includes adversarial design instructions for hard-difficulty prompts
4. Generated records validated against a strict schema
5. Near-duplicates removed via cosine similarity (threshold 0.97)
6. Domain and dimension splits enforced to match target distribution

## Citation

If you use this dataset, please cite:
```
@misc{{bfsi_llm_eval,
  title={{LLM Behavioral Evaluation Dataset — BFSI Edition}},
  year={{2025}},
  url={{https://github.com/sabyasm/bfsi-llm-eval}}
}}
```
"""


def _make_table(data: dict[str, int]) -> str:
    """Generate a markdown table from a dict."""
    lines = ["| Category | Count |", "|----------|-------|"]
    for k, v in sorted(data.items()):
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


class CardGenerator:
    """Generate HuggingFace dataset card README.md."""

    def __init__(self, output_dir: str = "data/final"):
        self.output_dir = Path(output_dir)

    def generate(self, stats: dict, version: str = "1.0.0") -> Path:
        """Generate the dataset card and write to output_dir/README.md."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "README.md"

        content = CARD_TEMPLATE.format(
            total=stats["total"],
            version=version,
            dimension_table=_make_table(stats.get("by_dimension", {})),
            domain_table=_make_table(stats.get("by_domain", {})),
            difficulty_table=_make_table(stats.get("by_difficulty", {})),
        )

        path.write_text(content)
        logger.info("Generated dataset card: %s", path)
        return path
