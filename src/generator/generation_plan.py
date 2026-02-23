"""Compute generation targets per (template, domain, difficulty)."""

from __future__ import annotations

import math
from dataclasses import dataclass


# Template -> (dimension, subdimension, base_count)
# base_count is the target from spec Section 6
TEMPLATE_SPEC: dict[str, tuple[str, str, int]] = {
    "H1": ("hallucination", "closed_book_truthfulness", 350),
    "H2": ("hallucination", "open_book_faithfulness", 250),
    "H3": ("hallucination", "uncertainty_calibration", 150),
    "H4": ("hallucination", "self_contradiction", 300),  # 100 base * 3 variants
    "H5": ("hallucination", "citation_fidelity", 100),
    "C1": ("consistency", "repeat_stability", 250),  # includes temperature_sensitivity subset
    "C2": ("consistency", "position_coherence", 100),  # 50 linked pairs
    "C3": ("consistency", "refusal_stability", 100),
    "P1": ("robustness", "phrasing_variants", 150),  # 150 core * direct style
    "P2": ("robustness", "phrasing_variants", 150),  # polite/verbose
    "P3": ("robustness", "phrasing_variants", 150),  # role-play
    "P4": ("robustness", "phrasing_variants", 150),  # few-shot
    "P5": ("robustness", "phrasing_variants", 150),  # noisy/typo
    "P6": ("robustness", "instruction_following", 150),
    "P7a": ("robustness", "zero_shot_vs_few_shot", 50),
    "P7b": ("robustness", "zero_shot_vs_few_shot", 50),
    "P8": ("robustness", "ambiguity_handling", 80),
    "S1": ("safety", "should_refuse", 250),
    "S2": ("safety", "should_comply_with_triggers", 100),
    "S3": ("safety", "toxicity_susceptibility", 200),
    "S4": ("safety", "bias_bbq_style", 80),
    "S5": ("safety", "tone_professionalism", 60),
}

# Difficulty distribution: 20% easy, 40% medium, 40% hard
DIFFICULTY_WEIGHTS = {"easy": 0.20, "medium": 0.40, "hard": 0.40}

# Domain weights from spec
DEFAULT_DOMAIN_WEIGHTS = {
    "insurance": 0.25,
    "banking": 0.20,
    "investments_wealth": 0.15,
    "tax_retirement": 0.15,
    "regulatory_compliance": 0.10,
    "general_financial_literacy": 0.10,
    "financial_history_events": 0.05,
}


@dataclass
class GenerationTarget:
    template_id: str
    dimension: str
    subdimension: str
    domain: str
    difficulty: str
    count: int


class GenerationPlan:
    """Compute how many prompts to generate per (template, domain, difficulty)."""

    def __init__(self, config: dict):
        self.domain_weights = config.get("domain_split", DEFAULT_DOMAIN_WEIGHTS)
        self.difficulty_weights = DIFFICULTY_WEIGHTS

    def compute(self, filter_domain: str | None = None,
                filter_dimension: str | None = None) -> list[GenerationTarget]:
        """Return list of GenerationTargets. Optionally filter by domain/dimension."""
        targets: list[GenerationTarget] = []

        for tid, (dim, subdim, base_count) in TEMPLATE_SPEC.items():
            if filter_dimension and dim != filter_dimension:
                continue

            for domain, domain_weight in self.domain_weights.items():
                if filter_domain and domain != filter_domain:
                    continue

                domain_count = base_count * domain_weight

                for diff, diff_weight in self.difficulty_weights.items():
                    count = max(1, round(domain_count * diff_weight))
                    targets.append(GenerationTarget(
                        template_id=tid,
                        dimension=dim,
                        subdimension=subdim,
                        domain=domain,
                        difficulty=diff,
                        count=count,
                    ))

        return targets

    def total_count(self, filter_domain: str | None = None,
                    filter_dimension: str | None = None) -> int:
        return sum(t.count for t in self.compute(filter_domain, filter_dimension))

    def summary(self) -> dict:
        """Return summary stats of the plan."""
        targets = self.compute()
        by_dim: dict[str, int] = {}
        by_domain: dict[str, int] = {}
        by_diff: dict[str, int] = {}
        for t in targets:
            by_dim[t.dimension] = by_dim.get(t.dimension, 0) + t.count
            by_domain[t.domain] = by_domain.get(t.domain, 0) + t.count
            by_diff[t.difficulty] = by_diff.get(t.difficulty, 0) + t.count
        return {
            "total": sum(t.count for t in targets),
            "by_dimension": by_dim,
            "by_domain": by_domain,
            "by_difficulty": by_diff,
        }
