"""Schema validation for BFSI LLM evaluation records using Pydantic v2."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# --- Enums as Literal types ---

Dimension = Literal[
    "hallucination",
    "consistency",
    "robustness",
    "safety",
]

Subdimension = Literal[
    # hallucination
    "closed_book_truthfulness",
    "open_book_faithfulness",
    "uncertainty_calibration",
    "self_contradiction",
    "citation_fidelity",
    # consistency
    "repeat_stability",
    "temperature_sensitivity",
    "position_coherence",
    "refusal_stability",
    # robustness
    "phrasing_variants",
    "instruction_following",
    "zero_shot_vs_few_shot",
    "ambiguity_handling",
    # safety
    "should_refuse",
    "should_comply_with_triggers",
    "toxicity_susceptibility",
    "bias_bbq_style",
    "tone_professionalism",
]

SourceDomain = Literal[
    "insurance",
    "banking",
    "investments_wealth",
    "tax_retirement",
    "regulatory_compliance",
    "general_financial_literacy",
    "financial_history_events",
]

Geography = Literal["canada", "usa", "general"]

Difficulty = Literal["easy", "medium", "hard"]

RubricType = Literal[
    "faithfulness_1_5",
    "safety_refusal_1_5",
    "robustness_1_5",
    "consistency_binary",
]

AdversarialStrategy = Literal[
    "plausible_misinformation",
    "conflicting_context_injection",
    "leading_question_framing",
    "near_miss_regulatory_confusion",
    "confident_sounding_unanswerable",
    "multi_step_reasoning_trap",
    "semantic_similarity_trap",
    "policy_boundary_probing",
    "over_refusal_trap",
]

# --- Dimension -> valid subdimensions mapping ---

DIMENSION_SUBDIMENSIONS: dict[str, set[str]] = {
    "hallucination": {
        "closed_book_truthfulness",
        "open_book_faithfulness",
        "uncertainty_calibration",
        "self_contradiction",
        "citation_fidelity",
    },
    "consistency": {
        "repeat_stability",
        "temperature_sensitivity",
        "position_coherence",
        "refusal_stability",
    },
    "robustness": {
        "phrasing_variants",
        "instruction_following",
        "zero_shot_vs_few_shot",
        "ambiguity_handling",
    },
    "safety": {
        "should_refuse",
        "should_comply_with_triggers",
        "toxicity_susceptibility",
        "bias_bbq_style",
        "tone_professionalism",
    },
}


# --- Pydantic models ---


class PassCriteria(BaseModel):
    rubric_type: RubricType
    min_passing_score: int = Field(ge=1, le=5)
    scoring_description: str = Field(min_length=1)
    hard_fail_conditions: list[str] = Field(default_factory=list)
    auto_scoreable: bool
    scoring_notes: str = ""


class EvalRecord(BaseModel):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=4000)
    dimension: Dimension
    subdimension: Subdimension
    source_domain: SourceDomain
    geography: Geography
    difficulty: Difficulty
    language: str = Field(default="en")
    expected_behavior: str = Field(min_length=1)
    pass_criteria: PassCriteria
    prompt_template_id: str = Field(min_length=1)
    reference_context: str | None = None
    is_adversarial: bool
    adversarial_strategy: AdversarialStrategy | None = None
    expected_failure_modes: list[str] = Field(default_factory=list)
    linked_prompt_ids: list[str] = Field(default_factory=list)

    @field_validator("subdimension")
    @classmethod
    def subdimension_matches_dimension(cls, v: str, info) -> str:
        dimension = info.data.get("dimension")
        if dimension and v not in DIMENSION_SUBDIMENSIONS.get(dimension, set()):
            raise ValueError(
                f"subdimension '{v}' is not valid for dimension '{dimension}'"
            )
        return v

    @field_validator("adversarial_strategy")
    @classmethod
    def hard_requires_adversarial(cls, v, info):
        difficulty = info.data.get("difficulty")
        if difficulty == "hard" and v is None:
            raise ValueError(
                "hard prompts must have a non-null adversarial_strategy"
            )
        return v

    @field_validator("expected_failure_modes")
    @classmethod
    def hard_requires_failure_modes(cls, v: list[str], info) -> list[str]:
        difficulty = info.data.get("difficulty")
        if difficulty == "hard" and len(v) < 2:
            raise ValueError(
                "hard prompts must have at least 2 expected_failure_modes"
            )
        return v


def validate_record(data: dict) -> tuple[bool, list[str]]:
    """Validate a record dict against the EvalRecord schema.

    Returns (True, []) on success or (False, [error_messages]) on failure.
    """
    try:
        EvalRecord.model_validate(data)
        return True, []
    except Exception as e:
        errors = []
        if hasattr(e, "errors"):
            for err in e.errors():
                loc = " -> ".join(str(l) for l in err["loc"])
                errors.append(f"{loc}: {err['msg']}")
        else:
            errors.append(str(e))
        return False, errors
