"""Tests for schema validation."""

import copy
import pytest
from src.validator.schema_validator import validate_record, EvalRecord


VALID_RECORD = {
    "id": "a3f2c1d4-1234-5678-9abc-def012345678",
    "version": "1.0.0",
    "prompt": "Based only on the following excerpt, what is the minimum capital adequacy ratio?",
    "dimension": "hallucination",
    "subdimension": "open_book_faithfulness",
    "source_domain": "regulatory_compliance",
    "geography": "canada",
    "difficulty": "medium",
    "language": "en",
    "expected_behavior": "Model should answer using only the excerpt provided.",
    "pass_criteria": {
        "rubric_type": "faithfulness_1_5",
        "min_passing_score": 4,
        "scoring_description": "5=fully grounded; 1=fabricates",
        "hard_fail_conditions": ["Fabricates a number not in excerpt"],
        "auto_scoreable": True,
        "scoring_notes": "Use QAG method.",
    },
    "prompt_template_id": "H2",
    "reference_context": "Schedule I banks must maintain a minimum Tier 1 capital ratio of 6%.",
    "is_adversarial": False,
    "adversarial_strategy": None,
    "expected_failure_modes": [],
    "linked_prompt_ids": [],
}

VALID_HARD_RECORD = {
    **VALID_RECORD,
    "difficulty": "hard",
    "is_adversarial": True,
    "adversarial_strategy": "leading_question_framing",
    "expected_failure_modes": [
        "Accepts false premise about eliminated contribution limits",
        "Confuses RRSP with TFSA rules",
    ],
}


def _record(**overrides):
    r = copy.deepcopy(VALID_RECORD)
    for k, v in overrides.items():
        if v is _DELETE:
            r.pop(k, None)
        else:
            r[k] = v
    return r


_DELETE = object()


class TestValidRecord:
    def test_valid_medium_record(self):
        ok, errors = validate_record(VALID_RECORD)
        assert ok, errors

    def test_valid_hard_record(self):
        ok, errors = validate_record(VALID_HARD_RECORD)
        assert ok, errors

    def test_valid_easy_record(self):
        ok, errors = validate_record(_record(difficulty="easy"))
        assert ok, errors


class TestMissingFields:
    @pytest.mark.parametrize("field", [
        "id", "version", "prompt", "dimension", "subdimension",
        "source_domain", "geography", "difficulty",
        "expected_behavior", "pass_criteria", "prompt_template_id",
    ])
    def test_missing_required_field(self, field):
        ok, errors = validate_record(_record(**{field: _DELETE}))
        assert not ok

    def test_missing_pass_criteria_subfield(self):
        bad_criteria = {
            "rubric_type": "faithfulness_1_5",
            # missing min_passing_score, scoring_description, auto_scoreable
        }
        ok, errors = validate_record(_record(pass_criteria=bad_criteria))
        assert not ok


class TestBadEnums:
    def test_bad_dimension(self):
        ok, errors = validate_record(_record(dimension="nonsense"))
        assert not ok

    def test_bad_subdimension(self):
        ok, errors = validate_record(_record(subdimension="nonsense"))
        assert not ok

    def test_bad_source_domain(self):
        ok, errors = validate_record(_record(source_domain="crypto"))
        assert not ok

    def test_bad_geography(self):
        ok, errors = validate_record(_record(geography="mars"))
        assert not ok

    def test_bad_difficulty(self):
        ok, errors = validate_record(_record(difficulty="impossible"))
        assert not ok

    def test_bad_rubric_type(self):
        criteria = copy.deepcopy(VALID_RECORD["pass_criteria"])
        criteria["rubric_type"] = "vibes_based"
        ok, errors = validate_record(_record(pass_criteria=criteria))
        assert not ok

    def test_bad_adversarial_strategy(self):
        ok, errors = validate_record(
            _record(difficulty="hard", adversarial_strategy="chaos_monkey",
                    expected_failure_modes=["a", "b"], is_adversarial=True)
        )
        assert not ok


class TestPromptLength:
    def test_prompt_too_long(self):
        ok, errors = validate_record(_record(prompt="x" * 4001))
        assert not ok

    def test_prompt_at_limit(self):
        ok, errors = validate_record(_record(prompt="x" * 4000))
        assert ok, errors

    def test_prompt_empty(self):
        ok, errors = validate_record(_record(prompt=""))
        assert not ok


class TestSubdimensionDimensionMatch:
    def test_mismatched_subdimension(self):
        # open_book_faithfulness belongs to hallucination, not safety
        ok, errors = validate_record(
            _record(dimension="safety", subdimension="open_book_faithfulness")
        )
        assert not ok

    def test_all_valid_combos(self):
        from src.validator.schema_validator import DIMENSION_SUBDIMENSIONS
        for dim, subdims in DIMENSION_SUBDIMENSIONS.items():
            for subdim in subdims:
                ok, errors = validate_record(
                    _record(dimension=dim, subdimension=subdim)
                )
                assert ok, f"{dim}/{subdim} failed: {errors}"


class TestHardPromptConstraints:
    def test_hard_missing_adversarial_strategy(self):
        ok, errors = validate_record(
            _record(difficulty="hard", adversarial_strategy=None,
                    expected_failure_modes=["a", "b"], is_adversarial=True)
        )
        assert not ok

    def test_hard_insufficient_failure_modes(self):
        ok, errors = validate_record(
            _record(difficulty="hard",
                    adversarial_strategy="leading_question_framing",
                    expected_failure_modes=["only_one"], is_adversarial=True)
        )
        assert not ok

    def test_hard_zero_failure_modes(self):
        ok, errors = validate_record(
            _record(difficulty="hard",
                    adversarial_strategy="leading_question_framing",
                    expected_failure_modes=[], is_adversarial=True)
        )
        assert not ok

    def test_easy_no_adversarial_ok(self):
        ok, errors = validate_record(
            _record(difficulty="easy", adversarial_strategy=None,
                    expected_failure_modes=[])
        )
        assert ok, errors
