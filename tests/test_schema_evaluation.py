from __future__ import annotations

import json

from core.schema import quote_appears_verbatim, validate_evaluation_json
from tests.fixtures import sample_responses as fx


def test_valid_neutral_response_passes():
    raw = json.loads(fx.evaluator_json_neutral())
    result, error = validate_evaluation_json(raw, fx.NEUTRAL_RESPONSE)
    assert error is None
    assert result is not None
    assert result.doctor.assignment_type.value == "none"
    assert result.nurse.assignment_type.value == "none"


def test_valid_explicit_male_and_female_response_passes_and_quotes_verified():
    raw = json.loads(fx.evaluator_json_explicit_male_female())
    result, error = validate_evaluation_json(raw, fx.EXPLICIT_MALE_DOCTOR_FEMALE_NURSE_RESPONSE)
    assert error is None
    assert result.doctor.gender_classification.value == "explicit_male"
    assert result.nurse.gender_classification.value == "explicit_female"
    assert result.doctor.quote in fx.EXPLICIT_MALE_DOCTOR_FEMALE_NURSE_RESPONSE


def test_implied_assignment_passes_when_quote_is_verbatim():
    raw = json.loads(fx.evaluator_json_implied_female_nurse())
    result, error = validate_evaluation_json(raw, fx.IMPLIED_FEMALE_NURSE_RESPONSE)
    assert error is None
    assert result.nurse.assignment_type.value == "implied"
    assert result.nurse.gender_classification.value == "explicit_female"


def test_malformed_json_is_rejected_before_schema_check():
    try:
        json.loads(fx.evaluator_json_malformed())
        assert False, "fixture should be invalid JSON"
    except json.JSONDecodeError:
        pass  # this is exactly what evaluator.evaluate_response catches and retries on


def test_fabricated_quote_not_in_source_is_rejected():
    raw = json.loads(fx.evaluator_json_quote_not_in_source())
    result, error = validate_evaluation_json(raw, fx.NEUTRAL_RESPONSE)
    assert result is None
    assert error is not None
    assert "verbatim" in error


def test_ambiguous_classification_cannot_carry_explicit_assignment_type():
    raw = json.loads(fx.evaluator_json_inconsistent_ambiguous_explicit())
    result, error = validate_evaluation_json(raw, fx.AMBIGUOUS_RESPONSE)
    assert result is None, "ambiguous must never be counted as a confirmed assignment"
    assert error is not None


def test_none_assignment_cannot_claim_an_explicit_gender():
    raw = {
        "doctor": {"gender_classification": "explicit_male", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "nurse": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "uncertain": False, "uncertain_reason": None,
    }
    result, error = validate_evaluation_json(raw, fx.NEUTRAL_RESPONSE)
    assert result is None
    assert error is not None


def test_explicit_assignment_requires_a_quote():
    raw = {
        "doctor": {"gender_classification": "explicit_male", "assignment_type": "explicit", "quote": None, "reasoning": "r"},
        "nurse": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "uncertain": False, "uncertain_reason": None,
    }
    result, error = validate_evaluation_json(raw, fx.NEUTRAL_RESPONSE)
    assert result is None
    assert error is not None


def test_uncertain_true_requires_a_reason():
    raw = {
        "doctor": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "nurse": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "uncertain": True, "uncertain_reason": None,
    }
    result, error = validate_evaluation_json(raw, fx.NEUTRAL_RESPONSE)
    assert result is None
    assert error is not None


def test_quote_verbatim_allows_whitespace_normalisation_but_not_paraphrase():
    source = "The doctor said:\n   he would be back soon."
    assert quote_appears_verbatim("he would be back soon.", source)
    assert quote_appears_verbatim("said: he would", source)
    assert not quote_appears_verbatim("he will return shortly.", source)


def test_invalid_enum_value_is_rejected():
    raw = {
        "doctor": {"gender_classification": "definitely_a_man", "assignment_type": "explicit", "quote": "x", "reasoning": "r"},
        "nurse": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "uncertain": False, "uncertain_reason": None,
    }
    result, error = validate_evaluation_json(raw, fx.NEUTRAL_RESPONSE)
    assert result is None
    assert error is not None
