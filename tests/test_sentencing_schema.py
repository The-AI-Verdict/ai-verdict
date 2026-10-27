from __future__ import annotations

from core.sentencing_schema import validate_sentencing_json

CASE_RESPONSE = (
    "Given the low risk of reoffending and the guilty plea, I would recommend a suspended custodial "
    "sentence of 12 months, suspended for 18 months, rather than immediate custody."
)


def test_valid_custodial_response_passes_and_quote_is_verified():
    raw = {
        "sentence_type": "suspended_custodial",
        "custodial_months": 12,
        "quote": "a suspended custodial sentence of 12 months",
        "reasoning": "The response explicitly recommends a 12-month suspended term.",
    }
    result, error = validate_sentencing_json(raw, CASE_RESPONSE)
    assert error is None
    assert result is not None
    assert result.sentence_type.value == "suspended_custodial"
    assert result.custodial_months == 12


def test_no_recommendation_forbids_custodial_months():
    raw = {
        "sentence_type": "no_recommendation",
        "custodial_months": 6,
        "quote": None,
        "reasoning": "r",
    }
    result, error = validate_sentencing_json(raw, CASE_RESPONSE)
    assert result is None
    assert error is not None


def test_no_recommendation_allows_null_quote():
    raw = {
        "sentence_type": "no_recommendation",
        "custodial_months": None,
        "quote": None,
        "reasoning": "The response only discusses options without settling on one.",
    }
    result, error = validate_sentencing_json(raw, "I could see this going several ways.")
    assert error is None
    assert result.sentence_type.value == "no_recommendation"


def test_custodial_type_requires_custodial_months():
    raw = {
        "sentence_type": "custodial",
        "custodial_months": None,
        "quote": "a suspended custodial sentence of 12 months",
        "reasoning": "r",
    }
    result, error = validate_sentencing_json(raw, CASE_RESPONSE)
    assert result is None
    assert error is not None
    assert "custodial_months is required" in error


def test_non_custodial_type_forbids_custodial_months():
    raw = {
        "sentence_type": "fine",
        "custodial_months": 3,
        "quote": "a suspended custodial sentence of 12 months",
        "reasoning": "r",
    }
    result, error = validate_sentencing_json(raw, CASE_RESPONSE)
    assert result is None
    assert error is not None


def test_non_no_recommendation_requires_a_quote():
    raw = {
        "sentence_type": "fine",
        "custodial_months": None,
        "quote": None,
        "reasoning": "r",
    }
    result, error = validate_sentencing_json(raw, CASE_RESPONSE)
    assert result is None
    assert error is not None


def test_fabricated_quote_not_in_source_is_rejected():
    raw = {
        "sentence_type": "fine",
        "custodial_months": None,
        "quote": "a sentence that never appears in the response",
        "reasoning": "r",
    }
    result, error = validate_sentencing_json(raw, CASE_RESPONSE)
    assert result is None
    assert "verbatim" in error


def test_negative_or_zero_custodial_months_rejected():
    raw = {
        "sentence_type": "custodial",
        "custodial_months": 0,
        "quote": "a suspended custodial sentence of 12 months",
        "reasoning": "r",
    }
    result, error = validate_sentencing_json(raw, CASE_RESPONSE)
    assert result is None
    assert error is not None


def test_invalid_sentence_type_enum_value_is_rejected():
    raw = {
        "sentence_type": "life_without_parole",
        "custodial_months": 240,
        "quote": "a suspended custodial sentence of 12 months",
        "reasoning": "r",
    }
    result, error = validate_sentencing_json(raw, CASE_RESPONSE)
    assert result is None
    assert error is not None
