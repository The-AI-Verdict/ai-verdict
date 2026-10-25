from __future__ import annotations

import math

from core import analysis


def _row(**overrides):
    base = {
        "run_id": 1, "model_id": "openai/gpt-4o-mini", "prompt_variant": "baseline",
        "run_status": "success", "has_evaluation": True, "evaluation_is_valid": True,
        "has_human_review": False, "is_countable": True,
        "doctor_final_classification": "unspecified", "doctor_final_assignment_type": "none",
        "nurse_final_classification": "unspecified", "nurse_final_assignment_type": "none",
        "doctor_disagreement": False, "nurse_disagreement": False,
    }
    base.update(overrides)
    return base


def test_wilson_ci_zero_n_returns_zero_zero():
    assert analysis.wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_bounds_contain_point_estimate_region():
    lo, hi = analysis.wilson_ci(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0


def test_wilson_ci_narrows_as_n_grows():
    lo_small, hi_small = analysis.wilson_ci(50, 100)
    lo_big, hi_big = analysis.wilson_ci(5000, 10000)
    assert (hi_small - lo_small) > (hi_big - lo_big)


def test_role_summary_denominator_excludes_failed_and_unresolved_invalid_rows():
    rows = [
        _row(run_id=1),  # countable, unspecified/none for both
        _row(run_id=2, run_status="failed", is_countable=False),  # excluded: API failure
        _row(run_id=3, evaluation_is_valid=False, has_human_review=False, is_countable=False),  # excluded: invalid + unreviewed
        _row(run_id=4, doctor_final_classification="explicit_male", doctor_final_assignment_type="explicit"),
    ]
    df = analysis.rows_to_dataframe(rows)
    summary = analysis.role_summary(df, "doctor")

    assert summary["explicit"].denominator == 2, "only the 2 countable rows should form the denominator"
    assert summary["explicit"].numerator == 1
    assert summary["explicit"].proportion == 0.5


def test_role_summary_keeps_explicit_and_implied_separate():
    rows = [
        _row(run_id=1, doctor_final_classification="explicit_male", doctor_final_assignment_type="explicit"),
        _row(run_id=2, doctor_final_classification="explicit_female", doctor_final_assignment_type="implied"),
        _row(run_id=3),
    ]
    df = analysis.rows_to_dataframe(rows)
    summary = analysis.role_summary(df, "doctor")

    assert summary["explicit"].numerator == 1
    assert summary["implied"].numerator == 1
    assert summary["any_assignment"].numerator == 2, "any_assignment must equal explicit + implied, never silently different"


def test_ambiguous_rows_are_not_counted_as_an_assignment():
    rows = [_row(run_id=1, doctor_final_classification="ambiguous", doctor_final_assignment_type="none")]
    df = analysis.rows_to_dataframe(rows)
    summary = analysis.role_summary(df, "doctor")

    assert summary["explicit"].numerator == 0
    assert summary["implied"].numerator == 0
    assert summary["ambiguous"].numerator == 1


def test_small_sample_is_flagged_exploratory():
    rows = [_row(run_id=i) for i in range(5)]
    df = analysis.rows_to_dataframe(rows)
    summary = analysis.role_summary(df, "doctor")
    assert summary["explicit"].is_exploratory
    assert summary["explicit"].denominator == 5 < analysis.SMALL_SAMPLE_THRESHOLD


def test_missing_and_invalid_counts():
    rows = [
        _row(run_id=1),
        _row(run_id=2, run_status="failed", is_countable=False),
        _row(run_id=3, evaluation_is_valid=False, has_human_review=False, is_countable=False),
        _row(run_id=4, has_human_review=False),
    ]
    df = analysis.rows_to_dataframe(rows)
    missing = analysis.missing_and_invalid_counts(df)
    assert missing["failed_runs"] == 1
    assert missing["invalid_unreviewed"] == 1
    assert missing["unreviewed_total"] == 3  # rows 1, 3, 4 - row 2 is failed, so it can never be reviewed


def test_disagreement_rows_only_includes_flagged_rows():
    rows = [
        _row(run_id=1, has_human_review=True, doctor_disagreement=True),
        _row(run_id=2, has_human_review=True, doctor_disagreement=False, nurse_disagreement=False),
    ]
    df = analysis.rows_to_dataframe(rows)
    disagreements = analysis.disagreement_rows(df)
    assert len(disagreements) == 1
    assert disagreements.iloc[0]["run_id"] == 1


def test_summarize_by_groups_correctly_by_model():
    rows = [
        _row(run_id=1, model_id="model-a", doctor_final_classification="explicit_male", doctor_final_assignment_type="explicit"),
        _row(run_id=2, model_id="model-a"),
        _row(run_id=3, model_id="model-b"),
    ]
    df = analysis.rows_to_dataframe(rows)
    by_model = analysis.summarize_by(df, "doctor", ["model_id"])
    a = by_model[by_model["model_id"] == "model-a"].iloc[0]
    b = by_model[by_model["model_id"] == "model-b"].iloc[0]
    assert a["explicit_n"] == 2 and a["explicit_count"] == 1
    assert b["explicit_n"] == 1 and b["explicit_count"] == 0
