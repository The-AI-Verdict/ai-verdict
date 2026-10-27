from __future__ import annotations

import math

from core import analysis, db


def _row(**overrides):
    base = {
        "run_id": 1, "model_id": "openai/gpt-4o-mini", "prompt_variant": "baseline",
        "run_status": "success", "has_evaluation": True, "evaluation_is_valid": True,
        "has_human_review": False, "is_countable": True, "evidence_insufficient": False,
        "doctor_final_classification": "unspecified", "doctor_final_assignment_type": "none",
        "nurse_final_classification": "unspecified", "nurse_final_assignment_type": "none",
        "doctor_disagreement": False, "nurse_disagreement": False,
        "response_model": "openai/gpt-4o-mini", "evaluation_cost_usd": None,
    }
    base.update(overrides)
    return base


def test_cost_summary_sums_known_costs_and_counts_missing_ones():
    rows = [
        _row(run_id=1, cost_usd=0.001),
        _row(run_id=2, cost_usd=0.002),
        _row(run_id=3, cost_usd=None),  # provider didn't report a cost for this one
        _row(run_id=4, run_status="failed", cost_usd=None),
    ]
    df = analysis.rows_to_dataframe(rows)
    summary = analysis.cost_summary(df)
    assert round(summary.total_usd, 3) == 0.003
    assert summary.runs_with_known_cost == 2
    assert summary.runs_missing_cost == 1, "failed runs shouldn't count as 'missing cost' - they never ran"


def test_reproducibility_drift_rows_flags_mismatched_model_strings():
    rows = [
        _row(run_id=1, model_id="openai/gpt-4o-mini", response_model="openai/gpt-4o-mini"),  # exact match
        _row(run_id=2, model_id="openai/gpt-4o-mini", response_model="openai/gpt-4o-mini-2024-07-18"),  # drift
        _row(run_id=3, model_id="openai/gpt-4o-mini", response_model=None),  # no response_model recorded
        _row(run_id=4, model_id="openai/gpt-4o-mini", response_model="openai/gpt-4o-mini-2024-07-18",
             run_status="failed"),  # drift on a failed run shouldn't count
    ]
    df = analysis.rows_to_dataframe(rows)
    drift = analysis.reproducibility_drift_rows(df)
    assert list(drift["run_id"]) == [2]


def test_cost_summary_includes_evaluation_cost_in_the_combined_total():
    """The real bug this guards against: evaluator cost was never tracked or
    summed anywhere, so every 'cost so far' figure silently excluded it and
    undercounted real spend. total_usd must be tested-model cost PLUS
    evaluator cost, not just the tested-model half."""
    rows = [
        _row(run_id=1, cost_usd=0.001, evaluation_cost_usd=0.0005),
        _row(run_id=2, cost_usd=0.002, evaluation_cost_usd=0.0007),
        _row(run_id=3, cost_usd=0.001, has_evaluation=False, evaluation_cost_usd=None),  # no evaluation at all
    ]
    df = analysis.rows_to_dataframe(rows)
    summary = analysis.cost_summary(df)
    assert round(summary.tested_model_cost_usd, 4) == 0.004
    assert round(summary.evaluation_cost_usd, 4) == 0.0012
    assert round(summary.total_usd, 4) == 0.0052
    assert summary.evaluations_with_known_cost == 2


def test_cost_summary_counts_evaluations_with_unrecorded_cost_as_missing():
    """A row created before evaluator cost tracking existed has
    has_evaluation=True but evaluation_cost_usd=None - must be counted as
    'missing', not silently treated as contributing $0 to the total."""
    rows = [
        _row(run_id=1, cost_usd=0.001, has_evaluation=True, evaluation_cost_usd=None),
        _row(run_id=2, cost_usd=0.001, has_evaluation=True, evaluation_cost_usd=0.0003),
    ]
    df = analysis.rows_to_dataframe(rows)
    summary = analysis.cost_summary(df)
    assert summary.evaluations_missing_cost == 1
    assert summary.evaluations_with_known_cost == 1
    assert round(summary.evaluation_cost_usd, 4) == 0.0003


def test_cost_summary_empty_df():
    df = analysis.rows_to_dataframe([])
    summary = analysis.cost_summary(df)
    assert summary.total_usd == 0.0
    assert summary.runs_with_known_cost == 0


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


def test_build_experiment_rows_counts_a_run_after_its_evaluation_is_retried(seeded_experiment, db_path):
    """End-to-end regression test for the real bug: a run whose first
    evaluation attempt errored out, then succeeded on retry, must show up as
    countable - not stuck looking unevaluated forever because an older
    attempt won a tie in the DB query."""
    exp_id = seeded_experiment["experiment_id"]
    with db.get_connection(db_path) as conn:
        run_id = db.create_run(
            conn, exp_id, seeded_experiment["prompt_version_id"], seeded_experiment["model_id"],
            run_number=1, sequence_number=1, request_params={},
        )
        db.mark_run_success(conn, run_id, "A neutral description.", "gen-1", "openai/gpt-4o-mini-2024",
                             "stop", 10, 5, 15, 0.0001, 100, 0)

        failed_eval_id = db.create_evaluation(conn, run_id, "some/dead-model", "v1")
        db.mark_evaluation_error(conn, failed_eval_id, "HTTP 404: model no longer available")

        retried_eval_id = db.create_evaluation(conn, run_id, "openai/gpt-4.1-mini", "v1")
        db.update_evaluation_result(
            conn, retried_eval_id, raw_evaluator_response="{}", is_valid=True, validation_errors=None,
            parsed={
                "doctor": {"gender_classification": "unspecified", "assignment_type": "none",
                           "quote": None, "reasoning": "r"},
                "nurse": {"gender_classification": "unspecified", "assignment_type": "none",
                          "quote": None, "reasoning": "r"},
                "uncertain": False, "uncertain_reason": None,
            },
        )

    with db.get_connection(db_path) as conn:
        rows = analysis.build_experiment_rows(conn, exp_id)
    df = analysis.rows_to_dataframe(rows)

    assert len(df) == 1
    assert bool(df.iloc[0]["is_countable"]) is True
    assert df.iloc[0]["doctor_final_classification"] == "unspecified"


def test_evidence_insufficient_excludes_row_even_with_a_valid_automated_evaluation():
    """A human marking 'evidence insufficient' must override a schema-valid
    automated classification - e.g. a truncated response that technically
    parsed as 'unspecified' isn't genuine evidence of a gender-neutral
    answer, it's evidence of nothing. This is the real-world case that
    motivated the check: reasoning-model responses cut off by max_tokens
    before producing visible text."""
    rows = [_row(run_id=1, has_human_review=True, evidence_insufficient=True, is_countable=False)]
    df = analysis.rows_to_dataframe(rows)
    missing = analysis.missing_and_invalid_counts(df)
    assert missing["evidence_insufficient"] == 1
    summary = analysis.role_summary(df, "doctor")
    assert summary["explicit"].denominator == 0


def test_evidence_insufficient_review_is_not_counted_as_a_disagreement(seeded_experiment, db_path):
    """Regression test: a reviewer correcting an obviously-wrong auto
    classification while ALSO marking the response evidence-insufficient
    (e.g. a truncated response) must not show up in the 'evaluator vs human
    disagreement' count - that count is meant to surface genuine judgement
    calls worth reviewing, not every row a human touched for data-quality
    reasons."""
    exp_id = seeded_experiment["experiment_id"]
    with db.get_connection(db_path) as conn:
        run_id = db.create_run(
            conn, exp_id, seeded_experiment["prompt_version_id"], seeded_experiment["model_id"],
            run_number=1, sequence_number=1, request_params={},
        )
        db.mark_run_success(conn, run_id, "\n*", "gen-1", "openai/gpt-4o-mini-2024",
                             "length", 10, 300, 310, 0.0001, 100, 0)
        eval_id = db.create_evaluation(conn, run_id, "openai/gpt-4.1-mini", "v1")
        db.update_evaluation_result(
            conn, eval_id, raw_evaluator_response="{}", is_valid=True, validation_errors=None,
            parsed={
                "doctor": {"gender_classification": "explicit_male", "assignment_type": "explicit",
                           "quote": "he", "reasoning": "r"},
                "nurse": {"gender_classification": "unspecified", "assignment_type": "none",
                          "quote": None, "reasoning": "r"},
                "uncertain": False, "uncertain_reason": None,
            },
        )
        db.create_human_review(
            conn, run_id, eval_id, "Alex", "unspecified", "none", "unspecified", "none",
            evidence_insufficient=True, reviewer_note="truncated - the 'he' the evaluator found isn't real evidence",
        )

    with db.get_connection(db_path) as conn:
        rows = analysis.build_experiment_rows(conn, exp_id)
    df = analysis.rows_to_dataframe(rows)

    assert bool(df.iloc[0]["doctor_disagreement"]) is False
    assert analysis.disagreement_rows(df).empty


def test_evidence_insufficient_excludes_row_end_to_end(seeded_experiment, db_path):
    exp_id = seeded_experiment["experiment_id"]
    with db.get_connection(db_path) as conn:
        run_id = db.create_run(
            conn, exp_id, seeded_experiment["prompt_version_id"], seeded_experiment["model_id"],
            run_number=1, sequence_number=1, request_params={},
        )
        db.mark_run_success(conn, run_id, "\n*", "gen-1", "openai/gpt-4o-mini-2024",
                             "length", 10, 300, 310, 0.0001, 100, 0)
        eval_id = db.create_evaluation(conn, run_id, "openai/gpt-4.1-mini", "v1")
        db.update_evaluation_result(
            conn, eval_id, raw_evaluator_response="{}", is_valid=True, validation_errors=None,
            parsed={
                "doctor": {"gender_classification": "unspecified", "assignment_type": "none",
                           "quote": None, "reasoning": "no text to evaluate"},
                "nurse": {"gender_classification": "unspecified", "assignment_type": "none",
                          "quote": None, "reasoning": "no text to evaluate"},
                "uncertain": False, "uncertain_reason": None,
            },
        )
        db.create_human_review(
            conn, run_id, eval_id, "Alex", "unspecified", "none", "unspecified", "none",
            evidence_insufficient=True, reviewer_note="response was truncated mid-generation (finish_reason=length)",
        )

    with db.get_connection(db_path) as conn:
        rows = analysis.build_experiment_rows(conn, exp_id)
    df = analysis.rows_to_dataframe(rows)

    assert len(df) == 1
    assert bool(df.iloc[0]["is_countable"]) is False
    missing = analysis.missing_and_invalid_counts(df)
    assert missing["evidence_insufficient"] == 1
    assert missing["unreviewed_total"] == 0, "it WAS reviewed - just marked insufficient, not left unreviewed"
