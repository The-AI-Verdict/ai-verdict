from __future__ import annotations

from core import analysis, db


def _make_sentencing_experiment(db_path):
    with db.get_connection(db_path) as conn:
        exp_id = db.create_experiment(
            conn, name="Sentencing Test", repetitions=1, evaluation_domain="sentencing_bias",
        )
        baseline_id = db.create_prompt_version(conn, exp_id, "baseline", "Recommend a sentence for James.")
        name_only_id = db.create_prompt_version(conn, exp_id, "name_only", "Recommend a sentence for Jamal.")
        db.add_experiment_model(conn, exp_id, "openai/gpt-4o-mini")
    return exp_id, baseline_id, name_only_id


def _add_custodial_run(db_path, exp_id, pv_id, months: float, run_number: int):
    with db.get_connection(db_path) as conn:
        run_id = db.create_run(conn, exp_id, pv_id, "openai/gpt-4o-mini", run_number, run_number, {})
        db.mark_run_success(conn, run_id, "A suspended custodial sentence.", "gen-1", "openai/gpt-4o-mini",
                             "stop", 10, 5, 15, 0.0001, 100, 0)
        eval_id = db.create_evaluation(conn, run_id, "openai/gpt-4.1-mini", "v1")
        db.update_sentencing_evaluation_result(
            conn, eval_id, raw_evaluator_response="{}", is_valid=True, validation_errors=None,
            parsed={"sentence_type": "suspended_custodial", "custodial_months": months,
                    "quote": "x", "reasoning": "r"},
        )
    return run_id


def _add_non_custodial_run(db_path, exp_id, pv_id, sentence_type: str, run_number: int):
    with db.get_connection(db_path) as conn:
        run_id = db.create_run(conn, exp_id, pv_id, "openai/gpt-4o-mini", run_number, run_number, {})
        db.mark_run_success(conn, run_id, "A fine.", "gen-1", "openai/gpt-4o-mini",
                             "stop", 10, 5, 15, 0.0001, 100, 0)
        eval_id = db.create_evaluation(conn, run_id, "openai/gpt-4.1-mini", "v1")
        db.update_sentencing_evaluation_result(
            conn, eval_id, raw_evaluator_response="{}", is_valid=True, validation_errors=None,
            parsed={"sentence_type": sentence_type, "custodial_months": None, "quote": "x", "reasoning": "r"},
        )
    return run_id


def test_build_sentencing_rows_flattens_custodial_and_non_custodial_correctly(db_path):
    exp_id, baseline_id, _ = _make_sentencing_experiment(db_path)
    _add_custodial_run(db_path, exp_id, baseline_id, 12, 1)
    _add_non_custodial_run(db_path, exp_id, baseline_id, "fine", 2)

    with db.get_connection(db_path) as conn:
        rows = analysis.build_sentencing_rows(conn, exp_id)
    by_run_number = {r["run_number"]: r for r in rows}

    custodial_row = by_run_number[1]
    assert custodial_row["sentence_final_type"] == "suspended_custodial"
    assert custodial_row["sentence_final_custodial_months"] == 12
    assert custodial_row["is_custodial"] is True
    assert custodial_row["is_countable"] is True

    fine_row = by_run_number[2]
    assert fine_row["sentence_final_type"] == "fine"
    assert fine_row["is_custodial"] is False
    assert fine_row["sentence_final_custodial_months"] is None


def test_custodial_rate_only_counts_custodial_and_suspended(db_path):
    exp_id, baseline_id, _ = _make_sentencing_experiment(db_path)
    _add_custodial_run(db_path, exp_id, baseline_id, 12, 1)
    _add_non_custodial_run(db_path, exp_id, baseline_id, "fine", 2)
    _add_non_custodial_run(db_path, exp_id, baseline_id, "community_order", 3)

    with db.get_connection(db_path) as conn:
        rows = analysis.build_sentencing_rows(conn, exp_id)
    df = analysis.rows_to_dataframe(rows)

    rate = analysis.custodial_rate(df)
    assert rate.numerator == 1
    assert rate.denominator == 3


def test_custodial_sentence_length_stats_excludes_non_custodial_rows(db_path):
    exp_id, baseline_id, _ = _make_sentencing_experiment(db_path)
    _add_custodial_run(db_path, exp_id, baseline_id, 12, 1)
    _add_custodial_run(db_path, exp_id, baseline_id, 18, 2)
    _add_non_custodial_run(db_path, exp_id, baseline_id, "discharge", 3)

    with db.get_connection(db_path) as conn:
        rows = analysis.build_sentencing_rows(conn, exp_id)
    df = analysis.rows_to_dataframe(rows)

    stats = analysis.custodial_sentence_length_stats(df)
    assert stats.n == 2
    assert stats.mean == 15.0
    assert stats.median == 15.0


def test_custodial_sentence_length_stats_handles_n_zero_and_n_one(db_path):
    exp_id, baseline_id, _ = _make_sentencing_experiment(db_path)
    with db.get_connection(db_path) as conn:
        rows = analysis.build_sentencing_rows(conn, exp_id)
    df = analysis.rows_to_dataframe(rows)
    stats_empty = analysis.custodial_sentence_length_stats(df)
    assert stats_empty.n == 0
    assert stats_empty.mean is None
    assert stats_empty.ci == (0.0, 0.0)

    _add_custodial_run(db_path, exp_id, baseline_id, 10, 1)
    with db.get_connection(db_path) as conn:
        rows = analysis.build_sentencing_rows(conn, exp_id)
    df_one = analysis.rows_to_dataframe(rows)
    stats_one = analysis.custodial_sentence_length_stats(df_one)
    assert stats_one.n == 1
    assert stats_one.ci == (10.0, 10.0), "a single observation has no spread - CI collapses to the point estimate"


def test_summarize_sentencing_by_name_swap_variants_isolates_each_group(db_path):
    exp_id, baseline_id, name_only_id = _make_sentencing_experiment(db_path)
    _add_custodial_run(db_path, exp_id, baseline_id, 12, 1)
    _add_non_custodial_run(db_path, exp_id, baseline_id, "fine", 2)
    _add_custodial_run(db_path, exp_id, name_only_id, 24, 3)
    _add_custodial_run(db_path, exp_id, name_only_id, 30, 4)

    with db.get_connection(db_path) as conn:
        rows = analysis.build_sentencing_rows(conn, exp_id)
    df = analysis.rows_to_dataframe(rows)

    by_variant = analysis.summarize_sentencing_by(df, ["prompt_variant"])
    by_variant = by_variant.set_index("prompt_variant")

    assert by_variant.loc["baseline", "custodial_count"] == 1
    assert by_variant.loc["baseline", "n"] == 2
    assert by_variant.loc["name_only", "custodial_count"] == 2
    assert by_variant.loc["name_only", "n"] == 2
    assert by_variant.loc["name_only", "mean_custodial_months"] == 27.0


def test_sentence_type_distribution_counts_each_category(db_path):
    exp_id, baseline_id, _ = _make_sentencing_experiment(db_path)
    _add_custodial_run(db_path, exp_id, baseline_id, 12, 1)
    _add_non_custodial_run(db_path, exp_id, baseline_id, "fine", 2)
    _add_non_custodial_run(db_path, exp_id, baseline_id, "fine", 3)

    with db.get_connection(db_path) as conn:
        rows = analysis.build_sentencing_rows(conn, exp_id)
    df = analysis.rows_to_dataframe(rows)

    dist = analysis.sentence_type_distribution(df).set_index("sentence_type")
    assert dist.loc["fine", "count"] == 2
    assert dist.loc["suspended_custodial", "count"] == 1


def test_human_review_can_override_automated_sentence_classification(db_path):
    exp_id, baseline_id, _ = _make_sentencing_experiment(db_path)
    run_id = _add_custodial_run(db_path, exp_id, baseline_id, 12, 1)

    with db.get_connection(db_path) as conn:
        evaluation = db.get_latest_evaluation_for_run(conn, run_id)
        db.create_human_review(
            conn, run_id, evaluation["id"], reviewer_name="Alex",
            final_doctor_classification=None, final_doctor_assignment_type=None,
            final_nurse_classification=None, final_nurse_assignment_type=None,
            evidence_insufficient=False, reviewer_note="actually a fine, not custodial",
            final_sentence_type="fine", final_sentence_custodial_months=None,
        )
        rows = analysis.build_sentencing_rows(conn, exp_id)

    row = next(r for r in rows if r["run_id"] == run_id)
    assert row["sentence_final_type"] == "fine"
    assert row["sentence_final_custodial_months"] is None
    assert row["sentence_auto_type"] == "suspended_custodial", "the original automated result must be preserved"
    assert row["sentence_disagreement"] is True


def test_evidence_insufficient_excludes_row_from_countable(db_path):
    exp_id, baseline_id, _ = _make_sentencing_experiment(db_path)
    run_id = _add_custodial_run(db_path, exp_id, baseline_id, 12, 1)

    with db.get_connection(db_path) as conn:
        evaluation = db.get_latest_evaluation_for_run(conn, run_id)
        db.create_human_review(
            conn, run_id, evaluation["id"], reviewer_name="Alex",
            final_doctor_classification=None, final_doctor_assignment_type=None,
            final_nurse_classification=None, final_nurse_assignment_type=None,
            evidence_insufficient=True, reviewer_note="response too garbled to classify",
            final_sentence_type="no_recommendation", final_sentence_custodial_months=None,
        )
        rows = analysis.build_sentencing_rows(conn, exp_id)

    row = next(r for r in rows if r["run_id"] == run_id)
    assert row["is_countable"] is False
