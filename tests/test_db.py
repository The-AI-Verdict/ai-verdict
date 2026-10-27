from __future__ import annotations

from core import db


def test_prompt_version_locks_and_is_immutable(db_path):
    with db.get_connection(db_path) as conn:
        exp_id = db.create_experiment(conn, name="E", repetitions=1)
        pv_id = db.create_prompt_version(conn, exp_id, "baseline", "Original text")

    with db.get_connection(db_path) as conn:
        pv = db.get_prompt_version(conn, pv_id)
        assert pv["is_locked"] == 0
        db.lock_prompt_version(conn, pv_id)

    with db.get_connection(db_path) as conn:
        pv = db.get_prompt_version(conn, pv_id)
        assert pv["is_locked"] == 1
        assert pv["prompt_text"] == "Original text"


def test_editing_a_locked_prompt_creates_new_version_without_mutating_old(db_path):
    with db.get_connection(db_path) as conn:
        exp_id = db.create_experiment(conn, name="E", repetitions=1)
        pv_id = db.create_prompt_version(conn, exp_id, "baseline", "v1 text")
        db.lock_prompt_version(conn, pv_id)
        new_id = db.edit_prompt_version(conn, pv_id, "v2 text", notes="fixed a typo")

    with db.get_connection(db_path) as conn:
        old = db.get_prompt_version(conn, pv_id)
        new = db.get_prompt_version(conn, new_id)
        assert old["prompt_text"] == "v1 text", "old version must never be mutated"
        assert old["superseded_by"] == new_id
        assert new["prompt_text"] == "v2 text"
        assert new["version_number"] == old["version_number"] + 1
        assert new["is_locked"] == 0


def test_run_persists_success_fields_exactly(seeded_experiment, db_path):
    with db.get_connection(db_path) as conn:
        run_id = db.create_run(
            conn, seeded_experiment["experiment_id"], seeded_experiment["prompt_version_id"],
            seeded_experiment["model_id"], run_number=1, sequence_number=1,
            request_params={"temperature": 0.7},
        )
        db.mark_run_success(
            conn, run_id, raw_response="The scene unfolds.", response_id="gen-123",
            response_model="openai/gpt-4o-mini-2024", finish_reason="stop",
            prompt_tokens=50, completion_tokens=20, total_tokens=70, cost_usd=0.0012,
            latency_ms=842, retry_count=1,
        )

    with db.get_connection(db_path) as conn:
        run = db.get_run(conn, run_id)
    assert run["status"] == "success"
    assert run["raw_response"] == "The scene unfolds."
    assert run["response_model"] == "openai/gpt-4o-mini-2024"
    assert run["prompt_tokens"] == 50
    assert run["total_tokens"] == 70
    assert run["cost_usd"] == 0.0012
    assert run["retry_count"] == 1
    assert run["error_message"] is None


def test_failed_run_is_distinguished_from_success_and_never_has_a_response(seeded_experiment, db_path):
    with db.get_connection(db_path) as conn:
        run_id = db.create_run(
            conn, seeded_experiment["experiment_id"], seeded_experiment["prompt_version_id"],
            seeded_experiment["model_id"], run_number=1, sequence_number=1, request_params={},
        )
        db.mark_run_failed(conn, run_id, error_message="HTTP 402: insufficient credits",
                            http_status=402, latency_ms=100, retry_count=0)

    with db.get_connection(db_path) as conn:
        run = db.get_run(conn, run_id)
    assert run["status"] == "failed"
    assert run["raw_response"] is None
    assert run["error_message"] == "HTTP 402: insufficient credits"
    assert run["http_status"] == 402


def test_human_review_revision_preserves_automated_evaluation(seeded_experiment, db_path):
    with db.get_connection(db_path) as conn:
        run_id = db.create_run(
            conn, seeded_experiment["experiment_id"], seeded_experiment["prompt_version_id"],
            seeded_experiment["model_id"], run_number=1, sequence_number=1, request_params={},
        )
        eval_id = db.create_evaluation(conn, run_id, "anthropic/claude-3.5-haiku", "v1")
        db.update_evaluation_result(
            conn, eval_id, raw_evaluator_response="{}", is_valid=True, validation_errors=None,
            parsed={
                "doctor": {"gender_classification": "explicit_male", "assignment_type": "explicit",
                           "quote": "his", "reasoning": "r"},
                "nurse": {"gender_classification": "unspecified", "assignment_type": "none",
                          "quote": None, "reasoning": "r"},
                "uncertain": False, "uncertain_reason": None,
            },
        )
        db.create_human_review(
            conn, run_id, eval_id, reviewer_name="Alex",
            final_doctor_classification="unspecified", final_doctor_assignment_type="none",
            final_nurse_classification="unspecified", final_nurse_assignment_type="none",
            evidence_insufficient=False, reviewer_note="model misread, no pronoun actually present",
        )

    with db.get_connection(db_path) as conn:
        evaluation = db.get_latest_evaluation_for_run(conn, run_id)
        review = db.get_latest_human_review(conn, run_id)

    assert evaluation["doctor_gender_classification"] == "explicit_male", "automated result must be preserved"
    assert review["final_doctor_classification"] == "unspecified"
    assert review["is_latest"] == 1


def test_second_human_review_supersedes_first_but_keeps_history(seeded_experiment, db_path):
    with db.get_connection(db_path) as conn:
        run_id = db.create_run(
            conn, seeded_experiment["experiment_id"], seeded_experiment["prompt_version_id"],
            seeded_experiment["model_id"], run_number=1, sequence_number=1, request_params={},
        )
        db.create_human_review(conn, run_id, None, "Alex", "unspecified", "none", "unspecified", "none",
                                False, "first pass")
        db.create_human_review(conn, run_id, None, "Jordan", "explicit_female", "explicit", "unspecified",
                                "none", False, "actually there is a 'she' here")

    with db.get_connection(db_path) as conn:
        history = db.list_human_review_history(conn, run_id)
        latest = db.get_latest_human_review(conn, run_id)

    assert len(history) == 2
    assert history[0]["reviewer_name"] == "Alex" and history[0]["is_latest"] == 0
    assert latest["reviewer_name"] == "Jordan" and latest["is_latest"] == 1


def test_list_evaluations_for_experiment_picks_newest_attempt_not_oldest(seeded_experiment, db_path):
    """Regression test: in production, every evaluation retry is recorded
    with attempt_number=1 (nothing increments it), so without an id DESC
    tiebreaker this query's ORDER BY has no reliable way to put the newest
    attempt first - and analysis.build_experiment_rows keeps the FIRST row
    it sees per run_id. This caused 150 real, successfully re-evaluated runs
    to still show up as uncountable because the stale, errored first
    attempt kept winning over the valid retry."""
    with db.get_connection(db_path) as conn:
        run_id = db.create_run(
            conn, seeded_experiment["experiment_id"], seeded_experiment["prompt_version_id"],
            seeded_experiment["model_id"], run_number=1, sequence_number=1, request_params={},
        )
        first_id = db.create_evaluation(conn, run_id, "some/dead-model", "v1")
        db.mark_evaluation_error(conn, first_id, "HTTP 404: model no longer available")

        second_id = db.create_evaluation(conn, run_id, "openai/gpt-4.1-mini", "v1")
        db.update_evaluation_result(
            conn, second_id, raw_evaluator_response="{}", is_valid=True, validation_errors=None,
            parsed={
                "doctor": {"gender_classification": "unspecified", "assignment_type": "none",
                           "quote": None, "reasoning": "r"},
                "nurse": {"gender_classification": "unspecified", "assignment_type": "none",
                          "quote": None, "reasoning": "r"},
                "uncertain": False, "uncertain_reason": None,
            },
        )

    with db.get_connection(db_path) as conn:
        evaluations = db.list_evaluations_for_experiment(conn, seeded_experiment["experiment_id"])

    assert len(evaluations) == 2, "both attempts must still exist - retries never overwrite history"
    assert evaluations[0]["id"] == second_id, "the newest attempt must sort first, not the stale error"
    assert evaluations[0]["status"] == "valid"


def test_get_latest_evaluations_by_run_matches_per_run_lookup(seeded_experiment, db_path):
    exp_id = seeded_experiment["experiment_id"]
    with db.get_connection(db_path) as conn:
        run_ids = []
        for i in range(2):
            run_id = db.create_run(
                conn, exp_id, seeded_experiment["prompt_version_id"], seeded_experiment["model_id"],
                run_number=i + 1, sequence_number=i + 1, request_params={},
            )
            run_ids.append(run_id)
            # First run gets a stale errored attempt then a valid retry;
            # second run gets no evaluation at all.
            if i == 0:
                old_id = db.create_evaluation(conn, run_id, "dead/model", "v1")
                db.mark_evaluation_error(conn, old_id, "gone")
                new_id = db.create_evaluation(conn, run_id, "openai/gpt-4.1-mini", "v1")
                db.update_evaluation_result(conn, new_id, "{}", True, None, {
                    "doctor": {"gender_classification": "unspecified", "assignment_type": "none",
                               "quote": None, "reasoning": "r"},
                    "nurse": {"gender_classification": "unspecified", "assignment_type": "none",
                              "quote": None, "reasoning": "r"},
                    "uncertain": False, "uncertain_reason": None,
                })

    with db.get_connection(db_path) as conn:
        batched = db.get_latest_evaluations_by_run(conn, exp_id)
        per_run = {rid: db.get_latest_evaluation_for_run(conn, rid) for rid in run_ids}

    assert batched[run_ids[0]]["status"] == per_run[run_ids[0]]["status"] == "valid"
    assert run_ids[1] not in batched
    assert per_run[run_ids[1]] is None


def test_create_experiment_defaults_to_gender_healthcare_domain(db_path):
    with db.get_connection(db_path) as conn:
        exp_id = db.create_experiment(conn, name="No domain specified")
        experiment = db.get_experiment(conn, exp_id)
    assert experiment["evaluation_domain"] == "gender_healthcare"


def test_create_experiment_persists_sentencing_domain(db_path):
    with db.get_connection(db_path) as conn:
        exp_id = db.create_experiment(conn, name="Sentencing", evaluation_domain="sentencing_bias")
        experiment = db.get_experiment(conn, exp_id)
    assert experiment["evaluation_domain"] == "sentencing_bias"


def test_update_sentencing_evaluation_result_round_trips_all_fields(db_path):
    with db.get_connection(db_path) as conn:
        exp_id = db.create_experiment(conn, name="S", evaluation_domain="sentencing_bias")
        pv_id = db.create_prompt_version(conn, exp_id, "baseline", "Recommend a sentence.")
        run_id = db.create_run(conn, exp_id, pv_id, "openai/gpt-4o-mini", 1, 1, {})
        eval_id = db.create_evaluation(conn, run_id, "openai/gpt-4.1-mini", "v1")
        db.update_sentencing_evaluation_result(
            conn, eval_id, raw_evaluator_response="{}", is_valid=True, validation_errors=None,
            parsed={"sentence_type": "custodial", "custodial_months": 24, "quote": "x", "reasoning": "r"},
        )

    with db.get_connection(db_path) as conn:
        ev = db.get_latest_evaluation_for_run(conn, run_id)
    assert ev["status"] == "valid"
    assert ev["sentence_type"] == "custodial"
    assert ev["sentence_custodial_months"] == 24
    assert ev["sentence_quote"] == "x"
    assert ev["sentence_no_recommendation"] == 0


def test_update_sentencing_evaluation_result_sets_no_recommendation_flag(db_path):
    with db.get_connection(db_path) as conn:
        exp_id = db.create_experiment(conn, name="S", evaluation_domain="sentencing_bias")
        pv_id = db.create_prompt_version(conn, exp_id, "baseline", "Recommend a sentence.")
        run_id = db.create_run(conn, exp_id, pv_id, "openai/gpt-4o-mini", 1, 1, {})
        eval_id = db.create_evaluation(conn, run_id, "openai/gpt-4.1-mini", "v1")
        db.update_sentencing_evaluation_result(
            conn, eval_id, raw_evaluator_response="{}", is_valid=True, validation_errors=None,
            parsed={"sentence_type": "no_recommendation", "custodial_months": None, "quote": None, "reasoning": "r"},
        )

    with db.get_connection(db_path) as conn:
        ev = db.get_latest_evaluation_for_run(conn, run_id)
    assert ev["sentence_no_recommendation"] == 1


def test_create_human_review_persists_sentencing_fields_without_touching_gender_columns(db_path):
    with db.get_connection(db_path) as conn:
        exp_id = db.create_experiment(conn, name="S", evaluation_domain="sentencing_bias")
        pv_id = db.create_prompt_version(conn, exp_id, "baseline", "Recommend a sentence.")
        run_id = db.create_run(conn, exp_id, pv_id, "openai/gpt-4o-mini", 1, 1, {})
        db.create_human_review(
            conn, run_id, None, "Alex",
            None, None, None, None, False, "overturned to a fine",
            final_sentence_type="fine", final_sentence_custodial_months=None,
        )

    with db.get_connection(db_path) as conn:
        review = db.get_latest_human_review(conn, run_id)
    assert review["final_sentence_type"] == "fine"
    assert review["final_sentence_custodial_months"] is None
    assert review["final_doctor_classification"] is None


def test_get_latest_human_reviews_by_run_reflects_most_recent_revision(seeded_experiment, db_path):
    exp_id = seeded_experiment["experiment_id"]
    with db.get_connection(db_path) as conn:
        run_id = db.create_run(
            conn, exp_id, seeded_experiment["prompt_version_id"], seeded_experiment["model_id"],
            run_number=1, sequence_number=1, request_params={},
        )
        db.create_human_review(conn, run_id, None, "Alex", "unspecified", "none", "unspecified", "none",
                                False, "first pass")
        db.create_human_review(conn, run_id, None, "Jordan", "explicit_female", "explicit", "unspecified",
                                "none", False, "corrected")

    with db.get_connection(db_path) as conn:
        batched = db.get_latest_human_reviews_by_run(conn, exp_id)

    assert batched[run_id]["reviewer_name"] == "Jordan"
