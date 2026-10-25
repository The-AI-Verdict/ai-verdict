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
