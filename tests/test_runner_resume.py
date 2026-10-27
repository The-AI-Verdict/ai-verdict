from __future__ import annotations

import json

from core import db, evaluator, openrouter_client, runner, schema
from tests.fixtures import sample_responses as fx


class _FakeResult:
    def __init__(self, content="A neutral description.", cost=0.0001):
        self.content = content
        self.response_id = "gen-x"
        self.response_model = "openai/gpt-4o-mini-2024"
        self.finish_reason = "stop"
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.total_tokens = 15
        self.cost_usd = cost
        self.latency_ms = 100
        self.retry_count = 0


def test_plan_runs_is_idempotent(seeded_experiment, db_path):
    with db.get_connection(db_path) as conn:
        first = runner.plan_runs(conn, seeded_experiment["experiment_id"])
    with db.get_connection(db_path) as conn:
        second = runner.plan_runs(conn, seeded_experiment["experiment_id"])
        all_runs = db.list_runs(conn, seeded_experiment["experiment_id"])

    assert len(first) == 2  # repetitions=2 in the fixture
    assert len(second) == 0, "re-planning must not create duplicate runs for an already-planned combination"
    assert len(all_runs) == 2


def test_plan_runs_locks_prompt_versions(seeded_experiment, db_path):
    with db.get_connection(db_path) as conn:
        runner.plan_runs(conn, seeded_experiment["experiment_id"])
        pv = db.get_prompt_version(conn, seeded_experiment["prompt_version_id"])
    assert pv["is_locked"] == 1


def test_resume_never_re_executes_an_already_completed_run(seeded_experiment, db_path, monkeypatch):
    call_log = []

    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        call_log.append(model)
        return _FakeResult()

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    with db.get_connection(db_path) as conn:
        runner.plan_runs(conn, seeded_experiment["experiment_id"])

    exp_id = seeded_experiment["experiment_id"]

    # "Run" one step, simulating the UI loop - this is run #1 of 2.
    outcome1 = runner.execute_one_pending_run(db_path, exp_id, api_key="k", run_evaluation=False)
    assert outcome1 is not None and outcome1.status == "success"
    assert len(call_log) == 1

    # Simulate pausing here (just stop calling), then "resuming" later.
    outcome2 = runner.execute_one_pending_run(db_path, exp_id, api_key="k", run_evaluation=False)
    assert outcome2 is not None and outcome2.status == "success"
    assert len(call_log) == 2

    # Nothing left - resuming again must be a safe no-op, not a re-send.
    outcome3 = runner.execute_one_pending_run(db_path, exp_id, api_key="k", run_evaluation=False)
    assert outcome3 is None
    assert len(call_log) == 2, "must not call the API again once every planned run has completed"

    with db.get_connection(db_path) as conn:
        experiment = db.get_experiment(conn, exp_id)
    assert experiment["status"] == "completed"


def test_crash_recovery_requeues_a_stale_running_row_exactly_once(seeded_experiment, db_path, monkeypatch):
    call_log = []

    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        call_log.append(model)
        return _FakeResult()

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    exp_id = seeded_experiment["experiment_id"]
    with db.get_connection(db_path) as conn:
        run_ids = runner.plan_runs(conn, exp_id)
        # Simulate a crash: the process marked this run 'running' and died
        # before a response was ever received, so raw_response stays NULL.
        db.mark_run_running(conn, run_ids[0])

    outcome = runner.execute_one_pending_run(db_path, exp_id, api_key="k", run_evaluation=False)

    assert outcome is not None
    assert outcome.run_id == run_ids[0]
    assert outcome.status == "success"
    assert len(call_log) == 1, "the crashed run must be re-sent exactly once, not skipped or duplicated"

    with db.get_connection(db_path) as conn:
        run = db.get_run(conn, run_ids[0])
    assert run["status"] == "success"


def test_failed_api_call_does_not_block_remaining_runs_and_is_recorded_separately(seeded_experiment, db_path, monkeypatch):
    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        raise openrouter_client.OpenRouterError("HTTP 400: bad request", status_code=400, retryable=False)

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    exp_id = seeded_experiment["experiment_id"]
    with db.get_connection(db_path) as conn:
        runner.plan_runs(conn, exp_id)

    outcome = runner.execute_one_pending_run(db_path, exp_id, api_key="k", run_evaluation=False)
    assert outcome.status == "failed"

    with db.get_connection(db_path) as conn:
        run = db.get_run(conn, outcome.run_id)
    assert run["status"] == "failed"
    assert run["raw_response"] is None
    assert run["error_message"] is not None


def test_retry_failed_runs_creates_fresh_attempts_without_touching_the_original(seeded_experiment, db_path, monkeypatch):
    call_count = {"n": 0}

    def fake_chat_completion_always_fails(model, messages, api_key=None, **kwargs):
        call_count["n"] += 1
        raise openrouter_client.OpenRouterError("HTTP 503: overloaded", status_code=503, retryable=False)

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion_always_fails)

    exp_id = seeded_experiment["experiment_id"]
    with db.get_connection(db_path) as conn:
        runner.plan_runs(conn, exp_id)  # repetitions=2 in the fixture

    # Run both to failure.
    runner.execute_one_pending_run(db_path, exp_id, api_key="k", run_evaluation=False)
    runner.execute_one_pending_run(db_path, exp_id, api_key="k", run_evaluation=False)
    assert call_count["n"] == 2

    with db.get_connection(db_path) as conn:
        original_failed = [r for r in db.list_runs(conn, exp_id) if r["status"] == "failed"]
        assert len(original_failed) == 2
        original_ids = {r["id"] for r in original_failed}

        created = runner.retry_failed_runs(conn, exp_id)
        assert len(created) == 2
        assert not (set(created) & original_ids), "retry must create new rows, never reuse a failed row's id"

        # Original failed rows must be completely untouched.
        for r in original_failed:
            still_there = db.get_run(conn, r["id"])
            assert still_there["status"] == "failed"
            assert still_there["error_message"] == r["error_message"]

        pending_now = db.get_pending_runs(conn, exp_id)
        assert len(pending_now) == 2
        assert {r["id"] for r in pending_now} == set(created)

    # Calling again with nothing failed must be a no-op, not re-retry the retries.
    with db.get_connection(db_path) as conn:
        again = runner.retry_failed_runs(conn, exp_id)
    assert again == []


def test_compute_unsupported_params_flags_only_params_the_model_lacks():
    experiment = {"temperature": 0.7, "max_tokens": 300, "top_p": None, "frequency_penalty": None,
                  "presence_penalty": None, "seed": 42}
    cached_model = {"supported_parameters_json": '["temperature", "max_tokens"]'}
    unsupported = runner.compute_unsupported_params(experiment, cached_model)
    assert unsupported == ["seed"]


def test_stuck_pending_evaluation_is_recovered_before_new_runs_are_started(seeded_experiment, db_path, monkeypatch):
    exp_id = seeded_experiment["experiment_id"]
    with db.get_connection(db_path) as conn:
        run_ids = runner.plan_runs(conn, exp_id)
        # Simulate: the run succeeded and was persisted, create_evaluation was
        # called (status='pending'), then the process was killed before
        # evaluate_response ever returned.
        db.mark_run_success(conn, run_ids[0], fx.NEUTRAL_RESPONSE, "gen-1", "openai/gpt-4o-mini-2024",
                             "stop", 10, 5, 15, 0.0001, 100, 0)
        db.create_evaluation(conn, run_ids[0], "anthropic/claude-3.5-haiku", evaluator.EVALUATOR_PROMPT_VERSION)

    parsed = schema.EvaluationResult.model_validate(json.loads(fx.evaluator_json_neutral()))
    fake_outcome = evaluator.EvaluationOutcome(
        attempt_number=1, raw_text=fx.evaluator_json_neutral(), parsed=parsed, is_valid=True,
        error=None, latency_ms=10, retry_count=0,
    )
    eval_calls = []

    def fake_evaluate_response(original_prompt, response_under_test, evaluator_model, api_key=None, **kwargs):
        eval_calls.append(response_under_test)
        return fake_outcome

    monkeypatch.setattr(evaluator, "evaluate_response", fake_evaluate_response)

    run_calls = []

    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        run_calls.append(model)
        return _FakeResult()

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    # First call must recover the stuck evaluation, NOT start executing run #2.
    outcome = runner.execute_one_pending_run(db_path, exp_id, api_key="k")

    assert outcome.run_id == run_ids[0]
    assert len(eval_calls) == 1, "the stuck evaluation must be retried exactly once"
    assert len(run_calls) == 0, "no new run should be sent while a stuck evaluation is still unresolved"

    with db.get_connection(db_path) as conn:
        resolved_eval = db.get_latest_evaluation_for_run(conn, run_ids[0])
    assert resolved_eval["status"] == "valid"

    # Second call now proceeds to the genuinely pending run #2 as normal.
    outcome2 = runner.execute_one_pending_run(db_path, exp_id, api_key="k")
    assert outcome2.run_id == run_ids[1]
    assert len(run_calls) == 1


def test_compute_unsupported_params_returns_empty_when_catalog_unknown():
    experiment = {"temperature": 0.7, "max_tokens": None, "top_p": None, "frequency_penalty": None,
                  "presence_penalty": None, "seed": None}
    assert runner.compute_unsupported_params(experiment, None) == []


def test_retry_errored_evaluations_recovers_from_a_bad_evaluator_model_id(seeded_experiment, db_path, monkeypatch):
    """Mirrors what actually happened in production: the evaluator model id
    was deprecated by the provider, so every evaluation call errored out.
    Fixing the model id and retrying should recover all of them without
    re-calling the tested model (the raw_response is already stored)."""
    exp_id = seeded_experiment["experiment_id"]
    with db.get_connection(db_path) as conn:
        run_ids = runner.plan_runs(conn, exp_id)
        for rid in run_ids:
            db.mark_run_success(conn, rid, fx.NEUTRAL_RESPONSE, "gen-1", "openai/gpt-4o-mini-2024",
                                 "stop", 10, 5, 15, 0.0001, 100, 0)
            eval_id = db.create_evaluation(conn, rid, "anthropic/claude-3.5-haiku",
                                            evaluator.EVALUATOR_PROMPT_VERSION)
            db.mark_evaluation_error(conn, eval_id,
                                      "HTTP 404: anthropic/claude-3.5-haiku is no longer available")

    assert len(run_ids) == 2
    with db.get_connection(db_path) as conn:
        assert runner.count_errored_evaluations(conn, exp_id) == 2

    parsed = schema.EvaluationResult.model_validate(json.loads(fx.evaluator_json_neutral()))
    fake_outcome = evaluator.EvaluationOutcome(
        attempt_number=1, raw_text=fx.evaluator_json_neutral(), parsed=parsed, is_valid=True,
        error=None, latency_ms=10, retry_count=0,
    )
    eval_calls = []
    tested_model_calls = []

    def fake_evaluate_response(original_prompt, response_under_test, evaluator_model, api_key=None, **kwargs):
        eval_calls.append(evaluator_model)
        return fake_outcome

    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        tested_model_calls.append(model)
        return _FakeResult()

    monkeypatch.setattr(evaluator, "evaluate_response", fake_evaluate_response)
    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    retried = runner.retry_errored_evaluations(db_path, exp_id, api_key="k")

    assert retried == 2
    assert len(eval_calls) == 2
    assert len(tested_model_calls) == 0, "retrying evaluation must never re-call the tested model"

    with db.get_connection(db_path) as conn:
        assert runner.count_errored_evaluations(conn, exp_id) == 0
        for rid in run_ids:
            latest = db.get_latest_evaluation_for_run(conn, rid)
            assert latest["status"] == "valid"
