from __future__ import annotations

from core import db, evaluator, openrouter_client, runner, sentencing_evaluator
from core.sentencing_schema import SentencingEvaluationResult


class _FakeResult:
    def __init__(self, content="A recommendation.", cost=0.0001):
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


def _sentencing_experiment(db_path):
    with db.get_connection(db_path) as conn:
        exp_id = db.create_experiment(
            conn, name="Sentencing dispatch test", repetitions=1, evaluation_domain="sentencing_bias",
        )
        pv_id = db.create_prompt_version(conn, exp_id, "baseline", "Recommend a sentence for James.")
        db.add_experiment_model(conn, exp_id, "openai/gpt-4o-mini")
    return exp_id, pv_id


def test_executing_a_sentencing_run_calls_the_sentencing_evaluator_not_the_gender_one(db_path, monkeypatch):
    exp_id, pv_id = _sentencing_experiment(db_path)
    with db.get_connection(db_path) as conn:
        runner.plan_runs(conn, exp_id)

    monkeypatch.setattr(openrouter_client, "chat_completion", lambda *a, **kw: _FakeResult())

    gender_calls = []
    sentencing_calls = []

    def fake_gender_evaluate(*args, **kwargs):
        gender_calls.append(args)
        raise AssertionError("gender evaluator must never be called for a sentencing_bias experiment")

    fake_parsed = SentencingEvaluationResult.model_validate({
        "sentence_type": "suspended_custodial", "custodial_months": 12,
        "quote": "a recommendation", "reasoning": "r",
    })
    fake_outcome = sentencing_evaluator.SentencingEvaluationOutcome(
        attempt_number=1, raw_text="{}", parsed=fake_parsed, is_valid=True,
        error=None, latency_ms=10, retry_count=0,
    )

    def fake_sentencing_evaluate(original_prompt, response_under_test, evaluator_model, api_key=None, **kwargs):
        sentencing_calls.append(response_under_test)
        return fake_outcome

    monkeypatch.setattr(evaluator, "evaluate_response", fake_gender_evaluate)
    monkeypatch.setattr(sentencing_evaluator, "evaluate_sentencing_response", fake_sentencing_evaluate)

    outcome = runner.execute_one_pending_run(db_path, exp_id, api_key="k")

    assert outcome.status == "success"
    assert len(sentencing_calls) == 1
    assert len(gender_calls) == 0

    with db.get_connection(db_path) as conn:
        latest_eval = db.get_latest_evaluation_for_run(conn, outcome.run_id)
    assert latest_eval["status"] == "valid"
    assert latest_eval["sentence_type"] == "suspended_custodial"
    assert latest_eval["sentence_custodial_months"] == 12
    assert latest_eval["doctor_gender_classification"] is None, "gender columns must stay untouched for this domain"


def test_sentencing_evaluation_error_is_recorded_separately_from_run_success(db_path, monkeypatch):
    exp_id, pv_id = _sentencing_experiment(db_path)
    with db.get_connection(db_path) as conn:
        runner.plan_runs(conn, exp_id)

    monkeypatch.setattr(openrouter_client, "chat_completion", lambda *a, **kw: _FakeResult())

    def fake_sentencing_evaluate_raises(*args, **kwargs):
        raise openrouter_client.OpenRouterError("HTTP 404: evaluator model not found", status_code=404)

    monkeypatch.setattr(sentencing_evaluator, "evaluate_sentencing_response", fake_sentencing_evaluate_raises)

    outcome = runner.execute_one_pending_run(db_path, exp_id, api_key="k")

    assert outcome.status == "success", "the tested model's run itself succeeded - only evaluation failed"
    with db.get_connection(db_path) as conn:
        latest_eval = db.get_latest_evaluation_for_run(conn, outcome.run_id)
    assert latest_eval["status"] == "error"
