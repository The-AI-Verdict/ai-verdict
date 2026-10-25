from __future__ import annotations

from core import evaluator, openrouter_client
from tests.fixtures import sample_responses as fx


class _FakeResult:
    def __init__(self, content: str):
        self.content = content
        self.latency_ms = 10
        self.retry_count = 0


def test_evaluate_response_valid_on_first_attempt(monkeypatch):
    calls = []

    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        calls.append(messages)
        return _FakeResult(fx.evaluator_json_explicit_male_female())

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    outcome = evaluator.evaluate_response(
        original_prompt="Describe the scene.",
        response_under_test=fx.EXPLICIT_MALE_DOCTOR_FEMALE_NURSE_RESPONSE,
        evaluator_model="anthropic/claude-3.5-haiku",
        api_key="k",
    )

    assert outcome.is_valid
    assert outcome.attempt_number == 1
    assert len(calls) == 1
    system_message = calls[0][0]["content"]
    assert "doctor" in system_message and "nurse" in system_message
    user_message = calls[0][1]["content"]
    assert "ORIGINAL_PROMPT" in user_message and "ORIGINAL_RESPONSE" in user_message


def test_evaluate_response_repairs_after_malformed_json(monkeypatch):
    attempts = {"n": 0}

    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _FakeResult(fx.evaluator_json_malformed())
        return _FakeResult(fx.evaluator_json_neutral())

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    outcome = evaluator.evaluate_response(
        original_prompt="Describe the scene.", response_under_test=fx.NEUTRAL_RESPONSE,
        evaluator_model="anthropic/claude-3.5-haiku", api_key="k", max_repair_attempts=1,
    )

    assert attempts["n"] == 2
    assert outcome.is_valid
    assert outcome.attempt_number == 2


def test_evaluate_response_gives_up_after_exhausting_repair_attempts_for_human_review(monkeypatch):
    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        return _FakeResult(fx.evaluator_json_quote_not_in_source())

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    outcome = evaluator.evaluate_response(
        original_prompt="Describe the scene.", response_under_test=fx.NEUTRAL_RESPONSE,
        evaluator_model="anthropic/claude-3.5-haiku", api_key="k", max_repair_attempts=1,
    )

    assert not outcome.is_valid, "invalid output must be flagged for human review, never silently accepted"
    assert outcome.error is not None
    assert outcome.parsed is None


def test_evaluator_strips_markdown_code_fences(monkeypatch):
    fenced = "```json\n" + fx.evaluator_json_neutral() + "\n```"

    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        return _FakeResult(fenced)

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    outcome = evaluator.evaluate_response(
        original_prompt="Describe the scene.", response_under_test=fx.NEUTRAL_RESPONSE,
        evaluator_model="anthropic/claude-3.5-haiku", api_key="k",
    )
    assert outcome.is_valid


def test_default_evaluator_model_falls_back_to_env(monkeypatch):
    monkeypatch.delenv("AI_VERDICT_EVALUATOR_MODEL", raising=False)
    assert evaluator.default_evaluator_model() == "anthropic/claude-3.5-haiku"
    monkeypatch.setenv("AI_VERDICT_EVALUATOR_MODEL", "some/other-model")
    assert evaluator.default_evaluator_model() == "some/other-model"
