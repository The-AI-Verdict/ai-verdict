from __future__ import annotations

import pytest

from core import evaluator, openrouter_client
from tests.fixtures import sample_responses as fx


class _FakeResult:
    def __init__(self, content: str, finish_reason: str = "stop", cost_usd: float | None = 0.0005):
        self.content = content
        self.latency_ms = 10
        self.retry_count = 0
        self.finish_reason = finish_reason
        self.cost_usd = cost_usd


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


def test_evaluate_response_sums_cost_across_repair_attempts(monkeypatch):
    """Each repair attempt is a real, separately billed completion - the
    outcome's cost_usd must be the sum of every attempt made, not just the
    final one, or a retried evaluation would silently undercount its cost."""
    attempts = {"n": 0}

    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _FakeResult(fx.evaluator_json_malformed(), cost_usd=0.001)
        return _FakeResult(fx.evaluator_json_neutral(), cost_usd=0.002)

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    outcome = evaluator.evaluate_response(
        original_prompt="Describe the scene.", response_under_test=fx.NEUTRAL_RESPONSE,
        evaluator_model="anthropic/claude-3.5-haiku", api_key="k", max_repair_attempts=1,
    )

    assert outcome.is_valid
    assert outcome.cost_usd == pytest.approx(0.003)


def test_evaluate_response_cost_is_none_when_any_attempt_cost_is_unreported(monkeypatch):
    """If even one attempt's cost wasn't reported by the provider, the total
    must become unknown (None) rather than silently treating the missing
    piece as $0 and under-reporting the real spend."""
    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        return _FakeResult(fx.evaluator_json_explicit_male_female(), cost_usd=None)

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    outcome = evaluator.evaluate_response(
        original_prompt="Describe the scene.", response_under_test=fx.EXPLICIT_MALE_DOCTOR_FEMALE_NURSE_RESPONSE,
        evaluator_model="anthropic/claude-3.5-haiku", api_key="k",
    )

    assert outcome.is_valid
    assert outcome.cost_usd is None


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


def test_evaluate_response_truncated_by_reasoning_model_gets_a_diagnosable_error(monkeypatch):
    """Regression test for the real failure mode this project hit with
    tested models (a reasoning model burning its whole token budget on
    hidden thinking before emitting visible text) - applied here to the
    EVALUATOR model. The error must name the actual cause, not just say
    'invalid JSON', since that's what made this so slow to diagnose the
    first time."""
    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        assert kwargs.get("max_tokens") == evaluator.EVALUATOR_MAX_TOKENS
        return _FakeResult("\n*", finish_reason="length")

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    outcome = evaluator.evaluate_response(
        original_prompt="Describe the scene.", response_under_test=fx.NEUTRAL_RESPONSE,
        evaluator_model="some/reasoning-model", api_key="k", max_repair_attempts=0,
    )

    assert not outcome.is_valid
    assert "truncated" in outcome.error.lower()
    assert "reasoning model" in outcome.error.lower()


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
    assert evaluator.default_evaluator_model() == "openai/gpt-4.1-mini"
    monkeypatch.setenv("AI_VERDICT_EVALUATOR_MODEL", "some/other-model")
    assert evaluator.default_evaluator_model() == "some/other-model"
