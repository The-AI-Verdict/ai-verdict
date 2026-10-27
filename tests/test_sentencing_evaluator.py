from __future__ import annotations

import pytest

from core import openrouter_client, sentencing_evaluator

CASE_RESPONSE = (
    "Given the low risk of reoffending and the guilty plea, I would recommend a suspended custodial "
    "sentence of 12 months, suspended for 18 months, rather than immediate custody."
)

VALID_JSON = (
    '{"sentence_type": "suspended_custodial", "custodial_months": 12, '
    '"quote": "a suspended custodial sentence of 12 months", "reasoning": "r"}'
)

MALFORMED_JSON = '{"sentence_type": "suspended_custodial", '  # truncated, invalid JSON

INVALID_SCHEMA_JSON = (
    '{"sentence_type": "custodial", "custodial_months": null, '
    '"quote": "a suspended custodial sentence of 12 months", "reasoning": "r"}'
)


class _FakeResult:
    def __init__(self, content: str, finish_reason: str = "stop", cost_usd: float | None = 0.0005):
        self.content = content
        self.latency_ms = 10
        self.retry_count = 0
        self.finish_reason = finish_reason
        self.cost_usd = cost_usd


def test_evaluate_sentencing_response_valid_on_first_attempt(monkeypatch):
    calls = []

    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        calls.append(messages)
        return _FakeResult(VALID_JSON)

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    outcome = sentencing_evaluator.evaluate_sentencing_response(
        original_prompt="Recommend a sentence.", response_under_test=CASE_RESPONSE,
        evaluator_model="openai/gpt-4.1-mini", api_key="k",
    )

    assert outcome.is_valid
    assert outcome.parsed.sentence_type.value == "suspended_custodial"
    assert len(calls) == 1
    system_message = calls[0][0]["content"]
    assert "sentence_type" in system_message
    user_message = calls[0][1]["content"]
    assert "ORIGINAL_PROMPT" in user_message and "ORIGINAL_RESPONSE" in user_message


def test_evaluate_sentencing_response_repairs_after_malformed_json(monkeypatch):
    attempts = {"n": 0}

    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _FakeResult(MALFORMED_JSON)
        return _FakeResult(VALID_JSON)

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    outcome = sentencing_evaluator.evaluate_sentencing_response(
        original_prompt="Recommend a sentence.", response_under_test=CASE_RESPONSE,
        evaluator_model="openai/gpt-4.1-mini", api_key="k", max_repair_attempts=1,
    )

    assert attempts["n"] == 2
    assert outcome.is_valid


def test_evaluate_sentencing_response_sums_cost_across_repair_attempts(monkeypatch):
    attempts = {"n": 0}

    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _FakeResult(MALFORMED_JSON, cost_usd=0.0015)
        return _FakeResult(VALID_JSON, cost_usd=0.0025)

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    outcome = sentencing_evaluator.evaluate_sentencing_response(
        original_prompt="Recommend a sentence.", response_under_test=CASE_RESPONSE,
        evaluator_model="openai/gpt-4.1-mini", api_key="k", max_repair_attempts=1,
    )

    assert outcome.is_valid
    assert outcome.cost_usd == pytest.approx(0.004)


def test_evaluate_sentencing_response_cost_is_none_when_any_attempt_cost_is_unreported(monkeypatch):
    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        return _FakeResult(VALID_JSON, cost_usd=None)

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    outcome = sentencing_evaluator.evaluate_sentencing_response(
        original_prompt="Recommend a sentence.", response_under_test=CASE_RESPONSE,
        evaluator_model="openai/gpt-4.1-mini", api_key="k",
    )

    assert outcome.is_valid
    assert outcome.cost_usd is None


def test_evaluate_sentencing_response_gives_up_after_exhausting_repair_attempts(monkeypatch):
    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        return _FakeResult(INVALID_SCHEMA_JSON)

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    outcome = sentencing_evaluator.evaluate_sentencing_response(
        original_prompt="Recommend a sentence.", response_under_test=CASE_RESPONSE,
        evaluator_model="openai/gpt-4.1-mini", api_key="k", max_repair_attempts=1,
    )

    assert not outcome.is_valid, "invalid output must be flagged for human review, never silently accepted"
    assert outcome.error is not None
    assert outcome.parsed is None


def test_evaluate_sentencing_response_truncated_by_reasoning_model_gets_a_diagnosable_error(monkeypatch):
    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        assert kwargs.get("max_tokens") == sentencing_evaluator.EVALUATOR_MAX_TOKENS
        return _FakeResult("\n*", finish_reason="length")

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    outcome = sentencing_evaluator.evaluate_sentencing_response(
        original_prompt="Recommend a sentence.", response_under_test=CASE_RESPONSE,
        evaluator_model="some/reasoning-model", api_key="k", max_repair_attempts=0,
    )

    assert not outcome.is_valid
    assert "truncated" in outcome.error.lower()
    assert "reasoning model" in outcome.error.lower()


def test_sentencing_evaluator_strips_markdown_code_fences(monkeypatch):
    fenced = "```json\n" + VALID_JSON + "\n```"

    def fake_chat_completion(model, messages, api_key=None, **kwargs):
        return _FakeResult(fenced)

    monkeypatch.setattr(openrouter_client, "chat_completion", fake_chat_completion)

    outcome = sentencing_evaluator.evaluate_sentencing_response(
        original_prompt="Recommend a sentence.", response_under_test=CASE_RESPONSE,
        evaluator_model="openai/gpt-4.1-mini", api_key="k",
    )
    assert outcome.is_valid


def test_default_sentencing_evaluator_model_matches_shared_default(monkeypatch):
    monkeypatch.delenv("AI_VERDICT_EVALUATOR_MODEL", raising=False)
    assert sentencing_evaluator.default_sentencing_evaluator_model() == "openai/gpt-4.1-mini"
