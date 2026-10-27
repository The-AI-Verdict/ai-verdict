from __future__ import annotations

import pytest
import requests

from core import openrouter_client as orc


class FakeResponse:
    def __init__(self, status_code: int, json_body: dict, text: str = ""):
        self.status_code = status_code
        self._json_body = json_body
        self.text = text or str(json_body)

    def json(self):
        return self._json_body


def _success_body():
    return {
        "id": "gen-abc123",
        "model": "openai/gpt-4o-mini-2024-07-18",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello world"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0003},
    }


def test_succeeds_on_first_try_no_retry(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        return FakeResponse(200, _success_body())

    monkeypatch.setattr(requests, "post", fake_post)
    result = orc.chat_completion("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}], api_key="k")

    assert result.content == "Hello world"
    assert result.response_id == "gen-abc123"
    assert result.cost_usd == 0.0003
    assert result.retry_count == 0
    assert len(calls) == 1


def test_retries_on_429_then_succeeds(monkeypatch):
    responses = [
        FakeResponse(429, {"error": {"code": 429, "message": "rate limited"}}),
        FakeResponse(429, {"error": {"code": 429, "message": "rate limited"}}),
        FakeResponse(200, _success_body()),
    ]
    call_count = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        resp = responses[call_count["n"]]
        call_count["n"] += 1
        return resp

    sleeps = []
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(orc.time, "sleep", lambda s: sleeps.append(s))

    result = orc.chat_completion("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}], api_key="k",
                                  backoff_base_seconds=0.01)

    assert result.content == "Hello world"
    assert result.retry_count == 2
    assert call_count["n"] == 3
    assert len(sleeps) == 2, "should sleep between the two failed attempts, not after success"


def test_does_not_retry_on_401(monkeypatch):
    call_count = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        call_count["n"] += 1
        return FakeResponse(401, {"error": {"code": 401, "message": "invalid api key"}})

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(orc.OpenRouterError) as exc_info:
        orc.chat_completion("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}], api_key="k")

    assert call_count["n"] == 1, "401 must not be retried"
    assert exc_info.value.status_code == 401
    assert not exc_info.value.retryable


def test_exhausts_retries_and_raises(monkeypatch):
    call_count = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        call_count["n"] += 1
        return FakeResponse(503, {"error": {"code": 503, "message": "overloaded"}})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(orc.time, "sleep", lambda s: None)

    with pytest.raises(orc.OpenRouterError) as exc_info:
        orc.chat_completion("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}], api_key="k",
                             max_retries=3, backoff_base_seconds=0.01)

    assert call_count["n"] == 4, "should attempt max_retries + 1 times total"
    assert exc_info.value.status_code == 503


def test_insufficient_credits_402_is_not_retried(monkeypatch):
    call_count = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        call_count["n"] += 1
        return FakeResponse(402, {"error": {"code": 402, "message": "insufficient credits"}})

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(orc.OpenRouterError):
        orc.chat_completion("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}], api_key="k")

    assert call_count["n"] == 1


def test_in_flight_credit_402_is_retried_since_it_is_transient(monkeypatch):
    """Observed in production against a free-tier OpenRouter key: this
    specific 402 variant clears on its own once prior in-flight requests
    settle, unlike a genuine zero-balance 402 - so it should be retried."""
    responses = [
        FakeResponse(402, {"error": {"code": 402, "message": (
            "This request would exceed your available credits given your current in-flight requests. "
            "Retry after in-flight requests settle, or add credits."
        )}}),
        FakeResponse(200, _success_body()),
    ]
    call_count = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        resp = responses[call_count["n"]]
        call_count["n"] += 1
        return resp

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(orc.time, "sleep", lambda s: None)

    result = orc.chat_completion("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}], api_key="k",
                                  backoff_base_seconds=0.01)
    assert result.content == "Hello world"
    assert call_count["n"] == 2


def test_list_models_returns_data_list(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        return FakeResponse(200, {"data": [{"id": "openai/gpt-4o-mini", "name": "GPT-4o mini"}]})

    monkeypatch.setattr(requests, "get", fake_get)
    models = orc.list_models(api_key="k")
    assert models == [{"id": "openai/gpt-4o-mini", "name": "GPT-4o mini"}]


def test_get_api_key_raises_clear_error_when_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(orc.OpenRouterError):
        orc.get_api_key()


def test_connect_timeout_is_retried_since_request_was_never_sent(monkeypatch):
    call_count = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise requests.exceptions.ConnectTimeout("connect timed out")
        return FakeResponse(200, _success_body())

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(orc.time, "sleep", lambda s: None)

    result = orc.chat_completion("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}], api_key="k")
    assert result.content == "Hello world"
    assert call_count["n"] == 2


def test_read_timeout_is_not_retried_to_avoid_duplicate_billing(monkeypatch):
    call_count = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        call_count["n"] += 1
        raise requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(orc.OpenRouterError) as exc_info:
        orc.chat_completion("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}], api_key="k")

    assert call_count["n"] == 1, "a read timeout must not be auto-retried - the server may have already billed it"
    assert not exc_info.value.retryable


def test_malformed_200_body_does_not_crash_and_is_not_retried(monkeypatch):
    class BadJsonResponse(FakeResponse):
        def json(self):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    call_count = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        call_count["n"] += 1
        return BadJsonResponse(200, {})

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(orc.OpenRouterError) as exc_info:
        orc.chat_completion("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}], api_key="k")

    assert call_count["n"] == 1
    assert exc_info.value.status_code == 200
    assert not exc_info.value.retryable
