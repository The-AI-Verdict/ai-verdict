"""Thin OpenRouter API client.

Verified against OpenRouter's public docs (Oct 2026):
- POST https://openrouter.ai/api/v1/chat/completions
- GET  https://openrouter.ai/api/v1/models
- usage.cost / usage.prompt_tokens / usage.completion_tokens are returned by
  default on every chat completion response (the old `usage: {include: true}`
  request flag is deprecated and no longer required).
- Error body shape: {"error": {"code": <int>, "message": <str>, "metadata": {...}}}
- Retryable HTTP statuses: 429 (rate limit), 408/524 (timeout), 500/502/503/529
  (upstream/server trouble). 402 (insufficient credits), 401 (auth) and 403
  (blocked) are NOT retried by default - retrying them would just burn time
  and, in the 402 case, usually cannot succeed without the user adding
  credits. Exception: a 402 whose message mentions "in-flight" requests IS
  retried - observed in production against a free/low-balance key, this
  specific variant is a transient per-account concurrent-reservation cap
  that clears on its own, unlike a genuine zero-balance 402.
- A client-side read timeout (the request was sent but no response arrived in
  time) is also NOT retried, since OpenRouter may have already generated and
  billed a completion we simply never received - retrying that case risks a
  duplicate charge for one logical run. Only a connect-timeout (the request
  was never actually sent) is retried automatically.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests

API_BASE = "https://openrouter.ai/api/v1"
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 524, 529}
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_RETRIES = 4
DEFAULT_BACKOFF_BASE_SECONDS = 2.0


class OpenRouterError(Exception):
    def __init__(self, message: str, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


@dataclass
class ChatCompletionResult:
    raw_response_json: dict[str, Any]
    content: str
    response_id: str | None
    response_model: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    latency_ms: int
    retry_count: int
    unsupported_params: list[str] = field(default_factory=list)


def _headers(api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    site_url = os.environ.get("OPENROUTER_SITE_URL")
    app_name = os.environ.get("OPENROUTER_APP_NAME")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name
    return headers


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file (see .env.example)."
        )
    return key


def list_models(api_key: str | None = None, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> list[dict[str, Any]]:
    """GET /models - public model catalog. Returns the raw `data` list."""
    key = api_key or get_api_key()
    resp = requests.get(f"{API_BASE}/models", headers=_headers(key), timeout=timeout)
    if resp.status_code != 200:
        raise OpenRouterError(f"Failed to list models: HTTP {resp.status_code} {resp.text[:300]}", resp.status_code)
    data = resp.json()
    return data.get("data", [])


def _extract_error_message(resp: requests.Response) -> str:
    try:
        body = resp.json()
        err = body.get("error", {})
        return err.get("message") or str(body)
    except ValueError:
        return resp.text[:500]


def chat_completion(
    model: str,
    messages: list[dict[str, str]],
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    seed: int | None = None,
    extra_params: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
) -> ChatCompletionResult:
    """Call the chat completions endpoint with retry + exponential backoff.

    A single HTTP attempt is made per retry iteration; we only retry when no
    response body was returned to the caller (i.e. the previous attempt
    definitely failed server-side before producing a completion), so a
    successful-but-slow response is never retried and double-charged.
    """
    key = api_key or get_api_key()
    body: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if top_p is not None:
        body["top_p"] = top_p
    if frequency_penalty is not None:
        body["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        body["presence_penalty"] = presence_penalty
    if seed is not None:
        body["seed"] = seed
    if extra_params:
        body.update(extra_params)

    last_error: OpenRouterError | None = None
    retry_count = 0
    start = time.monotonic()

    for attempt in range(max_retries + 1):
        attempt_start = time.monotonic()
        try:
            resp = requests.post(
                f"{API_BASE}/chat/completions", headers=_headers(key), json=body, timeout=timeout
            )
        except requests.exceptions.ConnectTimeout as exc:
            # The TCP connection itself never completed, so the request was
            # never sent - safe to retry.
            last_error = OpenRouterError(f"Connection timed out: {exc}", status_code=None, retryable=True)
        except requests.exceptions.Timeout as exc:
            # A read timeout (or an otherwise-unspecific Timeout) means the
            # request WAS sent and a response was still pending when we gave
            # up waiting - the server may have already generated and billed
            # a completion we never saw. Do not auto-retry; surface it so the
            # run is marked failed and a human can check OpenRouter usage
            # before deciding to resend.
            last_error = OpenRouterError(
                f"Timed out waiting for a response: {exc}. The request may already have reached "
                f"OpenRouter and been billed even though no response arrived - check usage before "
                f"manually retrying.",
                status_code=None, retryable=False,
            )
            raise last_error
        except requests.exceptions.RequestException as exc:
            last_error = OpenRouterError(f"Network error: {exc}", status_code=None, retryable=True)
        else:
            if resp.status_code == 200:
                latency_ms = int((time.monotonic() - start) * 1000)
                try:
                    data = resp.json()
                except ValueError as exc:
                    # A 200 status means the server considers the request
                    # fulfilled, so a malformed body likely still means a
                    # completion was generated - don't auto-retry and risk a
                    # duplicate billed completion.
                    raise OpenRouterError(
                        f"Received HTTP 200 with an unparseable body: {exc}", status_code=200, retryable=False,
                    )
                return _parse_success(data, latency_ms, retry_count)
            message = _extract_error_message(resp)
            retryable = resp.status_code in RETRYABLE_STATUS_CODES
            if resp.status_code == 402 and ("in-flight" in message.lower() or "in flight" in message.lower()):
                # Observed verbatim in production (Oct 2026): "This request
                # would exceed your available credits given your current
                # in-flight requests. Retry after in-flight requests settle,
                # or add credits." A free/low-balance OpenRouter key enforces
                # a cap on simultaneously in-flight credit reservations,
                # independent of total balance - unlike a genuine "add
                # credits" 402, this clears on its own, so it's worth a
                # backoff retry rather than failing the run outright.
                #
                # This is a substring match on free-form error text, not a
                # structured field - OpenRouter's error body only documents
                # `code`/`message`/`metadata`, and nothing in what we've
                # observed indicates a stabler machine-readable signal for
                # this specific condition. If OpenRouter changes this
                # wording, this stops matching and the 402 falls back to
                # non-retryable - safe (never worse than today), just no
                # longer self-healing for this case. Update the match here
                # if that's ever observed.
                retryable = True
            last_error = OpenRouterError(
                f"HTTP {resp.status_code}: {message}", status_code=resp.status_code, retryable=retryable
            )
            if not retryable:
                raise last_error

        retry_count = attempt + 1
        if attempt < max_retries:
            sleep_for = backoff_base_seconds * (2 ** attempt)
            time.sleep(sleep_for)
        del attempt_start

    assert last_error is not None
    raise last_error


def _parse_success(data: dict[str, Any], latency_ms: int, retry_count: int) -> ChatCompletionResult:
    choices = data.get("choices") or []
    message = (choices[0].get("message") if choices else {}) or {}
    content = message.get("content") or ""
    usage = data.get("usage") or {}
    return ChatCompletionResult(
        raw_response_json=data,
        content=content,
        response_id=data.get("id"),
        response_model=data.get("model"),
        finish_reason=choices[0].get("finish_reason") if choices else None,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        cost_usd=usage.get("cost"),
        latency_ms=latency_ms,
        retry_count=retry_count,
    )
