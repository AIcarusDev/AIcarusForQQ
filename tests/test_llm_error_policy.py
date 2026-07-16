from __future__ import annotations

from types import SimpleNamespace

from llm.core.error_policy import classify_llm_exception, normalize_llm_error


class FakeApiError(Exception):
    def __init__(self, message: str, status_code=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(
            status_code=status_code,
            headers=headers or {},
            text=message,
        )


def test_rate_limit_uses_retry_after_header():
    decision = classify_llm_exception(
        FakeApiError("Too many requests", status_code=429, headers={"retry-after": "7"})
    )

    assert decision.category == "rate_limit"
    assert decision.retryable is True
    assert decision.action == "cooldown"
    assert decision.cooldown_seconds == 7


def test_authentication_error_pauses_for_config_fix():
    decision = classify_llm_exception(FakeApiError("invalid api key", status_code=401))

    assert decision.category == "authentication"
    assert decision.retryable is False
    assert decision.action == "pause_until_config_fix"
    assert decision.cooldown_seconds >= 60


def test_payload_too_large_is_not_retried_immediately():
    decision = classify_llm_exception(FakeApiError("payload too large", status_code=413))

    assert decision.category == "payload_too_large"
    assert decision.retryable is False
    assert decision.action == "reduce_context_or_max_tokens"
    assert decision.cooldown_seconds > 0


def test_wrapped_upstream_400_is_retryable_server_error():
    decision = classify_llm_exception(
        FakeApiError(
            "Error from provider (Console Go): Upstream request failed "
            "(invalid_request_error)",
            status_code=400,
        )
    )

    assert decision.category == "server_error"
    assert decision.retryable is True
    assert decision.action == "cooldown"
    assert decision.cooldown_seconds < 300


def test_plain_400_remains_non_retryable_bad_request():
    decision = classify_llm_exception(
        FakeApiError("Invalid tool schema", status_code=400)
    )

    assert decision.category == "bad_request"
    assert decision.retryable is False
    assert decision.action == "fix_request_schema"


def test_server_error_is_retryable_with_cooldown():
    decision = classify_llm_exception(FakeApiError("bad gateway", status_code=502))

    assert decision.category == "server_error"
    assert decision.retryable is True
    assert decision.action == "cooldown"
    assert decision.cooldown_seconds > 0


def test_timeout_without_status_is_retryable():
    decision = classify_llm_exception(TimeoutError("request timed out"))

    assert decision.category == "timeout"
    assert decision.status_code is None
    assert decision.retryable is True


def test_normalize_stored_decision_dict():
    decision = normalize_llm_error(
        {
            "category": "rate_limit",
            "status_code": 429,
            "retryable": True,
            "cooldown_seconds": 2,
            "action": "cooldown",
            "summary": "limited",
            "detail": "x",
        }
    )

    assert decision is not None
    assert decision.category == "rate_limit"
    assert decision.status_code == 429
