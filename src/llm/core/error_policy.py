"""Structured LLM API error classification and runtime backoff policy."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from email.utils import parsedate_to_datetime
from typing import Any

logger = logging.getLogger("AICQ.llm.error_policy")


@dataclass(frozen=True)
class LLMErrorDecision:
    """Normalized decision for an LLM transport/API failure."""

    category: str
    status_code: int | None
    retryable: bool
    cooldown_seconds: float
    action: str
    summary: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DEFAULT_COOLDOWNS = {
    "rate_limit": 60.0,
    "server_error": 30.0,
    "timeout": 20.0,
    "network": 20.0,
    "payload_too_large": 300.0,
    "authentication": 900.0,
    "permission": 900.0,
    "not_found": 900.0,
    "bad_request": 300.0,
    "conflict": 15.0,
    "unprocessable": 300.0,
    "unknown": 30.0,
}


def classify_llm_exception(exc: Exception) -> LLMErrorDecision:
    """Classify provider/SDK exceptions into a small set of runtime actions."""
    status_code = _extract_status_code(exc)
    retry_after = _extract_retry_after_seconds(exc)
    detail = _compact_detail(exc)
    text = f"{type(exc).__name__} {detail}".lower()

    if status_code == 429:
        cooldown = retry_after or _DEFAULT_COOLDOWNS["rate_limit"]
        return _decision(
            "rate_limit",
            status_code,
            True,
            cooldown,
            "cooldown",
            "服务端限流，暂停 LLM 调用后再试。",
            detail,
        )
    if status_code in {401}:
        return _decision(
            "authentication",
            status_code,
            False,
            _DEFAULT_COOLDOWNS["authentication"],
            "pause_until_config_fix",
            "API key 缺失、无效或已过期，需要修正配置。",
            detail,
        )
    if status_code in {403}:
        return _decision(
            "permission",
            status_code,
            False,
            _DEFAULT_COOLDOWNS["permission"],
            "pause_until_config_fix",
            "账号、模型或端点权限不足，需要修正配置或供应商权限。",
            detail,
        )
    if status_code == 404:
        return _decision(
            "not_found",
            status_code,
            False,
            _DEFAULT_COOLDOWNS["not_found"],
            "pause_until_config_fix",
            "模型或接口路径不存在，需要修正 provider/model/base_url。",
            detail,
        )
    if status_code == 413:
        return _decision(
            "payload_too_large",
            status_code,
            False,
            _DEFAULT_COOLDOWNS["payload_too_large"],
            "reduce_context_or_max_tokens",
            "请求体超过供应商限制，需要压缩上下文或降低输出上限。",
            detail,
        )
    if status_code in {400}:
        return _decision(
            "bad_request",
            status_code,
            False,
            _DEFAULT_COOLDOWNS["bad_request"],
            "fix_request_schema",
            "请求参数或消息格式不被供应商接受，需要修正生成参数/工具 schema。",
            detail,
        )
    if status_code == 409:
        return _decision(
            "conflict",
            status_code,
            True,
            retry_after or _DEFAULT_COOLDOWNS["conflict"],
            "short_cooldown",
            "供应商返回冲突状态，短暂等待后重试。",
            detail,
        )
    if status_code == 422:
        return _decision(
            "unprocessable",
            status_code,
            False,
            _DEFAULT_COOLDOWNS["unprocessable"],
            "fix_request_schema",
            "请求语义无法处理，需要修正参数、工具 schema 或消息内容。",
            detail,
        )
    if status_code in {408, 500, 502, 503, 504}:
        return _decision(
            "server_error",
            status_code,
            True,
            retry_after or _DEFAULT_COOLDOWNS["server_error"],
            "cooldown",
            "供应商服务端或网关临时失败，等待后重试。",
            detail,
        )
    if _looks_like_timeout(text):
        return _decision(
            "timeout",
            status_code,
            True,
            _DEFAULT_COOLDOWNS["timeout"],
            "cooldown",
            "请求超时，等待后重试。",
            detail,
        )
    if _looks_like_network_error(text):
        return _decision(
            "network",
            status_code,
            True,
            _DEFAULT_COOLDOWNS["network"],
            "cooldown",
            "网络连接失败，等待后重试。",
            detail,
        )
    return _decision(
        "unknown",
        status_code,
        True,
        retry_after or _DEFAULT_COOLDOWNS["unknown"],
        "cooldown",
        "未知 LLM 调用错误，保守等待后重试。",
        detail,
    )


def normalize_llm_error(value: Any) -> LLMErrorDecision | None:
    """Rehydrate a stored decision dict into ``LLMErrorDecision``."""
    if isinstance(value, LLMErrorDecision):
        return value
    if not isinstance(value, dict):
        return None
    try:
        return LLMErrorDecision(
            category=str(value.get("category") or "unknown"),
            status_code=(
                int(value["status_code"])
                if value.get("status_code") is not None
                else None
            ),
            retryable=bool(value.get("retryable", True)),
            cooldown_seconds=max(0.0, float(value.get("cooldown_seconds") or 0.0)),
            action=str(value.get("action") or "cooldown"),
            summary=str(value.get("summary") or "LLM 调用失败。"),
            detail=str(value.get("detail") or ""),
        )
    except (TypeError, ValueError):
        logger.debug("invalid stored LLM error decision: %r", value)
        return None


def _decision(
    category: str,
    status_code: int | None,
    retryable: bool,
    cooldown_seconds: float,
    action: str,
    summary: str,
    detail: str,
) -> LLMErrorDecision:
    return LLMErrorDecision(
        category=category,
        status_code=status_code,
        retryable=retryable,
        cooldown_seconds=max(0.0, float(cooldown_seconds or 0.0)),
        action=action,
        summary=summary,
        detail=detail,
    )


def _extract_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        parsed = _parse_int(value)
        if parsed is not None:
            return parsed
    response = getattr(exc, "response", None)
    if response is not None:
        parsed = _parse_int(getattr(response, "status_code", None))
        if parsed is not None:
            return parsed
    match = re.search(r"\b([45]\d\d)\b", str(exc))
    if match:
        return int(match.group(1))
    return None


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is None:
        headers = getattr(exc, "headers", None)
    if not headers:
        return None
    value = None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None
    if value is None:
        return None
    numeric = _parse_float(value)
    if numeric is not None:
        return max(0.0, numeric)
    try:
        retry_at = parsedate_to_datetime(str(value))
        return max(0.0, retry_at.timestamp() - __import__("time").time())
    except (TypeError, ValueError, OverflowError):
        return None


def _compact_detail(exc: Exception) -> str:
    parts = [str(exc)]
    response = getattr(exc, "response", None)
    body = None
    if response is not None:
        body = getattr(response, "text", None)
        if callable(body):
            try:
                body = body()
            except Exception:
                body = None
    if body:
        parts.append(str(body))
    text = " | ".join(part for part in parts if part).strip()
    if not text:
        text = type(exc).__name__
    text = _json_minify(text)
    return text[:500]


def _json_minify(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith(("{", "[")):
        return " ".join(stripped.split())
    try:
        return json.dumps(json.loads(stripped), ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return " ".join(stripped.split())


def _parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _looks_like_timeout(text: str) -> bool:
    return any(marker in text for marker in ("timeout", "timed out", "readtimeout"))


def _looks_like_network_error(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "connection",
            "connecterror",
            "network",
            "dns",
            "proxy",
            "remoteprotocolerror",
        )
    )
