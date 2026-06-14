"""Duplicate full-response guard for model outputs.

The guard is intentionally conservative: it only treats normalized full raw
assistant responses as duplicates when they are exactly equal.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class DuplicateModelResponseGuardConfig:
    enabled: bool = False
    lookback_rounds: int = 3
    max_retries: int = 2
    normalize_whitespace: bool = True
    fallback_sleep_minutes: int = 30


def normalize_duplicate_model_response_guard_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    cfg = DuplicateModelResponseGuardConfig()

    def _int(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(raw.get(name, default))
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    return {
        "enabled": bool(raw.get("enabled", cfg.enabled)),
        "lookback_rounds": _int("lookback_rounds", cfg.lookback_rounds, 1, 20),
        "max_retries": _int("max_retries", cfg.max_retries, 1, 10),
        "normalize_whitespace": bool(raw.get("normalize_whitespace", cfg.normalize_whitespace)),
        "fallback_sleep_minutes": _int("fallback_sleep_minutes", cfg.fallback_sleep_minutes, 1, 600),
    }


def normalize_response_text(text: str, *, normalize_whitespace: bool = True) -> str:
    normalized = (text or "").strip()
    if normalize_whitespace:
        normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def build_duplicate_model_response_error(*, duplicate_count: int, max_retries: int) -> dict[str, Any]:
    return {
        "error": "DUPLICATE_MODEL_RESPONSE",
        "message": (
            "本轮模型输出与最近一次模型输出完全一致，包括 cognition 和 tool_call。"
            "系统未执行其中的工具。请重新评估当前 world，避免重复执行已完成的行为。"
        ),
        "tool_not_executed": True,
        "retryable": True,
        "duplicate_count": duplicate_count,
        "max_retries": max_retries,
    }


def build_duplicate_model_response_limit_error(*, duplicate_count: int) -> dict[str, Any]:
    return {
        "error": "DUPLICATE_MODEL_RESPONSE_LIMIT",
        "message": "模型连续输出完全相同内容，已停止重试并进入 sleep，等待新的外部输入。",
        "tool_not_executed": True,
        "retryable": False,
        "duplicate_count": duplicate_count,
        "fallback": "sleep",
    }
