"""Helpers for recording OpenAI-compatible token usage events."""

from __future__ import annotations

import json
import logging
from typing import Any

from database import save_llm_usage_event_sync

logger = logging.getLogger("AICQ.llm.usage")


def _to_plain(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return _to_plain(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _get_path(data: Any, *path: str) -> Any:
    cur = data
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
        if cur is None:
            return None
    return cur


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def parse_usage(usage: Any) -> dict[str, Any]:
    """Extract common OpenAI-compatible token fields from a usage object."""
    raw = _to_plain(usage)
    usage_available = usage is not None and bool(raw)
    raw_map = raw if isinstance(raw, dict) else {}

    input_tokens = _as_int(
        raw_map.get("prompt_tokens")
        if "prompt_tokens" in raw_map
        else raw_map.get("input_tokens")
    )
    output_tokens = _as_int(
        raw_map.get("completion_tokens")
        if "completion_tokens" in raw_map
        else raw_map.get("output_tokens")
    )
    total_tokens = _as_int(raw_map.get("total_tokens"))
    if not total_tokens and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens

    cached_input_tokens = max(
        _as_int(_get_path(raw_map, "prompt_tokens_details", "cached_tokens")),
        _as_int(_get_path(raw_map, "input_tokens_details", "cached_tokens")),
        _as_int(raw_map.get("prompt_cache_hit_tokens")),
    )
    reasoning_output_tokens = max(
        _as_int(_get_path(raw_map, "completion_tokens_details", "reasoning_tokens")),
        _as_int(_get_path(raw_map, "output_tokens_details", "reasoning_tokens")),
    )

    return {
        "usage_available": usage_available,
        "input_tokens": input_tokens if usage_available else 0,
        "output_tokens": output_tokens if usage_available else 0,
        "total_tokens": total_tokens if usage_available else 0,
        "cached_input_tokens": cached_input_tokens if usage_available else 0,
        "reasoning_output_tokens": reasoning_output_tokens if usage_available else 0,
        "raw_usage_json": json.dumps(raw or {}, ensure_ascii=False, separators=(",", ":")),
    }


def record_llm_usage(
    *,
    provider: str,
    model: str,
    feature: str,
    subfeature: str = "",
    usage: Any = None,
    status: str = "success",
) -> None:
    """Record usage without allowing persistence errors to affect LLM calls."""
    counts = parse_usage(usage)
    ok = save_llm_usage_event_sync(
        provider=provider,
        model=model,
        feature=feature,
        subfeature=subfeature,
        input_tokens=counts["input_tokens"],
        output_tokens=counts["output_tokens"],
        total_tokens=counts["total_tokens"],
        cached_input_tokens=counts["cached_input_tokens"],
        reasoning_output_tokens=counts["reasoning_output_tokens"],
        usage_available=bool(counts["usage_available"]),
        status=status,
        raw_usage_json=counts["raw_usage_json"],
    )
    if not ok:
        logger.debug(
            "LLM usage event was not persisted provider=%s model=%s feature=%s",
            provider,
            model,
            feature,
        )
