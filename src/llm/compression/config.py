"""Configuration normalization for cognition-flow compression."""

from __future__ import annotations

DEFAULT_LLM_CONTENTS_MAX_ROUNDS = 10
MIN_LLM_CONTENTS_MAX_ROUNDS = 6
DEFAULT_COMPRESSION_TRIGGER_ROUNDS = 5
MIN_COMPRESSION_TRIGGER_ROUNDS = 3
DEFAULT_WORLD_MULTIMODAL_IMAGE_LIMIT = 5


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_world_multimodal_image_limit(value) -> int:
    """Normalize the per-world real multimodal image limit.

    ``-1`` means unlimited. ``0`` means keep text labels/descriptions but do not
    send any real image_url parts.
    """
    normalized = _to_int(value, DEFAULT_WORLD_MULTIMODAL_IMAGE_LIMIT)
    if normalized == -1:
        return -1
    if normalized < -1:
        return DEFAULT_WORLD_MULTIMODAL_IMAGE_LIMIT
    return normalized


def normalize_generation_config(gen: dict | None) -> dict:
    """Return a copy with bounded flow-retention and compression settings."""
    normalized = dict(gen or {})
    max_rounds = max(
        MIN_LLM_CONTENTS_MAX_ROUNDS,
        _to_int(
            normalized.get("llm_contents_max_rounds"),
            DEFAULT_LLM_CONTENTS_MAX_ROUNDS,
        ),
    )
    trigger_rounds = max(
        MIN_COMPRESSION_TRIGGER_ROUNDS,
        _to_int(
            normalized.get("cognition_compression_trigger_rounds"),
            DEFAULT_COMPRESSION_TRIGGER_ROUNDS,
        ),
    )
    if trigger_rounds >= max_rounds:
        trigger_rounds = max_rounds - 1
    normalized["llm_contents_max_rounds"] = max_rounds
    normalized["cognition_compression_trigger_rounds"] = trigger_rounds
    normalized["world_multimodal_image_limit"] = normalize_world_multimodal_image_limit(
        normalized.get("world_multimodal_image_limit"),
    )
    return normalized
