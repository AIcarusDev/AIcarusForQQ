"""Prompt-level diagnostics that do not know tool internals."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("AICQ.llm.provider")


def estimate_token_count(text: str) -> int:
    """Cheap mixed Chinese/ASCII token estimate for prompt diagnostics."""
    ascii_chars = 0
    cjk_chars = 0
    other_chars = 0
    for char in text:
        codepoint = ord(char)
        if (
            0x4E00 <= codepoint <= 0x9FFF
            or 0x3400 <= codepoint <= 0x4DBF
            or 0x20000 <= codepoint <= 0x2A6DF
        ):
            cjk_chars += 1
        elif codepoint < 128:
            ascii_chars += 1
        else:
            other_chars += 1
    estimate = round((ascii_chars / 4) + cjk_chars + (other_chars / 2))
    return max(1, estimate)


def serialize_prompt_prefix(messages: list[dict[str, Any]]) -> str:
    """Serialize the already-built stable prompt prefix for change detection."""
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))


def log_prompt_prefix_comparison(
    *,
    provider: str,
    previous_prefix: str | None,
    current_prefix: str,
) -> None:
    """Log whether the stable prompt prefix changed since the previous round."""
    if previous_prefix is not None and previous_prefix == current_prefix:
        logger.info(
            "[%s] prompt prefix — stable prefix unchanged, about %d tokens (estimated)",
            provider,
            estimate_token_count(current_prefix),
        )
        return
    logger.debug(
        "[%s] prompt prefix — %s",
        provider,
        "first stable prefix snapshot" if previous_prefix is None else "stable prefix changed",
    )
