"""Typed internal tool results with optional model-visible text payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TextPayloadResult:
    """A structured tool result plus text that must not be JSON-escaped."""

    meta: Mapping[str, Any]
    text_payload: str


__all__ = ["TextPayloadResult"]
