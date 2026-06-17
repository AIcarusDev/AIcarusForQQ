"""Prompt-v2 archive output parser.

The parser intentionally accepts only the prompt-native extract block.  It does
not try to repair malformed model output because bad archive writes are much
harder to diagnose after they enter the memory graph.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


_EXTRACT_RE = re.compile(r"<extract\b[^>]*>(.*?)</extract>", re.IGNORECASE | re.DOTALL)
_EVENT_RE = re.compile(r"<event\b[^>]*>(.*?)</event>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"^\s*```|```\s*$", re.MULTILINE)
_TAG_RE = re.compile(r"<\s*(/)?\s*([A-Za-z][\w:-]*)\b[^>]*>", re.DOTALL)


@dataclass(slots=True)
class ParsedArchiveEvent:
    event: dict[str, Any]
    raw_json: str


@dataclass(slots=True)
class ArchiveParseResult:
    events: list[ParsedArchiveEvent] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class ArchiveParseFatalError(ValueError):
    """Raised when the whole archive output is structurally invalid."""


def parse_archive_output(text: str | None) -> ArchiveParseResult:
    """Parse prompt-v2 archive output.

    Fatal errors:
    - missing ``<extract>``
    - duplicated ``<extract>``
    - structurally unparseable extract block

    Per-event errors are returned in ``ArchiveParseResult.errors`` while valid
    sibling events are still accepted.
    """

    if not isinstance(text, str) or not text.strip():
        raise ArchiveParseFatalError("archive output is empty")

    extracts = _top_level_extract_bodies(text)
    if not extracts:
        raise ArchiveParseFatalError("missing <extract> block")
    if len(extracts) != 1:
        raise ArchiveParseFatalError("duplicated <extract> blocks")

    extract_body = extracts[0]
    result = ArchiveParseResult()
    for index, match in enumerate(_EVENT_RE.finditer(extract_body), start=1):
        payload = match.group(1).strip()
        if not payload:
            result.errors.append(f"event#{index}: empty payload")
            continue
        if _FENCE_RE.search(payload):
            result.errors.append(f"event#{index}: markdown fence is not allowed")
            continue
        if not (payload.startswith("{") and payload.endswith("}")):
            result.errors.append(f"event#{index}: payload must be one JSON object")
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            result.errors.append(f"event#{index}: invalid JSON: {exc}")
            continue
        if not isinstance(event, dict):
            result.errors.append(f"event#{index}: JSON payload is not an object")
            continue
        err = _validate_event(event)
        if err:
            result.errors.append(f"event#{index}: {err}")
            continue
        result.events.append(ParsedArchiveEvent(event=event, raw_json=payload))
    return result


def _top_level_extract_bodies(text: str) -> list[str]:
    bodies: list[str] = []
    stack: list[str] = []
    pos = 0
    while True:
        match = _TAG_RE.search(text, pos)
        if not match:
            break
        closing = bool(match.group(1))
        name = match.group(2).lower()
        full = match.group(0)
        self_closing = full.rstrip().endswith("/>")
        if name == "extract" and not closing and not stack:
            end = re.search(r"</\s*extract\s*>", text[match.end() :], re.IGNORECASE)
            if end is None:
                raise ArchiveParseFatalError("unclosed <extract> block")
            body_start = match.end()
            body_end = match.end() + end.start()
            bodies.append(text[body_start:body_end])
            pos = match.end() + end.end()
            continue
        if not self_closing:
            if closing:
                if stack and stack[-1] == name:
                    stack.pop()
            else:
                stack.append(name)
        pos = match.end()
    return bodies


def _validate_event(event: dict[str, Any]) -> str:
    summary = event.get("summary")
    event_type = event.get("event_type")
    roles = event.get("roles")
    if not isinstance(summary, str) or not summary.strip():
        return "summary must be a non-empty string"
    if not isinstance(event_type, str) or not event_type.strip():
        return "event_type must be a non-empty string"
    if not isinstance(roles, list):
        return "roles must be a list"
    return ""
