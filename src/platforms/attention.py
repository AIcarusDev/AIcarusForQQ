"""Cross-platform attention event collection and prompt rendering."""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


ATTENTION_LEVELS = {"normal", "mention"}
_LEVEL_PRIORITY = {"normal": 0, "mention": 1}

_ATTENTION_EVENTS_DES = (
    "Cross-platform attention events from platforms that are not currently open.\n"
    "- `level=\"normal\"` indicates a standard message (e.g., a message from someone else in a group chat).\n"
    "- `level=\"mention\"` indicates an event that may require attention (e.g., a direct private message or an @mention).\n"
    "Platforms that are currently open will not appear here."
)


@dataclass(frozen=True)
class AttentionEvent:
    name: str
    level: str = "normal"
    event_type: str = "platform"
    age: str = ""
    occurred_at: datetime | int | float | str | None = None


def _normalize_level(level: object) -> str:
    text = str(level or "normal").strip().lower()
    return text if text in ATTENTION_LEVELS else "normal"


def _coerce_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


def _age_seconds(occurred_at: datetime, now: datetime) -> int:
    event_time = occurred_at
    compare_now = now
    if event_time.tzinfo is None and compare_now.tzinfo is not None:
        event_time = event_time.replace(tzinfo=compare_now.tzinfo)
    elif event_time.tzinfo is not None and compare_now.tzinfo is None:
        compare_now = compare_now.replace(tzinfo=event_time.tzinfo)
    return max(0, int((compare_now - event_time).total_seconds()))


def _format_age(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    return f"{hours // 24}d"


def _resolved_age(event: AttentionEvent, now: datetime) -> str:
    if event.age:
        return str(event.age)
    occurred_at = _coerce_datetime(event.occurred_at)
    if occurred_at is None:
        return ""
    return _format_age(_age_seconds(occurred_at, now))


def _event_timestamp(event: AttentionEvent) -> float:
    occurred_at = _coerce_datetime(event.occurred_at)
    if occurred_at is None:
        return -1.0
    if occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=timezone.utc)
    return occurred_at.timestamp()


def normalize_attention_event(value: object) -> AttentionEvent | None:
    if isinstance(value, AttentionEvent):
        name = str(value.name or "").strip()
        if not name:
            return None
        return AttentionEvent(
            name=name,
            level=_normalize_level(value.level),
            event_type=str(value.event_type or "platform").strip() or "platform",
            age=str(value.age or "").strip(),
            occurred_at=value.occurred_at,
        )
    if not isinstance(value, dict):
        return None

    name = str(value.get("name") or value.get("platform") or "").strip()
    if not name:
        return None
    event_type = (
        str(
            value.get("event_type")
            or value.get("type")
            or value.get("tpye")
            or "platform"
        ).strip()
        or "platform"
    )
    occurred_at = value.get("occurred_at", value.get("timestamp"))
    return AttentionEvent(
        name=name,
        level=_normalize_level(value.get("level")),
        event_type=event_type,
        age=str(value.get("age") or "").strip(),
        occurred_at=occurred_at,
    )


def select_attention_events(
    events: list[object],
    *,
    current_platform: str = "",
) -> list[AttentionEvent]:
    """Filter current-platform events and de-duplicate by event type/name."""

    current = str(current_platform or "").strip()
    selected: dict[tuple[str, str], AttentionEvent] = {}
    for raw_event in events:
        event = normalize_attention_event(raw_event)
        if event is None:
            continue
        if event.event_type == "platform" and event.name == current:
            continue
        key = (event.event_type, event.name)
        previous = selected.get(key)
        if previous is None:
            selected[key] = event
            continue
        previous_rank = (_LEVEL_PRIORITY[previous.level], _event_timestamp(previous))
        event_rank = (_LEVEL_PRIORITY[event.level], _event_timestamp(event))
        if event_rank > previous_rank:
            selected[key] = event
    return list(selected.values())


def collect_attention_events(
    *,
    current_platform: str = "",
    now: datetime | None = None,
    registry: Any | None = None,
) -> list[AttentionEvent]:
    if registry is None:
        try:
            import app_state

            registry = getattr(app_state, "platform_registry", None)
        except Exception:
            registry = None
    if registry is None:
        return []

    raw_events: list[object] = []
    for runtime in getattr(registry, "runtimes", {}).values():
        getter = getattr(runtime, "attention_events", None)
        if not callable(getter):
            continue
        try:
            events = getter(now=now)
        except TypeError:
            events = getter()
        except Exception:
            continue
        if events and isinstance(events, Iterable):
            raw_events.extend(list(events))
    return select_attention_events(raw_events, current_platform=current_platform)


def render_attention_events(events: Sequence[object], *, now: datetime | None = None) -> str:
    selected = [
        event for event in (normalize_attention_event(item) for item in events) if event is not None
    ]
    if not selected:
        return "<attention_events/>"

    compare_now = now or datetime.now(timezone.utc)
    lines = [
        "<attention_events>",
        f"  <des>{html.escape(_ATTENTION_EVENTS_DES, quote=False)}</des>",
    ]
    for event in selected:
        attrs = [
            f'type="{html.escape(event.event_type, quote=True)}"',
            f'name="{html.escape(event.name, quote=True)}"',
        ]
        if age := _resolved_age(event, compare_now):
            attrs.append(f'age="{html.escape(age, quote=True)}"')
        attrs.append(f'level="{html.escape(event.level, quote=True)}"')
        lines.append(f"  <event {' '.join(attrs)}/>")
    lines.append("</attention_events>")
    return "\n".join(lines)


def build_attention_events_xml(
    *,
    current_platform: str = "",
    now: datetime | None = None,
    registry: Any | None = None,
) -> str:
    events = collect_attention_events(
        current_platform=current_platform,
        now=now,
        registry=registry,
    )
    return render_attention_events(events, now=now)
