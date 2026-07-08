"""Memory XML rendering helpers.

Normal render output is intentionally minimal: summary, relative time, and
confidence are visible to the model.  Scores, ids, predicates, participants,
and paths are debug-only data and must not be injected here.
"""

from __future__ import annotations

from datetime import datetime, timezone
import html


def _format_absolute_event_time(created_at_ms: int, now: datetime) -> str:
    del now
    try:
        dt = datetime.fromtimestamp(int(created_at_ms) / 1000, tz=timezone.utc)
    except Exception:
        dt = datetime.fromtimestamp(0, tz=timezone.utc)
    return dt.isoformat()


def _format_relative_event_time(created_at_ms: int, now: datetime) -> str:
    try:
        event_ms = int(created_at_ms)
    except Exception:
        event_ms = 0
    try:
        now_ms = int(now.timestamp() * 1000)
    except Exception:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    delta_seconds = max(0, (now_ms - event_ms) // 1000)
    if delta_seconds < 60:
        return f"{delta_seconds}秒前"
    minutes = delta_seconds // 60
    if minutes < 60:
        return f"{minutes}分钟前"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}小时前"
    days = hours // 24
    if days < 30:
        return f"{days}天前"
    months = days // 30
    if months < 12:
        return f"{months}个月前"
    years = days // 365
    return f"{max(1, years)}年前"


def _format_confidence(value: object) -> str:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return f"{confidence:.2f}"


def _render_memory_items(
    events: list[dict],
    now: datetime,
    sender_entity: str = "",
    nickname_map: dict[str, str] | None = None,
) -> str:
    del sender_entity, nickname_map
    if not events:
        return ""
    lines = []
    for event in events:
        summary = html.escape(str(event.get("summary", "")))
        occurred_at = int(event.get("occurred_at") or event.get("created_at") or 0)
        when = html.escape(_format_relative_event_time(occurred_at, now))
        confidence = html.escape(_format_confidence(event.get("confidence")))
        kind = "summary" if event.get("memory_kind") == "summary" else ""
        kind_attr = ' kind="summary"' if kind else ""
        lines.append(f'  <mem{kind_attr} when="{when}" confidence="{confidence}">{summary}</mem>')
    return "\n".join(lines)


def build_memory_xml(
    now: datetime | None = None,
    recalled_events: list[dict] | None = None,
    sender_entity: str = "",
    nickname_map: dict[str, str] | None = None,
) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    return _render_memory_items(
        recalled_events or [],
        now,
        sender_entity=sender_entity,
        nickname_map=nickname_map,
    )


def build_memory_debug_xml(
    now: datetime | None = None,
    recalled_events: list[dict] | None = None,
) -> str:
    """Render recall internals for logs/devtools only; never inject into model context."""

    if now is None:
        now = datetime.now(timezone.utc)
    events = recalled_events or []
    if not events:
        return '<memory_debug items="0"/>'
    lines = [f'<memory_debug items="{len(events)}">']
    for event in events:
        event_id = html.escape(str(event.get("event_id", "")))
        score = html.escape(str(event.get("recall_score", "")))
        path_cost = html.escape(str(event.get("recall_path_cost", "")))
        depth = html.escape(str(event.get("recall_path_depth", "")))
        reasons = html.escape(",".join(str(x) for x in event.get("recall_reasons", []) or []))
        event_type = html.escape(str(event.get("event_type", "")))
        when = html.escape(_format_absolute_event_time(int(event.get("occurred_at") or event.get("created_at") or 0), now))
        path = html.escape(" -> ".join(str(x) for x in event.get("recall_path", []) or []))
        summary = html.escape(str(event.get("summary", "")))
        lines.append(
            f'  <event id="{event_id}" when="{when}" score="{score}" '
            f'path_cost="{path_cost}" depth="{depth}" reasons="{reasons}" predicate="{event_type}">'
        )
        lines.append(f"    <summary>{summary}</summary>")
        lines.append(f"    <path>{path}</path>")
        lines.append("  </event>")
    lines.append("</memory_debug>")
    return "\n".join(lines)
