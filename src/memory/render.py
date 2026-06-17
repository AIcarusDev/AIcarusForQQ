"""Memory XML rendering helpers.

Normal render output is intentionally minimal: only summary plus absolute time
is visible to the model.  Scores, ids, predicates, participants, and paths are
debug-only data and must not be injected here.
"""

from __future__ import annotations

from datetime import datetime, timezone
import html


def _format_event_time(created_at_ms: int, now: datetime) -> str:
    del now
    try:
        dt = datetime.fromtimestamp(int(created_at_ms) / 1000, tz=timezone.utc)
    except Exception:
        dt = datetime.fromtimestamp(0, tz=timezone.utc)
    return dt.isoformat()


def _render_events_block(
    events: list[dict],
    now: datetime,
    sender_entity: str = "",
    nickname_map: dict[str, str] | None = None,
) -> str:
    del sender_entity, nickname_map
    if not events:
        return '<recent_events items="0"/>'
    lines = [f'<recent_events items="{len(events)}">']
    for event in events:
        summary = html.escape(str(event.get("summary", "")))
        occurred_at = int(event.get("occurred_at") or event.get("created_at") or 0)
        when = html.escape(_format_event_time(occurred_at, now))
        lines.append(f'  <event when="{when}">{summary}</event>')
    lines.append("</recent_events>")
    return "\n".join(lines)


def build_memory_xml(
    now: datetime | None = None,
    recalled_events: list[dict] | None = None,
    sender_entity: str = "",
    nickname_map: dict[str, str] | None = None,
) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    return _render_events_block(
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
        when = html.escape(_format_event_time(int(event.get("occurred_at") or event.get("created_at") or 0), now))
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
