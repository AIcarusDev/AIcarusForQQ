"""Memory XML rendering helpers.

Event render output is intentionally minimal: ID, summary, relative time, and
confidence. All memories expose a readable internal ID;
scores, external source IDs, predicates and participants stay internal.
"""

from __future__ import annotations

from datetime import datetime, timezone
import html
import xml.etree.ElementTree as ET

from memory.access import memory_id


def omit_visible_tool_memories(block: str, responses: list) -> str:
    """Prefer an unchanged full tool result over its duplicate automatic memory.

    Recompute from visible history on each request; never hide future revisions
    or keep exclusions after the tool response has left the raw context.
    """
    returned = set()
    for response in responses:
        if response.namespace != "memory_manage" or not isinstance(response.response, dict):
            continue
        payload = response.response
        if response.name == "memory_search":
            items = payload.get("items", [])
        elif response.name == "memory_read" and payload.get("found"):
            items = [payload.get("item")]
        else:
            continue
        for item in items:
            if isinstance(item, dict) and item.get("id") and isinstance(item.get("content"), str):
                returned.add((str(item["id"]), item["content"]))
    if not returned or not isinstance(block, str) or not block.strip():
        return block
    try:
        root = ET.fromstring(block)
    except ET.ParseError:
        return block
    if root.tag != "memory":
        return block
    changed = False
    for entry in list(root):
        if entry.tag == "mem" and (entry.get("id", ""), entry.text or "") in returned:
            root.remove(entry)
            changed = True
    return ET.tostring(root, encoding="unicode") if changed else block


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
        identity = html.escape(memory_id(event), quote=True)
        id_attr = f' id="{identity}"' if identity else ""
        if event.get("memory_kind") == "active":
            lines.append(f'  <mem kind="active"{id_attr}>{summary}</mem>')
            continue
        occurred_at = int(event.get("occurred_at") or event.get("created_at") or 0)
        when = html.escape(_format_relative_event_time(occurred_at, now))
        confidence = html.escape(_format_confidence(event.get("confidence")))
        kind = "summary" if event.get("memory_kind") == "summary" else ""
        kind_attr = ' kind="summary"' if kind else ""
        lines.append(f'  <mem{kind_attr}{id_attr} when="{when}" confidence="{confidence}">{summary}</mem>')
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
