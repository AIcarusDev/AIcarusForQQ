"""Memory XML rendering helpers (events-only)."""

from datetime import datetime, timezone
import html


def _age_text(created_at_ms: int, now: datetime) -> str:
    delta_sec = int(now.timestamp() - created_at_ms / 1000)
    if delta_sec < 60:
        return "刚刚"
    if delta_sec < 3600:
        return f"{delta_sec // 60}分钟前"
    if delta_sec < 86400:
        return f"{delta_sec // 3600}小时前"
    if delta_sec < 86400 * 7:
        return f"{delta_sec // 86400}天前"
    if delta_sec < 86400 * 30:
        return f"{delta_sec // (86400 * 7)}周前"
    return f"{delta_sec // (86400 * 30)}个月前"


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
    lines.append(
        "  <des>"
        "\n  人物标识: \"我\"=自己; 其他人=nickname#qq_id (无昵称仅 qq_id); "
        "若同名，则凭借 qq_id 区分。"
        "\n    confidence: 0.9+ 基本当事实用; 0.7-0.9 可用但措辞稍留余地; "
        "0.4-0.7 仅作参考,需验证; <0.4 是八卦/玩笑。"
        "\n  若多条事件矛盾: 看 confidence 与 when (越新越优先)"
        "</des>"
    )

    for event in events:
        event_id = event.get("event_id", "?")
        summary = event.get("summary", "")
        confidence = float(event.get("confidence", 0.6))
        modality = str(event.get("modality") or "actual")
        context_type = str(event.get("context_type") or "episodic")
        recall_score = event.get("recall_score")
        recall_path = event.get("recall_path") or []
        age = _age_text(event.get("occurred_at", 0), now)
        attrs = (
            f'id="{html.escape(str(event_id))}" '
            f'confidence="{confidence:.2f}" '
            f'modality="{html.escape(modality)}" '
            f'context="{html.escape(context_type)}" '
            f'when="{html.escape(age)}"'
        )
        if recall_score is not None:
            attrs += f' recall_score="{html.escape(str(recall_score))}"'
        lines.append(f"  <event {attrs}>")
        lines.append(f"    {html.escape(str(summary))}")
        if recall_path:
            lines.append(f"    <recall_path>{html.escape(' -> '.join(map(str, recall_path)))}</recall_path>")
        lines.append("  </event>")
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
