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
    # 7天以上：附上绝对日期，便于主模型判断记忆的时效性
    dt = datetime.fromtimestamp(created_at_ms / 1000, tz=timezone.utc)
    date_str = dt.strftime("%Y-%m-%d")
    if delta_sec < 86400 * 30:
        return f"{date_str}({delta_sec // (86400 * 7)}周前)"
    return f"{date_str}({delta_sec // (86400 * 30)}个月前)"


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
        "\n  confidence: 0.9+ 基本当事实用; 0.7-0.9 可用但措辞稍留余地; "
        "0.4-0.7 仅作参考,需验证; <0.4 是八卦/玩笑。"
        "\n  scene: 事件发生的群/私聊场景；type: 事件语义类型。"
        "\n  modality=hypothetical/possible: 该事件是假设或推测，不要当作已发生事实。"
        "\n  context=evidence: 该事件是他人转述/推测，非当事人直接陈述，措辞须留余地（\"据说\"\"好像\"等）。"
        "\n  若多条事件矛盾: 看 confidence 与 when (越新越优先)"
        "</des>"
    )

    for event in events:
        event_id = event.get("event_id", "?")
        summary = event.get("summary", "")
        confidence = float(event.get("confidence", 0.6))
        age = _age_text(event.get("occurred_at", 0), now)
        modality = event.get("modality", "actual")
        context_type = event.get("context_type", "episodic")
        event_type = (event.get("event_type") or "").strip()
        conv_name = (event.get("conv_name") or "").strip()
        attrs = (
            f'id="{html.escape(str(event_id))}" '
            f'confidence="{confidence:.2f}" when="{html.escape(age)}"'
        )
        if event_type:
            attrs += f' type="{html.escape(event_type)}"'
        if conv_name:
            attrs += f' scene="{html.escape(conv_name)}"'
        # 假设/反事实事件显式标注，避免主模型将其当作事实
        if modality != "actual":
            attrs += f' modality="{html.escape(modality)}"'
        if context_type == "hypothetical":
            attrs += ' context="hypothetical"'
        elif context_type == "evidence":
            attrs += ' context="evidence"'
        lines.append(f"  <event {attrs}>")
        lines.append(f"    {html.escape(str(summary))}")
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
