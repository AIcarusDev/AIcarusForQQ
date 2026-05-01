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


def _humanize(entity: str, nickname_map: dict[str, str]) -> str:
    if not entity:
        return ""
    if entity == "Bot:self":
        return "我"
    if entity.startswith("User:qq_"):
        qq_id = entity[len("User:qq_"):]
        nickname = nickname_map.get(qq_id, "")
        return f"{nickname}#qq_{qq_id}" if nickname else f"qq_{qq_id}"
    if entity.startswith("Group:qq_"):
        qq_id = entity[len("Group:qq_"):]
        return f"群#qq_{qq_id}"
    return entity


def _render_events_block(
    events: list[dict],
    now: datetime,
    sender_entity: str = "",
    nickname_map: dict[str, str] | None = None,
) -> str:
    if not events:
        return '<recent_events items="0"/>'

    nickname_map = nickname_map or {}

    lines = [f'<recent_events items="{len(events)}">']
    lines.append(
        "  <des>多角色事件检索结果，是我的长期记忆。"
        "\n  角色: agent=施事 patient=受事 theme=内容/客体 recipient=接收方 "
        "instrument=工具 location=地点 time=时间 attribute=补充修饰。"
        "\n  人物标识: \"我\"=自己; 其他人=nickname#qq_id (无昵称仅 qq_id); "
        "同名一律靠 qq_id 区分,绝不合并不同 qq_id 的人。"
        "\n  字段读法:"
        "\n    pol=positive 正常陈述; pol=negative 当事人在表达拒绝/反对/否认 (例: 不喜欢香菜),"
        "复述时要保留否定语气,不要反过来当作肯定事实。"
        "\n    mod=actual 真实发生; mod=possible 仅是推测(\"可能/也许\"),"
        "复述时必须带不确定语气,不可断言; mod=hypothetical 是反事实假设(\"如果...\"),"
        "禁止当作真实事件引用。"
        "\n    ctx=meta 是永久自我认知;"
        "ctx=contract 是临时角色扮演承诺;"
        "ctx=episodic 是某次对话的具体事件。"
        "冲突时优先级 meta > contract > episodic。"
        "\n    confidence: 0.9+ 直接当事实用; 0.7-0.9 可用但措辞稍留余地; "
        "0.4-0.7 仅作参考,需追问验证; <0.4 是八卦/玩笑,不要主动复述。"
        "\n  当多条事件矛盾: 看 confidence 与 when (越新越优先),"
        "context_type=meta 的条目永远优先于 episodic。</des>"
    )

    role_order = [
        "agent", "recipient", "patient", "theme",
        "instrument", "location", "time", "attribute",
    ]
    for event in events:
        event_id = event.get("event_id", "?")
        event_type = event.get("event_type", "")
        summary = event.get("summary", "")
        polarity = event.get("polarity", "positive")
        modality = event.get("modality", "actual")
        context_type = event.get("context_type", "episodic")
        confidence = float(event.get("confidence", 0.6))
        age = _age_text(event.get("occurred_at", 0), now)
        attrs = (
            f'id="{event_id}" type="{html.escape(event_type)}" '
            f'ctx="{context_type}" pol="{polarity}" mod="{modality}" '
            f'confidence="{confidence:.2f}" when="{age}"'
        )

        roles = event.get("roles") or []
        roles_sorted = sorted(
            roles,
            key=lambda role: role_order.index(role["role"])
            if role.get("role") in role_order else 99,
        )
        role_parts: list[str] = []
        for role in roles_sorted:
            role_name = role.get("role", "")
            payload = _humanize(role.get("entity") or "", nickname_map) or (role.get("value_text") or "")
            if not payload:
                continue
            role_parts.append(f"{role_name}={html.escape(str(payload))}")
        roles_text = " / ".join(role_parts) if role_parts else "—"
        lines.append(f'  <event {attrs}>{html.escape(summary)} :: {roles_text}</event>')
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
