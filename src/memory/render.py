"""Memory XML rendering helpers."""

from datetime import datetime, timezone
import html

from . import runtime

_SELF_HIGH_CONF: float = 0.65
_RELATIONSHIP_KEYWORDS: frozenset[str] = frozenset({
    "对我", "对bot", "对Bot", "对AI", "对ai",
    "关系", "印象", "期待", "期望", "看法",
    "评价", "信任", "亲近", "依赖", "态度",
    "感情", "好感", "不满", "喜欢", "讨厌",
})


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


def _source_display(source: str, conv_name: str, conv_id: str) -> str:
    if conv_id:
        suffix = f"{conv_name}({conv_id})" if conv_name else conv_id
        return f"{source} · {suffix}"
    return source


def _is_relationship_predicate(predicate: str) -> bool:
    return any(keyword in predicate for keyword in _RELATIONSHIP_KEYWORDS)


def _render_memory_block(
    tag: str,
    entries: list[dict],
    now: datetime,
    total_hint: int = 0,
) -> str:
    if not entries:
        return f'<{tag} items="0"/>'

    lines = [f'<{tag} items="{len(entries)}">  <!-- total_pool={total_hint} -->']
    if tag == "about_relationship":
        lines.append("  <des>这是对方对我的态度和我们之间关系的认知，不一定准确无误</des>")
    for memory in entries:
        subject = memory.get("subject", "")
        predicate = memory.get("predicate", "")
        content = memory.get("object_text", memory.get("content", ""))
        age = _age_text(memory.get("created_at", 0), now)
        source_text = _source_display(
            memory.get("source", ""),
            memory.get("conv_name", ""),
            memory.get("conv_id", ""),
        )
        memory_id = str(memory.get("id", "?"))
        lines.append(f'  <item id="{memory_id}">')
        if predicate and not (predicate.startswith("[") and predicate.endswith("]")):
            lines.append(f'    <subject>{html.escape(subject)}</subject>')
            lines.append(f'    <predicate>{html.escape(predicate)}</predicate>')
        lines.append(f'    <content>{html.escape(content)}</content>')
        lines.append(f'    <age>{age}</age>')
        lines.append(f'    <source>{html.escape(source_text)}</source>')
        if memory.get("reason"):
            lines.append(f'    <reason>{html.escape(memory["reason"])}</reason>')
        lines.append("  </item>")
    lines.append(f"</{tag}>")
    return "\n".join(lines)


def _render_self_block(entries: list[dict], now: datetime) -> str:
    if not entries:
        return '<about_self items="0"/>'

    lines = [f'<about_self items="{len(entries)}">']
    lines.append("  <des>这是我自己在过去对话中表达过的喜好、感受与观点</des>")
    for memory in entries:
        predicate = memory.get("predicate", "")
        content = memory.get("object_text", memory.get("content", ""))
        confidence = float(memory.get("confidence", 0.6))
        age = _age_text(memory.get("created_at", 0), now)
        memory_id = str(memory.get("id", "?"))
        if confidence >= _SELF_HIGH_CONF:
            summary = f"我 {predicate} {content}"
        else:
            summary = f"我在{age}说过：{predicate} {content}"
        lines.append(
            f'  <item id="{memory_id}" confidence="{confidence:.2f}">{html.escape(summary)}</item>'
        )
    lines.append("</about_self>")
    return "\n".join(lines)


def _render_events_block(
    events: list[dict],
    now: datetime,
    sender_entity: str = "",
    nickname_map: dict[str, str] | None = None,
    bot_nickname: str = "",
) -> str:
    if not events:
        return '<recent_events items="0"/>'

    nickname_map = nickname_map or {}

    def _humanize(entity: str) -> str:
        if not entity:
            return ""
        if entity == "Bot:self":
            return "我"
        if entity.startswith("User:qq_"):
            qq_id = entity[len("User:qq_"):]
            nickname = nickname_map.get(qq_id, "")
            return f"{nickname}#qq_{qq_id}" if nickname else f"qq_{qq_id}"
        if entity.startswith("Person:"):
            return entity
        if entity.startswith("Group:qq_"):
            qq_id = entity[len("Group:qq_"):]
            return f"群#qq_{qq_id}"
        return entity

    lines = [f'<recent_events items="{len(events)}">']
    lines.append(
        "  <des>这些是被检索到的多角色事件(agent=施事者 patient=受事者 "
        "theme=内容/客体 recipient=接收者)。\"我\" 指 Bot 自己;"
        "其他人一律以 nickname#qq_id 形式标识(无昵称时仅 qq_id),"
        "同名靠 qq_id 后缀区分,绝不要把不同 qq_id 的人当成同一人。"
        "polarity=negative 表示否定,modality=hypothetical/possible 表示假设而非事实,"
        "context_type=meta 是我永久自我认知,context_type=contract 是临时角色扮演承诺。</des>"
    )
    if sender_entity:
        lines.append(f'  <current_speaker>{html.escape(_humanize(sender_entity))}</current_speaker>')

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
            payload = _humanize(role.get("entity") or "") or (role.get("value_text") or "")
            if not payload:
                continue
            role_parts.append(f"{role_name}={html.escape(str(payload))}")
        roles_text = " / ".join(role_parts) if role_parts else "—"
        lines.append(f'  <event {attrs}>{html.escape(summary)} :: {roles_text}</event>')
    lines.append("</recent_events>")
    return "\n".join(lines)


def build_memory_xml(
    now: datetime | None = None,
    recalled: list[dict] | None = None,
    recalled_events: list[dict] | None = None,
    sender_entity: str = "",
    nickname_map: dict[str, str] | None = None,
    bot_nickname: str = "",
) -> str:
    if now is None:
        now = datetime.now(timezone.utc)

    if recalled is not None:
        runtime.set_last_recalled_ids({row["id"] for row in recalled if "id" in row})
        all_entries = recalled
    else:
        all_entries = runtime.get_all_cached_entries()

    self_entries: list[dict] = []
    user_entries: list[dict] = []
    relationship_entries: list[dict] = []
    for memory in all_entries:
        if memory.get("subject", "") == "Bot:self":
            self_entries.append(memory)
        elif _is_relationship_predicate(memory.get("predicate", "")):
            relationship_entries.append(memory)
        else:
            user_entries.append(memory)

    user_block = _render_memory_block(
        "about_user",
        user_entries,
        now,
        total_hint=runtime.get_total_pool_size(),
    )
    relationship_block = _render_memory_block(
        "about_relationship",
        relationship_entries,
        now,
        total_hint=runtime.get_total_pool_size(),
    )
    self_block = _render_self_block(self_entries, now)
    events_block = _render_events_block(
        recalled_events or [],
        now,
        sender_entity=sender_entity,
        nickname_map=nickname_map,
        bot_nickname=bot_nickname,
    )
    return f"{user_block}\n{relationship_block}\n{self_block}\n{events_block}"