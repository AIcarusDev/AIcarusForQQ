"""Automatic background memory archiving."""

import asyncio
import logging

logger = logging.getLogger("AICQ.archiver")

_SEM = asyncio.Semaphore(2)
_DEFAULT_CONTEXT_TURNS = 5
_DEFAULT_MAX_PER_TURN = 3

_ARCHIVE_TOOL_DECLARATION: dict = {
    "name": "archive_memories",
    "description": (
        "从给定对话片段中提取两类结构化记忆: events(多角色事件) 与 assertions(静态本体二元事实), "
        "通过本工具的参数返回。无可提取内容时仍调用本工具并把两个数组都填空。\n\n"
        "=== EVENTS (首选) ===\n"
        "涉及多方参与者、Bot 自我承诺、临时状态、会随时间变化的事实，全部用 event。\n"
        "=== ASSERTIONS (仅限永久本体) ===\n"
        "只用于绝不会随时间改变的本体属性，如 'Python isA 编程语言'、'User 职业是 程序员'。\n"
        "拿不准是否永久 -> 用 event 而非 assertion。\n\n"
        "=== 黄金规则 (违反会被静默丢弃) ===\n"
        "1. 涉及「教/学/告诉/纠正/问/反驳/答应/拒绝」的句子必须用 event, 且: "
        "agent=实施动作的人 / recipient=听众 / theme=内容。\n"
        "   反例(错): subject='User', predicate='学习到', object='X' (搞错主语视角)\n"
        "   正例(对): event_type='teaching', roles=[{role:'agent',entity:'User'},"
        "{role:'recipient',entity:'Bot'},{role:'theme',value_text:'X'}]\n"
        "2. 否定不要造一个「不喜欢」谓词, 用 polarity='negative'。\n"
        "3. 假设/反事实用 modality='hypothetical', 不要丢弃也不要当作事实。\n"
        "4. 会随时间变化的事实 (年龄/状态/今天的天气/正在做某事) 必须是 event, 禁止进 assertions。\n"
        "5. Bot 在角色扮演中说的话, event 应标 context_type='contract', 不要污染 meta。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "description": "Neo-Davidsonian 多角色事件列表。",
                "items": {
                    "type": "object",
                    "properties": {
                        "event_type": {
                            "type": "string",
                            "description": "简短事件标签, 如 teaching/correcting/asking/sharing/liking/disliking/promising/experiencing。",
                        },
                        "summary": {
                            "type": "string",
                            "description": "一句话事件摘要 (<=30 字), 用于检索与渲染。",
                        },
                        "polarity": {
                            "type": "string",
                            "enum": ["positive", "negative"],
                            "description": "表达否定意图时用 negative, 不要塞进 event_type。",
                        },
                        "modality": {
                            "type": "string",
                            "enum": ["actual", "hypothetical", "possible"],
                            "description": "事实用 actual, 「如果」「可能」用对应值。",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "0.0~1.0, 事实约 0.7, 推测约 0.4。",
                        },
                        "context_type": {
                            "type": "string",
                            "enum": ["meta", "contract", "episodic"],
                            "description": (
                                "meta=Bot 永久自我认知 (跨所有会话恒激活); "
                                "contract=角色扮演/临时承诺 (可被撤销); "
                                "episodic=普通对话事件 (默认)。"
                            ),
                        },
                        "recall_scope": {
                            "type": "string",
                            "description": "global | group:qq_{group_id} | private:qq_{user_id} (依对话片段开头 [场景:] 决定)。",
                        },
                        "roles": {
                            "type": "array",
                            "description": "参与者数组。",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {
                                        "type": "string",
                                        "enum": [
                                            "agent", "patient", "theme", "recipient",
                                            "instrument", "location", "time", "attribute",
                                        ],
                                        "description": "角色名, 仅这 8 个取值。",
                                    },
                                    "entity": {
                                        "type": "string",
                                        "description": (
                                            "实体标识。'User' -> 自动替换为当前发言用户; "
                                            "'Bot' -> 自动替换为 Bot:self; 其他保留为外部实体。"
                                        ),
                                    },
                                    "value_text": {
                                        "type": "string",
                                        "description": "当承载是一段文本/概念而非已知实体时使用 (如 theme 是被传授的内容)。",
                                    },
                                },
                                "required": ["role"],
                            },
                        },
                    },
                    "required": ["event_type", "summary", "roles"],
                },
            },
            "assertions": {
                "type": "array",
                "description": "静态本体二元事实列表 (仅限永久属性)。",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "'User' (当前用户) / 'Bot' (Bot 自己) / 其他外部实体名。",
                        },
                        "predicate": {
                            "type": "string",
                            "description": "二元谓词, 如 'isA' / '职业是' / '生于'。",
                        },
                        "object_text": {
                            "type": "string",
                            "description": "宾语文本。",
                        },
                        "recall_scope": {
                            "type": "string",
                            "description": "global | group:qq_{group_id} | private:qq_{user_id}。",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "0.0~1.0。",
                        },
                    },
                    "required": ["subject", "predicate", "object_text"],
                },
            },
        },
        "required": ["events", "assertions"],
    },
}

_EXTRACT_SYSTEM = (
    "你是记忆提取助手。本任务以函数调用形式工作: 你必须且只能调用工具 archive_memories, "
    "通过其参数返回从对话片段中提取的结构化记忆 (events 与 assertions)。\n"
    "目标: 让 Bot 在未来对话中能正确召回「谁对谁做了什么」, 以及不变的本体属性。\n"
    "工具的字段含义、取值范围与黄金规则参见 archive_memories 的描述与参数 schema, "
    "严格遵守; 无可提取内容时仍要调用工具并把两个数组都填空。"
)

_ARCHIVE_GEN: dict = {
    "temperature": 0.3,
    "max_output_tokens": 5000,
}


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content) if content else ""


async def archive_turn_memories(
    session,
    sender_id: str,
    tool_calls_log: list[dict],
) -> None:
    async with _SEM:
        import app_state

        from .repo.events import write_event as _db_write_event
        from .service import add_memory
        from .tokenizer import register_word as _register, tokenize as _tokenize

        cfg = app_state.config.get("memory", {}).get("auto_archive", {})
        if not cfg.get("enabled", True):
            return

        context_turns = int(cfg.get("context_turns", _DEFAULT_CONTEXT_TURNS))
        max_per_turn = int(cfg.get("max_per_turn", _DEFAULT_MAX_PER_TURN))

        already_written: set[tuple[str, str]] = set()
        for call in tool_calls_log:
            if call.get("function") == "write_memory" and not call.get("circuit_broken"):
                args = call.get("arguments", {})
                predicate = args.get("predicate", "")
                object_text = args.get("object_text", "") or args.get("content", "")
                if predicate and object_text:
                    already_written.add((predicate, object_text))

        msgs = session.context_messages[-(context_turns * 2):]
        if not any(message.get("role") == "user" for message in msgs):
            return

        lines: list[str] = []
        for message in msgs:
            role = message.get("role", "")
            content = _extract_text(message.get("content", ""))
            if not content:
                continue
            if role == "user":
                name = message.get("sender_name") or "User"
                lines.append(f"User({name}): {content}")
            elif role == "bot":
                lines.append(f"Bot: {content}")

        if not lines:
            return

        dialogue = f"[场景: {session.conv_type}/{session.conv_id}]\n" + "\n".join(lines)

        adapter = app_state.adapter
        if adapter is None:
            return

        try:
            raw = await asyncio.to_thread(
                adapter._call_forced_tool,
                _EXTRACT_SYSTEM,
                dialogue,
                _ARCHIVE_GEN,
                _ARCHIVE_TOOL_DECLARATION,
                "archiver",
            )
        except Exception:
            logger.debug("[archiver] archive_memories 调用异常", exc_info=True)
            return

        if not isinstance(raw, dict):
            return

        events_in = raw.get("events") or []
        assertions_in = raw.get("assertions") or []
        if not isinstance(events_in, list):
            events_in = []
        if not isinstance(assertions_in, list):
            assertions_in = []
        if not events_in and not assertions_in:
            return

        written = 0
        for event in events_in:
            if written >= max_per_turn:
                break
            if not isinstance(event, dict):
                continue

            event_type = str(event.get("event_type", "")).strip() or "unspecified"
            summary = str(event.get("summary", "")).strip()
            if not summary:
                continue

            polarity = str(event.get("polarity", "positive")).strip().lower()
            modality = str(event.get("modality", "actual")).strip().lower()
            context_type = str(event.get("context_type", "episodic")).strip().lower()
            recall_scope = str(event.get("recall_scope") or "global").strip()
            try:
                confidence = float(event.get("confidence", 0.6))
            except (TypeError, ValueError):
                confidence = 0.6
            confidence = max(0.1, min(1.0, confidence))

            valid_prefixes = ("group:qq_", "private:qq_")
            if not (
                recall_scope == "global"
                or any(recall_scope.startswith(prefix) for prefix in valid_prefixes)
            ):
                recall_scope = "global"

            roles_in = event.get("roles") or []
            if not isinstance(roles_in, list):
                continue
            normalized_roles: list[dict] = []
            for role in roles_in:
                if not isinstance(role, dict):
                    continue
                role_name = str(role.get("role", "")).strip().lower()
                entity = role.get("entity")
                value_text = role.get("value_text")
                if entity:
                    entity_text = str(entity).strip()
                    if entity_text in ("User", "Self"):
                        if not sender_id:
                            continue
                        entity_text = f"User:qq_{sender_id}"
                    elif entity_text == "Bot":
                        entity_text = "Bot:self"
                    entity = entity_text
                if value_text is not None:
                    value_text = str(value_text).strip() or None
                if not entity and not value_text:
                    continue
                normalized_roles.append({
                    "role": role_name,
                    "entity": entity,
                    "value_text": value_text,
                    "value_tok": _tokenize(value_text) if value_text else "",
                })

            if not normalized_roles:
                logger.debug("[archiver] event 无有效角色边，跳过：%s", summary)
                continue

            try:
                _register(summary)
                summary_tok = _tokenize(summary)
                event_id = await _db_write_event(
                    event_type=event_type,
                    summary=summary,
                    summary_tok=summary_tok,
                    polarity=polarity,
                    modality=modality,
                    confidence=confidence,
                    context_type=context_type,
                    recall_scope=recall_scope,
                    source="自动归档",
                    reason="从对话中自动提取",
                    conv_type=session.conv_type,
                    conv_id=session.conv_id,
                    conv_name=session.conv_name,
                    roles=normalized_roles,
                )
                role_brief = "/".join(
                    f"{role['role']}:{role['entity'] or role['value_text']}"
                    for role in normalized_roles
                )
                logger.info(
                    "[archiver] 写入 event#%d type=%s ctx=%s | %s | %s",
                    event_id,
                    event_type,
                    context_type,
                    summary,
                    role_brief,
                )
                written += 1
            except Exception:
                logger.warning("[archiver] event 写入失败：%s", summary, exc_info=True)

        for item in assertions_in:
            if written >= max_per_turn:
                break
            if not isinstance(item, dict):
                continue

            predicate = str(item.get("predicate", "")).strip()
            object_text = str(item.get("object_text", "")).strip()
            recall_scope = str(item.get("recall_scope") or "global").strip()
            valid_prefixes = ("global", "group:qq_", "private:qq_")
            if not any(recall_scope == "global" or recall_scope.startswith(prefix) for prefix in valid_prefixes):
                recall_scope = "global"
            try:
                confidence = float(item.get("confidence", 0.6))
            except (TypeError, ValueError):
                confidence = 0.6
            confidence = max(0.1, min(1.0, confidence))
            if not predicate or not object_text:
                continue

            raw_subject = str(item.get("subject", "User")).strip() or "User"
            if raw_subject in ("User", "Self"):
                if sender_id:
                    subject = f"User:qq_{sender_id}"
                else:
                    logger.warning(
                        "[archiver] sender_id 为空，subject 回退为 UnknownUser；"
                        "同 predicate/object 记忆将被 UNIQUE 索引静默去重"
                    )
                    subject = "UnknownUser"
            elif raw_subject == "Bot":
                subject = "Bot:self"
            else:
                subject = raw_subject

            if (predicate, object_text) in already_written:
                logger.debug("[archiver] 跳过（本轮已写）: %s / %s", predicate, object_text)
                continue

            try:
                await add_memory(
                    content=object_text,
                    predicate=predicate,
                    source="自动归档",
                    reason="从对话中自动提取",
                    conv_type=session.conv_type,
                    conv_id=session.conv_id,
                    conv_name=session.conv_name,
                    subject=subject,
                    origin="passive",
                    recall_scope=recall_scope,
                    confidence=confidence,
                )
                logger.info("[archiver] 写入: [%s] %s → %s", subject, predicate, object_text)
                written += 1
            except Exception:
                logger.warning(
                    "[archiver] 写入失败 %s / %s",
                    predicate,
                    object_text,
                    exc_info=True,
                )

        if written:
            logger.info("[archiver] 本轮自动归档 %d 条记忆", written)