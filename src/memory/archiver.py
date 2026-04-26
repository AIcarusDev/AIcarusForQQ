"""Automatic background memory archiving."""

import asyncio
import logging

logger = logging.getLogger("AICQ.archiver")

from .archive_memories import ARCHIVE_GEN, TOOL as ARCHIVE_TOOL, read_result as read_archive_result
from .archive_prompt import ARCHIVE_SYSTEM_PROMPT

_SEM = asyncio.Semaphore(2)
_DEFAULT_CONTEXT_TURNS = 5
_DEFAULT_MAX_PER_TURN = 3


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
                sid = str(message.get("sender_id") or "").strip()
                if sid:
                    lines.append(f"User:qq_{sid}({name}): {content}")
                else:
                    lines.append(f"User({name}): {content}")
            elif role == "bot":
                lines.append(f"我 (Bot:self): {content}")

        if not lines:
            return

        dialogue = f"[场景: {session.conv_type}/{session.conv_id}]\n" + "\n".join(lines)

        adapter = app_state.adapter
        if adapter is None:
            return

        try:
            raw = await asyncio.to_thread(
                adapter._call_forced_tool,
                ARCHIVE_SYSTEM_PROMPT,
                dialogue,
                ARCHIVE_GEN,
                ARCHIVE_TOOL,
                "archiver",
            )
        except Exception:
            logger.debug("[archiver] archive_memories 调用异常", exc_info=True)
            return

        events_in, assertions_in = read_archive_result(raw)
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
                    # User#qq_xxx -> User:qq_xxx (容错书写差异)
                    if entity_text.startswith("User#qq_"):
                        entity_text = "User:qq_" + entity_text[len("User#qq_"):]
                    if entity_text in ("User", "Self"):
                        if not sender_id:
                            continue
                        entity_text = f"User:qq_{sender_id}"
                    elif entity_text == "Bot":
                        entity_text = "Bot:self"
                    elif entity_text.startswith("User(") and entity_text.endswith(")"):
                        # 丢弃只有昵称没有 qq_id 的引用 (该误误导致独立节点)
                        logger.debug("[archiver] 丢弃无 qq_id 的 User 引用: %s", entity_text)
                        continue
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