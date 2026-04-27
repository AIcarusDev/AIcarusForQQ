"""Automatic background memory archiving."""

import asyncio
import logging

logger = logging.getLogger("AICQ.memory.archiver")

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

        from .repo.events import (
            merge_event_occurrence as _db_merge_occurrence,
            prefetch_candidates_for_archiver as _db_prefetch,
            write_event as _db_write_event,
        )
        from .tokenizer import register_word as _register, tokenize as _tokenize

        cfg = app_state.config.get("memory", {}).get("auto_archive", {})
        if not cfg.get("enabled", True):
            return

        context_turns = int(cfg.get("context_turns", _DEFAULT_CONTEXT_TURNS))
        max_per_turn = int(cfg.get("max_per_turn", _DEFAULT_MAX_PER_TURN))

        # tool_calls_log 保留参数位以兼容调用点；write_memory 已下线，本函数不再读取。
        del tool_calls_log

        msgs = session.context_messages[-(context_turns * 2):]
        if not any(message.get("role") == "user" for message in msgs):
            return

        # 直接复用主循环用的 XML 聊天记录格式（已含 <self>/<other>/<sender id="..."/> 身份信息），
        # 让抽取模型与主循环看到同样的视角，减少"我 vs 别人"的人称翻译负担。
        try:
            chat_xml = session.get_chat_log_display()
        except Exception:
            logger.debug("[archiver] get_chat_log_display 失败，回退到简化文本", exc_info=True)
            chat_xml = ""

        if chat_xml:
            dialogue = f"[场景: {session.conv_type}/{session.conv_id}]\n{chat_xml}"
        else:
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

        # ── 变化触发：窗口内容自上次归档以来未变化（例如本轮 bot 只是 wait/sleep）则跳过 ──
        import hashlib
        signature = hashlib.md5(dialogue.encode("utf-8", errors="ignore")).hexdigest()
        if signature == getattr(session, "_last_archived_signature", ""):
            logger.debug("[archiver] 窗口未变化，跳过本次归档 (sig=%s...)", signature[:8])
            return

        # ── Read-Before-Write: 预取可能重复的旧事件，注入 <existing_candidates> ──
        sender_entity = f"User:qq_{sender_id}" if sender_id else ""
        if session.conv_type == "group":
            context_scope = f"group:qq_{session.conv_id}"
        elif session.conv_type == "private":
            context_scope = f"private:qq_{session.conv_id}"
        else:
            context_scope = ""

        candidates: list[dict] = []
        try:
            candidates = await _db_prefetch(
                sender_entity=sender_entity,
                context_scope=context_scope,
                dialogue_text=dialogue,
                limit=8,
            )
        except Exception:
            logger.debug("[archiver] 候选预取失败，跳过 Read-Before-Write", exc_info=True)
            candidates = []

        valid_candidate_ids: set[int] = {int(c["event_id"]) for c in candidates}

        if candidates:
            cand_lines: list[str] = ["<existing_candidates>"]
            for c in candidates:
                role_brief = ", ".join(
                    f"{r['role']}=" + (
                        r["entity"] if r.get("entity")
                        else (f'"{r["value_text"]}"' if r.get("value_text") else f"→#{r.get('target_event')}")
                    )
                    for r in (c.get("roles") or [])
                )
                cand_lines.append(
                    f"#{c['event_id']}  ctx={c.get('context_type','')} "
                    f"pol={c.get('polarity','')}  | {c.get('summary','')} "
                    f"| roles: {role_brief}"
                )
            cand_lines.append("</existing_candidates>")
            dialogue = dialogue + "\n\n" + "\n".join(cand_lines)

        adapter = app_state.adapter
        if adapter is None:
            return

        gen_cfg = cfg.get("generation", {})
        archive_gen = {
            "temperature": float(gen_cfg.get("temperature", ARCHIVE_GEN["temperature"])),
            "max_output_tokens": int(gen_cfg.get("max_output_tokens", ARCHIVE_GEN["max_output_tokens"])),
        }

        try:
            raw = await asyncio.to_thread(
                adapter._call_forced_tool,
                ARCHIVE_SYSTEM_PROMPT,
                dialogue,
                archive_gen,
                ARCHIVE_TOOL,
                "archiver",
            )
        except Exception:
            logger.debug("[archiver] archive_memories 调用异常", exc_info=True)
            return

        events_in = read_archive_result(raw)
        # LLM 调用已成功（无论是否提取到事件），将窗口标记为已处理，
        # 避免下一 round 在窗口未变时重复调用。
        session._last_archived_signature = signature
        if not events_in:
            return

        written = 0
        merged = 0
        for event in events_in:
            if written + merged >= max_per_turn:
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

            # ── Read-Before-Write 决策 ──
            merge_into_raw = event.get("merge_into")
            supersedes_raw = event.get("supersedes")
            merge_into_id: int | None = None
            supersedes_id: int | None = None
            try:
                if merge_into_raw is not None:
                    mid = int(merge_into_raw)
                    if mid in valid_candidate_ids:
                        merge_into_id = mid
                    else:
                        logger.debug(
                            "[archiver] 丢弃越权 merge_into=%s (不在候选 %s 内)",
                            mid, sorted(valid_candidate_ids),
                        )
            except (TypeError, ValueError):
                pass
            try:
                if supersedes_raw is not None:
                    sid_v = int(supersedes_raw)
                    if sid_v in valid_candidate_ids:
                        supersedes_id = sid_v
                    else:
                        logger.debug(
                            "[archiver] 丢弃越权 supersedes=%s (不在候选 %s 内)",
                            sid_v, sorted(valid_candidate_ids),
                        )
            except (TypeError, ValueError):
                pass

            # merge_into 优先于 supersedes (同时给时按合并处理)
            if merge_into_id is not None:
                try:
                    ok = await _db_merge_occurrence(merge_into_id)
                    if ok:
                        logger.info(
                            "[archiver] 合并到 event#%d (occurrences+1) | %s",
                            merge_into_id, summary,
                        )
                        merged += 1
                    else:
                        logger.debug("[archiver] merge_into=%d 已失效", merge_into_id)
                except Exception:
                    logger.warning("[archiver] merge 失败 id=%s", merge_into_id, exc_info=True)
                continue

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
                    supersedes=supersedes_id,
                )
                role_brief = "/".join(
                    f"{role['role']}:{role['entity'] or role['value_text']}"
                    for role in normalized_roles
                )
                supersedes_note = f" supersedes#{supersedes_id}" if supersedes_id else ""
                logger.info(
                    "[archiver] 写入 event#%d type=%s ctx=%s%s | %s | %s",
                    event_id,
                    event_type,
                    context_type,
                    supersedes_note,
                    summary,
                    role_brief,
                )
                written += 1
            except Exception:
                logger.warning("[archiver] event 写入失败：%s", summary, exc_info=True)

        if written or merged:
            logger.info(
                "[archiver] 本轮自动归档：新增 %d / 合并 %d 条事件",
                written, merged,
            )