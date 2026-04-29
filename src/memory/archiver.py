"""Automatic background memory archiving."""

import asyncio
import logging

logger = logging.getLogger("AICQ.memory.archiver")

from .archive_memories import ARCHIVE_GEN, TOOL as ARCHIVE_TOOL, read_result as read_archive_result
from .archive_prompt import ARCHIVE_SYSTEM_PROMPT

_SEM = asyncio.Semaphore(2)
_DEFAULT_CONTEXT_TURNS = 5
_DEFAULT_MAX_PER_TURN = 3

# 各会话最近一次成功归档时的窗口指纹：key=(conv_type, conv_id), value=md5
# 用模块级哈希表而非 session 字段，可彻底避免不同会话/同一 session 对象被复用时的串扰。
# 持久化到 archive_signatures 表，进程重启后不会丢失。
_LAST_ARCHIVED_SIG: dict[tuple[str, str], str] = {}
_sig_loaded: bool = False


async def _ensure_sig_loaded() -> None:
    """首次使用时从数据库加载签名缓存（懒加载，只跑一次）。"""
    global _sig_loaded
    if _sig_loaded:
        return
    try:
        from database import load_archive_signatures
        loaded = await load_archive_signatures()
        _LAST_ARCHIVED_SIG.update(loaded)
        logger.debug("[archiver] 从数据库加载了 %d 条归档签名", len(loaded))
    except Exception:
        logger.warning("[archiver] 加载归档签名失败，本次按空签名运行", exc_info=True)
    _sig_loaded = True


async def _persist_signature(sess_key: tuple[str, str], signature: str) -> None:
    """将签名变更持久化到数据库（fire-and-forget 风格，失败不阻塞归档）。"""
    try:
        from database import save_archive_signature
        await save_archive_signature(sess_key[0], sess_key[1], signature)
    except Exception:
        logger.debug("[archiver] 签名持久化失败 (%s/%s)", sess_key[0], sess_key[1], exc_info=True)

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
        # 仅以 message_id 集合作为指纹：稳定且不受相对时间戳/昵称刷新/_extract_text 文本归一化影响。
        # 只要本窗口内没有新消息进入（bot 选择 wait/sleep 时即此情形），签名就保持不变。
        # 抢占式写入：在发起 LLM 调用前就把签名占住，使并发排队的任务能在信号量内立即跳过，
        # 避免因 LLM 往返耗时长导致同一窗口被多个并发 task 重复抽取。
        # 用模块级 dict 按 (conv_type, conv_id) 维护，跨会话/换群天然隔离。
        # 签名持久化到 archive_signatures 表，进程重启后可恢复。
        await _ensure_sig_loaded()
        import hashlib
        sess_key: tuple[str, str] = (str(session.conv_type), str(session.conv_id))
        mid_list = [str(m.get("message_id", "")) for m in msgs if m.get("message_id") is not None]
        sig_src = f"{sess_key[0]}/{sess_key[1]}|" + ",".join(mid_list)
        signature = hashlib.md5(sig_src.encode("utf-8", errors="ignore")).hexdigest()
        if signature == _LAST_ARCHIVED_SIG.get(sess_key, ""):
            logger.debug("[archiver] 窗口未变化，跳过本次归档 (%s/%s sig=%s...)", sess_key[0], sess_key[1], signature[:8])
            return
        prev_signature = _LAST_ARCHIVED_SIG.get(sess_key, "")
        logger.debug(
            "[archiver] 签名变化，触发归档 (%s/%s new=%s... old=%s... mids=%d)",
            sess_key[0], sess_key[1], signature[:8], prev_signature[:8] if prev_signature else "<empty>", len(mid_list),
        )
        _LAST_ARCHIVED_SIG[sess_key] = signature
        await _persist_signature(sess_key, signature)

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
            _LAST_ARCHIVED_SIG[sess_key] = prev_signature
            await _persist_signature(sess_key, prev_signature)
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
            # LLM 失败：回滚抢占的签名，让下一轮可重试同一窗口。
            _LAST_ARCHIVED_SIG[sess_key] = prev_signature
            await _persist_signature(sess_key, prev_signature)
            return

        events_in = read_archive_result(raw)
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