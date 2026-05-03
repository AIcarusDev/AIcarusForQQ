"""Automatic background memory archiving.

设计要点（v2，支持优雅退出 + 重启续跑）
----------------------------------------------------
1. 入口 :func:`archive_turn_memories` 仍由 ``_schedule_archive`` (main_loop) 与
   ``routes_chat`` fire-and-forget 调用。它只负责"准备 payload"：
   - 检查/更新签名，触发条件是窗口有变化；
   - 拼装 dialogue（含 ``<existing_candidates>`` 内联）；
   - 把 payload 持久化到 ``pending_archive_jobs`` 表，拿到 ``job_id``；
   - 然后调用 :func:`_run_archive_job` 真正执行。

2. :func:`_run_archive_job` 负责 LLM 调用 + 事件写入：
   - LLM 调用走 :func:`_call_llm_in_daemon_thread`，daemon 线程承载阻塞 HTTP，
     这样进程退出时无需等待线程结束，Ctrl+C 不会被 LLM 套牢。
   - 完成（成功 / LLM 异常）后删除 ``pending_archive_jobs`` 行。
   - 若 task 被 ``cancel()``（shutdown 触发），异常向上抛出但 **不删除** job 行，
     由下次启动 :func:`resume_pending_jobs` 续跑。

3. shutdown 流程仅做 cancel + 短超时等待，不阻塞退出。
"""

import asyncio
import json
import logging
import re
from concurrent.futures import Future as _CFuture
from typing import Any

logger = logging.getLogger("AICQ.memory.archiver")

from llm.core.daemon_thread import call_in_daemon_thread

# event_type 归一化：把常见的进行时/错误形式映射到闭合词表原形
_EVENT_TYPE_NORMALIZE: dict[str, str] = {
    "teaching": "teach",
    "correcting": "correct",
    "asking": "ask",
    "answering": "answer",
    "promising": "promise",
    "refusing": "refuse",
    "agreeing": "agree",
    "liking": "like",
    "disliking": "dislike",
    "feeling": "feel",
    "experiencing": "experience",
    "sharing": "share",
    "complaining": "complain",
    "joking": "joke",
    "updating": "update",
    "saying": "say",
    "telling": "tell",
    "doing": "do",
    "being": "be",
    "owning": "own",
    "understanding": "understand",
}

from .archive_memories import ARCHIVE_GEN, TOOL as ARCHIVE_TOOL, read_result as read_archive_result
from .archive_prompt import ARCHIVE_SYSTEM_PROMPT

_SEM = asyncio.Semaphore(2)
_DEFAULT_CONTEXT_TURNS = 5
_DEFAULT_MAX_PER_TURN = 3

# 各会话最近一次成功归档时的窗口指纹：key=(conv_type, conv_id), value=md5
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


def _call_llm_in_daemon_thread(fn, *args, **kwargs) -> _CFuture:
    """在 daemon 线程里跑阻塞函数，返回 concurrent.futures.Future。

    与 ``asyncio.to_thread`` 的区别：daemon 线程在进程退出时被 OS 直接收走，
    不会阻塞 Python 解释器的 ThreadPoolExecutor.shutdown(wait=True)。
    这是 Ctrl+C 能立刻退出的关键。
    """
    return call_in_daemon_thread(fn, *args, thread_name="archive-llm", **kwargs)


def _track_archive_task(coro) -> asyncio.Task:
    """创建归档任务并登记到 app_state.archive_tasks，便于 shutdown 统一 cancel。"""
    import app_state
    task = asyncio.create_task(coro, name="archive-job")
    app_state.archive_tasks.add(task)
    task.add_done_callback(app_state.archive_tasks.discard)
    return task


# ── 准备阶段：从 session 构建 payload，并持久化为 pending job ─────────────────


async def archive_turn_memories(
    session,
    sender_id: str,
    tool_calls_log: list[dict],
) -> None:
    async with _SEM:
        import app_state
        from database import enqueue_archive_job

        from .repo.events import prefetch_candidates_for_archiver as _db_prefetch

        cfg = app_state.config.get("memory", {}).get("auto_archive", {})
        if not cfg.get("enabled", True):
            return

        context_turns = int(cfg.get("context_turns", _DEFAULT_CONTEXT_TURNS))

        # tool_calls_log 保留参数位以兼容调用点；write_memory 已下线，本函数不再读取。
        del tool_calls_log

        msgs = session.context_messages[-(context_turns * 2):]
        if not any(message.get("role") == "user" for message in msgs):
            return

        # 直接复用主循环用的 XML 聊天记录格式
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

        # ── 变化触发 + 抢占式签名 ────────────────────────────
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

        # ── Read-Before-Write：内联 candidates 到 dialogue ──
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

        valid_candidate_ids: list[int] = [int(c["event_id"]) for c in candidates]

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

        adapter = getattr(app_state, "archiver_adapter", None)
        if adapter is None:
            logger.warning("[memory_archiver] 未配置专用适配器，跳过本轮归档")
            _LAST_ARCHIVED_SIG[sess_key] = prev_signature
            await _persist_signature(sess_key, prev_signature)
            return

        # ── 持久化 pending job：先入库再跑 LLM ──
        try:
            job_id = await enqueue_archive_job(
                conv_type=str(session.conv_type),
                conv_id=str(session.conv_id),
                conv_name=str(session.conv_name or ""),
                sender_id=str(sender_id or ""),
                dialogue=dialogue,
                signature=signature,
                prev_signature=prev_signature,
                valid_candidate_ids=valid_candidate_ids,
            )
        except Exception:
            logger.warning("[archiver] enqueue_archive_job 失败，回滚签名占位", exc_info=True)
            _LAST_ARCHIVED_SIG[sess_key] = prev_signature
            await _persist_signature(sess_key, prev_signature)
            return

        payload: dict[str, Any] = {
            "job_id": job_id,
            "conv_type": str(session.conv_type),
            "conv_id": str(session.conv_id),
            "conv_name": str(session.conv_name or ""),
            "sender_id": str(sender_id or ""),
            "dialogue": dialogue,
            "signature": signature,
            "prev_signature": prev_signature,
            "valid_candidate_ids": valid_candidate_ids,
        }
        await _run_archive_job(payload)


# ── 执行阶段：跑 LLM + 写事件 + 删除 pending job ───────────────────────────


async def _run_archive_job(payload: dict[str, Any]) -> None:
    """执行单条归档任务。

    - 正常完成（成功 or LLM 调用异常）：删除 pending_archive_jobs 行。
    - 被 ``CancelledError`` 中断（shutdown 触发）：保留 job 行，向上抛出。
    - LLM 调用异常：回滚签名占位，让下次仍能重试同一窗口。
    """
    import app_state
    from database import delete_archive_job

    from .repo.events import (
        merge_event_occurrence as _db_merge_occurrence,
        write_event as _db_write_event,
    )
    from .tokenizer import register_word as _register, tokenize as _tokenize

    cfg = app_state.config.get("memory", {}).get("auto_archive", {})
    max_per_turn = int(cfg.get("max_per_turn", _DEFAULT_MAX_PER_TURN))
    gen_cfg = cfg.get("generation", {})
    archive_gen = {
        "temperature": float(gen_cfg.get("temperature", ARCHIVE_GEN["temperature"])),
        "max_output_tokens": int(gen_cfg.get("max_output_tokens", ARCHIVE_GEN["max_output_tokens"])),
    }

    job_id: int = int(payload["job_id"])
    conv_type: str = payload["conv_type"]
    conv_id: str = payload["conv_id"]
    conv_name: str = payload["conv_name"]
    sender_id: str = payload["sender_id"]
    dialogue: str = payload["dialogue"]
    signature: str = payload["signature"]
    prev_signature: str = payload["prev_signature"]
    valid_candidate_ids: set[int] = {int(x) for x in payload.get("valid_candidate_ids", [])}
    sess_key: tuple[str, str] = (conv_type, conv_id)

    adapter = app_state.archiver_adapter
    if adapter is None:
        # archiver_adapter 尚未就绪等场景：保留 job 行，下次再说
        logger.debug("[archiver] archiver_adapter 尚未就绪，保留 job#%d", job_id)
        return

    # 同步刷新内存签名缓存（resume 路径可能进来时缓存里没有）
    _LAST_ARCHIVED_SIG[sess_key] = signature

    # ── LLM 调用（daemon 线程）──
    try:
        fut = _call_llm_in_daemon_thread(
            adapter._call_forced_tool,
            ARCHIVE_SYSTEM_PROMPT,
            dialogue,
            archive_gen,
            ARCHIVE_TOOL,
            "archiver",
        )
        raw = await asyncio.wrap_future(fut)
    except asyncio.CancelledError:
        # 被 shutdown cancel：保留 job 行供下次启动续跑
        logger.info("[archiver] job#%d 被取消（shutdown），保留待下次启动续跑", job_id)
        raise
    except Exception:
        logger.debug("[archiver] archive_memories 调用异常 job#%d", job_id, exc_info=True)
        _LAST_ARCHIVED_SIG[sess_key] = prev_signature
        await _persist_signature(sess_key, prev_signature)
        try:
            await delete_archive_job(job_id)
        except Exception:
            logger.debug("[archiver] delete_archive_job 失败 job#%d", job_id, exc_info=True)
        return

    try:
        events_in = read_archive_result(raw)
        if not events_in:
            return

        written = 0
        merged = 0
        # 批内去重：记录已写入的 (agent实体, 归一化summary)，防止同窗口同义重复
        _batch_written: list[tuple[str, str]] = []
        for event in events_in:
            if written + merged >= max_per_turn:
                break
            if not isinstance(event, dict):
                continue

            event_type = str(event.get("event_type", "")).strip() or "unspecified"
            event_type = _EVENT_TYPE_NORMALIZE.get(event_type, event_type)
            summary = str(event.get("summary", "")).strip()
            if not summary:
                continue

            polarity = str(event.get("polarity", "positive")).strip().lower()
            modality = str(event.get("modality", "actual")).strip().lower()
            context_type = str(event.get("context_type", "episodic")).strip().lower()
            recall_scope = str(event.get("recall_scope") or "global").strip()
            reason = str(event.get("reason") or "").strip()
            try:
                confidence = float(event.get("confidence", 0.6))
            except (TypeError, ValueError):
                confidence = 0.6
            confidence = max(0.1, min(1.0, confidence))

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
                    if entity_text.startswith("User#qq_"):
                        entity_text = "User:qq_" + entity_text[len("User#qq_"):]
                    if entity_text in ("User", "Self"):
                        if not sender_id:
                            continue
                        entity_text = f"User:qq_{sender_id}"
                    elif entity_text == "Bot":
                        entity_text = "Bot:self"
                    elif entity_text.startswith("User(") and entity_text.endswith(")"):
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

            # ── 批内去重 ─────────────────────────────────────────────────────
            _ba = next(
                (r["entity"] or "" for r in normalized_roles if r.get("role") == "agent"),
                "",
            )
            _bn = re.sub(r"\s+", "", summary.lower())
            if any(
                ba == _ba and (_bn in bs or bs in _bn)
                for ba, bs in _batch_written
            ):
                logger.debug("[archiver] 批内重复，跳过：%s", summary)
                continue
            # ─────────────────────────────────────────────────────────────────

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
                    reason=reason or "从对话中自动提取",
                    conv_type=conv_type,
                    conv_id=conv_id,
                    conv_name=conv_name,
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
                _batch_written.append((_ba, _bn))
            except Exception:
                logger.warning("[archiver] event 写入失败：%s", summary, exc_info=True)

        if written or merged:
            logger.info(
                "[archiver] job#%d 完成：新增 %d / 合并 %d 条事件",
                job_id, written, merged,
            )
    finally:
        # 无论 LLM 后处理结果如何（除被 cancel 外），都应清掉 job 行；
        # 被 cancel 的情况已经在前面 raise 出去了，不会走到这里。
        try:
            await delete_archive_job(job_id)
        except Exception:
            logger.debug("[archiver] delete_archive_job 失败 job#%d", job_id, exc_info=True)


# ── 启动续跑 ─────────────────────────────────────────────────────────────


async def resume_pending_jobs() -> int:
    """startup 时调用：把上次未完成的归档任务重新调度起来。

    返回续跑的任务数。每个任务在 :data:`app_state.archive_tasks` 内登记，
    后续 shutdown 会统一 cancel。
    """
    try:
        from database import load_pending_archive_jobs
        jobs = await load_pending_archive_jobs()
    except Exception:
        logger.warning("[archiver] 加载 pending_archive_jobs 失败", exc_info=True)
        return 0

    if not jobs:
        return 0

    # 把签名缓存抢占占位，避免恢复期间又被新消息触发同一窗口
    await _ensure_sig_loaded()
    for job in jobs:
        sess_key = (job["conv_type"], job["conv_id"])
        _LAST_ARCHIVED_SIG[sess_key] = job["signature"]

    for job in jobs:
        async def _runner(payload=job) -> None:
            async with _SEM:
                await _run_archive_job(payload)
        _track_archive_task(_runner())

    logger.info("[archiver] 续跑了 %d 条上次未完成的归档任务", len(jobs))
    return len(jobs)


# 导出给调用方使用的便捷调度器
def schedule_archive(session, sender_id: str, tool_calls_log: list[dict] | None = None) -> asyncio.Task:
    """fire-and-forget 调度归档任务，并登记到 app_state.archive_tasks。"""
    return _track_archive_task(
        archive_turn_memories(session, str(sender_id or ""), list(tool_calls_log or []))
    )


__all__ = [
    "archive_turn_memories",
    "resume_pending_jobs",
    "schedule_archive",
]
