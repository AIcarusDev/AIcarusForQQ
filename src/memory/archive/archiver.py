"""Automatic background memory archiving.

设计要点（支持优雅退出 + 重启续跑）
----------------------------------------------------
1. 主入口 :func:`archive_cognition_flow_range` 由意识流压缩 worker 在冻结
   raw cognition 区间后 fire-and-forget 调用。它只负责"准备 payload"：
   - 检查/更新区间签名；
   - 按 ``prompt.py`` 约定拼装干净的 ``<task><cognition ...>`` 序列；
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
import logging
import re
from concurrent.futures import Future as _CFuture
from datetime import datetime, timezone
from html import unescape
from typing import Any, Iterable

from llm.core.daemon_thread import call_in_daemon_thread

from .prompt import ARCHIVE_SYSTEM_PROMPT
from .parser import ArchiveParseFatalError, parse_archive_output

logger = logging.getLogger("AICQ.memory.archive.archiver")

_SEM = asyncio.Semaphore(2)
_DEFAULT_CONTEXT_TURNS = 5
_ARCHIVE_GEN_DEFAULTS: dict[str, Any] = {
    "temperature": 0.3,
    "max_output_tokens": 10000,
}
_COGNITION_RE = re.compile(
    r"<cognition\b([^>]*)>(.*?)</cognition>",
    re.IGNORECASE | re.DOTALL,
)
_XML_ATTR_RE = re.compile(r"""([A-Za-z_:][\w:.-]*)\s*=\s*(['"])(.*?)\2""", re.DOTALL)

# 各会话最近一次成功归档时的窗口指纹：key=(conv_type, conv_id), value=md5
_LAST_ARCHIVED_SIG: dict[tuple[str, str], str] = {}
_sig_loaded: bool = False


def _build_prompt_task(content: str, timestamp: datetime | None = None) -> str:
    normalized = (content or "").strip()
    if normalized.startswith("<task"):
        return normalized
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    timestamp_text = timestamp.isoformat()
    return (
        "<task>\n"
        f'<cognition id="1" timestamp="{timestamp_text}">\n'
        f"{normalized}\n"
        "</cognition>\n"
        "</task>"
    )


def _auto_archive_cfg() -> dict[str, Any]:
    import app_state

    memory_cfg = getattr(app_state, "config", {}).get("memory", {})
    if not isinstance(memory_cfg, dict):
        return {}
    cfg = memory_cfg.get("auto_archive", {})
    return cfg if isinstance(cfg, dict) else {}


def _auto_archive_enabled() -> bool:
    return bool(_auto_archive_cfg().get("enabled", True))


def _raw_turn_archive_enabled() -> bool:
    cfg = _auto_archive_cfg()
    return bool(cfg.get("enabled", True)) and bool(cfg.get("raw_turn_archive_enabled", False))


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


def _normalize_entity_id(entity: Any) -> str | None:
    entity_text = str(entity or "").strip()
    if not entity_text:
        return None
    return entity_text


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


def schedule_cognition_flow_range_archive(
    rounds: list | tuple,
    *,
    coverage_start_seq: int,
    coverage_end_seq: int,
) -> None:
    """Schedule memory extraction from the raw cognition-flow range being compressed."""

    if not rounds:
        return
    _track_archive_task(
        archive_cognition_flow_range(
            rounds,
            coverage_start_seq=coverage_start_seq,
            coverage_end_seq=coverage_end_seq,
        )
    )


async def archive_cognition_flow_range(
    rounds: list | tuple,
    *,
    coverage_start_seq: int,
    coverage_end_seq: int,
) -> None:
    """Extract durable memories from frozen raw cognition-flow rounds."""

    async with _SEM:
        import hashlib
        import app_state
        from database import enqueue_archive_job

        cfg = _auto_archive_cfg()
        if not cfg.get("enabled", True):
            return

        task_prompt = _format_cognition_flow_task_xml(
            rounds,
            coverage_start_seq=coverage_start_seq,
            coverage_end_seq=coverage_end_seq,
        )
        if not task_prompt.strip():
            return
        candidates = _memory_candidates_from_rounds(rounds)
        valid_candidate_ids = _candidate_event_ids(candidates)
        if candidates:
            task_prompt = task_prompt + "\n\n" + _format_existing_candidates(candidates)
        logger.debug(
            "[archiver] cognition-flow existing candidates rounds=%d merged=%d event_ids=%d",
            len(rounds),
            len(candidates),
            len(valid_candidate_ids),
        )

        await _ensure_sig_loaded()
        range_id = f"cognition_flow_range:{int(coverage_start_seq)}-{int(coverage_end_seq)}"
        sess_key: tuple[str, str] = ("flow", range_id)
        signature = hashlib.md5(
            f"{int(coverage_start_seq)}|{int(coverage_end_seq)}|{task_prompt}".encode(
                "utf-8",
                errors="ignore",
            )
        ).hexdigest()
        if signature == _LAST_ARCHIVED_SIG.get(sess_key, ""):
            logger.debug(
                "[archiver] cognition-flow range unchanged, skip coverage=%d..%d sig=%s...",
                coverage_start_seq,
                coverage_end_seq,
                signature[:8],
            )
            return

        prev_signature = _LAST_ARCHIVED_SIG.get(sess_key, "")
        _LAST_ARCHIVED_SIG[sess_key] = signature
        await _persist_signature(sess_key, signature)

        adapter = getattr(app_state, "archiver_adapter", None)
        if adapter is None:
            logger.warning("[memory_archiver] archiver adapter missing; skip cognition-flow archive")
            _LAST_ARCHIVED_SIG[sess_key] = prev_signature
            await _persist_signature(sess_key, prev_signature)
            return

        try:
            job_id = await enqueue_archive_job(
                conv_type=sess_key[0],
                conv_id=sess_key[1],
                conv_name=f"Cognition flow range {int(coverage_start_seq)}-{int(coverage_end_seq)}",
                sender_id="",
                dialogue=task_prompt,
                signature=signature,
                prev_signature=prev_signature,
                valid_candidate_ids=valid_candidate_ids,
            )
        except Exception:
            logger.warning("[archiver] enqueue cognition-flow archive failed", exc_info=True)
            _LAST_ARCHIVED_SIG[sess_key] = prev_signature
            await _persist_signature(sess_key, prev_signature)
            return

        await _run_archive_job({
            "job_id": job_id,
            "conv_type": sess_key[0],
            "conv_id": sess_key[1],
            "conv_name": f"Cognition flow range {int(coverage_start_seq)}-{int(coverage_end_seq)}",
            "sender_id": "",
            "dialogue": task_prompt,
            "signature": signature,
            "prev_signature": prev_signature,
            "valid_candidate_ids": valid_candidate_ids,
            "archive_mode": "cognition_flow_range",
        })


async def _load_recent_member_aliases(limit: int = 500) -> dict[str, str]:
    """Return unambiguous recent nickname -> User:qq_id mappings."""

    try:
        import aiosqlite
        from database import DB_PATH

        seen: dict[str, set[str]] = {}
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                """SELECT sender_name, sender_id
                   FROM chat_messages
                   WHERE sender_name != '' AND sender_id != ''
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (int(limit),),
            ) as cur:
                rows = await cur.fetchall()
        for name, sid in rows:
            name = str(name or "").strip()
            sid = str(sid or "").strip()
            if not name or not sid or sid.lower() == "self":
                continue
            seen.setdefault(name, set()).add(sid)
        return {
            name: f"User:qq_{next(iter(ids))}"
            for name, ids in seen.items()
            if len(ids) == 1
        }
    except Exception:
        logger.debug("[archiver] failed to load recent member aliases", exc_info=True)
        return {}


def _format_existing_candidates(candidates: list[dict]) -> str:
    if not candidates:
        return ""
    lines: list[str] = ["<existing_candidates>"]
    for c in candidates:
        roles = c.get("roles") or []
        role_brief = ""
        if roles:
            role_brief = ", ".join(
                f"{r['role']}=" + (
                    r["entity"] if r.get("entity")
                    else (f'"{r["value_text"]}"' if r.get("value_text") else f"->#{r.get('target_event')}")
                )
                for r in roles
                if isinstance(r, dict)
            )
        elif c.get("core_entities"):
            role_brief = ", ".join(str(item) for item in c.get("core_entities") or () if item)
        item_id = c.get("event_id") or c.get("summary_id") or c.get("source_id") or "unknown"
        source_ids = _candidate_event_ids_from_item(c)
        source_brief = f" source_events={','.join(str(x) for x in source_ids)}" if source_ids else ""
        lines.append(
            f"#{item_id}  kind={c.get('memory_kind') or 'event'} ctx={c.get('context_type','')}{source_brief} "
            f"| {c.get('summary','')} "
            f"| roles: {role_brief}"
        )
    lines.append("</existing_candidates>")
    return "\n".join(lines)


def _candidate_event_ids_from_item(item: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for key in ("event_id", "source_event_ids", "contributing_event_ids"):
        values = item.get(key)
        if values is None:
            continue
        if isinstance(values, (list, tuple, set)):
            raw_values = values
        else:
            raw_values = (values,)
        for value in raw_values:
            try:
                event_id = int(value)
            except (TypeError, ValueError):
                continue
            if event_id > 0 and event_id not in ids:
                ids.append(event_id)
    return ids


def _candidate_event_ids(candidates: list[dict]) -> list[int]:
    ids: list[int] = []
    for item in candidates:
        for event_id in _candidate_event_ids_from_item(item):
            if event_id not in ids:
                ids.append(event_id)
    return ids


def _merge_existing_candidates(*groups: Iterable[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for group in groups:
        for item in group or ():
            if not isinstance(item, dict):
                continue
            ids = _candidate_event_ids_from_item(item)
            if ids:
                key = "events:" + ",".join(str(event_id) for event_id in ids)
            else:
                key = str(item.get("summary_id") or item.get("event_id") or item.get("summary") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(dict(item))
            if len(out) >= 64:
                return out
    return out


def _memory_candidates_from_rounds(rounds: Iterable[Any]) -> list[dict]:
    groups: list[list[dict]] = []
    for rnd in rounds or ():
        candidates = getattr(rnd, "memory_candidates", None)
        if candidates is None and isinstance(rnd, dict):
            candidates = rnd.get("memory_candidates")
        if isinstance(candidates, list):
            groups.append(candidates)
    return _merge_existing_candidates(*groups)


def _format_member_aliases(aliases: dict[str, str]) -> str:
    if not aliases:
        return ""
    lines = ["<member_aliases>"]
    for name, entity in sorted(aliases.items()):
        lines.append(f'  "{name}" -> {entity}')
    lines.append("</member_aliases>")
    return "\n".join(lines)


def _format_cognition_timestamp(value: object) -> str:
    if value is None:
        return ""
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()  # type: ignore[arg-type]
    except (TypeError, ValueError, OSError, OverflowError):
        return str(value)


def _format_cognition_flow_task_xml(
    rounds: list | tuple,
    *,
    coverage_start_seq: int,
    coverage_end_seq: int,
) -> str:
    from xml.sax.saxutils import escape

    del coverage_start_seq, coverage_end_seq
    lines = ["<task>"]
    item_id = 0
    for rnd in rounds:
        cognition = str(getattr(rnd, "cognition", "") or "").strip()
        if not cognition:
            continue
        item_id += 1
        source_id = escape(str(item_id))
        timestamp = escape(_format_cognition_timestamp(getattr(rnd, "timestamp", None)))
        lines.append(f'<cognition id="{source_id}" timestamp="{timestamp}">')
        lines.append(escape(cognition))
        lines.append("</cognition>")
    if item_id == 0:
        return ""
    lines.append("</task>")
    return "\n".join(lines)


def _extract_cognition_source_map(task_xml: str) -> dict[str, dict[str, str]]:
    sources: dict[str, dict[str, str]] = {}
    for match in _COGNITION_RE.finditer(task_xml or ""):
        attrs = {
            name: unescape(value.strip())
            for name, _quote, value in _XML_ATTR_RE.findall(match.group(1) or "")
        }
        source_id = str(attrs.get("id") or "").strip()
        if not source_id:
            continue
        sources[source_id] = {
            "timestamp": str(attrs.get("timestamp") or "").strip(),
            "text": unescape(match.group(2) or "").strip(),
        }
    return sources


def _normalize_event_source_ids(event: dict[str, Any]) -> list[str]:
    raw = event.get("source_id")
    if not isinstance(raw, str):
        return []
    return list(dict.fromkeys(re.findall(r"\d+", raw)))


async def _run_post_archive_mount_workflow(
    new_event_ids: list[int],
    candidate_event_ids: list[int],
) -> dict[str, Any]:
    def _write() -> dict[str, Any]:
        import database

        from ..post_archive.mount_workflow import run_post_archive_mount_workflow

        return run_post_archive_mount_workflow(
            database.DB_PATH,
            new_event_ids=new_event_ids,
            candidate_event_ids=candidate_event_ids,
            max_mounts_per_atom=3,
        )

    return await asyncio.to_thread(_write)


# ── 准备阶段：从 session 构建 payload，并持久化为 pending job ─────────────────


async def archive_turn_memories(
    session,
    sender_id: str,
    tool_calls_log: list[dict],
) -> None:
    async with _SEM:
        import app_state
        from database import enqueue_archive_job

        from ..repo.events import prefetch_candidates_for_archiver as _db_prefetch

        cfg = _auto_archive_cfg()
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
                    lines.append(f"我 (self): {content}")
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

        recalled_candidates = _merge_existing_candidates(getattr(session, "recalled_events", []) or [])
        prefetch_candidates: list[dict] = []
        try:
            prefetch_candidates = await _db_prefetch(
                sender_entity=sender_entity,
                context_scope=context_scope,
                dialogue_text=dialogue,
                limit=8,
            )
        except Exception:
            logger.debug("[archiver] 候选预取失败，跳过 Read-Before-Write", exc_info=True)
            prefetch_candidates = []

        candidates = _merge_existing_candidates(recalled_candidates, prefetch_candidates)
        valid_candidate_ids = _candidate_event_ids(candidates)

        if candidates:
            dialogue = dialogue + "\n\n" + _format_existing_candidates(candidates)
        logger.debug(
            "[archiver] existing candidates recalled=%d prefetch=%d merged=%d event_ids=%d",
            len(recalled_candidates),
            len(prefetch_candidates),
            len(candidates),
            len(valid_candidate_ids),
        )

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
    - auto_archive.enabled=false 时：保留 job 行，不调用 LLM。
    """
    import app_state
    from database import delete_archive_job

    from consciousness.sources import upsert_cognition_sources as _db_upsert_cognition_sources

    from ..repo.events import write_prompt_event as _db_write_prompt_event

    cfg = _auto_archive_cfg()
    if not cfg.get("enabled", True):
        logger.debug("[archiver] auto_archive.enabled=false，保留 job#%d 不执行", int(payload["job_id"]))
        return
    job_id: int = int(payload["job_id"])
    archive_mode = str(payload.get("archive_mode") or "")
    if not archive_mode and payload.get("conv_type") == "flow" and payload.get("conv_id") == "compression_summary":
        archive_mode = "compression_summary"
    if (
        not archive_mode
        and payload.get("conv_type") == "flow"
        and str(payload.get("conv_id") or "").startswith("cognition_flow_range")
    ):
        archive_mode = "cognition_flow_range"
    if archive_mode == "compression_summary":
        logger.info("[archiver] skip legacy compression-summary archive job#%d", job_id)
        try:
            await delete_archive_job(job_id)
        except Exception:
            logger.debug("[archiver] delete legacy compression-summary job#%d failed", job_id, exc_info=True)
        return
    gen_cfg = cfg.get("generation", {})
    archive_gen = {
        "temperature": float(gen_cfg.get("temperature", _ARCHIVE_GEN_DEFAULTS["temperature"])),
        "max_output_tokens": int(gen_cfg.get("max_output_tokens", _ARCHIVE_GEN_DEFAULTS["max_output_tokens"])),
    }
    for key, value in gen_cfg.items():
        if key not in ("temperature", "max_output_tokens"):
            archive_gen[key] = value

    conv_type: str = payload["conv_type"]
    conv_id: str = payload["conv_id"]
    conv_name: str = payload["conv_name"]
    sender_id: str = payload["sender_id"]
    dialogue: str = payload["dialogue"]
    signature: str = payload["signature"]
    prev_signature: str = payload["prev_signature"]
    valid_candidate_ids: set[int] = {int(x) for x in payload.get("valid_candidate_ids", [])}
    sess_key: tuple[str, str] = (conv_type, conv_id)

    async def _rollback_failed_generation() -> None:
        _LAST_ARCHIVED_SIG[sess_key] = prev_signature
        await _persist_signature(sess_key, prev_signature)
        try:
            await delete_archive_job(job_id)
        except Exception:
            logger.debug("[archiver] delete_archive_job 失败 job#%d", job_id, exc_info=True)

    adapter = app_state.archiver_adapter
    if adapter is None:
        # archiver_adapter 尚未就绪等场景：保留 job 行，下次再说
        logger.debug("[archiver] archiver_adapter 尚未就绪，保留 job#%d", job_id)
        return

    # 同步刷新内存签名缓存（resume 路径可能进来时缓存里没有）
    _LAST_ARCHIVED_SIG[sess_key] = signature
    cognition_time = datetime.now(timezone.utc)
    cognition_time_ms = int(cognition_time.timestamp() * 1000)
    task_payload = _build_prompt_task(dialogue, cognition_time)
    source_meta = _extract_cognition_source_map(task_payload)
    if source_meta:
        try:
            source_meta = await _db_upsert_cognition_sources(
                source_meta,
                origin_type=conv_type,
                origin_id=conv_id,
            )
        except Exception:
            logger.warning("[archiver] cognition source upsert failed job#%d", job_id, exc_info=True)
            source_meta = {}

    # ── LLM 调用（daemon 线程）──
    try:
        system_prompt = ARCHIVE_SYSTEM_PROMPT
        fut = _call_llm_in_daemon_thread(
            adapter.call_simple_text,
            system_prompt,
            task_payload,
            archive_gen,
            "archiver",
        )
        raw = await asyncio.wrap_future(fut)
    except asyncio.CancelledError:
        # 被 shutdown cancel：保留 job 行供下次启动续跑
        logger.info("[archiver] job#%d 被取消（shutdown），保留待下次启动续跑", job_id)
        raise
    except Exception:
        logger.debug("[archiver] prompt archive 调用异常 job#%d", job_id, exc_info=True)
        await _rollback_failed_generation()
        return

    if not isinstance(raw, str) or not raw.strip():
        logger.debug("[archiver] prompt archive 无输出 job#%d，按生成失败处理", job_id)
        await _rollback_failed_generation()
        return

    try:
        try:
            parsed = parse_archive_output(raw)
        except ArchiveParseFatalError:
            logger.warning("[archiver] prompt 输出结构无效 job#%d", job_id, exc_info=True)
            return
        for err in parsed.errors:
            logger.warning("[archiver] prompt event rejected job#%d: %s", job_id, err)
        events_in = [item.event | {"_raw_event_json": item.raw_json} for item in parsed.events]
        if not events_in:
            return

        written = 0
        merged = 0
        mounts_staged = 0
        written_event_ids: list[int] = []
        # 批内去重：记录已写入的 (agent实体, 归一化summary)，防止同窗口同义重复
        _batch_written: list[tuple[str, str]] = []
        for event in events_in:
            if not isinstance(event, dict):
                continue

            event_type = str(event.get("event_type", "")).strip() or "unspecified"
            summary = str(event.get("summary", "")).strip()
            if not summary:
                continue
            source_ids = _normalize_event_source_ids(event)
            if source_meta:
                invalid_source_ids = [sid for sid in source_ids if sid not in source_meta]
                if invalid_source_ids:
                    logger.debug(
                        "[archiver] 丢弃 event 无效 source_id=%s valid=%s summary=%s",
                        invalid_source_ids,
                        sorted(source_meta),
                        summary,
                    )
                source_ids = [sid for sid in source_ids if sid in source_meta]

            reason = str(event.get("reason") or "").strip()

            supersedes_raw = event.get("supersedes")
            supersedes_id: int | None = None
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
                    entity = _normalize_entity_id(entity)
                    if not entity:
                        continue
                if value_text is not None:
                    value_text = str(value_text).strip() or None
                if not entity and not value_text:
                    continue
                normalized_roles.append({
                        "role": role_name,
                        "entity": entity,
                        "value_text": value_text,
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
                event_for_store = dict(event)
                event_for_store["event_type"] = event_type
                event_for_store["summary"] = summary
                event_for_store["roles"] = normalized_roles
                event_source = (
                    "compression_summary"
                    if archive_mode == "compression_summary"
                    else (
                        "cognition_flow_range"
                        if archive_mode == "cognition_flow_range"
                        else "自动归档"
                    )
                )
                event_reason = reason or (
                    "extracted from cognition-flow compression summary"
                    if archive_mode == "compression_summary"
                    else (
                        "extracted raw cognition-flow range"
                        if archive_mode == "cognition_flow_range"
                        else "从对话中自动提取"
                    )
                )
                event_id = await _db_write_prompt_event(
                    event_for_store,
                    raw_event_json=str(event.get("_raw_event_json") or ""),
                    source=event_source,
                    reason=event_reason,
                    conv_type=conv_type,
                    conv_id=conv_id,
                    conv_name=conv_name,
                    occurred_at=cognition_time_ms,
                    supersedes=supersedes_id,
                    source_ids=source_ids,
                    source_meta=source_meta,
                )
                role_brief = "/".join(
                    f"{role['role']}:{role['entity'] or role['value_text']}"
                    for role in normalized_roles
                )
                supersedes_note = f" supersedes#{supersedes_id}" if supersedes_id else ""
                logger.info(
                    "[archiver] 写入 event#%d type=%s status=%s%s | %s | %s",
                    event_id,
                    event_type,
                    str(event.get("status") or "actual"),
                    supersedes_note,
                    summary,
                    role_brief,
                )
                written += 1
                written_event_ids.append(int(event_id))
                _batch_written.append((_ba, _bn))
            except Exception:
                logger.warning("[archiver] event 写入失败：%s", summary, exc_info=True)

        if written_event_ids:
            try:
                mount_stats = await _run_post_archive_mount_workflow(
                    written_event_ids,
                    sorted(valid_candidate_ids),
                )
                mounts_staged = int(mount_stats.get("mounts_staged") or 0)
                logger.info(
                    "[archiver] job#%d 二步挂载：mode=%s new_events=%d candidate_events=%d historical_atoms=%d cluster_summaries=%d proposed=%d staged=%d atom_links=%d atom_link_pending=%d local_cluster_pending=%d summary_ready=%d model_errors=%d mount_errors=%d atom_link_errors=%d local_cluster_errors=%d",
                    job_id,
                    str(mount_stats.get("mount_mode") or "rules"),
                    int(mount_stats.get("new_events_loaded") or 0),
                    int(mount_stats.get("candidate_event_ids") or 0),
                    int(mount_stats.get("historical_atoms_loaded") or 0),
                    int(mount_stats.get("cluster_summaries_loaded") or 0),
                    int(mount_stats.get("mounts_proposed") or 0),
                    mounts_staged,
                    int(mount_stats.get("atom_links_proposed") or 0),
                    int(mount_stats.get("atom_links_staged") or 0),
                    int(mount_stats.get("local_clusters_staged") or 0),
                    int(mount_stats.get("summaries_ready") or 0),
                    len(mount_stats.get("model_errors") or ()),
                    len(mount_stats.get("mount_errors") or ()),
                    len(mount_stats.get("atom_link_errors") or ()),
                    len(mount_stats.get("local_cluster_errors") or ()),
                )
            except Exception:
                logger.warning("[archiver] post-archive mount workflow failed job#%d", job_id, exc_info=True)

        if written or merged:
            logger.info(
                "[archiver] job#%d 完成：新增 %d / 合并 %d 条事件 / pending mount %d 条",
                job_id, written, merged, mounts_staged,
            )
        elif events_in:
            logger.warning(
                "[archiver] job#%d parsed %d valid events but wrote none",
                job_id, len(events_in),
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
    if not _auto_archive_enabled():
        logger.info("[archiver] auto_archive.enabled=false，跳过待归档任务续跑")
        return 0

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


__all__ = [
    "archive_turn_memories",
    "archive_cognition_flow_range",
    "resume_pending_jobs",
    "schedule_cognition_flow_range_archive",
]
