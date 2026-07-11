"""Memory storyline-summary worker.

The worker turns structured storyline-summary tasks into ready storyline-summary
rows. Event-storyline summaries are always generated or refreshed by the
memory-consolidation LLM; deterministic code may only prepare the LLM payload.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from html import escape, unescape
from typing import Any, Callable, Iterable

from .prompt import STORYLINE_SUMMARY_SYSTEM_PROMPT
from .consolidation import ensure_preprocessing_schema


STORYLINE_SUMMARY_EVENT_LIMIT = 24
STORYLINE_SUMMARY_MODEL = "memory_consolidation.storyline_summary.v2"
_SOURCE_KIND_RE = re.compile(r"[^a-zA-Z0-9_.:-]+")
logger = logging.getLogger("AICQ.memory.sleep.summary_worker")


@dataclass(frozen=True)
class SummaryRefreshReport:
    summary_tasks_queued: int = 0
    summary_tasks_loaded: int = 0
    summary_tasks_done: int = 0
    summaries_ready: int = 0
    summary_tasks_failed: int = 0
    summary_tasks_retrying: int = 0
    summary_queue_paused: int = 0
    summary_llm_calls: int = 0

    def with_tasks_queued(self, count: int) -> "SummaryRefreshReport":
        return replace(self, summary_tasks_queued=int(count or 0))

    def to_dict(self) -> dict[str, int]:
        return {
            "summary_tasks_queued": self.summary_tasks_queued,
            "summary_tasks_loaded": self.summary_tasks_loaded,
            "summary_tasks_done": self.summary_tasks_done,
            "summaries_ready": self.summaries_ready,
            "summary_tasks_failed": self.summary_tasks_failed,
            "summary_tasks_retrying": self.summary_tasks_retrying,
            "summary_queue_paused": self.summary_queue_paused,
            "summary_llm_calls": self.summary_llm_calls,
        }


def run_summary_refresh_worker(
    con_or_path: sqlite3.Connection | str | os.PathLike[str],
    *,
    max_inputs: int = 32,
    storyline_ids: Iterable[str] = (),
    priority_task_ids: Iterable[str] = (),
    priority_storyline_ids: Iterable[str] = (),
    deadline_ms: int | None = None,
    should_continue: Callable[[], bool] | None = None,
    now_ms: int | None = None,
    model: str = STORYLINE_SUMMARY_MODEL,
) -> SummaryRefreshReport:
    """Process storyline-summary tasks produced by sleep solidification."""

    owns_connection = not isinstance(con_or_path, sqlite3.Connection)
    con = sqlite3.connect(os.fspath(con_or_path), timeout=30.0) if owns_connection else con_or_path
    try:
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        ensure_preprocessing_schema(con)
        con.commit()
        now = int(now_ms or _now_ms())
        target_storyline_ids = _unique_strings((*storyline_ids, *priority_storyline_ids))
        priority_storyline_ids = _unique_strings(priority_storyline_ids)
        target_task_ids = set(_unique_strings(priority_task_ids))
        target_task_ids.update(summary_id_for_source("storyline", storyline_id) for storyline_id in priority_storyline_ids)

        queued = queue_storyline_summary_refresh_tasks(
            con,
            max_storylines=len(target_storyline_ids),
            storyline_ids=target_storyline_ids,
            now_ms=now,
        )
        con.commit()
        input_limit = max(max(1, int(max_inputs or 1)), len(target_task_ids))
        processed = process_active_summary_inputs(
            con,
            max_inputs=input_limit,
            priority_task_ids=target_task_ids,
            deadline_ms=deadline_ms,
            should_continue=should_continue,
            now_ms=now,
            model=model,
        )
        if owns_connection:
            con.commit()
        return processed.with_tasks_queued(queued)
    finally:
        if owns_connection:
            con.close()


def queue_storyline_summary_refresh_tasks(
    con: sqlite3.Connection,
    *,
    max_storylines: int = 64,
    storyline_ids: Iterable[str] = (),
    now_ms: int | None = None,
) -> int:
    """Create structured summary tasks for explicitly affected storylines."""

    ensure_preprocessing_schema(con)
    if not _table_exists(con, "MemoryStorylines") or not _table_exists(con, "MemoryStorylineMembers"):
        return 0

    now = int(now_ms or _now_ms())
    previous_row_factory = con.row_factory
    queued = 0
    try:
        con.row_factory = sqlite3.Row
        target_ids = _unique_strings(storyline_ids)
        if not target_ids:
            return 0
        target_clause = ""
        params: list[Any] = []
        placeholders = ",".join("?" * len(target_ids))
        target_clause = f" AND storyline_id IN ({placeholders})"
        params.extend(target_ids)
        params.append(max(1, int(max_storylines or 1)))
        storylines = list(
            con.execute(
                f"""
                SELECT storyline_id, scope, scheme_name, anchor_key, profile,
                       revision, member_count, score, updated_at
                FROM MemoryStorylines
                WHERE status='active' AND member_count >= 2{target_clause}
                ORDER BY updated_at DESC, score DESC, storyline_id ASC
                LIMIT ?
                """,
                params,
            )
        )
        for row in storylines:
            storyline_id = str(row["storyline_id"] or "")
            summary_id = summary_id_for_source("storyline", storyline_id)
            source_revision = int(row["revision"] or 1)
            events = _load_storyline_event_window(
                con,
                storyline_id,
                max_events=STORYLINE_SUMMARY_EVENT_LIMIT,
            )
            if not events:
                continue
            input_hash = _storyline_summary_task_hash(
                storyline_id=storyline_id,
                storyline_revision=source_revision,
                event_ids=[int(item["event_id"]) for item in events],
            )
            ready = con.execute(
                """
                SELECT 1
                FROM MemorySummaryCache
                WHERE task_id=? AND input_hash=? AND status='ready'
                LIMIT 1
                """,
                (summary_id, input_hash),
            ).fetchone()
            if ready:
                continue
            existing_active = con.execute(
                """
                SELECT 1
                FROM MemoryStorylineSummaryTasks
                WHERE task_id=? AND input_hash=? AND status='active'
                LIMIT 1
                """,
                (summary_id, input_hash),
            ).fetchone()
            if existing_active:
                continue
            _upsert_storyline_summary_task(
                con,
                task_id=summary_id,
                task_type="refresh",
                storyline_id=storyline_id,
                storyline_revision=source_revision,
                input_hash=input_hash,
                priority=90,
                confidence_tier="medium",
                now_ms=now,
            )
            _replace_storyline_summary_task_events(con, summary_id, events)
            queued += 1
    finally:
        con.row_factory = previous_row_factory
    return queued


def process_active_summary_inputs(
    con: sqlite3.Connection,
    *,
    max_inputs: int = 32,
    priority_task_ids: Iterable[str] = (),
    deadline_ms: int | None = None,
    should_continue: Callable[[], bool] | None = None,
    now_ms: int | None = None,
    model: str = STORYLINE_SUMMARY_MODEL,
) -> SummaryRefreshReport:
    """Consume active storyline-summary tasks and write LLM-generated summaries."""

    ensure_preprocessing_schema(con)
    con.commit()
    now = int(now_ms or _now_ms())
    adapter, gen, max_retries = _summary_llm_runtime()
    previous_row_factory = con.row_factory
    stats = {
        "summary_tasks_loaded": 0,
        "summary_tasks_done": 0,
        "summaries_ready": 0,
        "summary_tasks_failed": 0,
        "summary_tasks_retrying": 0,
        "summary_queue_paused": 0,
        "summary_llm_calls": 0,
    }
    try:
        con.row_factory = sqlite3.Row
        priority_ids = _unique_strings(priority_task_ids)
        order_prefix = ""
        params: list[Any] = []
        if priority_ids:
            placeholders = ",".join("?" * len(priority_ids))
            order_prefix = f"CASE WHEN task_id IN ({placeholders}) THEN 0 ELSE 1 END,"
            params.extend(priority_ids)
        params.append(max(1, int(max_inputs or 1)))
        rows = list(
            con.execute(
                f"""
                SELECT *
                FROM MemoryStorylineSummaryTasks
                WHERE status='active'
                  AND task_type='refresh'
                ORDER BY {order_prefix} priority DESC, updated_at_ms ASC, task_id ASC
                LIMIT ?
                """,
                params,
            )
        )
        stats["summary_tasks_loaded"] = len(rows)
        for row in rows:
            if (deadline_ms and _now_ms() >= int(deadline_ms)) or (
                should_continue is not None and not should_continue()
            ):
                stats["summary_queue_paused"] = 1
                break
            task_id = str(row["task_id"] or "")
            try:
                if adapter is None:
                    raise RuntimeError("memory_consolidation_adapter unavailable")
                events = _load_storyline_summary_task_events(con, task_id)
                previous_summary = _load_previous_storyline_summary(con, task_id)
                generated_summary = _generate_storyline_summary(
                    adapter,
                    previous_summary=previous_summary or "",
                    events=events,
                    gen=gen,
                )
                stats["summary_llm_calls"] += 1
                summary = generated_summary or previous_summary or ""
                _write_summary_cache(
                    con,
                    summary_id=task_id,
                    summary=summary,
                    input_hash=str(row["input_hash"] or ""),
                    model=model,
                    now_ms=now,
                )
                con.execute(
                    """
                    UPDATE MemoryStorylineSummaryTasks
                    SET status='done', updated_at_ms=?
                    WHERE task_id=?
                    """,
                    (now, task_id),
                )
                stats["summary_tasks_done"] += 1
                stats["summaries_ready"] += 1
                con.commit()
            except Exception as exc:
                retry_count = int(row["retry_count"] or 0) + 1
                should_retry = retry_count < max(1, int(max_retries or 1))
                con.execute(
                    """
                    UPDATE MemoryStorylineSummaryTasks
                    SET status=?, retry_count=?, last_error=?, updated_at_ms=?
                    WHERE task_id=?
                    """,
                    ("active" if should_retry else "failed", retry_count, str(exc)[:500], now, task_id),
                )
                if should_retry:
                    stats["summary_tasks_retrying"] += 1
                else:
                    stats["summary_tasks_failed"] += 1
                con.commit()
                logger.warning(
                    "[summary_worker] 故事线 summary 生成失败 task_id=%s retry=%d/%d error=%s",
                    task_id,
                    retry_count,
                    max_retries,
                    exc,
                )
    finally:
        con.row_factory = previous_row_factory
    return SummaryRefreshReport(
        summary_tasks_loaded=int(stats["summary_tasks_loaded"]),
        summary_tasks_done=int(stats["summary_tasks_done"]),
        summaries_ready=int(stats["summaries_ready"]),
        summary_tasks_failed=int(stats["summary_tasks_failed"]),
        summary_tasks_retrying=int(stats["summary_tasks_retrying"]),
        summary_queue_paused=int(stats["summary_queue_paused"]),
        summary_llm_calls=int(stats["summary_llm_calls"]),
    )


def summary_id_for_source(source_kind: str, source_id: str) -> str:
    kind = _SOURCE_KIND_RE.sub("_", str(source_kind or "summary").strip() or "summary")
    return f"summary:{kind}:{_sha1('summary-source', kind, str(source_id or ''))[:20]}"


def _storyline_summary_task_hash(
    *,
    storyline_id: str,
    storyline_revision: int,
    event_ids: Iterable[int],
) -> str:
    return _sha1(
        "storyline-summary-task",
        storyline_id,
        str(int(storyline_revision or 0)),
        ",".join(str(int(event_id)) for event_id in event_ids if int(event_id) > 0),
    )


def _generate_storyline_summary(
    adapter: Any,
    *,
    previous_summary: str,
    events: Iterable[dict[str, Any]],
    gen: dict[str, Any],
) -> str:
    user_prompt = _build_storyline_summary_user_prompt(previous_summary, events)
    raw = adapter.call_simple_text(
        STORYLINE_SUMMARY_SYSTEM_PROMPT,
        user_prompt,
        gen,
        log_tag="memory_consolidation/summary",
    )
    return _parse_storyline_summary_response(raw)


def _build_storyline_summary_user_prompt(
    previous_summary: str,
    events: Iterable[dict[str, Any]],
) -> str:
    normalized_events = sorted(
        (dict(event) for event in events if isinstance(event, dict)),
        key=lambda event: (
            _int(event.get("occurred_at") or event.get("created_at"), 0),
            _int(event.get("event_id"), 0),
        ),
    )
    lines = ["<task>", "<previous_storyline>"]
    if previous_summary:
        lines.append(f"  {escape(str(previous_summary).strip())}")
    lines.extend(["</previous_storyline>", "", "<events>"])
    for event in normalized_events:
        event_time = _format_event_time(event.get("occurred_at") or event.get("created_at"))
        confidence = _format_confidence(event.get("confidence"))
        summary = escape(str(event.get("summary") or "").strip())
        lines.append(
            f'  <event occurred_at="{escape(event_time, quote=True)}" confidence="{confidence}">'
        )
        lines.append(f"    {summary}")
        lines.append("  </event>")
    lines.extend(["</events>", "</task>"])
    return "\n".join(lines)


def _load_previous_storyline_summary(con: sqlite3.Connection, summary_id: str) -> str | None:
    row = con.execute(
        """
        SELECT summary
        FROM MemorySummaryCache
        WHERE summary_id=? OR task_id=?
        ORDER BY
          CASE status
            WHEN 'ready' THEN 0
            WHEN 'stale' THEN 1
            ELSE 2
          END,
          updated_at_ms DESC
        LIMIT 1
        """,
        (summary_id, summary_id),
    ).fetchone()
    if not row:
        return None
    return str(row[0] or "")


def _load_storyline_summary_task_events(con: sqlite3.Connection, task_id: str) -> list[dict[str, Any]]:
    if not _table_exists(con, "MemoryEvents"):
        return []
    rows = list(
        con.execute(
            """
            SELECT e.event_id, e.summary, e.confidence, e.occurred_at, e.created_at
            FROM MemoryStorylineSummaryTaskEvents te
            JOIN MemoryEvents e ON e.event_id=te.event_id
            WHERE te.task_id=? AND te.status='active' AND e.is_deleted=0
            ORDER BY te.rank ASC, e.occurred_at ASC, e.event_id ASC
            """,
            (task_id,),
        )
    )
    return [
        {
            "event_id": int(row["event_id"]),
            "summary": str(row["summary"] or ""),
            "confidence": float(row["confidence"] or 0.0),
            "occurred_at": int(row["occurred_at"] or 0),
            "created_at": int(row["created_at"] or 0),
        }
        for row in rows
    ]


def _parse_storyline_summary_response(raw: object) -> str:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty LLM summary response")
    match = re.search(r"<storyline\s*>(.*?)</storyline\s*>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return unescape(match.group(1)).strip()
    no_update = re.fullmatch(
        r"(?:<analysis\s*>.*?</analysis\s*>\s*)?</storyline\s*>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if no_update:
        return ""
    raise ValueError("LLM summary response missing <storyline> output")


def _summary_llm_runtime() -> tuple[Any | None, dict[str, Any], int]:
    try:
        import app_state

        cfg = getattr(app_state, "memory_consolidation_cfg", None)
        if not isinstance(cfg, dict) or not cfg:
            root = getattr(app_state, "config", {}) or {}
            memory = root.get("memory", {}) if isinstance(root, dict) else {}
            cfg = memory.get("consolidation", {}) if isinstance(memory, dict) else {}
        cfg = dict(cfg or {})
        adapter = getattr(app_state, "memory_consolidation_adapter", None)
        gen = dict(cfg.get("generation", {}) if isinstance(cfg.get("generation"), dict) else {})
        gen.setdefault("temperature", 0.2)
        gen.setdefault("max_output_tokens", 4000)
        retries = _int(cfg.get("summary_max_retries"), 3)
        return adapter, gen, max(1, min(10, retries))
    except Exception:
        return None, {"temperature": 0.2, "max_output_tokens": 4000}, 3


def _format_event_time(value: object) -> str:
    timestamp_ms = _int(value, 0)
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def _format_confidence(value: object) -> str:
    confidence = max(0.0, min(1.0, _float(value, 0.0)))
    return f"{confidence:.2f}"


def _write_summary_cache(
    con: sqlite3.Connection,
    *,
    summary_id: str,
    summary: str,
    input_hash: str,
    model: str,
    now_ms: int,
) -> None:
    con.execute(
        """
        INSERT INTO MemorySummaryCache (
            summary_id, task_id, input_hash, model, status, summary,
            created_at_ms, updated_at_ms, error_json
        ) VALUES (?, ?, ?, ?, 'ready', ?, ?, ?, '{}')
        ON CONFLICT(summary_id) DO UPDATE SET
            task_id=excluded.task_id,
            input_hash=excluded.input_hash,
            model=excluded.model,
            status='ready',
            summary=excluded.summary,
            updated_at_ms=excluded.updated_at_ms,
            error_json='{}'
        """,
        (
            summary_id,
            summary_id,
            input_hash,
            model,
            str(summary or "").strip(),
            now_ms,
            now_ms,
        ),
    )


def _upsert_storyline_summary_task(
    con: sqlite3.Connection,
    *,
    task_id: str,
    task_type: str,
    storyline_id: str,
    storyline_revision: int,
    input_hash: str,
    priority: int,
    confidence_tier: str,
    now_ms: int,
) -> None:
    con.execute(
        """
        INSERT INTO MemoryStorylineSummaryTasks (
            task_id, task_type, storyline_id, storyline_revision, input_hash,
            priority, confidence_tier, status, retry_count, last_error,
            created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', 0, '', ?, ?)
        ON CONFLICT(task_id) DO UPDATE SET
            task_type=excluded.task_type,
            storyline_id=excluded.storyline_id,
            storyline_revision=excluded.storyline_revision,
            input_hash=excluded.input_hash,
            priority=excluded.priority,
            confidence_tier=excluded.confidence_tier,
            status='active',
            retry_count=0,
            last_error='',
            updated_at_ms=excluded.updated_at_ms
        """,
        (
            task_id,
            task_type,
            storyline_id,
            int(storyline_revision),
            input_hash,
            int(priority),
            confidence_tier,
            now_ms,
            now_ms,
        ),
    )


def _replace_storyline_summary_task_events(
    con: sqlite3.Connection,
    task_id: str,
    events: list[dict[str, Any]],
) -> None:
    con.execute("DELETE FROM MemoryStorylineSummaryTaskEvents WHERE task_id=?", (task_id,))
    con.executemany(
        """
        INSERT OR REPLACE INTO MemoryStorylineSummaryTaskEvents (task_id, event_id, rank, role, status)
        VALUES (?, ?, ?, ?, 'active')
        """,
        [
            (
                task_id,
                int(event["event_id"]),
                index,
                str(event.get("window_role") or "storyline_member"),
            )
            for index, event in enumerate(events, start=1)
            if int(event.get("event_id") or 0) > 0
        ],
    )


def _load_storyline_event_window(
    con: sqlite3.Connection,
    storyline_id: str,
    *,
    max_events: int,
) -> list[dict[str, Any]]:
    rows = list(
        con.execute(
            """
            SELECT e.event_id, e.summary, e.confidence, e.occurred_at, e.created_at,
                   m.rank
            FROM MemoryStorylineMembers m
            JOIN MemoryEvents e ON e.event_id=m.event_id
            WHERE m.storyline_id=? AND m.status='active' AND e.is_deleted=0
            ORDER BY m.rank ASC, e.occurred_at ASC, e.event_id ASC
            LIMIT ?
            """,
            (storyline_id, max(1, int(max_events or 1))),
        )
    )
    return [
        {
            "event_id": int(row["event_id"]),
            "summary": str(row["summary"] or ""),
            "confidence": float(row["confidence"] or 0.0),
            "occurred_at": int(row["occurred_at"] or 0),
            "created_at": int(row["created_at"] or 0),
            "window_role": "storyline_member",
        }
        for row in rows
    ]


def _unique_strings(values: Iterable[Any]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return bool(
        con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        ).fetchone()
    )


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sha1(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "STORYLINE_SUMMARY_MODEL",
    "SummaryRefreshReport",
    "process_active_summary_inputs",
    "queue_storyline_summary_refresh_tasks",
    "run_summary_refresh_worker",
    "summary_id_for_source",
]
