"""Memory storyline-summary worker.

The worker turns structured storyline-summary tasks into ready storyline-summary
rows. Event-storyline summaries are always generated or refreshed by the
memory-consolidation LLM; deterministic code may only prepare the LLM payload.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Iterable

from .prompt import STORYLINE_SUMMARY_SYSTEM_PROMPT
from .consolidation import (
    StorylineSummaryRecord,
    ensure_preprocessing_schema,
    storyline_summary_from_json,
    storyline_summary_to_json,
)


STORYLINE_SUMMARY_EVENT_LIMIT = 24
STORYLINE_SUMMARY_MODEL = "memory_consolidation.storyline_summary.v1"
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
            relations = _load_storyline_relation_briefs(con, storyline_id)
            input_hash = _storyline_summary_task_hash(
                storyline_id=storyline_id,
                storyline_revision=source_revision,
                event_ids=[int(item["event_id"]) for item in events],
                relation_ids=[str(item.get("relation_id") or "") for item in relations],
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
            _replace_storyline_summary_task_links(con, summary_id, events, relations)
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
                packet = _build_storyline_summary_task_packet(con, row)
                summary = _build_llm_storyline_summary_from_packet(adapter, packet, gen)
                stats["summary_llm_calls"] += 1
                if not summary.summary_id or not summary.short_summary:
                    raise ValueError("storyline summary is empty")
                _write_summary_cache(
                    con,
                    summary,
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
    relation_ids: Iterable[str],
) -> str:
    return _sha1(
        "storyline-summary-task",
        storyline_id,
        str(int(storyline_revision or 0)),
        ",".join(str(int(event_id)) for event_id in event_ids if int(event_id) > 0),
        ",".join(str(relation_id) for relation_id in relation_ids if str(relation_id or "").strip()),
    )


def _build_storyline_summary_from_packet(packet: dict[str, Any]) -> StorylineSummaryRecord:
    previous = _previous_storyline_summary(packet)
    task_type = str(packet.get("task_type") or "")
    source_kind = str(packet.get("source_kind") or (previous.source_kind if previous else "") or "storyline")
    source_id = str(packet.get("source_id") or (previous.source_id if previous else "") or "")
    summary_id = (
        (previous.summary_id if previous else "")
        or str(packet.get("summary_id") or "")
        or summary_id_for_source(source_kind, source_id)
    )
    source_revision = _int(packet.get("source_revision"), previous.revision if previous else 1)
    revision = max(1, source_revision)
    if previous and task_type == "summary_refresh":
        revision = max(previous.revision + 1, revision)

    events = _normalize_packet_events(packet.get("events") or ())
    event_ids = tuple(item["event_id"] for item in events if item.get("event_id"))
    boundary_notes = _derive_boundary_notes(packet, previous)

    return StorylineSummaryRecord(
        summary_id=summary_id,
        source_kind=source_kind,
        source_id=source_id,
        revision=revision,
        title=_clip(previous.title if previous else source_id, 80),
        short_summary=_clip(previous.short_summary if previous else "", 360),
        core_entities=previous.core_entities if previous else (),
        confirmed_claims=previous.confirmed_claims if previous else (),
        uncertain_claims=previous.uncertain_claims if previous else (),
        disputed_claims=previous.disputed_claims if previous else (),
        current_state=previous.current_state if previous else "",
        open_slots=previous.open_slots if previous else (),
        boundary_notes=tuple(boundary_notes[:8]),
        source_event_ids=event_ids,
    )


def _build_llm_storyline_summary_from_packet(adapter: Any, packet: dict[str, Any], gen: dict[str, Any]) -> StorylineSummaryRecord:
    base = _build_storyline_summary_from_packet(packet)
    payload = _storyline_summary_llm_payload(packet, base)
    raw = adapter.call_simple_text(
        STORYLINE_SUMMARY_SYSTEM_PROMPT,
        json.dumps(payload, ensure_ascii=False, indent=2),
        gen,
        log_tag="memory_consolidation/summary",
    )
    parsed = _parse_storyline_summary_response(raw)
    title = _clip(parsed.get("title") or base.title, 80)
    short_summary = _clip(parsed.get("summary") or parsed.get("short_summary") or base.short_summary, 720)
    if not short_summary:
        raise ValueError("LLM summary response missing summary")
    return StorylineSummaryRecord(
        summary_id=base.summary_id,
        source_kind=base.source_kind,
        source_id=base.source_id,
        revision=base.revision,
        title=title,
        short_summary=short_summary,
        core_entities=tuple(_string_list(parsed.get("core_entities"), fallback=base.core_entities)[:12]),
        confirmed_claims=tuple(_string_list(parsed.get("confirmed_claims"), fallback=base.confirmed_claims)[:16]),
        uncertain_claims=tuple(_string_list(parsed.get("uncertain_claims"), fallback=base.uncertain_claims)[:10]),
        disputed_claims=tuple(_string_list(parsed.get("disputed_claims"), fallback=base.disputed_claims)[:10]),
        current_state=_clean_state(parsed.get("current_state") or base.current_state),
        open_slots=tuple(_string_list(parsed.get("open_slots"), fallback=base.open_slots)[:8]),
        boundary_notes=tuple(_string_list(parsed.get("boundary_notes"), fallback=base.boundary_notes)[:8]),
        source_event_ids=base.source_event_ids,
    )


def _storyline_summary_llm_payload(packet: dict[str, Any], base: StorylineSummaryRecord) -> dict[str, Any]:
    return {
        "task": "generate_or_refresh_event_storyline_summary",
        "summary_id": base.summary_id,
        "source_kind": base.source_kind,
        "source_id": base.source_id,
        "source_revision": base.revision,
        "storyline": packet.get("storyline") if isinstance(packet.get("storyline"), dict) else {},
        "previous_storyline_summary_stale_prior": packet.get("previous_storyline_summary_stale_prior") or {},
        "events": packet.get("events") or [],
        "relations": packet.get("relations") or [],
        "policy": {
            "source_event_ids_must_remain": list(base.source_event_ids),
            "do_not_invent_facts": True,
            "newer_events_win_on_conflict": True,
        },
    }


def _build_storyline_summary_task_packet(con: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    task_id = str(row["task_id"] or "")
    storyline_id = str(row["storyline_id"] or "")
    storyline_revision = _int(row["storyline_revision"], 1)
    summary_id = task_id
    events = _load_storyline_summary_task_events(con, task_id)
    relations = _load_storyline_summary_task_relations(con, task_id)
    previous = _load_previous_storyline_summary(con, summary_id)
    storyline = _load_storyline_brief(con, storyline_id)
    return {
        "summary_id": summary_id,
        "task_type": f"summary_{row['task_type']}",
        "source_kind": "storyline",
        "source_id": storyline_id,
        "source_revision": storyline_revision,
        "storyline": storyline,
        "previous_storyline_summary_stale_prior": asdict(previous) if previous else {},
        "storyline_summary": asdict(previous) if previous else {},
        "events": events,
        "relations": relations,
    }


def _load_previous_storyline_summary(con: sqlite3.Connection, summary_id: str) -> StorylineSummaryRecord | None:
    row = con.execute(
        """
        SELECT storyline_summary_json
        FROM MemorySummaryCache
        WHERE (summary_id=? OR task_id=?) AND storyline_summary_json <> '{}'
        ORDER BY
          CASE status
            WHEN 'stale' THEN 0
            WHEN 'ready' THEN 1
            ELSE 2
          END,
          updated_at_ms DESC
        LIMIT 1
        """,
        (summary_id, summary_id),
    ).fetchone()
    if not row:
        return None
    try:
        return storyline_summary_from_json(str(row[0] or "{}"))
    except Exception:
        return None


def _load_storyline_brief(con: sqlite3.Connection, storyline_id: str) -> dict[str, Any]:
    if not _table_exists(con, "MemoryStorylines"):
        return {}
    row = con.execute(
        """
        SELECT scope, scheme_name, anchor_key, profile, member_count, score
        FROM MemoryStorylines
        WHERE storyline_id=?
        LIMIT 1
        """,
        (storyline_id,),
    ).fetchone()
    if not row:
        return {}
    return {
        "scope": str(row[0] or ""),
        "scheme_name": str(row[1] or ""),
        "anchor_key": str(row[2] or ""),
        "profile": str(row[3] or ""),
        "member_count": int(row[4] or 0),
        "score": float(row[5] or 0.0),
    }


def _load_storyline_summary_task_events(con: sqlite3.Connection, task_id: str) -> list[dict[str, Any]]:
    if not _table_exists(con, "MemoryEvents"):
        return []
    rows = list(
        con.execute(
            """
            SELECT e.event_id, e.summary, e.event_type_norm, e.status, e.confidence,
                   e.occurred_at, e.created_at, e.last_seen_at, e.last_accessed,
                   e.occurrences, e.access_count, te.rank, te.role
            FROM MemoryStorylineSummaryTaskEvents te
            JOIN MemoryEvents e ON e.event_id=te.event_id
            WHERE te.task_id=? AND te.status='active' AND e.is_deleted=0
            ORDER BY te.rank ASC, e.occurred_at ASC, e.event_id ASC
            """,
            (task_id,),
        )
    )
    event_ids = [int(row["event_id"]) for row in rows]
    roles_by_event = _load_event_role_briefs(con, event_ids)
    return [
        {
            "event_id": int(row["event_id"]),
            "summary": str(row["summary"] or ""),
            "event_type_norm": str(row["event_type_norm"] or ""),
            "status": str(row["status"] or ""),
            "confidence": float(row["confidence"] or 0.0),
            "occurred_at": int(row["occurred_at"] or 0),
            "created_at": int(row["created_at"] or 0),
            "last_seen_at": int(row["last_seen_at"] or 0),
            "last_accessed": int(row["last_accessed"] or 0),
            "occurrences": int(row["occurrences"] or 1),
            "access_count": int(row["access_count"] or 0),
            "window_role": str(row["role"] or "storyline_member"),
            "roles": roles_by_event.get(int(row["event_id"]), []),
        }
        for row in rows
    ]


def _load_storyline_summary_task_relations(con: sqlite3.Connection, task_id: str) -> list[dict[str, Any]]:
    if not _table_exists(con, "MemoryStorylineSummaryTaskRelations"):
        return []
    return [
        {
            "relation_id": str(row[0] or ""),
            "source_event_id": int(row[1] or 0),
            "target_event_id": int(row[2] or 0),
            "relation_type": str(row[3] or ""),
            "status": str(row[4] or ""),
            "confidence": float(row[5] or 0.0),
        }
        for row in con.execute(
            """
            SELECT relation_id, source_event_id, target_event_id, relation_type, status, confidence
            FROM MemoryStorylineSummaryTaskRelations
            WHERE task_id=? AND status IN ('active', 'weak')
            ORDER BY relation_id ASC
            """,
            (task_id,),
        )
    ]


def _parse_storyline_summary_response(raw: object) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty LLM summary response")
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    first = _extract_first_json_object(text)
    if first:
        candidates.insert(0, first)
    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(exc.msg)
            continue
        if isinstance(parsed, dict):
            return parsed
        errors.append("response JSON is not an object")
    raise ValueError("no parseable LLM summary JSON: " + "; ".join(errors[:3]))


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


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


def _string_list(value: object, *, fallback: Iterable[str] = ()) -> list[str]:
    source = value if isinstance(value, list) else list(fallback)
    out: list[str] = []
    for item in source:
        text = str(item or "").strip()
        if text:
            out.append(_clip(text, 240))
    return _dedupe_preserve_order(out)


def _clean_state(value: object) -> str:
    return str(value or "").strip()


def _previous_storyline_summary(packet: dict[str, Any]) -> StorylineSummaryRecord | None:
    for key in ("previous_storyline_summary_stale_prior", "storyline_summary"):
        value = packet.get(key)
        if isinstance(value, dict):
            try:
                summary = storyline_summary_from_json(value)
                if summary.summary_id:
                    return summary
            except Exception:
                continue
    return None


def _normalize_packet_events(values: Iterable[Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        event_id = _int(value.get("event_id"), 0)
        if event_id <= 0:
            continue
        item = dict(value)
        item["event_id"] = event_id
        item["summary"] = str(item.get("summary") or "").strip()
        item["occurred_at"] = _int(item.get("occurred_at"), 0)
        item["created_at"] = _int(item.get("created_at"), 0)
        item["confidence"] = _float(item.get("confidence"), 0.5)
        events.append(item)
    events.sort(key=lambda item: (_int(item.get("occurred_at"), 0), item["event_id"]))
    deduped: dict[int, dict[str, Any]] = {}
    for item in events:
        deduped[item["event_id"]] = item
    return list(deduped.values())


def _derive_boundary_notes(packet: dict[str, Any], previous: StorylineSummaryRecord | None) -> list[str]:
    notes = list(previous.boundary_notes if previous else ())
    if packet.get("previous_storyline_summary_stale_prior"):
        notes.append("previous_storyline_summary_stale_prior")
    policy = packet.get("window_policy")
    if isinstance(policy, dict) and policy.get("activation_score_is_relevance_not_truth"):
        notes.append("activation_score_is_relevance_not_truth")
    return _dedupe_preserve_order(notes)


def _write_summary_cache(
    con: sqlite3.Connection,
    summary: StorylineSummaryRecord,
    *,
    input_hash: str,
    model: str,
    now_ms: int,
) -> None:
    task_id = summary.summary_id
    con.execute(
        """
        UPDATE MemorySummaryCache
        SET status='stale', updated_at_ms=?
        WHERE task_id=? AND summary_id<>? AND status='ready'
        """,
        (now_ms, task_id, summary.summary_id),
    )
    con.execute(
        """
        INSERT INTO MemorySummaryCache (
            summary_id, task_id, input_hash, model, status, title, short_summary,
            digest_json, salient_entities_json, storyline_summary_json, created_at_ms,
            updated_at_ms, error_json
        ) VALUES (?, ?, ?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, '{}')
        ON CONFLICT(summary_id) DO UPDATE SET
            task_id=excluded.task_id,
            input_hash=excluded.input_hash,
            model=excluded.model,
            status='ready',
            title=excluded.title,
            short_summary=excluded.short_summary,
            digest_json=excluded.digest_json,
            salient_entities_json=excluded.salient_entities_json,
            storyline_summary_json=excluded.storyline_summary_json,
            updated_at_ms=excluded.updated_at_ms,
            error_json='{}'
        """,
        (
            summary.summary_id,
            task_id,
            input_hash,
            model,
            summary.title,
            summary.short_summary,
            _json(list(summary.confirmed_claims)),
            _json(list(summary.core_entities)),
            storyline_summary_to_json(summary),
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


def _replace_storyline_summary_task_links(
    con: sqlite3.Connection,
    task_id: str,
    events: list[dict[str, Any]],
    relations: list[dict[str, Any]],
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
    con.execute("DELETE FROM MemoryStorylineSummaryTaskRelations WHERE task_id=?", (task_id,))
    con.executemany(
        """
        INSERT OR REPLACE INTO MemoryStorylineSummaryTaskRelations (
            task_id, relation_id, source_event_id, target_event_id, relation_type, status, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                task_id,
                str(rel.get("relation_id") or ""),
                _int(rel.get("source_event_id"), 0),
                _int(rel.get("target_event_id"), 0),
                str(rel.get("relation_type") or ""),
                str(rel.get("status") or ""),
                _float(rel.get("confidence"), 0.0),
            )
            for rel in relations
            if str(rel.get("relation_id") or "")
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
            SELECT e.event_id, e.summary, e.event_type_norm, e.status, e.confidence,
                   e.occurred_at, e.created_at, e.last_seen_at, e.last_accessed,
                   e.occurrences, e.access_count, m.rank
            FROM MemoryStorylineMembers m
            JOIN MemoryEvents e ON e.event_id=m.event_id
            WHERE m.storyline_id=? AND m.status='active' AND e.is_deleted=0
            ORDER BY m.rank ASC, e.occurred_at ASC, e.event_id ASC
            LIMIT ?
            """,
            (storyline_id, max(1, int(max_events or 1))),
        )
    )
    event_ids = [int(row["event_id"]) for row in rows]
    roles_by_event = _load_event_role_briefs(con, event_ids)
    return [
        {
            "event_id": int(row["event_id"]),
            "summary": str(row["summary"] or ""),
            "event_type_norm": str(row["event_type_norm"] or ""),
            "status": str(row["status"] or ""),
            "confidence": float(row["confidence"] or 0.0),
            "occurred_at": int(row["occurred_at"] or 0),
            "created_at": int(row["created_at"] or 0),
            "last_seen_at": int(row["last_seen_at"] or 0),
            "last_accessed": int(row["last_accessed"] or 0),
            "occurrences": int(row["occurrences"] or 1),
            "access_count": int(row["access_count"] or 0),
            "window_role": "storyline_member",
            "roles": roles_by_event.get(int(row["event_id"]), []),
        }
        for row in rows
    ]


def _load_storyline_relation_briefs(con: sqlite3.Connection, storyline_id: str) -> list[dict[str, Any]]:
    if not _table_exists(con, "MemoryStorylineRelations"):
        return []
    return [
        {
            "relation_id": str(row[0] or ""),
            "source_event_id": int(row[1] or 0),
            "target_event_id": int(row[2] or 0),
            "relation_type": str(row[3] or ""),
            "status": str(row[4] or ""),
            "confidence": float(row[5] or 0.0),
        }
        for row in con.execute(
            """
            SELECT relation_id, source_event_id, target_event_id, relation_type, status, confidence
            FROM MemoryStorylineRelations
            WHERE storyline_id=? AND status IN ('active', 'weak', 'rejected')
            ORDER BY updated_at_ms ASC, relation_id ASC
            """,
            (storyline_id,),
        )
    ]


def _load_event_role_briefs(con: sqlite3.Connection, event_ids: list[int]) -> dict[int, list[dict[str, str]]]:
    if not event_ids or not _table_exists(con, "MemoryParticipants"):
        return {}
    placeholders = ",".join("?" * len(event_ids))
    roles: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in con.execute(
        f"""
        SELECT event_id, role, entity, value_text
        FROM MemoryParticipants
        WHERE event_id IN ({placeholders})
        ORDER BY event_id ASC, participant_id ASC
        """,
        event_ids,
    ):
        roles[int(row[0])].append(
            {
                "role": str(row[1] or ""),
                "entity": str(row[2] or ""),
                "value_text": str(row[3] or ""),
            }
        )
    return roles


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = re.sub(r"\s+", "", text.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _clip(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _safe_json_object(raw: object) -> dict[str, Any]:
    try:
        value = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


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


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
