"""Memory cluster-summary worker.

The worker turns queued summary inputs into ready cluster-summary cache rows.
When the memory-consolidation model is configured, generation/refresh is done
by that LLM.  A deterministic builder remains only as a compatibility fallback
for tests or explicitly disabled consolidation configs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any, Callable, Iterable

from .consolidation import (
    ClusterSummaryRecord,
    ensure_preprocessing_schema,
    cluster_summary_from_json,
    cluster_summary_to_json,
)


SUMMARY_BOOTSTRAP_EVENT_LIMIT = 24
CLUSTER_SUMMARY_MODEL = "memory_consolidation.cluster_summary.v1"
DETERMINISTIC_SUMMARY_MODEL = "deterministic.summary_worker.v1"
_SOURCE_KIND_RE = re.compile(r"[^a-zA-Z0-9_.:-]+")
logger = logging.getLogger("AICQ.memory.summary_worker")

CLUSTER_SUMMARY_SYSTEM_PROMPT = """\
你是长期记忆的事件簇 summary 生成器。

任务：根据输入的事件簇、事件窗口、关系和可选旧 summary，生成一个供长期记忆召回使用的事件簇 summary。

约束：
- 只能依据输入事件和关系，不要编造新事实。
- previous_cluster_summary_stale_prior 只能作为旧草稿；新事件和关系优先。
- 如果存在 correction/refutation/rejected 关系，要体现事实被修正或存在争议。
- 输出要短、准、可召回；不要写流水账。
- 不要输出 markdown，不要输出解释。

输出严格 JSON：
{
  "title": "短标题",
  "summary": "一段自然语言 summary",
  "core_entities": ["核心实体"],
  "confirmed_claims": ["确认事实"],
  "uncertain_claims": ["不确定事实"],
  "disputed_claims": ["争议/被修正事实"],
  "current_state": "observed|in_progress|completed|revised|unknown",
  "open_slots": ["后续可接续的槽位"],
  "boundary_notes": ["边界说明"]
}
"""


def run_summary_refresh_worker(
    con_or_path: sqlite3.Connection | str | os.PathLike[str],
    *,
    max_inputs: int = 32,
    max_bootstrap_clusters: int = 64,
    priority_packet_ids: Iterable[str] = (),
    priority_cluster_ids: Iterable[str] = (),
    deadline_ms: int | None = None,
    should_continue: Callable[[], bool] | None = None,
    now_ms: int | None = None,
    model: str = CLUSTER_SUMMARY_MODEL,
) -> dict[str, int]:
    """Queue missing cluster summaries and process active summary inputs."""

    owns_connection = not isinstance(con_or_path, sqlite3.Connection)
    con = sqlite3.connect(os.fspath(con_or_path), timeout=30.0) if owns_connection else con_or_path
    try:
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        ensure_preprocessing_schema(con)
        con.commit()
        now = int(now_ms or _now_ms())
        target_cluster_ids = _unique_strings(priority_cluster_ids)
        target_packet_ids = set(_unique_strings(priority_packet_ids))
        target_packet_ids.update(summary_id_for_source("cluster", cluster_id) for cluster_id in target_cluster_ids)

        priority_queued = 0
        if target_cluster_ids:
            priority_queued = queue_missing_cluster_summary_inputs(
                con,
                max_clusters=len(target_cluster_ids),
                cluster_ids=target_cluster_ids,
                now_ms=now,
            )
        queued = queue_missing_cluster_summary_inputs(
            con,
            max_clusters=max_bootstrap_clusters,
            now_ms=now,
        )
        con.commit()
        input_limit = max(max(1, int(max_inputs or 1)), len(target_packet_ids))
        processed = process_active_summary_inputs(
            con,
            max_inputs=input_limit,
            priority_packet_ids=target_packet_ids,
            deadline_ms=deadline_ms,
            should_continue=should_continue,
            now_ms=now,
            model=model,
        )
        if owns_connection:
            con.commit()
        return {"priority_bootstrap_inputs_queued": priority_queued, "bootstrap_inputs_queued": queued, **processed}
    finally:
        if owns_connection:
            con.close()


def queue_missing_cluster_summary_inputs(
    con: sqlite3.Connection,
    *,
    max_clusters: int = 64,
    cluster_ids: Iterable[str] = (),
    now_ms: int | None = None,
) -> int:
    """Create summary inputs for active clusters that lack a fresh ready summary."""

    ensure_preprocessing_schema(con)
    if not _table_exists(con, "MemoryClusters") or not _table_exists(con, "MemoryClusterMembers"):
        return 0

    now = int(now_ms or _now_ms())
    previous_row_factory = con.row_factory
    queued = 0
    try:
        con.row_factory = sqlite3.Row
        target_ids = _unique_strings(cluster_ids)
        target_clause = ""
        params: list[Any] = []
        if target_ids:
            placeholders = ",".join("?" * len(target_ids))
            target_clause = f" AND cluster_id IN ({placeholders})"
            params.extend(target_ids)
        params.append(max(1, int(max_clusters or 1)))
        clusters = list(
            con.execute(
                f"""
                SELECT cluster_id, scope, scheme_name, anchor_key, profile,
                       revision, member_count, score, updated_at
                FROM MemoryClusters
                WHERE status='active' AND member_count >= 2{target_clause}
                ORDER BY updated_at DESC, score DESC, cluster_id ASC
                LIMIT ?
                """,
                params,
            )
        )
        for row in clusters:
            cluster_id = str(row["cluster_id"] or "")
            summary_id = summary_id_for_source("cluster", cluster_id)
            source_revision = int(row["revision"] or 1)
            events = _load_cluster_event_window(
                con,
                cluster_id,
                max_events=SUMMARY_BOOTSTRAP_EVENT_LIMIT,
            )
            if not events:
                continue
            relations = _load_cluster_relation_briefs(con, cluster_id)
            packet = {
                "packet_id": summary_id,
                "packet_type": "summary_bootstrap_input",
                "source_kind": "cluster",
                "source_id": cluster_id,
                "source_revision": source_revision,
                "cluster": {
                    "scope": str(row["scope"] or ""),
                    "scheme_name": str(row["scheme_name"] or ""),
                    "anchor_key": str(row["anchor_key"] or ""),
                    "profile": str(row["profile"] or ""),
                    "member_count": int(row["member_count"] or 0),
                    "score": float(row["score"] or 0.0),
                },
                "cluster_summary": asdict(
                    _build_cluster_summary_from_packet(
                        {
                            "packet_id": summary_id,
                            "packet_type": "summary_bootstrap_input",
                            "source_kind": "cluster",
                            "source_id": cluster_id,
                            "source_revision": source_revision,
                            "events": events,
                            "relations": relations,
                        }
                    )
                ),
                "events": events,
                "relations": relations,
                "provenance": {"llm_used": False, "generator": "memory.summary_worker"},
            }
            input_hash = _sha1("summary-input", _json(packet))
            ready = con.execute(
                """
                SELECT 1
                FROM MemorySummaryCache
                WHERE packet_id=? AND input_hash=? AND status='ready'
                LIMIT 1
                """,
                (summary_id, input_hash),
            ).fetchone()
            if ready:
                continue
            existing_active = con.execute(
                """
                SELECT 1
                FROM MemorySummaryInputs
                WHERE packet_id=? AND input_hash=? AND status='active'
                LIMIT 1
                """,
                (summary_id, input_hash),
            ).fetchone()
            if existing_active:
                continue
            _upsert_summary_input(
                con,
                packet_id=summary_id,
                packet_type="summary_bootstrap_input",
                source_kind="cluster",
                source_id=cluster_id,
                source_revision=source_revision,
                input_hash=input_hash,
                priority=30,
                confidence_tier="medium",
                packet=packet,
                now_ms=now,
            )
            _replace_summary_input_links(con, summary_id, events, relations)
            queued += 1
    finally:
        con.row_factory = previous_row_factory
    return queued


def process_active_summary_inputs(
    con: sqlite3.Connection,
    *,
    max_inputs: int = 32,
    priority_packet_ids: Iterable[str] = (),
    deadline_ms: int | None = None,
    should_continue: Callable[[], bool] | None = None,
    now_ms: int | None = None,
    model: str = CLUSTER_SUMMARY_MODEL,
) -> dict[str, int]:
    """Consume active summary input packets and write ready cluster summaries."""

    ensure_preprocessing_schema(con)
    con.commit()
    now = int(now_ms or _now_ms())
    adapter, gen, llm_required, max_retries = _summary_llm_runtime()
    previous_row_factory = con.row_factory
    stats = {
        "summary_inputs_loaded": 0,
        "summary_inputs_done": 0,
        "summaries_ready": 0,
        "summary_inputs_failed": 0,
        "summary_inputs_retrying": 0,
        "summary_queue_paused": 0,
        "summary_llm_calls": 0,
    }
    try:
        con.row_factory = sqlite3.Row
        priority_ids = _unique_strings(priority_packet_ids)
        order_prefix = ""
        params: list[Any] = []
        if priority_ids:
            placeholders = ",".join("?" * len(priority_ids))
            order_prefix = f"CASE WHEN packet_id IN ({placeholders}) THEN 0 ELSE 1 END,"
            params.extend(priority_ids)
        params.append(max(1, int(max_inputs or 1)))
        rows = list(
            con.execute(
                f"""
                SELECT *
                FROM MemorySummaryInputs
                WHERE status='active'
                  AND packet_type IN ('summary_bootstrap_input', 'summary_refresh_input')
                ORDER BY {order_prefix} priority DESC, updated_at_ms ASC, packet_id ASC
                LIMIT ?
                """,
                params,
            )
        )
        stats["summary_inputs_loaded"] = len(rows)
        for row in rows:
            if (deadline_ms and _now_ms() >= int(deadline_ms)) or (
                should_continue is not None and not should_continue()
            ):
                stats["summary_queue_paused"] = 1
                break
            packet_id = str(row["packet_id"] or "")
            try:
                packet = json.loads(str(row["packet_json"] or "{}"))
                if not isinstance(packet, dict):
                    raise ValueError("packet_json must be an object")
                if llm_required:
                    if adapter is None:
                        raise RuntimeError("memory_consolidation_adapter unavailable")
                    summary = _build_llm_cluster_summary_from_packet(adapter, packet, gen)
                    stats["summary_llm_calls"] += 1
                else:
                    summary = _build_cluster_summary_from_packet(packet)
                if not summary.summary_id or not summary.short_summary:
                    raise ValueError("cluster summary is empty")
                _write_summary_cache(
                    con,
                    summary,
                    input_hash=str(row["input_hash"] or ""),
                    model=model if llm_required else DETERMINISTIC_SUMMARY_MODEL,
                    now_ms=now,
                )
                con.execute(
                    """
                    UPDATE MemorySummaryInputs
                    SET status='done', updated_at_ms=?
                    WHERE packet_id=?
                    """,
                    (now, packet_id),
                )
                stats["summary_inputs_done"] += 1
                stats["summaries_ready"] += 1
                con.commit()
            except Exception as exc:
                provenance = _safe_json_object(row["provenance_json"])
                retry_count = int(provenance.get("retry_count") or 0) + 1
                provenance["retry_count"] = retry_count
                provenance["last_error"] = str(exc)[:500]
                provenance["llm_required"] = bool(llm_required)
                should_retry = retry_count < max(1, int(max_retries or 1))
                con.execute(
                    """
                    UPDATE MemorySummaryInputs
                    SET status=?, updated_at_ms=?, provenance_json=?
                    WHERE packet_id=?
                    """,
                    ("active" if should_retry else "failed", now, _json(provenance), packet_id),
                )
                if should_retry:
                    stats["summary_inputs_retrying"] += 1
                else:
                    stats["summary_inputs_failed"] += 1
                con.commit()
                logger.warning(
                    "[summary_worker] 事件簇 summary 生成失败 packet_id=%s retry=%d/%d error=%s",
                    packet_id,
                    retry_count,
                    max_retries,
                    exc,
                )
    finally:
        con.row_factory = previous_row_factory
    return stats


def summary_id_for_source(source_kind: str, source_id: str) -> str:
    kind = _SOURCE_KIND_RE.sub("_", str(source_kind or "summary").strip() or "summary")
    return f"summary:{kind}:{_sha1('summary-source', kind, str(source_id or ''))[:20]}"


def _build_cluster_summary_from_packet(packet: dict[str, Any]) -> ClusterSummaryRecord:
    previous = _previous_cluster_summary(packet)
    packet_type = str(packet.get("packet_type") or "")
    source_kind = str(packet.get("source_kind") or (previous.source_kind if previous else "") or "cluster")
    source_id = str(packet.get("source_id") or (previous.source_id if previous else "") or "")
    summary_id = (
        (previous.summary_id if previous else "")
        or str(packet.get("summary_id") or packet.get("packet_id") or "")
        or summary_id_for_source(source_kind, source_id)
    )
    source_revision = _int(packet.get("source_revision"), previous.revision if previous else 1)
    revision = max(1, source_revision)
    if previous and packet_type == "summary_refresh_input":
        revision = max(previous.revision + 1, revision)

    events = _normalize_packet_events(packet.get("events") or ())
    relations = [rel for rel in packet.get("relations") or () if isinstance(rel, dict)]
    event_ids = tuple(item["event_id"] for item in events if item.get("event_id"))
    entities = _select_core_entities(events, previous.core_entities if previous else ())
    confirmed_claims = _select_claims(events, relations, status="confirmed", previous=previous)
    uncertain_claims = _select_claims(events, relations, status="uncertain", previous=previous)
    disputed_claims = _select_claims(events, relations, status="disputed", previous=previous)
    current_state = _derive_current_state(events, relations, previous)
    title = (previous.title if previous else "") or _derive_title(entities, events, source_kind, source_id)
    short_summary = _derive_short_summary(events, previous)
    open_slots = _derive_open_slots(current_state, events, previous)
    boundary_notes = _derive_boundary_notes(packet, previous)

    return ClusterSummaryRecord(
        summary_id=summary_id,
        source_kind=source_kind,
        source_id=source_id,
        revision=revision,
        title=_clip(title, 80),
        short_summary=_clip(short_summary, 360),
        core_entities=tuple(entities[:12]),
        confirmed_claims=tuple(confirmed_claims[:16]),
        uncertain_claims=tuple(uncertain_claims[:10]),
        disputed_claims=tuple(disputed_claims[:10]),
        current_state=current_state,
        open_slots=tuple(open_slots[:8]),
        boundary_notes=tuple(boundary_notes[:8]),
        source_event_ids=event_ids,
    )


def _build_llm_cluster_summary_from_packet(adapter: Any, packet: dict[str, Any], gen: dict[str, Any]) -> ClusterSummaryRecord:
    base = _build_cluster_summary_from_packet(packet)
    payload = _cluster_summary_llm_payload(packet, base)
    raw = adapter.call_simple_text(
        CLUSTER_SUMMARY_SYSTEM_PROMPT,
        json.dumps(payload, ensure_ascii=False, indent=2),
        gen,
        log_tag="memory_consolidation/summary",
    )
    parsed = _parse_cluster_summary_response(raw)
    title = _clip(parsed.get("title") or base.title, 80)
    short_summary = _clip(parsed.get("summary") or parsed.get("short_summary") or base.short_summary, 720)
    if not short_summary:
        raise ValueError("LLM summary response missing summary")
    return ClusterSummaryRecord(
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


def _cluster_summary_llm_payload(packet: dict[str, Any], base: ClusterSummaryRecord) -> dict[str, Any]:
    return {
        "task": "generate_or_refresh_event_cluster_summary",
        "summary_id": base.summary_id,
        "source_kind": base.source_kind,
        "source_id": base.source_id,
        "source_revision": base.revision,
        "cluster": packet.get("cluster") if isinstance(packet.get("cluster"), dict) else {},
        "previous_cluster_summary_stale_prior": packet.get("previous_cluster_summary_stale_prior") or {},
        "events": packet.get("events") or [],
        "relations": packet.get("relations") or [],
        "policy": {
            "source_event_ids_must_remain": list(base.source_event_ids),
            "do_not_invent_facts": True,
            "newer_events_win_on_conflict": True,
        },
    }


def _parse_cluster_summary_response(raw: object) -> dict[str, Any]:
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


def _summary_llm_runtime() -> tuple[Any | None, dict[str, Any], bool, int]:
    try:
        import app_state

        cfg = getattr(app_state, "memory_consolidation_cfg", None)
        if not isinstance(cfg, dict) or not cfg:
            root = getattr(app_state, "config", {}) or {}
            memory = root.get("memory", {}) if isinstance(root, dict) else {}
            cfg = memory.get("consolidation", {}) if isinstance(memory, dict) else {}
        cfg = dict(cfg or {})
        enabled = bool(cfg.get("enabled", False))
        adapter = getattr(app_state, "memory_consolidation_adapter", None)
        gen = dict(cfg.get("generation", {}) if isinstance(cfg.get("generation"), dict) else {})
        gen.setdefault("temperature", 0.2)
        gen.setdefault("max_output_tokens", 4000)
        retries = _int(cfg.get("summary_max_retries"), 3)
        return adapter, gen, enabled, max(1, min(10, retries))
    except Exception:
        return None, {"temperature": 0.2, "max_output_tokens": 4000}, False, 3


def _string_list(value: object, *, fallback: Iterable[str] = ()) -> list[str]:
    source = value if isinstance(value, list) else list(fallback)
    out: list[str] = []
    for item in source:
        text = str(item or "").strip()
        if text:
            out.append(_clip(text, 240))
    return _dedupe_preserve_order(out)


def _clean_state(value: object) -> str:
    text = str(value or "").strip().lower()
    allowed = {"observed", "in_progress", "completed", "revised", "unknown"}
    return text if text in allowed else "observed"


def _previous_cluster_summary(packet: dict[str, Any]) -> ClusterSummaryRecord | None:
    for key in ("previous_cluster_summary_stale_prior", "cluster_summary"):
        value = packet.get(key)
        if isinstance(value, dict):
            try:
                summary = cluster_summary_from_json(value)
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


def _select_core_entities(events: list[dict[str, Any]], prior: Iterable[str]) -> list[str]:
    counter: Counter[str] = Counter()
    display: dict[str, str] = {}
    for entity in prior:
        if _is_informative_entity(entity):
            key = _entity_key(entity)
            counter[key] += 2
            display.setdefault(key, str(entity))
    for event in events:
        for role in event.get("roles") or ():
            if not isinstance(role, dict):
                continue
            entity = str(role.get("entity") or "").strip()
            if not _is_informative_entity(entity):
                continue
            key = _entity_key(entity)
            counter[key] += 1
            display.setdefault(key, entity)
    ordered = sorted(counter, key=lambda key: (-counter[key], display[key]))
    return [display[key] for key in ordered]


def _select_claims(
    events: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    *,
    status: str,
    previous: ClusterSummaryRecord | None,
) -> list[str]:
    claims: list[str] = []
    if previous:
        source = {
            "confirmed": previous.confirmed_claims,
            "uncertain": previous.uncertain_claims,
            "disputed": previous.disputed_claims,
        }.get(status, ())
        claims.extend(str(item) for item in source if str(item).strip())

    disputed_event_ids = {
        _int(rel.get("source_event_id"), 0)
        for rel in relations
        if str(rel.get("relation_type") or "") in {"corrects", "corrects_identity", "refutes"}
    }
    weak_event_ids = {
        _int(rel.get("source_event_id"), 0)
        for rel in relations
        if str(rel.get("status") or "") == "weak"
    }
    for event in events:
        summary = str(event.get("summary") or "").strip()
        if not summary:
            continue
        event_status = str(event.get("status") or "").lower()
        event_id = _int(event.get("event_id"), 0)
        if status == "disputed" and event_id in disputed_event_ids:
            claims.append(summary)
        elif status == "uncertain" and (event_id in weak_event_ids or event_status in {"possible", "hypothetical", "conditional", "future"}):
            claims.append(summary)
        elif status == "confirmed" and event_id not in disputed_event_ids and event_id not in weak_event_ids and event_status not in {"hypothetical", "conditional", "future"}:
            claims.append(summary)
    return _dedupe_preserve_order(claims)


def _derive_current_state(
    events: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    previous: ClusterSummaryRecord | None,
) -> str:
    text = " ".join(str(event.get("summary") or "") for event in events[-3:]).lower()
    relation_types = {str(rel.get("relation_type") or "") for rel in relations if str(rel.get("status") or "") in {"active", "weak"}}
    if any(marker in text for marker in ("完成", "通关", "白金", "complete", "finished", "done")):
        return "completed"
    if relation_types & {"updates_state", "progresses", "continues"}:
        return "in_progress"
    if relation_types & {"corrects", "corrects_identity", "refutes"}:
        return "revised"
    return (previous.current_state if previous else "") or "observed"


def _derive_title(entities: list[str], events: list[dict[str, Any]], source_kind: str, source_id: str) -> str:
    if len(entities) >= 2:
        return f"{entities[0]} / {entities[1]}"
    if entities:
        return entities[0]
    for event in events:
        summary = str(event.get("summary") or "").strip()
        if summary:
            return summary[:40]
    return f"{source_kind}:{source_id}"[:80]


def _derive_short_summary(events: list[dict[str, Any]], previous: ClusterSummaryRecord | None) -> str:
    summaries = [str(event.get("summary") or "").strip() for event in events if str(event.get("summary") or "").strip()]
    delta = [
        str(event.get("summary") or "").strip()
        for event in events
        if str(event.get("window_role") or "") == "delta_new_evidence" and str(event.get("summary") or "").strip()
    ]
    if previous and delta:
        return "；".join(_dedupe_preserve_order([previous.short_summary, *delta[-3:]]))
    if summaries:
        selected = summaries[-4:] if previous else summaries[:4]
        return "；".join(_dedupe_preserve_order(selected))
    return previous.short_summary if previous else ""


def _derive_open_slots(
    current_state: str,
    events: list[dict[str, Any]],
    previous: ClusterSummaryRecord | None,
) -> list[str]:
    slots = list(previous.open_slots if previous else ())
    text = " ".join(str(event.get("summary") or "") for event in events[-3:]).lower()
    if current_state == "completed":
        slots.extend(["post_completion_comment", "correction"])
    elif current_state == "in_progress":
        slots.extend(["progress_update", "blocked_point", "completion", "correction"])
    elif any(marker in text for marker in ("吗", "?", "？", "how", "what", "why")):
        slots.extend(["answer", "follow_up", "correction"])
    else:
        slots.extend(["new_evidence", "correction"])
    return _dedupe_preserve_order(slots)


def _derive_boundary_notes(packet: dict[str, Any], previous: ClusterSummaryRecord | None) -> list[str]:
    notes = list(previous.boundary_notes if previous else ())
    if packet.get("previous_cluster_summary_stale_prior"):
        notes.append("previous_cluster_summary_stale_prior")
    policy = packet.get("window_policy")
    if isinstance(policy, dict) and policy.get("activation_score_is_relevance_not_truth"):
        notes.append("activation_score_is_relevance_not_truth")
    return _dedupe_preserve_order(notes)


def _write_summary_cache(
    con: sqlite3.Connection,
    summary: ClusterSummaryRecord,
    *,
    input_hash: str,
    model: str,
    now_ms: int,
) -> None:
    packet_id = summary.summary_id
    con.execute(
        """
        UPDATE MemorySummaryCache
        SET status='stale', updated_at_ms=?
        WHERE packet_id=? AND summary_id<>? AND status='ready'
        """,
        (now_ms, packet_id, summary.summary_id),
    )
    con.execute(
        """
        INSERT INTO MemorySummaryCache (
            summary_id, packet_id, input_hash, model, status, title, short_summary,
            digest_json, salient_entities_json, cluster_summary_json, created_at_ms,
            updated_at_ms, error_json
        ) VALUES (?, ?, ?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, '{}')
        ON CONFLICT(summary_id) DO UPDATE SET
            packet_id=excluded.packet_id,
            input_hash=excluded.input_hash,
            model=excluded.model,
            status='ready',
            title=excluded.title,
            short_summary=excluded.short_summary,
            digest_json=excluded.digest_json,
            salient_entities_json=excluded.salient_entities_json,
            cluster_summary_json=excluded.cluster_summary_json,
            updated_at_ms=excluded.updated_at_ms,
            error_json='{}'
        """,
        (
            summary.summary_id,
            packet_id,
            input_hash,
            model,
            summary.title,
            summary.short_summary,
            _json(list(summary.confirmed_claims)),
            _json(list(summary.core_entities)),
            cluster_summary_to_json(summary),
            now_ms,
            now_ms,
        ),
    )


def _upsert_summary_input(
    con: sqlite3.Connection,
    *,
    packet_id: str,
    packet_type: str,
    source_kind: str,
    source_id: str,
    source_revision: int,
    input_hash: str,
    priority: int,
    confidence_tier: str,
    packet: dict[str, Any],
    now_ms: int,
) -> None:
    con.execute(
        """
        INSERT INTO MemorySummaryInputs (
            packet_id, packet_type, source_kind, source_id, source_revision,
            input_hash, priority, confidence_tier, status, created_at_ms,
            updated_at_ms, packet_json, invalidation_json, provenance_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, '{}', ?)
        ON CONFLICT(packet_id) DO UPDATE SET
            packet_type=excluded.packet_type,
            source_kind=excluded.source_kind,
            source_id=excluded.source_id,
            source_revision=excluded.source_revision,
            input_hash=excluded.input_hash,
            priority=excluded.priority,
            confidence_tier=excluded.confidence_tier,
            status='active',
            updated_at_ms=excluded.updated_at_ms,
            packet_json=excluded.packet_json,
            provenance_json=excluded.provenance_json
        """,
        (
            packet_id,
            packet_type,
            source_kind,
            source_id,
            int(source_revision),
            input_hash,
            int(priority),
            confidence_tier,
            now_ms,
            now_ms,
            _json(packet),
            _json({"llm_used": False, "generator": "memory.summary_worker"}),
        ),
    )


def _replace_summary_input_links(
    con: sqlite3.Connection,
    packet_id: str,
    events: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> None:
    con.execute("DELETE FROM MemorySummaryInputEvents WHERE packet_id=?", (packet_id,))
    con.executemany(
        """
        INSERT OR REPLACE INTO MemorySummaryInputEvents (packet_id, event_id, rank, role, status)
        VALUES (?, ?, ?, ?, 'active')
        """,
        [
            (
                packet_id,
                int(event["event_id"]),
                index,
                str(event.get("window_role") or "cluster_member"),
            )
            for index, event in enumerate(events, start=1)
            if int(event.get("event_id") or 0) > 0
        ],
    )
    con.execute("DELETE FROM MemorySummaryInputRelations WHERE packet_id=?", (packet_id,))
    con.executemany(
        """
        INSERT OR REPLACE INTO MemorySummaryInputRelations (
            packet_id, relation_id, source_event_id, target_event_id, relation_type, status
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                packet_id,
                str(rel.get("relation_id") or ""),
                _int(rel.get("source_event_id"), 0),
                _int(rel.get("target_event_id"), 0),
                str(rel.get("relation_type") or ""),
                str(rel.get("status") or ""),
            )
            for rel in relations
            if str(rel.get("relation_id") or "")
        ],
    )


def _load_cluster_event_window(
    con: sqlite3.Connection,
    cluster_id: str,
    *,
    max_events: int,
) -> list[dict[str, Any]]:
    rows = list(
        con.execute(
            """
            SELECT e.event_id, e.summary, e.event_type_norm, e.status, e.confidence,
                   e.occurred_at, e.created_at, e.last_seen_at, e.last_accessed,
                   e.occurrences, e.access_count, m.rank
            FROM MemoryClusterMembers m
            JOIN MemoryEvents e ON e.event_id=m.event_id
            WHERE m.cluster_id=? AND m.status='active' AND e.is_deleted=0
            ORDER BY m.rank ASC, e.occurred_at ASC, e.event_id ASC
            LIMIT ?
            """,
            (cluster_id, max(1, int(max_events or 1))),
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
            "window_role": "cluster_member",
            "roles": roles_by_event.get(int(row["event_id"]), []),
        }
        for row in rows
    ]


def _load_cluster_relation_briefs(con: sqlite3.Connection, cluster_id: str) -> list[dict[str, Any]]:
    if not _table_exists(con, "MemoryClusterRelations"):
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
            FROM MemoryClusterRelations
            WHERE cluster_id=? AND status IN ('active', 'weak', 'rejected')
            ORDER BY updated_at_ms ASC, relation_id ASC
            """,
            (cluster_id,),
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


def _is_informative_entity(entity: object) -> bool:
    text = str(entity or "").strip()
    if not text or text.lower() in {"self", "person:self", "bot", "model"}:
        return False
    return not text.startswith(("Group:", "Platform:", "Location:", "Time:", "Session:", "group:", "platform:", "location:", "time:", "session:"))


def _entity_key(entity: object) -> str:
    return re.sub(r"\s+", "", str(entity or "").strip().lower())


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
    "CLUSTER_SUMMARY_MODEL",
    "process_active_summary_inputs",
    "queue_missing_cluster_summary_inputs",
    "run_summary_refresh_worker",
    "summary_id_for_source",
]
