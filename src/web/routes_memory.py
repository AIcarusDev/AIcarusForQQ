"""Memory graph routes.

The graph is derived only from Memory archive output. It does not
pre-create account, group, profile, or session nodes from legacy tables.
"""

from __future__ import annotations

import json
import logging
import re
import time
import asyncio
from pathlib import Path
from typing import Any

import aiosqlite
from quart import Blueprint, jsonify, render_template, request

from database import DB_PATH
from memory.repo.events import ensure_schema
from memory.tokenizer import build_fts_query

logger = logging.getLogger("AICQ.web.memory")

memory_bp = Blueprint("memory", __name__)

_GRAPH_CHUNK_DEFAULT = 80
_GRAPH_CHUNK_MAX = 250
_LEGACY_GRAPH_DEFAULT = 300
_SEARCH_LIMIT_DEFAULT = 60
_SEARCH_LIMIT_MAX = 120
_GRAPH_SNAPSHOT_PATH = Path(DB_PATH).resolve().parent / "memory_graph_snapshot.json"
_GRAPH_SNAPSHOT_MAX_BYTES = 32 * 1024 * 1024


def _read_graph_snapshot_file() -> dict[str, Any] | None:
    if not _GRAPH_SNAPSHOT_PATH.exists():
        return None
    if _GRAPH_SNAPSHOT_PATH.stat().st_size > _GRAPH_SNAPSHOT_MAX_BYTES:
        raise ValueError("snapshot too large")
    return json.loads(_GRAPH_SNAPSHOT_PATH.read_text(encoding="utf-8"))


def _write_graph_snapshot_file(data: dict[str, Any]) -> int:
    data["savedAt"] = int(time.time() * 1000)
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    byte_len = len(serialized.encode("utf-8"))
    if byte_len > _GRAPH_SNAPSHOT_MAX_BYTES:
        raise ValueError("snapshot too large")
    _GRAPH_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _GRAPH_SNAPSHOT_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(serialized, encoding="utf-8")
    tmp_path.replace(_GRAPH_SNAPSHOT_PATH)
    return byte_len


@memory_bp.route("/memory")
async def memory_page():
    return await render_template("memory.html", active_page="memory")


@memory_bp.route("/memory/3d")
async def memory_3d_page():
    """Experimental 3D memory graph workspace."""

    return await render_template("memory_3d.html", active_page="memory")


def _arg_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = request.args.get(name, "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _shorten(text: str, limit: int) -> str:
    text = str(text or "")
    return text[:limit] + "..." if len(text) > limit else text


def _edge_id(src: str, dst: str, label: str) -> str:
    return f"{src}::{dst}::{label}"


def _event_select_sql(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    from_sql = f"MemoryEvents {alias}" if alias else "MemoryEvents"
    return f"""
        SELECT {prefix}event_id AS event_id, {prefix}summary AS summary,
               {prefix}event_type AS event_type,
               {prefix}event_type_norm AS event_type_norm,
               {prefix}status AS status, {prefix}is_negated AS is_negated,
               {prefix}confidence AS confidence, {prefix}occurred_at AS occurred_at,
               {prefix}source AS source, {prefix}reason AS reason,
               {prefix}conv_type AS conv_type, {prefix}conv_id AS conv_id,
               {prefix}conv_name AS conv_name, {prefix}occurrences AS occurrences,
               {prefix}raw_event_json AS raw_event_json,
               {prefix}created_at AS created_at,
               {prefix}last_seen_at AS last_seen_at,
               {prefix}last_accessed AS last_accessed
FROM {from_sql}
WHERE {prefix}is_deleted=0
"""


def _event_graph_degree_sql(alias: str = "") -> str:
    prefix = f"{alias}." if alias else "MemoryEvents."
    return f"""
(
    SELECT COUNT(*)
    FROM MemoryParticipants p
    WHERE p.event_id={prefix}event_id
      AND (
        COALESCE(p.entity, '') <> ''
        OR COALESCE(p.value_text, '') <> ''
      )
)
+ (
    SELECT COUNT(*)
    FROM MemoryRelations r
    WHERE r.src_event_id={prefix}event_id OR r.dst_event_id={prefix}event_id
)
"""


async def _fetch_roles(
    db: aiosqlite.Connection, event_ids: list[int]
) -> dict[int, list[dict[str, Any]]]:
    roles_by_event: dict[int, list[dict[str, Any]]] = {}
    if not event_ids:
        return roles_by_event
    placeholders = ",".join("?" * len(event_ids))
    async with db.execute(
        f"""
        SELECT participant_id, event_id, role, entity, value_text, raw_participant_json
        FROM MemoryParticipants
        WHERE event_id IN ({placeholders})
        ORDER BY event_id, participant_id
        """,
        event_ids,
    ) as cur:
        async for row in cur:
            roles_by_event.setdefault(int(row["event_id"]), []).append(dict(row))
    return roles_by_event


async def _fetch_sources(
    db: aiosqlite.Connection, event_ids: list[int]
) -> dict[int, list[dict[str, Any]]]:
    sources_by_event: dict[int, list[dict[str, Any]]] = {}
    if not event_ids:
        return sources_by_event
    placeholders = ",".join("?" * len(event_ids))
    async with db.execute(
        f"""
        SELECT event_source_id, event_id, source_kind, source_uid, source_id,
               prompt_source_id, source_seq, source_timestamp, created_at
        FROM MemoryEventSources
        WHERE event_id IN ({placeholders})
        ORDER BY event_id, source_seq, source_id
        """,
        event_ids,
    ) as cur:
        async for row in cur:
            sources_by_event.setdefault(int(row["event_id"]), []).append(dict(row))
    return sources_by_event


async def _build_graph_payload(
    db: aiosqlite.Connection,
    events: list[dict[str, Any]],
    *,
    include_relations: bool = True,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()

    def add_node(node: dict[str, Any]) -> None:
        node_id = str(node["id"])
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        nodes.append(node)

    def add_edge(src: str, dst: str, label: str, title: str = "") -> None:
        src_s, dst_s, label_s = str(src), str(dst), str(label)
        key = (src_s, dst_s, label_s)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edge = {
            "id": _edge_id(src_s, dst_s, label_s),
            "from": src_s,
            "to": dst_s,
            "label": label_s,
        }
        if title:
            edge["title"] = title
        edges.append(edge)

    event_ids = [int(event["event_id"]) for event in events]
    roles_by_event = await _fetch_roles(db, event_ids)
    sources_by_event = await _fetch_sources(db, event_ids)

    for event in events:
        event_id = int(event["event_id"])
        event_node = f"ev-{event_id}"
        summary = str(event.get("summary") or "(empty)")
        predicate = str(event.get("event_type") or "event")
        predicate_norm = str(event.get("event_type_norm") or predicate)
        status = str(event.get("status") or "actual")
        roles = roles_by_event.get(event_id, [])
        role_text = " ".join(
            str(role.get("entity") or role.get("value_text") or role.get("role") or "")
            for role in roles
        )
        source_rows = sources_by_event.get(event_id, [])
        source_text = " ".join(
            str(row.get("source_id") or row.get("source_uid") or row.get("prompt_source_id") or "")
            for row in source_rows
        )

        add_node(
            {
                "id": event_node,
                "label": f"{predicate}\n#{event_id} {_shorten(summary, 24)}",
                "group": "event",
                "title": summary,
                "searchText": " ".join(
                    [
                        summary,
                        predicate,
                        predicate_norm,
                        str(event.get("reason") or ""),
                        str(event.get("source") or ""),
                        str(event.get("conv_name") or ""),
                        str(event.get("conv_id") or ""),
                        role_text,
                        source_text,
                    ]
                ),
                "extra": {
                    "event_id": event_id,
                    "summary": summary,
                    "event_type": predicate,
                    "event_type_norm": predicate_norm,
                    "status": status,
                    "is_negated": bool(event.get("is_negated")),
                    "confidence": float(event.get("confidence") or 0.0),
                    "occurred_at": int(event.get("occurred_at") or 0),
                    "created_at": int(event.get("created_at") or 0),
                    "last_seen_at": int(event.get("last_seen_at") or 0),
                    "last_accessed": int(event.get("last_accessed") or 0),
                    "source": event.get("source") or "",
                    "reason": event.get("reason") or "",
                    "conv_type": event.get("conv_type") or "",
                    "conv_id": event.get("conv_id") or "",
                    "conv_name": event.get("conv_name") or "",
                    "occurrences": int(event.get("occurrences") or 1),
                    "roles": roles,
                    "source_ids": [
                        str(row.get("source_id") or row.get("source_uid") or "")
                        for row in source_rows
                        if row.get("source_id") or row.get("source_uid")
                    ],
                    "sources": source_rows,
                },
            }
        )

        predicate_node = f"pred-{predicate_norm}"
        add_node(
            {
                "id": predicate_node,
                "label": predicate_norm,
                "group": "predicate",
                "title": f"Predicate: {predicate_norm}",
                "extra": {"event_type_norm": predicate_norm},
            }
        )
        add_edge(event_node, predicate_node, "predicate")

        for role in roles:
            role_name = str(role.get("role") or "role")
            entity = str(role.get("entity") or "").strip()
            value_text = str(role.get("value_text") or "").strip()
            participant_id = int(role.get("participant_id") or 0)
            if entity:
                participant_node = f"entity-{entity}"
                add_node(
                    {
                        "id": participant_node,
                        "label": entity[-36:] if len(entity) > 36 else entity,
                        "group": "participant",
                        "title": entity,
                        "searchText": f"{entity} {role_name}",
                        "extra": {"entity": entity, "role": role_name},
                    }
                )
                add_edge(participant_node, event_node, role_name)
            if value_text:
                value_node = f"value-{participant_id or f'{event_id}-{role_name}-{value_text[:48]}'}"
                add_node(
                    {
                        "id": value_node,
                        "label": _shorten(value_text, 38),
                        "group": "value",
                        "title": value_text,
                        "searchText": f"{value_text} {role_name}",
                        "extra": {
                            "event_id": event_id,
                            "role": role_name,
                            "value_text": value_text,
                        },
                    }
                )
                add_edge(value_node, event_node, role_name)

    if include_relations and event_ids:
        placeholders = ",".join("?" * len(event_ids))
        async with db.execute(
            f"""
            SELECT src_event_id, dst_event_id, relation_type, reason
            FROM MemoryRelations
            WHERE src_event_id IN ({placeholders}) OR dst_event_id IN ({placeholders})
            """,
            [*event_ids, *event_ids],
        ) as cur:
            async for row in cur:
                src = f"ev-{int(row['src_event_id'])}"
                dst = f"ev-{int(row['dst_event_id'])}"
                add_edge(
                    src,
                    dst,
                    str(row["relation_type"] or "relation"),
                    str(row["reason"] or ""),
                )

    return {"nodes": nodes, "edges": edges, "event_ids": event_ids}


async def _graph_chunk(limit_default: int) -> Any:
    offset = _arg_int("offset", 0, 0, 10_000_000)
    limit = _arg_int("limit", limit_default, 1, _GRAPH_CHUNK_MAX)
    after_event_id = _arg_int("after_event_id", 0, 0, 10_000_000_000)

    try:
        await ensure_schema()
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT COUNT(*) AS n FROM MemoryEvents WHERE is_deleted=0"
            ) as cur:
                total_events = int((await cur.fetchone())["n"] or 0)

            if after_event_id:
                async with db.execute(
                    "SELECT COUNT(*) AS n FROM MemoryEvents WHERE is_deleted=0 AND event_id > ?",
                    [after_event_id],
                ) as cur:
                    filtered_events = int((await cur.fetchone())["n"] or 0)
                event_sql = f"{_event_select_sql()} AND event_id > ? ORDER BY event_id ASC LIMIT ? OFFSET ?"
                event_params = [after_event_id, limit, offset]
            else:
                filtered_events = total_events
                event_sql = (
                    f"{_event_select_sql()} "
                    f"ORDER BY {_event_graph_degree_sql()} DESC, occurred_at DESC, event_id DESC LIMIT ? OFFSET ?"
                )
                event_params = [limit, offset]

            async with db.execute(event_sql, event_params) as cur:
                events = [dict(row) for row in await cur.fetchall()]

            payload = await _build_graph_payload(db, events)
            next_offset = offset + len(events)
            payload.update(
                {
                    "offset": offset,
                    "limit": limit,
                    "next_offset": next_offset,
                    "has_more": next_offset < filtered_events,
                    "total_events": total_events,
                    "total_new_events": filtered_events,
                    "after_event_id": after_event_id,
                    "batch_events": len(events),
                }
            )
            return jsonify(payload)
    except Exception as exc:
        logger.warning("memory graph chunk query failed: %s", exc, exc_info=True)
        return jsonify({"nodes": [], "edges": [], "error": str(exc)})


@memory_bp.route("/memory/graph")
async def memory_graph():
    """Compatibility route for older WebUI clients."""

    return await _graph_chunk(_LEGACY_GRAPH_DEFAULT)


@memory_bp.route("/memory/graph/chunk")
async def memory_graph_chunk():
    """Return one progressive Memory memory graph chunk."""

    return await _graph_chunk(_GRAPH_CHUNK_DEFAULT)


@memory_bp.route("/memory/graph/snapshot", methods=["GET", "PUT"])
async def memory_graph_snapshot():
    """Persist graph layout snapshot on the server for cross-device loading."""

    if request.method == "GET":
        try:
            data = await asyncio.to_thread(_read_graph_snapshot_file)
            if data is None:
                return jsonify({"exists": False, "snapshot": None})
            return jsonify({"exists": True, "snapshot": data})
        except Exception as exc:
            logger.warning("memory graph snapshot read failed: %s", exc, exc_info=True)
            return jsonify({"exists": False, "snapshot": None, "error": str(exc)})

    try:
        data = await request.get_json()
        if not isinstance(data, dict):
            return jsonify({"ok": False, "error": "snapshot must be an object"}), 400
        if int(data.get("version") or 0) != 1:
            return jsonify({"ok": False, "error": "unsupported snapshot version"}), 400
        if not isinstance(data.get("nodes"), list) or not isinstance(data.get("edges"), list):
            return jsonify({"ok": False, "error": "snapshot nodes/edges must be arrays"}), 400
        if not isinstance(data.get("positions"), dict):
            return jsonify({"ok": False, "error": "snapshot positions must be an object"}), 400

        try:
            byte_len = await asyncio.to_thread(_write_graph_snapshot_file, data)
        except ValueError:
            return jsonify({"ok": False, "error": "snapshot too large"}), 413
        return jsonify({"ok": True, "path": str(_GRAPH_SNAPSHOT_PATH), "bytes": byte_len})
    except Exception as exc:
        logger.warning("memory graph snapshot write failed: %s", exc, exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 500


@memory_bp.route("/memory/graph/status", methods=["GET", "POST"])
async def memory_graph_status():
    """Return lightweight event status used to gray stale snapshot nodes."""

    try:
        requested_ids: list[int] = []
        if request.method == "POST":
            body = await request.get_json()
            raw_ids = body.get("event_ids", []) if isinstance(body, dict) else []
            if isinstance(raw_ids, list):
                for raw_id in raw_ids:
                    try:
                        event_id = int(raw_id)
                    except (TypeError, ValueError):
                        continue
                    if event_id > 0:
                        requested_ids.append(event_id)
            requested_ids = sorted(set(requested_ids))
            if not requested_ids:
                return jsonify({"events": {}, "count": 0})

        await ensure_schema()
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            if requested_ids:
                placeholders = ",".join("?" * len(requested_ids))
                sql = f"""
                SELECT event_id, is_deleted, summary, event_type, event_type_norm,
                       status, confidence, occurred_at, created_at, last_seen_at,
                       occurrences
                FROM MemoryEvents
                WHERE event_id IN ({placeholders})
                """
                params: list[Any] = requested_ids
            else:
                sql = """
                SELECT event_id, is_deleted, summary, event_type, event_type_norm,
                       status, confidence, occurred_at, created_at, last_seen_at,
                       occurrences
                FROM MemoryEvents
                """
                params = []
            async with db.execute(sql, params) as cur:
                rows = await cur.fetchall()
        events: dict[str, dict[str, Any]] = {}
        for row in rows:
            event_id = int(row["event_id"])
            version = "|".join(
                [
                    str(row["summary"] or ""),
                    str(row["event_type"] or ""),
                    str(row["event_type_norm"] or ""),
                    str(row["status"] or ""),
                    str(row["confidence"] or ""),
                    str(row["occurred_at"] or 0),
                    str(row["created_at"] or 0),
                    str(row["last_seen_at"] or 0),
                    str(row["occurrences"] or 0),
                    str(row["is_deleted"] or 0),
                ]
            )
            events[str(event_id)] = {
                "is_deleted": bool(row["is_deleted"]),
                "version": version,
            }
        return jsonify({"events": events, "count": len(events)})
    except Exception as exc:
        logger.warning("memory graph status query failed: %s", exc, exc_info=True)
        return jsonify({"events": {}, "count": 0, "error": str(exc)})


@memory_bp.route("/memory/graph/meta")
async def memory_graph_meta():
    """Return graph-wide counters before the progressive graph load starts."""

    try:
        await ensure_schema()
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row

            async def scalar(sql: str, params: list[Any] | None = None) -> int:
                async with db.execute(sql, params or []) as cur:
                    row = await cur.fetchone()
                    return int((row[0] if row else 0) or 0)

            events = await scalar("SELECT COUNT(*) FROM MemoryEvents WHERE is_deleted=0")
            predicates = await scalar(
                """
                SELECT COUNT(DISTINCT event_type_norm)
                FROM MemoryEvents
                WHERE is_deleted=0 AND event_type_norm<>''
                """
            )
            participants = await scalar(
                """
                SELECT COUNT(DISTINCT p.entity)
                FROM MemoryParticipants p
                JOIN MemoryEvents e ON e.event_id=p.event_id
                WHERE e.is_deleted=0 AND p.entity IS NOT NULL AND p.entity<>''
                """
            )
            values = await scalar(
                """
                SELECT COUNT(*)
                FROM MemoryParticipants p
                JOIN MemoryEvents e ON e.event_id=p.event_id
                WHERE e.is_deleted=0 AND p.value_text IS NOT NULL AND p.value_text<>''
                """
            )
            relations = await scalar(
                """
                SELECT COUNT(*)
                FROM MemoryRelations r
                JOIN MemoryEvents s ON s.event_id=r.src_event_id
                JOIN MemoryEvents d ON d.event_id=r.dst_event_id
                WHERE s.is_deleted=0 AND d.is_deleted=0
                """
            )
            sources = await scalar(
                """
                SELECT COUNT(*)
                FROM MemoryEventSources s
                JOIN MemoryEvents e ON e.event_id=s.event_id
                WHERE e.is_deleted=0
                """
            )
            cognition_sources = await scalar("SELECT COUNT(*) FROM CognitionSources")
            async with db.execute(
                """
                SELECT MIN(occurred_at) AS min_ts, MAX(occurred_at) AS max_ts
                FROM MemoryEvents
                WHERE is_deleted=0
                """
            ) as cur:
                span_row = await cur.fetchone()
            total_nodes = events + predicates + participants + values
            return jsonify(
                {
                    "events": events,
                    "predicates": predicates,
                    "participants": participants,
                    "values": values,
                    "relations": relations,
                    "sources": sources,
                    "cognition_sources": cognition_sources,
                    "total_nodes": total_nodes,
                    "min_occurred_at": int((span_row["min_ts"] if span_row else 0) or 0),
                    "max_occurred_at": int((span_row["max_ts"] if span_row else 0) or 0),
                }
            )
    except Exception as exc:
        logger.warning("memory graph meta query failed: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)})


def _split_terms(query: str) -> list[str]:
    query = str(query or "").strip()
    if not query:
        return []
    parts = [p.strip() for p in re.split(r"\s+", query) if p.strip()]
    return parts[:8] if parts else [query[:200]]


def _like_pattern(term: str) -> str:
    escaped = (
        str(term or "")
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )
    return f"%{escaped}%"


def _search_where(terms: list[str]) -> tuple[str, list[Any]]:
    field_sql = """
        e.summary LIKE ? ESCAPE '\\'
        OR e.summary_tok LIKE ? ESCAPE '\\'
        OR e.event_type LIKE ? ESCAPE '\\'
        OR e.event_type_norm LIKE ? ESCAPE '\\'
        OR e.status LIKE ? ESCAPE '\\'
        OR e.source LIKE ? ESCAPE '\\'
        OR e.reason LIKE ? ESCAPE '\\'
        OR e.conv_type LIKE ? ESCAPE '\\'
        OR e.conv_id LIKE ? ESCAPE '\\'
        OR e.conv_name LIKE ? ESCAPE '\\'
        OR e.raw_event_json LIKE ? ESCAPE '\\'
        OR EXISTS (
            SELECT 1 FROM MemoryParticipants p
            WHERE p.event_id=e.event_id AND (
                p.role LIKE ? ESCAPE '\\'
                OR COALESCE(p.entity, '') LIKE ? ESCAPE '\\'
                OR COALESCE(p.value_text, '') LIKE ? ESCAPE '\\'
                OR COALESCE(p.raw_participant_json, '') LIKE ? ESCAPE '\\'
            )
        )
        OR EXISTS (
            SELECT 1 FROM MemoryEventSources s
            WHERE s.event_id=e.event_id AND (
                s.source_kind LIKE ? ESCAPE '\\'
                OR s.source_id LIKE ? ESCAPE '\\'
                OR COALESCE(s.source_uid, '') LIKE ? ESCAPE '\\'
                OR COALESCE(s.prompt_source_id, '') LIKE ? ESCAPE '\\'
                OR COALESCE(s.source_timestamp, '') LIKE ? ESCAPE '\\'
            )
        )
    """
    clauses: list[str] = []
    params: list[Any] = []
    for term in terms:
        pattern = _like_pattern(term)
        clauses.append(f"({field_sql})")
        params.extend([pattern] * 20)
    return " AND ".join(clauses), params


async def _search_like(
    db: aiosqlite.Connection, terms: list[str], limit: int
) -> list[dict[str, Any]]:
    where_sql, params = _search_where(terms)
    async with db.execute(
        f"""
        {_event_select_sql("e")}
          AND {where_sql}
        ORDER BY occurred_at DESC, event_id DESC
        LIMIT ?
        """,
        [*params, limit],
    ) as cur:
        return [dict(row) for row in await cur.fetchall()]


async def _search_fts(
    db: aiosqlite.Connection, query: str, limit: int
) -> list[dict[str, Any]]:
    fts_q = build_fts_query(query)
    if not fts_q:
        return []
    try:
        async with db.execute(
            """
            SELECT e.*, bm25(MemorySearch) AS rank
            FROM MemorySearch
            JOIN MemoryEvents e ON e.event_id=MemorySearch.rowid
            WHERE MemorySearch MATCH ? AND e.is_deleted=0
            ORDER BY rank
            LIMIT ?
            """,
            [fts_q, limit],
        ) as cur:
            return [dict(row) for row in await cur.fetchall()]
    except Exception:
        logger.debug("[memory_graph] FTS search failed: %r", fts_q, exc_info=True)
        return []


def _contains_any(text: Any, terms: list[str]) -> bool:
    haystack = str(text or "").casefold()
    return any(term.casefold() in haystack for term in terms)


def _snippet(text: Any, terms: list[str], radius: int = 46) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    folded = raw.casefold()
    pos = -1
    matched = ""
    for term in terms:
        pos = folded.find(term.casefold())
        if pos >= 0:
            matched = term
            break
    if pos < 0:
        return _shorten(raw, radius * 2)
    start = max(0, pos - radius)
    end = min(len(raw), pos + len(matched) + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(raw) else ""
    return f"{prefix}{raw[start:end]}{suffix}"


def _describe_hits(
    event: dict[str, Any],
    roles: list[dict[str, Any]],
    terms: list[str],
    *,
    fts_hit: bool,
    sources: list[dict[str, Any]] | None = None,
) -> tuple[list[str], str, float]:
    hit_fields: list[str] = []
    score = 0.0
    snippet = ""

    field_weights = [
        ("摘要", event.get("summary"), 10.0),
        ("谓词", event.get("event_type"), 7.0),
        ("谓词归一", event.get("event_type_norm"), 6.0),
        ("归档原因", event.get("reason"), 4.0),
        ("来源", event.get("source"), 3.0),
        ("会话", " ".join([str(event.get("conv_type") or ""), str(event.get("conv_id") or ""), str(event.get("conv_name") or "")]), 3.0),
        ("原始事件", event.get("raw_event_json"), 2.0),
    ]
    for label, value, weight in field_weights:
        if _contains_any(value, terms):
            hit_fields.append(label)
            score += weight
            if not snippet:
                snippet = _snippet(value, terms)

    for role in roles:
        role_blob = " ".join(
            [
                str(role.get("role") or ""),
                str(role.get("entity") or ""),
                str(role.get("value_text") or ""),
                str(role.get("raw_participant_json") or ""),
            ]
        )
        if _contains_any(role_blob, terms):
            hit_fields.append(f"角色:{role.get('role') or 'role'}")
            score += 6.0
            if not snippet:
                snippet = _snippet(role_blob, terms)

    source_blob = " ".join(
        " ".join(
            [
                str(source.get("source_kind") or ""),
                str(source.get("source_id") or ""),
                str(source.get("source_uid") or ""),
                str(source.get("prompt_source_id") or ""),
                str(source.get("source_timestamp") or ""),
            ]
        )
        for source in (sources or [])
    )
    if _contains_any(source_blob, terms):
        hit_fields.append("来源认知")
        score += 5.0
        if not snippet:
            snippet = _snippet(source_blob, terms)

    if fts_hit:
        hit_fields.append("FTS")
        score += 4.0
    if not snippet:
        snippet = _snippet(event.get("summary"), terms)
    score += min(1.5, float(event.get("confidence") or 0) * 1.5)
    score += min(1.0, int(event.get("occurrences") or 1) * 0.08)
    return hit_fields[:8], snippet, score


@memory_bp.route("/memory/search")
async def memory_search():
    """Search real Memory data, not the currently rendered graph nodes."""

    query = str(request.args.get("q", "") or "").strip()
    limit = _arg_int("limit", _SEARCH_LIMIT_DEFAULT, 1, _SEARCH_LIMIT_MAX)
    terms = _split_terms(query)
    if not terms:
        return jsonify({"query": query, "found": 0, "results": [], "nodes": [], "edges": []})

    try:
        await ensure_schema()
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            like_rows = await _search_like(db, terms, limit * 2)
            fts_rows = await _search_fts(db, query, limit * 2)

            rows_by_id: dict[int, dict[str, Any]] = {}
            fts_ids = {int(row["event_id"]) for row in fts_rows}
            for row in [*like_rows, *fts_rows]:
                rows_by_id.setdefault(int(row["event_id"]), row)

            rows = list(rows_by_id.values())
            event_row_ids = [int(row["event_id"]) for row in rows]
            roles_by_event = await _fetch_roles(db, event_row_ids)
            sources_by_event = await _fetch_sources(db, event_row_ids)
            ranked: list[tuple[float, dict[str, Any]]] = []
            for row in rows:
                event_id = int(row["event_id"])
                roles = roles_by_event.get(event_id, [])
                sources = sources_by_event.get(event_id, [])
                hit_fields, snippet, score = _describe_hits(
                    row, roles, terms, fts_hit=event_id in fts_ids, sources=sources
                )
                result = {
                    "event_id": event_id,
                    "node_id": f"ev-{event_id}",
                    "summary": row.get("summary") or "",
                    "event_type": row.get("event_type") or "",
                    "event_type_norm": row.get("event_type_norm") or "",
                    "status": row.get("status") or "",
                    "confidence": float(row.get("confidence") or 0.0),
                    "occurred_at": int(row.get("occurred_at") or 0),
                    "conv_type": row.get("conv_type") or "",
                    "conv_id": row.get("conv_id") or "",
                    "conv_name": row.get("conv_name") or "",
                    "occurrences": int(row.get("occurrences") or 1),
                    "hit_fields": hit_fields,
                    "snippet": snippet,
                    "score": round(score, 3),
                }
                ranked.append((score, result))

            ranked.sort(
                key=lambda item: (
                    item[0],
                    int(item[1].get("occurred_at") or 0),
                    int(item[1].get("event_id") or 0),
                ),
                reverse=True,
            )
            results = [item[1] for item in ranked[:limit]]
            event_ids = [int(result["event_id"]) for result in results]
            events_for_graph = [rows_by_id[event_id] for event_id in event_ids if event_id in rows_by_id]
            graph_payload = await _build_graph_payload(db, events_for_graph)

            return jsonify(
                {
                    "query": query,
                    "found": len(results),
                    "results": results,
                    "nodes": graph_payload["nodes"],
                    "edges": graph_payload["edges"],
                    "searched": {
                        "like_candidates": len(like_rows),
                        "fts_candidates": len(fts_rows),
                    },
                }
            )
    except Exception as exc:
        logger.warning("memory search failed: %s", exc, exc_info=True)
        return jsonify({"query": query, "found": 0, "results": [], "nodes": [], "edges": [], "error": str(exc)})
