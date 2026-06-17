"""Memory V2 graph routes.

The graph is derived only from Memory V2 archive output. It does not
pre-create account, group, profile, or session nodes from legacy tables.
"""

from __future__ import annotations

import logging

import aiosqlite
from quart import Blueprint, jsonify, render_template

from database import DB_PATH
from memory.repo.events_v2 import ensure_schema

logger = logging.getLogger("AICQ.web.memory")

memory_bp = Blueprint("memory", __name__)


@memory_bp.route("/memory")
async def memory_page():
    return await render_template("memory.html", active_page="memory")


@memory_bp.route("/memory/graph")
async def memory_graph():
    """Return V2 memory graph nodes/edges for the WebUI."""

    nodes: list[dict] = []
    edges: list[dict] = []
    seen_nodes: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()

    def add_node(node: dict) -> None:
        node_id = str(node["id"])
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        nodes.append(node)

    def add_edge(src: str, dst: str, label: str) -> None:
        key = (str(src), str(dst), str(label))
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"from": key[0], "to": key[1], "label": key[2]})

    try:
        await ensure_schema()
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row

            async with db.execute(
                """
                SELECT event_id, summary, event_type, event_type_norm, status,
                       is_negated, occurred_at, source, reason, conv_type,
                       conv_id, conv_name, occurrences
                FROM MemoryV2Events
                WHERE is_deleted=0
                ORDER BY occurred_at DESC
                LIMIT 300
                """
            ) as cur:
                events = [dict(row) for row in await cur.fetchall()]

            event_ids = [int(event["event_id"]) for event in events]
            roles_by_event: dict[int, list[dict]] = {}
            if event_ids:
                placeholders = ",".join("?" * len(event_ids))
                async with db.execute(
                    f"""
                    SELECT event_id, role, entity, value_text
                    FROM MemoryV2Participants
                    WHERE event_id IN ({placeholders})
                    """,
                    event_ids,
                ) as cur:
                    async for row in cur:
                        roles_by_event.setdefault(int(row["event_id"]), []).append(dict(row))

            for event in events:
                event_id = int(event["event_id"])
                event_node = f"ev-{event_id}"
                summary = str(event.get("summary") or "(empty)")
                short_summary = summary[:36] + "..." if len(summary) > 36 else summary
                predicate = str(event.get("event_type") or "event")
                status = str(event.get("status") or "actual")
                add_node(
                    {
                        "id": event_node,
                        "label": f"{predicate}\n{short_summary}",
                        "group": "event",
                        "title": summary,
                        "extra": {
                            "event_id": event_id,
                            "summary": summary,
                            "event_type": predicate,
                            "event_type_norm": event.get("event_type_norm") or "",
                            "status": status,
                            "is_negated": bool(event.get("is_negated")),
                            "occurred_at": int(event.get("occurred_at") or 0),
                            "source": event.get("source") or "",
                            "reason": event.get("reason") or "",
                            "conv_type": event.get("conv_type") or "",
                            "conv_id": event.get("conv_id") or "",
                            "conv_name": event.get("conv_name") or "",
                            "occurrences": int(event.get("occurrences") or 1),
                            "roles": roles_by_event.get(event_id, []),
                        },
                    }
                )

                predicate_norm = str(event.get("event_type_norm") or predicate)
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

                for role in roles_by_event.get(event_id, []):
                    role_name = str(role.get("role") or "role")
                    entity = str(role.get("entity") or "").strip()
                    value_text = str(role.get("value_text") or "").strip()
                    if entity:
                        participant_node = f"entity-{entity}"
                        label = entity[-32:] if len(entity) > 32 else entity
                        add_node(
                            {
                                "id": participant_node,
                                "label": label,
                                "group": "participant",
                                "title": entity,
                                "extra": {"entity": entity},
                            }
                        )
                        add_edge(participant_node, event_node, role_name)
                    if value_text:
                        value_node = f"value-{event_id}-{role_name}-{value_text[:48]}"
                        label = value_text[:32] + "..." if len(value_text) > 32 else value_text
                        add_node(
                            {
                                "id": value_node,
                                "label": label,
                                "group": "value",
                                "title": value_text,
                                "extra": {"value_text": value_text},
                            }
                        )
                        add_edge(value_node, event_node, role_name)

            if event_ids:
                placeholders = ",".join("?" * len(event_ids))
                async with db.execute(
                    f"""
                    SELECT src_event_id, dst_event_id, relation_type, reason
                    FROM MemoryV2Relations
                    WHERE src_event_id IN ({placeholders}) OR dst_event_id IN ({placeholders})
                    """,
                    [*event_ids, *event_ids],
                ) as cur:
                    async for row in cur:
                        src = f"ev-{int(row['src_event_id'])}"
                        dst = f"ev-{int(row['dst_event_id'])}"
                        if src in seen_nodes and dst in seen_nodes:
                            add_edge(src, dst, str(row["relation_type"] or "relation"))

    except Exception as exc:
        logger.warning("memory_graph query failed: %s", exc, exc_info=True)
        return jsonify({"nodes": [], "edges": [], "error": str(exc)})

    return jsonify({"nodes": nodes, "edges": edges})
