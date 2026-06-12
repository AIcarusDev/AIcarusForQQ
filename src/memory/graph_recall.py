"""Graph-based reranking for event memories.

This is deliberately a lightweight in-SQLite adaptation of the older
entitySystem activation idea: event memories are first-class nodes, entities
are connector nodes, and recall prefers cheap, explainable paths while
penalising hub entities and non-actual worlds.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Any


_ROLE_COST = {
    "agent": 0.12,
    "theme": 0.16,
    "patient": 0.20,
    "recipient": 0.20,
    "attribute": 0.28,
    "instrument": 0.34,
    "location": 0.36,
    "time": 0.42,
}


def _event_node(event_id: int) -> str:
    return f"E:{int(event_id)}"


def _entity_node(entity: str) -> str:
    return f"N:{entity}"


def _context_penalty(event: dict[str, Any]) -> float:
    context_type = str(event.get("context_type") or "episodic").lower()
    modality = str(event.get("modality") or "actual").lower()
    penalty = 0.0
    if modality == "possible":
        penalty += 0.35
    elif modality == "hypothetical":
        penalty += 1.4
    if context_type == "hypothetical":
        penalty += 1.6
    return penalty


def _confidence_bonus(event: dict[str, Any]) -> float:
    try:
        confidence = float(event.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return max(0.0, min(0.3, confidence * 0.3))


def _entity_label(node: str) -> str:
    return node[2:] if node.startswith("N:") else node


def _event_label(node: str, events: dict[int, dict[str, Any]]) -> str:
    if not node.startswith("E:"):
        return node
    event_id = int(node[2:])
    summary = str(events.get(event_id, {}).get("summary") or "")
    return f"event#{event_id}: {summary}"


def _render_path(path: list[str], events: dict[int, dict[str, Any]]) -> list[str]:
    rendered: list[str] = []
    for node in path:
        if node.startswith("N:"):
            rendered.append(_entity_label(node))
        elif node.startswith("E:"):
            rendered.append(_event_label(node, events))
        else:
            rendered.append(node)
    return rendered


async def rerank_events_by_graph(
    db,
    candidates: list[dict[str, Any]],
    *,
    seed_entities: list[str],
    context_scope: str = "",
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Rerank candidate events with a small bipartite event/entity Dijkstra.

    The input candidates still come from cheap retrieval (FTS, direct entity
    hits, recent fallback). This function expands their neighbouring entities
    once, then searches paths from current seed entities to event nodes.
    """

    if not candidates:
        return []

    candidate_ids = [int(ev["event_id"]) for ev in candidates]
    candidate_id_set = set(candidate_ids)
    candidate_rank = {int(ev["event_id"]): idx for idx, ev in enumerate(candidates)}
    events: dict[int, dict[str, Any]] = {int(ev["event_id"]): dict(ev) for ev in candidates}

    placeholders = ",".join("?" * len(candidate_ids))
    async with db.execute(
        f"SELECT * FROM MemoryRoles WHERE event_id IN ({placeholders})",
        candidate_ids,
    ) as cur:
        candidate_roles = [dict(row) for row in await cur.fetchall()]

    frontier_entities = {
        str(row.get("entity") or "").strip()
        for row in candidate_roles
        if str(row.get("entity") or "").strip()
    }
    frontier_entities.update(str(ent).strip() for ent in seed_entities if str(ent).strip())
    if not frontier_entities:
        return candidates[:limit]

    ent_placeholders = ",".join("?" * len(frontier_entities))
    scope_clause = "AND (e.recall_scope='global' OR e.recall_scope=?)" if context_scope else ""
    params: list[Any] = list(frontier_entities)
    if context_scope:
        params.append(context_scope)
    params.append(max(limit * 12, 80))

    async with db.execute(
        f"""
        SELECT DISTINCT e.*
        FROM MemoryEvents e
        JOIN MemoryRoles r ON r.event_id=e.event_id
        WHERE e.is_deleted=0
          AND r.entity IN ({ent_placeholders})
          {scope_clause}
        ORDER BY e.last_seen_at DESC, e.occurred_at DESC
        LIMIT ?
        """,
        params,
    ) as cur:
        for row in await cur.fetchall():
            ev = dict(row)
            events[int(ev["event_id"])] = ev

    event_ids = sorted(events)
    event_placeholders = ",".join("?" * len(event_ids))
    async with db.execute(
        f"SELECT * FROM MemoryRoles WHERE event_id IN ({event_placeholders})",
        event_ids,
    ) as cur:
        role_rows = [dict(row) for row in await cur.fetchall()]

    roles_by_event: dict[int, list[dict[str, Any]]] = defaultdict(list)
    entity_degree: dict[str, int] = defaultdict(int)
    seen_entity_event: set[tuple[str, int]] = set()
    for role in role_rows:
        event_id = int(role["event_id"])
        roles_by_event[event_id].append(role)
        entity = str(role.get("entity") or "").strip()
        if entity and (entity, event_id) not in seen_entity_event:
            seen_entity_event.add((entity, event_id))
            entity_degree[entity] += 1

    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for event_id, roles in roles_by_event.items():
        event = events[event_id]
        event_node = _event_node(event_id)
        event_penalty = _context_penalty(event)
        confidence_bonus = _confidence_bonus(event)
        for role in roles:
            entity = str(role.get("entity") or "").strip()
            if not entity:
                continue
            role_name = str(role.get("role") or "").strip()
            hub_penalty = math.log1p(entity_degree.get(entity, 1)) * 0.12
            cost = 0.55 + _ROLE_COST.get(role_name, 0.30) + hub_penalty + event_penalty - confidence_bonus
            cost = max(0.08, cost)
            entity_node = _entity_node(entity)
            adj[entity_node].append((event_node, cost))
            if str(event.get("context_type") or "") != "hypothetical":
                adj[event_node].append((entity_node, cost + 0.08))

    dist: dict[str, float] = {}
    parent: dict[str, str | None] = {}
    pq: list[tuple[float, str]] = []
    for entity in seed_entities:
        entity = str(entity or "").strip()
        if not entity:
            continue
        node = _entity_node(entity)
        dist[node] = 0.0
        parent[node] = None
        heapq.heappush(pq, (0.0, node))

    # FTS-only candidates are still legitimate direct hits. They start with a
    # modest virtual path instead of being invisible to entity graph search.
    for event_id in candidate_ids:
        node = _event_node(event_id)
        direct_cost = 1.25 + _context_penalty(events[event_id]) - _confidence_bonus(events[event_id])
        if direct_cost < dist.get(node, float("inf")):
            dist[node] = direct_cost
            parent[node] = "query"
            heapq.heappush(pq, (direct_cost, node))

    max_cost = 5.0
    while pq:
        cost, node = heapq.heappop(pq)
        if cost > max_cost:
            continue
        if cost != dist.get(node):
            continue
        for next_node, edge_cost in adj.get(node, []):
            next_cost = cost + edge_cost
            if next_cost < dist.get(next_node, float("inf")) and next_cost <= max_cost:
                dist[next_node] = next_cost
                parent[next_node] = node
                heapq.heappush(pq, (next_cost, next_node))

    def build_node_path(node: str) -> list[str]:
        out: list[str] = []
        while node:
            out.append(node)
            prev = parent.get(node)
            if prev == "query":
                out.append("query")
                break
            if prev is None:
                break
            node = prev
        out.reverse()
        return out

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for event_id in candidate_ids:
        event = events[event_id]
        node = _event_node(event_id)
        cost = dist.get(node, 2.75 + _context_penalty(event))
        rank_bonus = max(0.0, (len(candidate_ids) - candidate_rank[event_id]) * 0.015)
        occurrence_bonus = min(0.12, max(0, int(event.get("occurrences") or 1) - 1) * 0.03)
        score = (1.0 / (1.0 + cost)) + rank_bonus + occurrence_bonus
        event["recall_score"] = round(score, 4)
        event["recall_cost"] = round(cost, 4)
        event["recall_path"] = _render_path(build_node_path(node), events)
        event["recall_reason"] = (
            "graph_path" if event["recall_path"] and event["recall_path"][0] != "query" else "direct_query"
        )
        event["roles"] = roles_by_event.get(event_id, [])
        scored.append((score, -candidate_rank[event_id], event))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [event for _, _, event in scored[:limit]]
