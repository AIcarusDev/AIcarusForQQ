"""Post-archive memory mounting workflow.

The archive model remains event-only.  This module runs after events have been
written, reloads those durable events from SQLite, compares them with summary
anchors associated with the archive recall candidates, and stages only pending
mounts.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
import hashlib
from collections import defaultdict
from dataclasses import replace
from typing import Any, Iterable

from .consolidation import (
    ALLOWED_MEMORY_MOUNT_RELATION_TYPES,
    LocalClusterMount,
    MemoryAtom,
    MemoryMount,
    ClusterSummaryRecord,
    ensure_preprocessing_schema,
    propose_memory_mounts,
    stage_memory_mount_candidates,
    cluster_summary_from_json,
    write_local_cluster_mounts,
    write_memory_mounts,
)


GUARDED_STATUSES = {"hypothetical", "conditional", "future"}
logger = logging.getLogger("AICQ.memory.mount_workflow")

MOUNT_PROPOSER_BASE_PROMPT = """\
你是长期记忆二步挂载判断器。

任务：判断一批新记忆事件是否应该挂载到候选历史记忆上。

候选历史记忆分两类：
- anchors：已有事件簇的 summary anchor。
- historical_atoms：候选历史原子节点。

只在新事件明确延续、回答、纠正、反驳、更新或完成某个候选历史记忆的未决内容时输出挂载。
不要因为同一个人物、同一天、同一个群、泛泛相似词或时间词相同就挂载。
如果没有高质量关系，返回空数组。

允许的 relation_type:
- updates_state: 新事件更新了 anchor 的状态、进度、阶段或结果。
- progresses: 新事件推进了 anchor 中的目标或任务。
- causes_or_results: 新事件是 anchor 的直接原因或结果。
- answers: 新事件回答了 anchor 的明确问题或 follow-up。
- corrects: 新事件纠正了 anchor 中的事实。
- corrects_identity: 新事件纠正了 anchor 的人物/对象身份。
- refutes: 新事件反驳了 anchor。
- same_object: 新事件确实是同一对象/主题的新证据，但不是单纯同人名。

如果新事件应该连接到已有事件簇 anchor，写入 mounts。
如果新事件应该连接到历史原子节点，写入 atom_links。"""

MOUNT_PROPOSER_LOCAL_CLUSTER_PROMPT = """
如果多个新事件彼此构成一个新的同一话题/同一 episode，但没有合适已有 anchor，写入 local_clusters。
local_clusters 只表示待 sleep 整合的 pending 候选，不会在归档阶段直接固化 summary。"""

MOUNT_PROPOSER_OUTPUT_PROMPT = """
输出必须是严格 JSON：
{"mounts":[{"new_atom_local_id":"N1","anchor_summary_id":"...","anchor_revision":1,"relation_type":"answers","confidence":0.72,"evidence_text":"...","uncertainty_reason":""}],"atom_links":[{"new_atom_local_id":"N1","historical_atom_local_id":"H1","relation_type":"same_object","confidence":0.72,"evidence_text":"...","uncertainty_reason":""}],"local_clusters":[{"new_atom_local_ids":["N1","N2"],"title":"...","confidence":0.78,"evidence_text":"..."}]}

要求：
- new_atom_local_id 必须来自输入的 new_atoms。
- historical_atom_local_id 必须来自输入的 historical_atoms。
- local_clusters 的 new_atom_local_ids 必须全部来自输入的 new_atoms，且至少 2 条。
- anchor_summary_id 和 anchor_revision 必须来自输入的 anchors。
- confidence 使用 0 到 1；弱关系低于 0.62。
- evidence_text 用一句话说明为什么应该挂载。
- 没有合适已有事件簇 anchor 时，优先考虑是否存在高质量 atom_links 或 local_clusters，而不是勉强输出 mounts。
- 不要输出 markdown，不要输出解释。"""


def run_post_archive_mount_workflow(
    con_or_path: sqlite3.Connection | str | os.PathLike[str],
    *,
    new_event_ids: Iterable[int],
    candidate_event_ids: Iterable[int] = (),
    max_mounts_per_atom: int = 3,
    max_cluster_summaries: int = 32,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Stage pending mounts for events produced by one archive extraction pass."""

    owns_connection = not isinstance(con_or_path, sqlite3.Connection)
    con = (
        sqlite3.connect(os.fspath(con_or_path), timeout=30.0)
        if owns_connection
        else con_or_path
    )
    try:
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        stats = _run_post_archive_mount_workflow(
            con,
            new_event_ids=new_event_ids,
            candidate_event_ids=candidate_event_ids,
            max_mounts_per_atom=max_mounts_per_atom,
            max_cluster_summaries=max_cluster_summaries,
            now_ms=now_ms,
        )
        if owns_connection:
            con.commit()
        return stats
    finally:
        if owns_connection:
            con.close()


def load_memory_atoms(con: sqlite3.Connection, event_ids: Iterable[int]) -> list[MemoryAtom]:
    ids = _unique_ints(event_ids)
    if not ids:
        return []
    previous_row_factory = con.row_factory
    try:
        con.row_factory = sqlite3.Row
        placeholders = ",".join("?" * len(ids))
        rows = list(
            con.execute(
                f"""
                SELECT event_id, summary, event_type, event_type_norm, occurred_at, status
                FROM MemoryV2Events
                WHERE event_id IN ({placeholders}) AND is_deleted=0
                """,
                ids,
            )
        )
        if not rows:
            return []

        roles_by_event: dict[int, list[str]] = defaultdict(list)
        for row in con.execute(
            f"""
            SELECT event_id, entity
            FROM MemoryV2Participants
            WHERE event_id IN ({placeholders}) AND entity IS NOT NULL AND entity <> ''
            ORDER BY participant_id ASC
            """,
            ids,
        ):
            roles_by_event[int(row["event_id"])].append(str(row["entity"] or ""))
    finally:
        con.row_factory = previous_row_factory

    atoms: list[MemoryAtom] = []
    for row in rows:
        status = str(row["status"] or "actual").strip().lower()
        if status in GUARDED_STATUSES:
            continue
        event_id = int(row["event_id"])
        entities = tuple(dict.fromkeys(item for item in roles_by_event.get(event_id, []) if item))
        atoms.append(
            MemoryAtom(
                event_id=event_id,
                summary=str(row["summary"] or ""),
                event_type_norm=str(row["event_type_norm"] or row["event_type"] or ""),
                entities=entities,
                occurred_at=int(row["occurred_at"] or 0),
                source="post_archive",
            )
        )
    return atoms


def load_candidate_cluster_summaries(
    con: sqlite3.Connection,
    candidate_event_ids: Iterable[int],
    *,
    exclude_event_ids: Iterable[int] = (),
    max_cluster_summaries: int = 32,
) -> list[ClusterSummaryRecord]:
    candidate_ids = set(_unique_ints(candidate_event_ids))
    if not candidate_ids:
        return []
    excluded = set(_unique_ints(exclude_event_ids))
    limit = max(1, int(max_cluster_summaries or 1))
    cards: dict[str, ClusterSummaryRecord] = {}

    query_limit = max(limit * 4, 32)
    placeholders = ",".join("?" * len(candidate_ids))
    for row in con.execute(
        f"""
        SELECT c.cluster_summary_json
        FROM MemoryV2SummaryInputEvents ie
        JOIN MemoryV2SummaryCache c ON c.packet_id=ie.packet_id
        WHERE ie.status='active'
          AND c.status='ready'
          AND c.cluster_summary_json <> '{{}}'
          AND ie.event_id IN ({placeholders})
        ORDER BY c.updated_at_ms DESC, c.summary_id DESC
        LIMIT ?
        """,
        [*sorted(candidate_ids), query_limit],
    ):
        _add_card(cards, row[0], candidate_ids, excluded, limit)
        if len(cards) >= limit:
            return list(cards.values())

    resolved_summary_ids = _resolve_summary_ids_from_input_links(con, candidate_ids)
    resolved_summary_ids.difference_update(cards.keys())
    if resolved_summary_ids:
        summary_placeholders = ",".join("?" * len(resolved_summary_ids))
        for row in con.execute(
            f"""
            SELECT cluster_summary_json
            FROM MemoryV2SummaryCache
            WHERE status='ready'
              AND cluster_summary_json <> '{{}}'
              AND summary_id IN ({summary_placeholders})
            ORDER BY updated_at_ms DESC, summary_id DESC
            LIMIT ?
            """,
            [*sorted(resolved_summary_ids), query_limit],
        ):
            _add_card(cards, row[0], candidate_ids, excluded, limit)
            if len(cards) >= limit:
                return list(cards.values())

    scan_limit = query_limit
    for row in con.execute(
        """
        SELECT cluster_summary_json
        FROM MemoryV2SummaryCache
        WHERE status='ready' AND cluster_summary_json <> '{}'
        ORDER BY updated_at_ms DESC, summary_id DESC
        LIMIT ?
        """,
        (scan_limit,),
    ):
        _add_card(cards, row[0], candidate_ids, excluded, limit)
        if len(cards) >= limit:
            break
    return list(cards.values())


def load_recent_cluster_summaries(
    con: sqlite3.Connection,
    *,
    exclude_event_ids: Iterable[int] = (),
    max_cluster_summaries: int = 32,
) -> list[ClusterSummaryRecord]:
    """Load recent ready cluster summaries without requiring candidate event anchors."""

    excluded = set(_unique_ints(exclude_event_ids))
    limit = max(1, int(max_cluster_summaries or 1))
    cards: dict[str, ClusterSummaryRecord] = {}
    for row in con.execute(
        """
        SELECT cluster_summary_json
        FROM MemoryV2SummaryCache
        WHERE status='ready' AND cluster_summary_json <> '{}'
        ORDER BY updated_at_ms DESC, summary_id DESC
        LIMIT ?
        """,
        (max(limit * 3, 32),),
    ):
        _add_card_any_source(cards, row[0], excluded, limit)
        if len(cards) >= limit:
            break
    return list(cards.values())


def _resolve_summary_ids_from_input_links(
    con: sqlite3.Connection,
    candidate_ids: set[int],
) -> set[str]:
    if not candidate_ids:
        return set()
    placeholders = ",".join("?" * len(candidate_ids))
    summary_ids: set[str] = set()
    for packet_id, invalidation_json, packet_json in con.execute(
        f"""
        SELECT DISTINCT ie.packet_id, si.invalidation_json, si.packet_json
        FROM MemoryV2SummaryInputEvents ie
        LEFT JOIN MemoryV2SummaryInputs si ON si.packet_id=ie.packet_id
        WHERE ie.status='active'
          AND ie.event_id IN ({placeholders})
        """,
        sorted(candidate_ids),
    ):
        summary_ids.update(_summary_ids_from_input_metadata(packet_id, invalidation_json, packet_json))
    return summary_ids


def _summary_ids_from_input_metadata(
    packet_id: object,
    invalidation_json: object,
    packet_json: object,
) -> set[str]:
    ids: set[str] = set()
    packet_key = str(packet_id or "").strip()
    if packet_key:
        ids.add(packet_key)
        if packet_key.startswith("summary-refresh:"):
            ids.add(packet_key.removeprefix("summary-refresh:").strip())

    invalidation = _safe_json_object(invalidation_json)
    _add_summary_id(ids, invalidation.get("summary_id"))

    packet = _safe_json_object(packet_json)
    _add_summary_id(ids, packet.get("summary_id"))
    cluster_summary = packet.get("cluster_summary")
    if isinstance(cluster_summary, dict):
        _add_summary_id(ids, cluster_summary.get("summary_id"))
    previous = packet.get("previous_cluster_summary_stale_prior")
    if isinstance(previous, dict):
        _add_summary_id(ids, previous.get("summary_id"))

    return {item for item in ids if item}


def _add_summary_id(ids: set[str], value: object) -> None:
    text = str(value or "").strip()
    if text:
        ids.add(text)


def _safe_json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _run_post_archive_mount_workflow(
    con: sqlite3.Connection,
    *,
    new_event_ids: Iterable[int],
    candidate_event_ids: Iterable[int],
    max_mounts_per_atom: int,
    max_cluster_summaries: int,
    now_ms: int | None,
) -> dict[str, Any]:
    ensure_preprocessing_schema(con)
    con.commit()
    new_ids = _unique_ints(new_event_ids)
    new_id_set = set(new_ids)
    candidate_ids = [item for item in _unique_ints(candidate_event_ids) if item not in new_id_set]
    atoms = load_memory_atoms(con, new_ids)
    historical_atoms = load_memory_atoms(con, candidate_ids)
    cards = load_candidate_cluster_summaries(
        con,
        candidate_ids,
        exclude_event_ids=new_ids,
        max_cluster_summaries=max_cluster_summaries,
    )
    mode = "rules"
    proposed_count = 0
    written = 0
    model_errors: list[str] = []
    mount_errors: list[str] = []
    atom_links_proposed = 0
    atom_links_staged = 0
    atom_link_errors: list[str] = []
    if _llm_mount_enabled():
        mode = "llm"
        cards = _expand_cluster_summaries_for_llm(
            con,
            cards,
            exclude_event_ids=new_ids,
            max_cluster_summaries=max_cluster_summaries,
        )
        llm_stats = _run_llm_mount_proposer(
            con,
            atoms,
            cards,
            historical_atoms,
            max_mounts_per_atom=max_mounts_per_atom,
            now_ms=now_ms,
        )
        proposed_count = int(llm_stats.get("mount_candidates") or 0)
        written = int(llm_stats.get("mounts_staged") or 0)
        atom_links_proposed = int(llm_stats.get("atom_link_candidates") or 0)
        atom_links_staged = int(llm_stats.get("atom_links_staged") or 0)
        atom_link_errors = [str(item) for item in llm_stats.get("atom_link_errors") or ()]
        model_errors = [str(item) for item in llm_stats.get("model_errors") or ()]
        mount_errors = [str(item) for item in llm_stats.get("mount_errors") or ()]
        local_clusters_proposed = int(llm_stats.get("local_cluster_candidates") or 0)
        local_clusters_staged = int(llm_stats.get("local_clusters_staged") or 0)
        local_cluster_errors = [str(item) for item in llm_stats.get("local_cluster_errors") or ()]
        summary_inputs_queued = int(llm_stats.get("summary_inputs_queued") or 0)
        summaries_ready = int(llm_stats.get("summaries_ready") or 0)
    else:
        local_clusters_proposed = 0
        local_clusters_staged = 0
        local_cluster_errors = []
        summary_inputs_queued = 0
        summaries_ready = 0
        proposed = propose_memory_mounts(
            cards,
            atoms,
            max_mounts_per_atom=max_mounts_per_atom,
        )
        proposed_count = len(proposed)
        staged = [_with_workflow_evidence(item) for item in proposed]
        written = write_memory_mounts(con, staged, now_ms=int(now_ms or time.time() * 1000))
    return {
        "mount_mode": mode,
        "new_event_ids": len(new_ids),
        "new_events_loaded": len(atoms),
        "candidate_event_ids": len(candidate_ids),
        "historical_atoms_loaded": len(historical_atoms),
        "cluster_summaries_loaded": len(cards),
        "mounts_proposed": proposed_count,
        "mounts_staged": written,
        "atom_links_proposed": atom_links_proposed,
        "atom_links_staged": atom_links_staged,
        "atom_link_errors": atom_link_errors,
        "local_clusters_proposed": local_clusters_proposed,
        "local_clusters_staged": local_clusters_staged,
        "local_cluster_errors": local_cluster_errors,
        "summary_inputs_queued": summary_inputs_queued,
        "summaries_ready": summaries_ready,
        "model_errors": model_errors,
        "mount_errors": mount_errors,
    }


def _add_card(
    cards: dict[str, ClusterSummaryRecord],
    payload: object,
    candidate_ids: set[int],
    excluded_ids: set[int],
    limit: int,
) -> None:
    if len(cards) >= limit:
        return
    try:
        card = cluster_summary_from_json(str(payload or "{}"))
    except Exception:
        return
    if not card.summary_id or card.summary_id in cards:
        return
    source_ids = set(int(x) for x in card.source_event_ids)
    if not source_ids.intersection(candidate_ids):
        return
    if source_ids.intersection(excluded_ids):
        return
    cards[card.summary_id] = card


def _add_card_any_source(
    cards: dict[str, ClusterSummaryRecord],
    payload: object,
    excluded_ids: set[int],
    limit: int,
) -> None:
    if len(cards) >= limit:
        return
    try:
        card = cluster_summary_from_json(str(payload or "{}"))
    except Exception:
        return
    if not card.summary_id or card.summary_id in cards:
        return
    source_ids = set(int(x) for x in card.source_event_ids)
    if source_ids.intersection(excluded_ids):
        return
    cards[card.summary_id] = card


def _expand_cluster_summaries_for_llm(
    con: sqlite3.Connection,
    cards: list[ClusterSummaryRecord],
    *,
    exclude_event_ids: Iterable[int],
    max_cluster_summaries: int,
) -> list[ClusterSummaryRecord]:
    limit = max(1, int(max_cluster_summaries or 1))
    by_id = {card.summary_id: card for card in cards if card.summary_id}
    if len(by_id) < limit:
        for card in load_recent_cluster_summaries(
            con,
            exclude_event_ids=exclude_event_ids,
            max_cluster_summaries=limit,
        ):
            by_id.setdefault(card.summary_id, card)
            if len(by_id) >= limit:
                break
    return list(by_id.values())[:limit]


def _llm_mount_enabled() -> bool:
    cfg = _memory_consolidation_cfg()
    return bool(cfg.get("enabled", False) and cfg.get("llm_mount_enabled", False))


def _memory_consolidation_cfg() -> dict[str, Any]:
    try:
        import app_state

        cfg = getattr(app_state, "memory_consolidation_cfg", None)
        if isinstance(cfg, dict):
            return dict(cfg)
        root = getattr(app_state, "config", {}) or {}
        memory = root.get("memory", {}) if isinstance(root, dict) else {}
        consolidation = memory.get("consolidation", {}) if isinstance(memory, dict) else {}
        return dict(consolidation) if isinstance(consolidation, dict) else {}
    except Exception:
        return {}


def _run_llm_mount_proposer(
    con: sqlite3.Connection,
    atoms: list[MemoryAtom],
    cards: list[ClusterSummaryRecord],
    historical_atoms: list[MemoryAtom],
    *,
    max_mounts_per_atom: int,
    now_ms: int | None,
) -> dict[str, Any]:
    if not atoms:
        return {
            "mount_candidates": 0,
            "mounts_staged": 0,
            "atom_link_candidates": 0,
            "atom_links_staged": 0,
            "atom_link_errors": [],
            "local_cluster_candidates": 0,
            "local_clusters_staged": 0,
            "local_cluster_errors": [],
            "summary_inputs_queued": 0,
            "summaries_ready": 0,
            "mount_errors": [],
            "model_errors": [],
        }
    try:
        import app_state
    except Exception:
        return {
            "mount_candidates": 0,
            "mounts_staged": 0,
            "atom_link_candidates": 0,
            "atom_links_staged": 0,
            "atom_link_errors": [],
            "local_cluster_candidates": 0,
            "local_clusters_staged": 0,
            "local_cluster_errors": [],
            "summary_inputs_queued": 0,
            "summaries_ready": 0,
            "mount_errors": [],
            "model_errors": ["app_state unavailable"],
        }

    adapter = getattr(app_state, "memory_consolidation_adapter", None)
    if adapter is None:
        return {
            "mount_candidates": 0,
            "mounts_staged": 0,
            "atom_link_candidates": 0,
            "atom_links_staged": 0,
            "atom_link_errors": [],
            "local_cluster_candidates": 0,
            "local_clusters_staged": 0,
            "local_cluster_errors": [],
            "summary_inputs_queued": 0,
            "summaries_ready": 0,
            "mount_errors": [],
            "model_errors": ["memory_consolidation_adapter unavailable"],
        }

    local_event_ids = {f"N{index}": atom.event_id for index, atom in enumerate(atoms, start=1)}
    historical_event_ids = {f"H{index}": atom.event_id for index, atom in enumerate(historical_atoms, start=1)}
    user_payload = _build_llm_mount_user_payload(atoms, cards, historical_atoms)
    cfg = _memory_consolidation_cfg()
    gen = dict(cfg.get("generation", {}) if isinstance(cfg.get("generation"), dict) else {})
    gen.setdefault("temperature", 0.2)
    gen.setdefault("max_output_tokens", 4000)
    logger.info(
        "[mount_workflow] 调用 LLM 挂载 proposer atoms=%d historical_atoms=%d anchors=%d",
        len(atoms),
        len(historical_atoms),
        len(cards),
    )
    try:
        raw = adapter.call_simple_text(
            _mount_proposer_system_prompt(),
            user_payload,
            gen,
            log_tag="memory_consolidation/mount",
        )
    except Exception as exc:
        logger.warning("[mount_workflow] LLM 挂载 proposer 调用失败: %s", exc)
        return {
            "mount_candidates": 0,
            "mounts_staged": 0,
            "atom_link_candidates": 0,
            "atom_links_staged": 0,
            "atom_link_errors": [],
            "local_cluster_candidates": 0,
            "local_clusters_staged": 0,
            "local_cluster_errors": [],
            "summary_inputs_queued": 0,
            "summaries_ready": 0,
            "mount_errors": [],
            "model_errors": [f"adapter call failed: {exc}"],
        }
    parsed, parse_errors = _parse_llm_mount_response(raw)
    stage_stats = stage_memory_mount_candidates(
        con,
        parsed["mounts"],
        local_event_ids=local_event_ids,
        now_ms=now_ms,
        max_mounts_per_atom=max_mounts_per_atom,
    )
    atom_link_stats = _stage_historical_atom_link_candidates(
        con,
        parsed["atom_links"],
        local_event_ids=local_event_ids,
        historical_event_ids=historical_event_ids,
        now_ms=now_ms,
        max_links_per_batch=int(cfg.get("max_atom_links_per_archive", cfg.get("max_local_clusters_per_archive", 8)) or 8),
        min_confidence=float(cfg.get("local_cluster_min_confidence", 0.62) or 0.62),
    )
    local_stats = _stage_local_cluster_candidates(
        con,
        parsed["local_clusters"],
        local_event_ids=local_event_ids,
        now_ms=now_ms,
        max_clusters_per_batch=int(cfg.get("max_local_clusters_per_archive", 8) or 8),
        min_confidence=float(cfg.get("local_cluster_min_confidence", 0.62) or 0.62),
    )
    logger.info(
        "[mount_workflow] LLM 挂载 proposer 完成 mount_candidates=%d mounts_staged=%d atom_link_candidates=%d atom_links_staged=%d local_cluster_candidates=%d local_clusters_staged=%d model_errors=%d mount_errors=%d atom_link_errors=%d local_cluster_errors=%d",
        int(stage_stats.get("mount_candidates") or 0),
        int(stage_stats.get("mounts_staged") or 0),
        int(atom_link_stats.get("atom_link_candidates") or 0),
        int(atom_link_stats.get("atom_links_staged") or 0),
        int(local_stats.get("local_cluster_candidates") or 0),
        int(local_stats.get("local_clusters_staged") or 0),
        len(parse_errors),
        len(stage_stats.get("mount_errors") or ()),
        len(atom_link_stats.get("atom_link_errors") or ()),
        len(local_stats.get("local_cluster_errors") or ()),
    )
    return {
        **stage_stats,
        **atom_link_stats,
        **local_stats,
        "model_errors": parse_errors,
    }


def _mount_proposer_system_prompt() -> str:
    parts = [
        MOUNT_PROPOSER_BASE_PROMPT,
        MOUNT_PROPOSER_LOCAL_CLUSTER_PROMPT,
        MOUNT_PROPOSER_OUTPUT_PROMPT,
    ]
    return "\n\n".join(part.strip() for part in parts if part.strip())


def _build_llm_mount_user_payload(
    atoms: list[MemoryAtom],
    cards: list[ClusterSummaryRecord],
    historical_atoms: list[MemoryAtom],
) -> str:
    payload = {
        "new_atoms": [
            {
                "local_id": f"N{index}",
                "event_type": atom.event_type_norm,
                "summary": atom.summary,
                "entities": list(atom.entities),
                "occurred_at": atom.occurred_at,
            }
            for index, atom in enumerate(atoms, start=1)
        ],
        "historical_atoms": [
            {
                "local_id": f"H{index}",
                "event_id": atom.event_id,
                "event_type": atom.event_type_norm,
                "summary": atom.summary,
                "entities": list(atom.entities),
                "occurred_at": atom.occurred_at,
            }
            for index, atom in enumerate(historical_atoms, start=1)
        ],
        "anchors": [
            {
                "summary_id": card.summary_id,
                "revision": card.revision,
                "source_kind": card.source_kind,
                "source_id": card.source_id,
                "title": card.title,
                "short_summary": card.short_summary,
                "core_entities": list(card.core_entities),
                "confirmed_claims": list(card.confirmed_claims),
                "uncertain_claims": list(card.uncertain_claims),
                "current_state": card.current_state,
                "open_slots": list(card.open_slots),
            }
            for card in cards
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_llm_mount_response(raw: object) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    text = str(raw or "").strip()
    if not text:
        return {"mounts": [], "atom_links": [], "local_clusters": []}, ["empty model response"]
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    first_obj = _extract_first_json_container(text)
    if first_obj:
        candidates.insert(0, first_obj)
    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(f"json parse failed: {exc.msg}")
            continue
        mounts = parsed.get("mounts") if isinstance(parsed, dict) else parsed
        atom_links = parsed.get("atom_links") if isinstance(parsed, dict) else []
        local_clusters = parsed.get("local_clusters") if isinstance(parsed, dict) else []
        if mounts is None:
            mounts = []
        if atom_links is None:
            atom_links = []
        if local_clusters is None:
            local_clusters = []
        if not isinstance(mounts, list):
            errors.append("parsed JSON does not contain a mounts array")
            continue
        if not isinstance(atom_links, list):
            errors.append("parsed JSON atom_links is not an array")
            atom_links = []
        if not isinstance(local_clusters, list):
            errors.append("parsed JSON local_clusters is not an array")
            local_clusters = []
        out_mounts: list[dict[str, Any]] = []
        for item in mounts:
            if isinstance(item, dict):
                item = dict(item)
                item["_raw_mount_json"] = json.dumps(item, ensure_ascii=False, sort_keys=True)
                out_mounts.append(item)
        out_atom_links: list[dict[str, Any]] = []
        for item in atom_links:
            if isinstance(item, dict):
                item = dict(item)
                item["_raw_atom_link_json"] = json.dumps(item, ensure_ascii=False, sort_keys=True)
                out_atom_links.append(item)
        out_clusters: list[dict[str, Any]] = []
        for item in local_clusters:
            if isinstance(item, dict):
                item = dict(item)
                item["_raw_cluster_json"] = json.dumps(item, ensure_ascii=False, sort_keys=True)
                out_clusters.append(item)
        return {"mounts": out_mounts, "atom_links": out_atom_links, "local_clusters": out_clusters}, errors
    return {"mounts": [], "atom_links": [], "local_clusters": []}, errors or ["no parseable JSON response"]


def _stage_historical_atom_link_candidates(
    con: sqlite3.Connection,
    candidates: Iterable[dict[str, Any]],
    *,
    local_event_ids: dict[str, int],
    historical_event_ids: dict[str, int],
    now_ms: int | None,
    max_links_per_batch: int,
    min_confidence: float,
) -> dict[str, Any]:
    ensure_preprocessing_schema(con)
    candidate_list = [item for item in candidates if isinstance(item, dict)]
    if not candidate_list:
        return {
            "atom_link_candidates": 0,
            "atom_links_staged": 0,
            "atom_link_errors": [],
        }

    now = int(now_ms or time.time() * 1000)
    errors: list[str] = []
    mounts: list[LocalClusterMount] = []
    seen_event_sets: set[tuple[int, ...]] = set()
    limit = max(1, min(64, int(max_links_per_batch or 1)))
    threshold = max(0.0, min(1.0, float(min_confidence)))
    for index, candidate in enumerate(candidate_list, start=1):
        new_local_id = str(candidate.get("new_atom_local_id") or candidate.get("new_event_local_id") or "").strip()
        if not new_local_id or new_local_id not in local_event_ids:
            errors.append(f"atom_link#{index}: unknown new_atom_local_id {new_local_id!r}")
            continue
        historical_local_id = str(
            candidate.get("historical_atom_local_id") or candidate.get("existing_atom_local_id") or ""
        ).strip()
        if not historical_local_id or historical_local_id not in historical_event_ids:
            errors.append(f"atom_link#{index}: unknown historical_atom_local_id {historical_local_id!r}")
            continue
        relation_type = str(candidate.get("relation_type") or "").strip()
        if relation_type == "unrelated" or relation_type not in ALLOWED_MEMORY_MOUNT_RELATION_TYPES:
            errors.append(f"atom_link#{index}: relation_type is not allowed {relation_type!r}")
            continue
        confidence_raw = candidate.get("confidence")
        if isinstance(confidence_raw, bool):
            errors.append(f"atom_link#{index}: confidence must be numeric")
            continue
        try:
            confidence = max(0.0, min(1.0, float(confidence_raw)))
        except (TypeError, ValueError):
            errors.append(f"atom_link#{index}: confidence must be numeric")
            continue
        if confidence < threshold:
            errors.append(f"atom_link#{index}: confidence below threshold {confidence:.3f}")
            continue
        evidence_text = str(candidate.get("evidence_text") or "").strip()
        if not evidence_text:
            errors.append(f"atom_link#{index}: evidence_text is required")
            continue
        event_key = tuple(sorted({int(local_event_ids[new_local_id]), int(historical_event_ids[historical_local_id])}))
        if len(event_key) < 2:
            errors.append(f"atom_link#{index}: new and historical atoms resolve to the same event")
            continue
        if event_key in seen_event_sets:
            errors.append(f"atom_link#{index}: duplicate atom link event set")
            continue
        title = _clean_cluster_title(candidate.get("title") or candidate.get("cluster_title") or evidence_text)
        proposal_id = _sha1_local(
            "historical-atom-link",
            *(str(event_id) for event_id in event_key),
            relation_type,
            evidence_text,
        )[:24]
        evidence = {
            "generator": "post_archive_mount_workflow.atom_link_candidate",
            "event_ids": list(event_key),
            "new_atom_local_id": new_local_id,
            "historical_atom_local_id": historical_local_id,
            "relation_type": relation_type,
            "title": title,
            "evidence_text": evidence_text,
            "raw_atom_link_json": str(candidate.get("_raw_atom_link_json") or ""),
        }
        mounts.append(
            LocalClusterMount(
                proposal_id=proposal_id,
                event_ids=event_key,
                title=title,
                confidence=round(confidence, 6),
                evidence_text=evidence_text,
                uncertainty_reason=str(candidate.get("uncertainty_reason") or ""),
                evidence=evidence,
            )
        )
        seen_event_sets.add(event_key)
        if len(mounts) >= limit:
            break

    written = write_local_cluster_mounts(con, mounts, now_ms=now) if mounts else 0
    return {
        "atom_link_candidates": len(candidate_list),
        "atom_links_staged": written,
        "atom_link_errors": errors,
    }


def _stage_local_cluster_candidates(
    con: sqlite3.Connection,
    candidates: Iterable[dict[str, Any]],
    *,
    local_event_ids: dict[str, int],
    now_ms: int | None,
    max_clusters_per_batch: int,
    min_confidence: float,
) -> dict[str, Any]:
    ensure_preprocessing_schema(con)
    candidate_list = [item for item in candidates if isinstance(item, dict)]
    if not candidate_list:
        return {
            "local_cluster_candidates": 0,
            "local_clusters_staged": 0,
            "local_cluster_errors": [],
            "summary_inputs_queued": 0,
            "summaries_ready": 0,
        }

    now = int(now_ms or time.time() * 1000)
    errors: list[str] = []
    mounts: list[LocalClusterMount] = []
    seen_event_sets: set[tuple[int, ...]] = set()
    limit = max(1, min(32, int(max_clusters_per_batch or 1)))
    threshold = max(0.0, min(1.0, float(min_confidence)))
    for index, candidate in enumerate(candidate_list, start=1):
        raw_ids = candidate.get("new_atom_local_ids")
        if raw_ids is None:
            raw_ids = candidate.get("new_event_local_ids")
        if not isinstance(raw_ids, list):
            errors.append(f"local_cluster#{index}: new_atom_local_ids must be an array")
            continue
        event_ids: list[int] = []
        for raw_id in raw_ids:
            local_id = str(raw_id or "").strip()
            event_id = local_event_ids.get(local_id)
            if event_id is None:
                errors.append(f"local_cluster#{index}: unknown new_atom_local_id {local_id!r}")
                continue
            if event_id not in event_ids:
                event_ids.append(event_id)
        if len(event_ids) < 2:
            errors.append(f"local_cluster#{index}: at least two known new atoms are required")
            continue
        event_key = tuple(sorted(event_ids))
        if event_key in seen_event_sets:
            errors.append(f"local_cluster#{index}: duplicate local event set")
            continue
        try:
            confidence = max(0.0, min(1.0, float(candidate.get("confidence"))))
        except (TypeError, ValueError):
            errors.append(f"local_cluster#{index}: confidence must be numeric")
            continue
        if confidence < threshold:
            errors.append(f"local_cluster#{index}: confidence below threshold {confidence:.3f}")
            continue
        evidence_text = str(candidate.get("evidence_text") or "").strip()
        if not evidence_text:
            errors.append(f"local_cluster#{index}: evidence_text is required")
            continue
        title = _clean_cluster_title(candidate.get("title") or candidate.get("cluster_title"))
        proposal_id = _sha1_local(
            "local-cluster-mount",
            *(str(event_id) for event_id in event_key),
            evidence_text,
        )[:24]
        evidence = {
            "generator": "post_archive_mount_workflow.local_cluster_candidate",
            "event_ids": list(event_key),
            "title": title,
            "evidence_text": evidence_text,
            "raw_cluster_json": str(candidate.get("_raw_cluster_json") or ""),
        }
        mounts.append(
            LocalClusterMount(
                proposal_id=proposal_id,
                event_ids=event_key,
                title=title,
                confidence=round(confidence, 6),
                evidence_text=evidence_text,
                uncertainty_reason=str(candidate.get("uncertainty_reason") or ""),
                evidence=evidence,
            )
        )
        seen_event_sets.add(event_key)
        if len(mounts) >= limit:
            break

    if not mounts:
        return {
            "local_cluster_candidates": len(candidate_list),
            "local_clusters_staged": 0,
            "local_cluster_errors": errors,
            "summary_inputs_queued": 0,
            "summaries_ready": 0,
        }
    written = write_local_cluster_mounts(con, mounts, now_ms=now)
    return {
        "local_cluster_candidates": len(candidate_list),
        "local_clusters_staged": written,
        "local_cluster_errors": errors,
        "summary_inputs_queued": 0,
        "summaries_ready": 0,
    }


def _clean_cluster_title(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())[:80]


def _sha1_local(*parts: str) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _extract_first_json_container(text: str) -> str:
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not starts:
        return ""
    start = min(starts)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
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
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _with_workflow_evidence(mount: MemoryMount) -> MemoryMount:
    evidence = dict(mount.evidence)
    evidence["generator"] = "post_archive_mount_workflow.rules"
    return replace(mount, evidence=evidence)


def _unique_ints(values: Iterable[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values or ():
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item <= 0 or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out




__all__ = [
    "load_candidate_cluster_summaries",
    "load_recent_cluster_summaries",
    "load_memory_atoms",
    "run_post_archive_mount_workflow",
]
