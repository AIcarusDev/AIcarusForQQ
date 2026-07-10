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
from dataclasses import dataclass
from typing import Any, Iterable

from .prompt import MOUNT_PROPOSER_SYSTEM_PROMPT
from ..sleep.consolidation import (
    AttachAtomToClusterResult,
    LocalClusterMount,
    MemoryAtom,
    ClusterSummaryRecord,
    ensure_preprocessing_schema,
    stage_atom_to_cluster_mounts,
    cluster_summary_from_json,
    write_local_cluster_mounts,
)


logger = logging.getLogger("AICQ.memory.post_archive.mount_workflow")


@dataclass(frozen=True)
class LinkAtomToHistoricalAtomResult:
    candidates: int = 0
    staged: int = 0
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_link_candidates": self.candidates,
            "atom_links_staged": self.staged,
            "atom_link_errors": list(self.errors),
        }


@dataclass(frozen=True)
class ProposeLocalClusterResult:
    candidates: int = 0
    staged: int = 0
    errors: tuple[str, ...] = ()
    summary_tasks_queued: int = 0
    summaries_ready: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_cluster_candidates": self.candidates,
            "local_clusters_staged": self.staged,
            "local_cluster_errors": list(self.errors),
            "summary_tasks_queued": self.summary_tasks_queued,
            "summaries_ready": self.summaries_ready,
        }


@dataclass(frozen=True)
class MountProposalResult:
    attach: AttachAtomToClusterResult = AttachAtomToClusterResult()
    historical_links: LinkAtomToHistoricalAtomResult = LinkAtomToHistoricalAtomResult()
    local_clusters: ProposeLocalClusterResult = ProposeLocalClusterResult()
    model_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.attach.to_dict(),
            **self.historical_links.to_dict(),
            **self.local_clusters.to_dict(),
            "model_errors": list(self.model_errors),
        }


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
                FROM MemoryEvents
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
            FROM MemoryParticipants
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
        event_id = int(row["event_id"])
        entities = tuple(dict.fromkeys(item for item in roles_by_event.get(event_id, []) if item))
        atoms.append(
            MemoryAtom(
                event_id=event_id,
                summary=str(row["summary"] or ""),
                event_type_norm=str(row["event_type_norm"] or row["event_type"] or ""),
                status=status,
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
        FROM MemoryClusterSummaryTaskEvents te
        JOIN MemorySummaryCache c ON c.task_id=te.task_id
        WHERE te.status='active'
          AND c.status='ready'
          AND c.cluster_summary_json <> '{{}}'
          AND te.event_id IN ({placeholders})
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
            FROM MemorySummaryCache
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
        FROM MemorySummaryCache
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
        FROM MemorySummaryCache
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
    for task_id, in con.execute(
        f"""
        SELECT DISTINCT task_id
        FROM MemoryClusterSummaryTaskEvents
        WHERE status='active'
          AND event_id IN ({placeholders})
        """,
        sorted(candidate_ids),
    ):
        text = str(task_id or "").strip()
        if text:
            summary_ids.add(text)
    return summary_ids


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
    mode = "disabled"
    proposed_count = 0
    written = 0
    model_errors: list[str] = []
    mount_errors: list[str] = []
    atom_links_proposed = 0
    atom_links_staged = 0
    atom_link_errors: list[str] = []
    local_clusters_proposed = 0
    local_clusters_staged = 0
    local_cluster_errors: list[str] = []
    summary_tasks_queued = 0
    summaries_ready = 0
    if _llm_mount_enabled():
        mode = "llm"
        cards = _expand_cluster_summaries_for_llm(
            con,
            cards,
            exclude_event_ids=new_ids,
            max_cluster_summaries=max_cluster_summaries,
        )
        llm_result = _run_llm_mount_proposer(
            con,
            atoms,
            cards,
            historical_atoms,
            max_mounts_per_atom=max_mounts_per_atom,
            now_ms=now_ms,
        )
        proposed_count = llm_result.attach.candidates
        written = llm_result.attach.staged
        atom_links_proposed = llm_result.historical_links.candidates
        atom_links_staged = llm_result.historical_links.staged
        atom_link_errors = list(llm_result.historical_links.errors)
        model_errors = list(llm_result.model_errors)
        mount_errors = list(llm_result.attach.errors)
        local_clusters_proposed = llm_result.local_clusters.candidates
        local_clusters_staged = llm_result.local_clusters.staged
        local_cluster_errors = list(llm_result.local_clusters.errors)
        summary_tasks_queued = llm_result.local_clusters.summary_tasks_queued
        summaries_ready = llm_result.local_clusters.summaries_ready
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
        "summary_tasks_queued": summary_tasks_queued,
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
) -> MountProposalResult:
    if not atoms:
        return MountProposalResult()
    try:
        import app_state
    except Exception:
        return MountProposalResult(model_errors=("app_state unavailable",))

    adapter = getattr(app_state, "memory_consolidation_adapter", None)
    if adapter is None:
        return MountProposalResult(model_errors=("memory_consolidation_adapter unavailable",))

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
            MOUNT_PROPOSER_SYSTEM_PROMPT,
            user_payload,
            gen,
            log_tag="memory_consolidation/mount",
        )
    except Exception as exc:
        logger.warning("[mount_workflow] LLM 挂载 proposer 调用失败: %s", exc)
        return MountProposalResult(model_errors=(f"adapter call failed: {exc}",))
    parsed, parse_errors = _parse_llm_mount_response(raw)
    attach = stage_atom_to_cluster_mounts(
        con,
        parsed["mounts"],
        local_event_ids=local_event_ids,
        now_ms=now_ms,
        max_mounts_per_atom=max_mounts_per_atom,
    )
    atom_links = _stage_historical_atom_link_candidates(
        con,
        parsed["atom_links"],
        local_event_ids=local_event_ids,
        historical_event_ids=historical_event_ids,
        now_ms=now_ms,
        max_links_per_batch=int(cfg.get("max_atom_links_per_archive", cfg.get("max_local_clusters_per_archive", 8)) or 8),
        min_confidence=float(cfg.get("local_cluster_min_confidence", 0.62) or 0.62),
    )
    local_clusters = _stage_local_cluster_candidates(
        con,
        parsed["local_clusters"],
        local_event_ids=local_event_ids,
        now_ms=now_ms,
        max_clusters_per_batch=int(cfg.get("max_local_clusters_per_archive", 8) or 8),
        min_confidence=float(cfg.get("local_cluster_min_confidence", 0.62) or 0.62),
    )
    logger.info(
        "[mount_workflow] LLM 挂载 proposer 完成 mount_candidates=%d mounts_staged=%d atom_link_candidates=%d atom_links_staged=%d local_cluster_candidates=%d local_clusters_staged=%d model_errors=%d mount_errors=%d atom_link_errors=%d local_cluster_errors=%d",
        attach.candidates,
        attach.staged,
        atom_links.candidates,
        atom_links.staged,
        local_clusters.candidates,
        local_clusters.staged,
        len(parse_errors),
        len(attach.errors),
        len(atom_links.errors),
        len(local_clusters.errors),
    )
    return MountProposalResult(
        attach=attach,
        historical_links=atom_links,
        local_clusters=local_clusters,
        model_errors=tuple(parse_errors),
    )


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
                "status": atom.status,
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
                "status": atom.status,
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
) -> LinkAtomToHistoricalAtomResult:
    ensure_preprocessing_schema(con)
    candidate_list = [item for item in candidates if isinstance(item, dict)]
    if not candidate_list:
        return LinkAtomToHistoricalAtomResult()

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
        if not relation_type:
            errors.append(f"atom_link#{index}: relation_type must describe a mount relation")
            continue
        confidence_raw = candidate.get("confidence")
        if confidence_raw is None:
            errors.append(f"atom_link#{index}: confidence is required")
            continue
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
    return LinkAtomToHistoricalAtomResult(
        candidates=len(candidate_list),
        staged=written,
        errors=tuple(errors),
    )


def _stage_local_cluster_candidates(
    con: sqlite3.Connection,
    candidates: Iterable[dict[str, Any]],
    *,
    local_event_ids: dict[str, int],
    now_ms: int | None,
    max_clusters_per_batch: int,
    min_confidence: float,
) -> ProposeLocalClusterResult:
    ensure_preprocessing_schema(con)
    candidate_list = [item for item in candidates if isinstance(item, dict)]
    if not candidate_list:
        return ProposeLocalClusterResult()

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
        confidence_val = candidate.get("confidence")
        if confidence_val is None:
            errors.append(f"local_cluster#{index}: confidence is required")
            continue
        try:
            confidence = max(0.0, min(1.0, float(confidence_val)))
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
        return ProposeLocalClusterResult(
            candidates=len(candidate_list),
            errors=tuple(errors),
        )
    written = write_local_cluster_mounts(con, mounts, now_ms=now)
    return ProposeLocalClusterResult(
        candidates=len(candidate_list),
        staged=written,
        errors=tuple(errors),
    )


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
