"""Post-archive event linking and candidate-storyline workflow."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .prompt import POST_ARCHIVE_TIDY_SYSTEM_PROMPT
from ..sleep.consolidation import ensure_preprocessing_schema


logger = logging.getLogger("AICQ.memory.post_archive.tidy_workflow")
_RESPONSE_RE = re.compile(
    r"^\s*<analysis>(?P<analysis>.*?)</analysis>\s*"
    r"<tidy>\s*<link>(?P<link>.*?)</link>\s*"
    r"<candidate_storyline>(?P<candidate_storyline>.*?)</candidate_storyline>\s*</tidy>\s*$",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class EventAtom:
    event_id: int
    summary: str
    entities: tuple[str, ...] = ()


@dataclass(frozen=True)
class TidyResult:
    links: tuple[tuple[str, str], ...] = ()
    candidate_storylines: tuple[tuple[str, ...], ...] = ()
    errors: tuple[str, ...] = ()


def run_post_archive_tidy_workflow(
    con_or_path: sqlite3.Connection | str | os.PathLike[str],
    *,
    new_event_ids: Iterable[int],
    candidate_event_ids: Iterable[int] = (),
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Run the second-step LLM and persist event links and candidate storylines."""

    owns_connection = not isinstance(con_or_path, sqlite3.Connection)
    con = sqlite3.connect(os.fspath(con_or_path), timeout=30.0) if owns_connection else con_or_path
    try:
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        ensure_preprocessing_schema(con)
        con.commit()

        new_ids = _unique_ints(new_event_ids)
        new_id_set = set(new_ids)
        existing_ids = [value for value in _unique_ints(candidate_event_ids) if value not in new_id_set]
        new_atoms = load_event_atoms(con, new_ids)
        historical_atoms = load_event_atoms(con, existing_ids)

        stats: dict[str, Any] = {
            "tidy_mode": "disabled",
            "new_event_ids": len(new_ids),
            "new_events_loaded": len(new_atoms),
            "candidate_event_ids": len(existing_ids),
            "historical_events_loaded": len(historical_atoms),
            "links_proposed": 0,
            "links_written": 0,
            "candidate_storylines_proposed": 0,
            "candidate_storylines_staged": 0,
            "model_errors": [],
        }
        if not _llm_tidy_enabled() or not new_atoms:
            return stats

        stats["tidy_mode"] = "llm"
        result = _run_llm_tidy(new_atoms, historical_atoms)
        stats["links_proposed"] = len(result.links)
        stats["candidate_storylines_proposed"] = len(result.candidate_storylines)
        stats["model_errors"] = list(result.errors)
        if result.errors and not result.links and not result.candidate_storylines:
            return stats

        new_map = {f"N{index}": atom.event_id for index, atom in enumerate(new_atoms, start=1)}
        historical_map = {f"H{index}": atom.event_id for index, atom in enumerate(historical_atoms, start=1)}
        now = int(now_ms or time.time() * 1000)
        stats["links_written"] = _write_event_links(
            con,
            result.links,
            new_event_ids=new_map,
            historical_event_ids=historical_map,
            now_ms=now,
        )
        stats["candidate_storylines_staged"] = _write_candidate_storylines(
            con,
            result.candidate_storylines,
            new_event_ids=new_map,
            now_ms=now,
        )
        con.commit()
        return stats
    finally:
        if owns_connection:
            con.close()


def load_event_atoms(con: sqlite3.Connection, event_ids: Iterable[int]) -> list[EventAtom]:
    ids = _unique_ints(event_ids)
    if not ids:
        return []
    previous_row_factory = con.row_factory
    try:
        con.row_factory = sqlite3.Row
        placeholders = ",".join("?" * len(ids))
        rows = list(
            con.execute(
                f"SELECT event_id, summary FROM MemoryEvents WHERE event_id IN ({placeholders}) AND is_deleted=0",
                ids,
            )
        )
        entities: dict[int, list[str]] = defaultdict(list)
        for row in con.execute(
            f"""
            SELECT event_id, entity
            FROM MemoryParticipants
            WHERE event_id IN ({placeholders}) AND entity IS NOT NULL AND entity <> ''
            ORDER BY participant_id ASC
            """,
            ids,
        ):
            entities[int(row["event_id"])].append(str(row["entity"]))
    finally:
        con.row_factory = previous_row_factory
    by_id = {
        int(row["event_id"]): EventAtom(
            event_id=int(row["event_id"]),
            summary=str(row["summary"] or ""),
            entities=tuple(dict.fromkeys(entities.get(int(row["event_id"]), ()))),
        )
        for row in rows
    }
    return [by_id[event_id] for event_id in ids if event_id in by_id]


def build_tidy_user_payload(new_atoms: list[EventAtom], historical_atoms: list[EventAtom]) -> str:
    payload = {
        "new_events": [
            {"id": f"N{index}", "summary": atom.summary, "entities": list(atom.entities)}
            for index, atom in enumerate(new_atoms, start=1)
        ],
        "existing_events": [
            {"id": f"H{index}", "summary": atom.summary, "entities": list(atom.entities)}
            for index, atom in enumerate(historical_atoms, start=1)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_tidy_response(
    raw: object,
    *,
    new_ids: set[str],
    historical_ids: set[str],
) -> TidyResult:
    text = str(raw or "")
    text = re.sub(r"<link\s*/>", "<link></link>", text)
    text = re.sub(
        r"<candidate_storyline\s*/>",
        "<candidate_storyline></candidate_storyline>",
        text,
    )
    match = _RESPONSE_RE.fullmatch(text)
    if match is None:
        return TidyResult(errors=("response must exactly match analysis/tidy/link/candidate_storyline structure",))

    link_value, link_error = _parse_json_block(match.group("link"), block="link")
    candidate_storyline_value, candidate_storyline_error = _parse_json_block(
        match.group("candidate_storyline"),
        block="candidate_storyline",
    )
    errors = [error for error in (link_error, candidate_storyline_error) if error]
    if errors:
        return TidyResult(errors=tuple(errors))
    if not isinstance(link_value, list):
        errors.append("link must contain a JSON array")
        link_value = []
    if not isinstance(candidate_storyline_value, list):
        errors.append("candidate_storyline must contain a JSON array")
        candidate_storyline_value = []

    links: list[tuple[str, str]] = []
    seen_links: set[tuple[str, str]] = set()
    for index, item in enumerate(link_value, start=1):
        if not isinstance(item, dict) or set(item) != {"new_event", "existing_event"}:
            errors.append(f"link#{index} must contain exactly new_event and existing_event")
            continue
        new_id = item.get("new_event")
        historical_id = item.get("existing_event")
        if not isinstance(new_id, str) or new_id not in new_ids:
            errors.append(f"link#{index} has unknown new_event {new_id!r}")
            continue
        if not isinstance(historical_id, str) or historical_id not in historical_ids:
            errors.append(f"link#{index} has unknown existing_event {historical_id!r}")
            continue
        key = (new_id, historical_id)
        if key not in seen_links:
            links.append(key)
            seen_links.add(key)

    candidate_storylines: list[tuple[str, ...]] = []
    seen_candidate_storylines: set[tuple[str, ...]] = set()
    for index, item in enumerate(candidate_storyline_value, start=1):
        if not isinstance(item, list):
            errors.append(f"candidate_storyline#{index} must be an array")
            continue
        ids: list[str] = []
        invalid = False
        for value in item:
            if not isinstance(value, str) or value not in new_ids:
                errors.append(f"candidate_storyline#{index} has unknown new event id {value!r}")
                invalid = True
                break
            if value not in ids:
                ids.append(value)
        if invalid:
            continue
        if len(ids) < 2:
            errors.append(f"candidate_storyline#{index} must contain at least two distinct new event ids")
            continue
        key = tuple(sorted(ids, key=_local_id_sort_key))
        if key not in seen_candidate_storylines:
            candidate_storylines.append(key)
            seen_candidate_storylines.add(key)
    return TidyResult(tuple(links), tuple(candidate_storylines), tuple(errors))


def _run_llm_tidy(new_atoms: list[EventAtom], historical_atoms: list[EventAtom]) -> TidyResult:
    try:
        import app_state
    except Exception:
        return TidyResult(errors=("app_state unavailable",))
    adapter = getattr(app_state, "memory_consolidation_adapter", None)
    if adapter is None:
        return TidyResult(errors=("memory_consolidation_adapter unavailable",))
    cfg = _memory_consolidation_cfg()
    gen = dict(cfg.get("generation", {}) if isinstance(cfg.get("generation"), dict) else {})
    gen.setdefault("temperature", 0.2)
    gen.setdefault("max_output_tokens", 4000)
    try:
        raw = adapter.call_simple_text(
            POST_ARCHIVE_TIDY_SYSTEM_PROMPT,
            build_tidy_user_payload(new_atoms, historical_atoms),
            gen,
            log_tag="memory_consolidation/tidy",
        )
    except Exception as exc:
        logger.warning("[tidy_workflow] LLM event tidy call failed: %s", exc)
        return TidyResult(errors=(f"adapter call failed: {exc}",))
    return parse_tidy_response(
        raw,
        new_ids={f"N{index}" for index in range(1, len(new_atoms) + 1)},
        historical_ids={f"H{index}" for index in range(1, len(historical_atoms) + 1)},
    )


def _write_event_links(
    con: sqlite3.Connection,
    links: Iterable[tuple[str, str]],
    *,
    new_event_ids: dict[str, int],
    historical_event_ids: dict[str, int],
    now_ms: int,
) -> int:
    written = 0
    for new_local_id, historical_local_id in links:
        source_id = int(new_event_ids[new_local_id])
        target_id = int(historical_event_ids[historical_local_id])
        cur = con.execute(
            """
            INSERT INTO MemoryRelations (src_event_id, dst_event_id, relation_type, created_at, reason)
            SELECT ?, ?, 'related', ?, 'post_archive_tidy'
            WHERE NOT EXISTS (
                SELECT 1 FROM MemoryRelations
                WHERE src_event_id=? AND dst_event_id=? AND relation_type='related'
            )
            """,
            (source_id, target_id, now_ms, source_id, target_id),
        )
        written += max(0, int(cur.rowcount or 0))
    return written


def _write_candidate_storylines(
    con: sqlite3.Connection,
    candidate_storylines: Iterable[tuple[str, ...]],
    *,
    new_event_ids: dict[str, int],
    now_ms: int,
) -> int:
    rows = []
    for local_ids in candidate_storylines:
        event_ids = tuple(sorted({int(new_event_ids[value]) for value in local_ids}))
        candidate_storyline_id = "candidate_storyline:" + _sha1(
            "post-archive-candidate-storyline",
            *(str(value) for value in event_ids),
        )[:24]
        rows.append((candidate_storyline_id, json.dumps(event_ids), now_ms, now_ms))
    con.executemany(
        """
        INSERT INTO MemoryCandidateStorylines (
            candidate_storyline_id, event_ids_json, status, created_at_ms, updated_at_ms
        ) VALUES (?, ?, 'pending', ?, ?)
        ON CONFLICT(candidate_storyline_id) DO UPDATE SET
            event_ids_json=excluded.event_ids_json,
            status=CASE WHEN MemoryCandidateStorylines.status='pending' THEN 'pending' ELSE MemoryCandidateStorylines.status END,
            updated_at_ms=excluded.updated_at_ms
        """,
        rows,
    )
    return len(rows)


def _parse_json_block(value: str, *, block: str) -> tuple[Any, str]:
    text = str(value or "").strip()
    if not text:
        return [], ""
    try:
        return json.loads(text), ""
    except json.JSONDecodeError as exc:
        return [], f"{block} contains invalid JSON: {exc.msg}"


def _memory_consolidation_cfg() -> dict[str, Any]:
    try:
        import app_state

        cfg = getattr(app_state, "memory_consolidation_cfg", {})
        return dict(cfg) if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _llm_tidy_enabled() -> bool:
    cfg = _memory_consolidation_cfg()
    return bool(cfg.get("enabled", False)) and bool(cfg.get("llm_tidy_enabled", False))


def _local_id_sort_key(value: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", value)
    return (match.group(1), int(match.group(2))) if match else (value, 0)


def _sha1(*parts: str) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _unique_ints(values: Iterable[int]) -> list[int]:
    result: list[int] = []
    for value in values or ():
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item > 0 and item not in result:
            result.append(item)
    return result


__all__ = [
    "EventAtom",
    "TidyResult",
    "build_tidy_user_payload",
    "load_event_atoms",
    "parse_tidy_response",
    "run_post_archive_tidy_workflow",
]
