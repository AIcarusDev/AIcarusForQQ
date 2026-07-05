"""Durable cognition-source identity storage."""

from __future__ import annotations

import hashlib
import json

from cognition_sources_schema import COGNITION_SOURCES_SCHEMA_SQL
from database import _connect, _ms


async def ensure_schema() -> None:
    async with _connect() as db:
        await db.executescript(COGNITION_SOURCES_SCHEMA_SQL)
        await db.commit()


async def upsert_cognition_sources(
    sources: dict[str, dict[str, str]],
    *,
    origin_type: str = "",
    origin_id: str = "",
) -> dict[str, dict[str, str]]:
    """Persist prompt-local cognition blocks and attach durable source uids."""

    now = _ms()
    out: dict[str, dict[str, str]] = {}
    async with _connect() as db:
        await db.executescript(COGNITION_SOURCES_SCHEMA_SQL)
        for prompt_source_id, raw_meta in sources.items():
            prompt_id = str(prompt_source_id or "").strip()
            if not prompt_id:
                continue
            meta = dict(raw_meta or {})
            timestamp = str(meta.get("timestamp") or "")
            cognition_text = str(meta.get("text") or "")
            cognition_hash = _source_hash(cognition_text)
            source_uid = cognition_source_uid(
                origin_type=origin_type,
                origin_id=origin_id,
                timestamp=timestamp,
                cognition_hash=cognition_hash,
            )
            source_seq = _source_seq(prompt_id)
            metadata_json = json.dumps(
                {
                    "prompt_source_id": prompt_id,
                    "origin_type": origin_type,
                    "origin_id": origin_id,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            await db.execute(
                """
                INSERT INTO CognitionSources (
                    source_uid, source_kind, origin_type, origin_id, prompt_source_id,
                    source_seq, source_timestamp, cognition_text, cognition_hash,
                    created_at, last_seen_at, metadata_json
                ) VALUES (?, 'cognition', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_uid) DO UPDATE SET
                    last_seen_at=excluded.last_seen_at,
                    prompt_source_id=excluded.prompt_source_id,
                    source_seq=excluded.source_seq,
                    source_timestamp=excluded.source_timestamp,
                    cognition_text=excluded.cognition_text,
                    cognition_hash=excluded.cognition_hash,
                    metadata_json=excluded.metadata_json
                """,
                (
                    source_uid,
                    origin_type,
                    origin_id,
                    prompt_id,
                    source_seq,
                    timestamp,
                    cognition_text,
                    cognition_hash,
                    now,
                    now,
                    metadata_json,
                ),
            )
            meta["source_uid"] = source_uid
            meta["prompt_source_id"] = prompt_id
            meta["timestamp"] = timestamp
            out[prompt_id] = meta
        await db.commit()
    return out


def cognition_source_uid(
    *,
    origin_type: str,
    origin_id: str,
    timestamp: str,
    cognition_hash: str,
) -> str:
    payload = json.dumps(
        {
            "v": 1,
            "kind": "cognition",
            "origin_type": str(origin_type or ""),
            "origin_id": str(origin_id or ""),
            "timestamp": str(timestamp or ""),
            "cognition_hash": str(cognition_hash or ""),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "cog_" + _source_hash(payload)[:32]


def _source_seq(source_id: str) -> int | None:
    try:
        return int(str(source_id).strip())
    except (TypeError, ValueError):
        return None


def _source_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()
