"""Durable reference reservations and immutable content bindings (metadata only).

This ledger has its own SQLite transaction, independent of chat persistence.
Reservations survive crashes and deletion: a published ref is never recycled.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class MediaRefConflict(ValueError):
    """An existing reference cannot be rebound to different bytes."""


class MediaIdentityUnavailable(RuntimeError):
    """Identity persistence failed; callers must not publish an unbound payload."""


@contextmanager
def _ledger():
    from . import media_storage as storage

    root = storage.MEDIA_ROOT
    conn = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        path = root / "references.sqlite3"
        if not storage._inside_media_root(path):
            raise MediaIdentityUnavailable("Reference ledger is outside media root")
        conn = sqlite3.connect(path, timeout=30)
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("CREATE TABLE IF NOT EXISTS refs (ref TEXT PRIMARY KEY, sha256 TEXT, locator TEXT)")
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except BaseException as exc:
        if conn is not None:
            conn.rollback()
        if isinstance(exc, Exception) and not isinstance(exc, (MediaRefConflict, MediaIdentityUnavailable)):
            raise MediaIdentityUnavailable("Media identity persistence failed") from exc
        raise
    finally:
        if conn is not None:
            conn.close()


def _historical_binding(ref: str) -> tuple[bool, set[str]]:
    """Consult pre-ledger identities without migrating or editing old stores."""
    import database
    from . import media_storage as storage, sticker_collection as stickers

    occupied = False
    digests: set[str] = set()
    path = storage.locate_media_file(ref)
    if path is not None:
        occupied = True
        digests.add(hashlib.sha256(path.read_bytes()).hexdigest())
    if stickers._IMAGES_DIR.is_dir():
        for candidate in stickers._IMAGES_DIR.glob(f"{ref}.*"):
            if candidate.suffix.lower() in stickers._VALID_EXTENSIONS and candidate.is_file():
                path = stickers._image_path(candidate.name)
                occupied = True
                digests.add(hashlib.sha256(path.read_bytes()).hexdigest())
    if stickers._INDEX_PATH.is_file():
        document = json.loads(stickers._INDEX_PATH.read_text(encoding="utf-8"))
        bindings = document.get("ref_hashes", {})
        if ref in bindings:
            occupied = True
            digests.add(bindings[ref])
    db_path = Path(database.DB_PATH)
    if db_path.is_file():
        conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
        try:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='media_registry'").fetchone():
                row = conn.execute(
                    "SELECT sha256, source_type, locator FROM media_registry WHERE image_ref=?", (ref,),
                ).fetchone()
                if row:
                    occupied = True
                    digest, source, locator = row
                    if digest:
                        digests.add(digest)
                    if source == "chat" and "::" in locator:
                        session, message = locator.split("::", 1)
                        payload_row = conn.execute(
                            "SELECT images FROM chat_messages WHERE session_key=? AND message_id=? LIMIT 1",
                            (session, message),
                        ).fetchone()
                        if payload_row:
                            for candidate, info in database._media_payload_items(json.loads(payload_row[0])):
                                if candidate == ref:
                                    if info.get("base64"):
                                        digests.add(hashlib.sha256(base64.b64decode(info["base64"], validate=True)).hexdigest())
                                    elif info.get("file_path") and Path(info["file_path"]).is_file():
                                        digests.add(hashlib.sha256(Path(info["file_path"]).read_bytes()).hexdigest())
                    elif locator and Path(locator).is_file():
                        digests.add(hashlib.sha256(Path(locator).read_bytes()).hexdigest())
        finally:
            conn.close()
    return occupied, digests


def reserve_time_ref(dt: datetime | None = None, *, initial_length: int = 5) -> str:
    """Reserve before publishing: eight attempts per length, at most 12 hex digits."""
    if not 4 <= initial_length <= 12:
        raise ValueError("Reference suffix must start at 4..12 digits")
    prefix = (dt or datetime.now(timezone.utc)).strftime("%y%m%d")
    with _ledger() as conn:
        for length in range(initial_length, 13):
            for _ in range(8):
                ref = f"{prefix}_{uuid.uuid4().hex[:length]}"
                if conn.execute("SELECT 1 FROM refs WHERE ref=?", (ref,)).fetchone():
                    continue
                occupied, _ = _historical_binding(ref)
                if occupied:
                    conn.execute("INSERT INTO refs(ref) VALUES (?)", (ref,))
                    continue
                conn.execute("INSERT INTO refs(ref) VALUES (?)", (ref,))
                return ref
    raise MediaRefConflict("Media reference allocation exhausted")


def bind_media_identity(ref: str, digest: str | None) -> None:
    """First payload binds the reservation; later retries must match its digest."""
    from .media_storage import _SAFE_REF_PATTERN

    if not _SAFE_REF_PATTERN.fullmatch(ref):
        raise ValueError("Invalid media reference")
    with _ledger() as conn:
        row = conn.execute("SELECT sha256 FROM refs WHERE ref=?", (ref,)).fetchone()
        # Bound ledger entries remain authoritative even if backing files vanish.
        if row and row[0]:
            if digest and row[0] != digest:
                raise MediaRefConflict("Media reference is bound to different content")
            return
        _, old_digests = _historical_binding(ref)
        if len(old_digests) > 1 or (digest and old_digests and digest not in old_digests):
            raise MediaRefConflict("Historical media reference has conflicting content")
        bound = digest or next(iter(old_digests), None)
        conn.execute(
            "INSERT INTO refs(ref, sha256) VALUES (?, ?) ON CONFLICT(ref) DO UPDATE SET sha256=excluded.sha256",
            (ref, bound),
        )


def claim_media_path(ref: str, suggested: Path) -> Path:
    """Select one filename per reference, even across MIME types and processes."""
    from .media_storage import MEDIA_ROOT, _inside_media_root

    with _ledger() as conn:
        row = conn.execute("SELECT locator FROM refs WHERE ref=?", (ref,)).fetchone()
        if row is None:
            raise MediaRefConflict("Media reference has not been bound")
        path = MEDIA_ROOT / row[0] if row[0] else suggested
        if not _inside_media_root(path):
            raise ValueError("Media file is outside media root")
        if not row[0]:
            conn.execute("UPDATE refs SET locator=? WHERE ref=?", (path.relative_to(MEDIA_ROOT).as_posix(), ref))
        return path
