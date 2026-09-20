"""Durable reference reservations and immutable content bindings (metadata only).

This ledger has its own SQLite transaction, independent of chat persistence.
Reservations survive crashes and deletion: a published ref is never recycled.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path


class MediaRefConflict(ValueError):
    """An existing reference cannot be rebound to different bytes."""


class MediaIdentityUnavailable(RuntimeError):
    """Identity persistence failed; callers must not publish an unbound payload."""


def _historical_binding(ref: str) -> tuple[bool, set[str]]:
    """Consult pre-ledger identities without migrating or editing old stores."""
    import database
    from . import media_storage as storage, sticker_collection as stickers

    db_path = Path(database.DB_PATH)
    if db_path.is_file():
        with sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='media_state'").fetchone():
                if db.execute("SELECT 1 FROM media_state WHERE key='legacy_import_verified'").fetchone():
                    return False, set()

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
        if ref in document.get('stickers', {}):
            occupied = True
        if ref in bindings:
            occupied = True
            digests.add(bindings[ref])
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
    from .image_store import reserve_ref
    return reserve_ref(dt, initial_length=initial_length)


def bind_media_identity(ref: str, digest: str | None) -> None:
    from .image_store import bind_ref
    bind_ref(ref, digest)
