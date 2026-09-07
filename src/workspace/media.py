"""Workspace media ref registration and utilities."""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

from database import register_media_ref
from llm.media.media_cache import cache_recent_image

_EXT_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".avif": "image/avif",
    ".ico": "image/x-icon",
}


async def register_workspace_image(
    file_path: str | Path,
    *,
    image_ref: str | None = None,
    mime: str | None = None,
) -> str:
    """Register an image from the workspace/filesystem into media_registry and L1 cache.

    Returns the assigned `image_ref`.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Workspace image file not found: {file_path}")

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    inferred_mime = mime or _EXT_TO_MIME.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "image/jpeg"

    # Bind an immutable copy; editing the workspace original must not change ref.
    from llm.media.media_storage import save_media_bytes
    ref, stored_path = save_media_bytes(raw, mime=inferred_mime, image_ref=image_ref)

    # Register into L2 SQLite media_registry
    await register_media_ref(
        image_ref=ref,
        source_type="workspace",
        locator=str(stored_path),
        mime=inferred_mime,
        sha256=digest,
    )

    # Register into L1 memory cache
    cache_recent_image(ref, {"data": raw, "mime": inferred_mime}, source="workspace")

    return ref


__all__ = ["register_workspace_image"]
