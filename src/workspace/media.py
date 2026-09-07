"""Workspace media ref registration and utilities."""

from __future__ import annotations

import mimetypes
from pathlib import Path


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


async def register_workspace_bytes(
    raw: bytes,
    *,
    mime: str | None = None,
    image_ref: str | None = None,
) -> str:
    """Register image bytes from the workspace into media_registry and L1 cache.

    Deduplicates against existing media_registry records by sha256.
    Returns the assigned `image_ref`.
    """
    from llm.media.image_store import register_image
    import asyncio
    record = await asyncio.to_thread(register_image, raw, "workspace", image_ref)
    return record["image_ref"]


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
    inferred_mime = mime or _EXT_TO_MIME.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return await register_workspace_bytes(raw, mime=inferred_mime, image_ref=image_ref)


__all__ = ["register_workspace_bytes", "register_workspace_image"]
