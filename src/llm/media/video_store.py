"""Lightweight video reference storage and on-demand download interface.

Videos share the same ref namespace as images (generated via generate_time_ref).
Video bytes are not downloaded eagerly and are not stored in SQLite tables;
they are fetched on-demand to the local filesystem (MEDIA_ROOT) when requested.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any

from .media_identity import bind_media_identity
from .media_storage import (
    MEDIA_ROOT,
    get_media_dir,
    locate_media_file,
    parse_time_ref_date,
)

logger = logging.getLogger("AICQ.video_store")


def locate_video(video_ref: str) -> Path | None:
    """Locate an existing local video file for a video_ref, if downloaded."""
    ref = str(video_ref or "").strip()
    if not ref:
        return None
    path = locate_media_file(ref)
    if path is not None and path.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv", ".avi", ".flv"}:
        return path
    for ext in (".mp4", ".webm", ".mov", ".mkv", ".avi", ".flv"):
        candidate = MEDIA_ROOT / f"{ref}{ext}"
        if candidate.is_file():
            return candidate
    return None


async def download_video_for_ref(
    video_ref: str,
    url: str,
    *,
    timeout: float = 60.0,
) -> Path | None:
    """Download a video on-demand to the local media directory for the given video_ref.

    Uses stream writing to keep memory overhead low and does_not store video blobs in SQLite.
    """
    ref = str(video_ref or "").strip()
    src_url = str(url or "").strip()
    if not ref or not src_url:
        return None

    existing = locate_video(ref)
    if existing is not None:
        return existing

    import httpx

    date_parts = parse_time_ref_date(ref)
    target_dir = get_media_dir(date_parts[0], date_parts[1]) if date_parts else MEDIA_ROOT
    target_path = target_dir / f"{ref}.mp4"
    temp_path = target_dir / f"{ref}.mp4.tmp"
    hasher = hashlib.sha256()

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", src_url) as resp:
                if resp.status_code != 200:
                    logger.warning("[video_store] Download failed for ref=%s url=%s status=%s", ref, src_url, resp.status_code)
                    return None
                with open(temp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        f.write(chunk)
                        hasher.update(chunk)

        temp_path.replace(target_path)
        digest = hasher.hexdigest()
        bind_media_identity(ref, digest)
        logger.info("[video_store] Successfully downloaded video ref=%s size=%d sha256=%s", ref, target_path.stat().st_size, digest)
        return target_path
    except Exception as exc:
        logger.warning("[video_store] Error downloading video ref=%s from url=%s: %s", ref, src_url, exc)
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        return None
