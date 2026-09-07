"""Immutable media files and reserved time references (5 hex digits by default)."""

from __future__ import annotations

import logging
import hashlib
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("AICQ.media_storage")

MEDIA_ROOT = Path("data/media")
_SAFE_REF_PATTERN = re.compile(r"[A-Za-z0-9_-]{4,128}\Z")


def _inside_media_root(path: Path) -> bool:
    try:
        return path.resolve().is_relative_to(MEDIA_ROOT.resolve())
    except (OSError, RuntimeError, ValueError):
        return False

_TIME_REF_PATTERN = re.compile(r"^(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})_(?P<rand>[0-9a-fA-F]{4,12})$")

_MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/avif": ".avif",
    "image/x-icon": ".ico",
}

_EXT_TO_MIME = {ext: mime for mime, ext in _MIME_TO_EXT.items()}
_EXT_TO_MIME[".jpeg"] = "image/jpeg"


def generate_time_ref(dt: datetime | None = None) -> str:
    """Durably reserve a short time reference before exposing it to a caller."""
    from .media_identity import reserve_time_ref
    return reserve_time_ref(dt)


def parse_time_ref_date(image_ref: str) -> tuple[int, int, int] | None:
    """Parse (YYYY, MM, DD) from a time-sorted image_ref, or None if legacy format."""
    match = _TIME_REF_PATTERN.match(str(image_ref or "").strip())
    if not match:
        return None
    yy = int(match.group("yy"))
    mm = int(match.group("mm"))
    dd = int(match.group("dd"))
    year = 2000 + yy if yy < 70 else 1900 + yy
    return year, mm, dd


def get_media_dir(year: int, month: int) -> Path:
    """Return data/media/YYYY/MM directory Path."""
    return MEDIA_ROOT / f"{year:04d}" / f"{month:02d}"


def locate_media_file(image_ref: str) -> Path | None:
    """Locate existing disk file for image_ref without knowing the extension in advance."""
    ref = str(image_ref or "").strip()
    if not _SAFE_REF_PATTERN.fullmatch(ref):
        return None

    date_parts = parse_time_ref_date(ref)
    if date_parts:
        year, month, _ = date_parts
        folder = get_media_dir(year, month)
        if folder.is_dir():
            for candidate in folder.glob(f"{ref}.*"):
                if candidate.suffix.lower() in _EXT_TO_MIME and _inside_media_root(candidate) and candidate.is_file():
                    return candidate
            return None

    # Fallback search in data/media directly or across subdirectories if needed
    if MEDIA_ROOT.is_dir():
        for candidate in MEDIA_ROOT.glob(f"**/{ref}.*"):
            if candidate.suffix.lower() in _EXT_TO_MIME and _inside_media_root(candidate) and candidate.is_file():
                return candidate

    return None


def save_media_bytes(
    raw: bytes,
    mime: str = "image/jpeg",
    *,
    image_ref: str | None = None,
    dt: datetime | None = None,
) -> tuple[str, Path]:
    """Atomically save raw media bytes into data/media/YYYY/MM/<image_ref>.<ext>.

    Returns (image_ref, absolute_or_relative_path).
    """
    ref = str(image_ref or "").strip()
    target_dt = dt or datetime.now(timezone.utc)
    if not ref:
        ref = generate_time_ref(target_dt)
    if not _SAFE_REF_PATTERN.fullmatch(ref):
        raise ValueError("Invalid media reference")
    from .media_identity import bind_media_identity, MediaRefConflict
    bind_media_identity(ref, hashlib.sha256(raw).hexdigest())
    existing = locate_media_file(ref)
    if existing is not None:
        if existing.read_bytes() != raw:
            raise MediaRefConflict("Media file contains different content")
        return ref, existing

    date_parts = parse_time_ref_date(ref)
    if date_parts:
        year, month, _ = date_parts
    else:
        year, month = target_dt.year, target_dt.month

    target_dir = get_media_dir(year, month)
    if not _inside_media_root(target_dir):
        raise ValueError("Media directory is outside media root")
    target_dir.mkdir(parents=True, exist_ok=True)

    normalized_mime = str(mime or "image/jpeg").split(";", 1)[0].strip().lower()
    ext = _MIME_TO_EXT.get(normalized_mime, ".jpg")
    from .media_identity import claim_media_path
    dest_path = claim_media_path(ref, target_dir / f"{ref}{ext}")
    if not _inside_media_root(dest_path):
        raise ValueError("Media file is outside media root")
    target_dir = dest_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    # Atomic write
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target_dir, prefix=f".{ref}-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, dest_path)
        except FileExistsError:
            if dest_path.read_bytes() != raw:
                raise MediaRefConflict("Media file contains different content")
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)

    return ref, dest_path


def read_media_bytes(image_ref: str) -> tuple[bytes, str] | None:
    """Fast read media bytes and infer MIME for image_ref."""
    path = locate_media_file(image_ref)
    if path is None or not path.is_file():
        return None
    try:
        raw = path.read_bytes()
        mime = _EXT_TO_MIME.get(path.suffix.lower(), "image/jpeg")
        return raw, mime
    except OSError:
        logger.debug("Failed to read media file %s", path, exc_info=True)
        return None


__all__ = [
    "MEDIA_ROOT",
    "generate_time_ref",
    "get_media_dir",
    "locate_media_file",
    "parse_time_ref_date",
    "read_media_bytes",
    "save_media_bytes",
]
