"""Immutable media files and reserved time references (5 hex digits by default)."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("AICQ.media_storage")

MEDIA_ROOT = Path(__file__).resolve().parents[3] / "data" / "media"
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
    from .image_store import register_image
    record = register_image(raw, "chat", image_ref, dt=dt)
    return record["image_ref"], Path(record["locator"])


def read_media_bytes(image_ref: str) -> tuple[bytes, str] | None:
    """Fast read media bytes and infer MIME for image_ref."""
    from .image_store import read_image
    record = read_image(image_ref)
    if record and not record.get("unavailable_status"):
        return record["data"], record["mime"]
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
