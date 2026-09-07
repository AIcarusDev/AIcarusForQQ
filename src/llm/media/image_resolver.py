"""Resolve image references from the model-visible world without side effects."""

from __future__ import annotations

import base64
import binascii
import io
import logging
import re
import warnings
from dataclasses import dataclass
from typing import Any, Callable

from browser.session import read_browser_image_file
from platforms.chat.history_window import load_history_window
from PIL import Image, UnidentifiedImageError


logger = logging.getLogger("AICQ.tools")

HistoryLoader = Callable[[Any, int, int], list[dict[str, Any]]]
BrowserImageReader = Callable[[str], tuple[bytes, str] | None]

_FORMAT_MIME = {
    "AVIF": "image/avif",
    "BMP": "image/bmp",
    "GIF": "image/gif",
    "ICO": "image/x-icon",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class ImagePayloadError(ValueError):
    def __init__(self, code: str, *, details: dict[str, int | str] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True, slots=True)
class ImagePayloadInfo:
    mime_type: str
    width: int
    height: int
    frame_count: int


def inspect_image_payload(
    raw: bytes,
    *,
    max_bytes: int,
    max_pixels: int,
) -> ImagePayloadInfo:
    """Validate raster bytes and return content-derived image metadata."""

    if not raw:
        raise ImagePayloadError("empty_image")
    if len(raw) > max_bytes:
        raise ImagePayloadError(
            "image_too_large",
            details={"size_bytes": len(raw), "limit_bytes": max_bytes},
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                detected_format = str(image.format or "").upper()
                mime_type = _FORMAT_MIME.get(detected_format, "")
                width, height = int(image.width), int(image.height)
                frame_count = int(getattr(image, "n_frames", 1) or 1)
                image.verify()
    except ImagePayloadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImagePayloadError("invalid_image") from exc
    if not mime_type:
        raise ImagePayloadError(
            "unsupported_image_format",
            details={"detected_format": detected_format or "unknown"},
        )
    if width <= 0 or height <= 0 or width * height > max_pixels:
        raise ImagePayloadError(
            "image_dimensions_exceeded",
            details={"width": width, "height": height, "limit_pixels": max_pixels},
        )
    return ImagePayloadInfo(
        mime_type=mime_type,
        width=width,
        height=height,
        frame_count=frame_count,
    )


class ImageResolver:
    """Resolve an ``image_ref`` via fast L1 memory cache and reliable L2 SQLite media_registry."""

    def __init__(
        self,
        session: Any = None,
        *,
        history_loader: HistoryLoader | None = None,
        browser_image_reader: BrowserImageReader | None = None,
    ) -> None:
        self.session = session
        self.history_loader = history_loader or load_history_window
        self.browser_image_reader = browser_image_reader or read_browser_image_file

    def resolve(self, image_ref: object, *, include_browser: bool = True) -> tuple[dict[str, Any], str] | None:
        """Return the first resolved image and its source, or ``None``."""

        normalized_ref = normalize_image_ref(image_ref)
        if not normalized_ref:
            return None

        from .image_store import read_image
        record = read_image(normalized_ref)
        if record is not None:
            return record, str(record.get("source_type") or "media")
        return None

    @staticmethod
    def _visible_result(image_ref: str, image: dict[str, Any], source: str) -> tuple[dict[str, Any], str]:
        from .media_cache import cache_recent_image

        cache_recent_image(image_ref, image, source=source)
        return image, source

    @staticmethod
    def payload(image: dict[str, Any]) -> tuple[str | bytes, str] | None:
        return image_payload(image)

    @staticmethod
    def unavailable_status(image: dict[str, Any]) -> str:
        return image_unavailable_status(image)


def normalize_image_ref(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"\b(?:image_ref|ref)\s*=\s*['\"]([^'\"]+)['\"]", text)
    if match:
        text = match.group(1).strip()
    return text.strip("`'\"[] ")


def visible_forward_entries(session: Any) -> list[dict[str, Any]]:
    stack = getattr(session, "forward_browser_stack", None) or []
    if not stack:
        return []
    frame = stack[-1] or {}
    nodes = [node for node in (frame.get("nodes") or []) if isinstance(node, dict)]
    try:
        page_size = int(frame.get("page_size") or 8)
        page_offset = max(0, int(frame.get("page_offset") or 0))
    except (TypeError, ValueError):
        page_size = 8
        page_offset = 0
    return nodes[page_offset:page_offset + page_size]


def image_from_entry(entry: dict[str, Any], image_ref: str) -> dict[str, Any] | None:
    images = entry.get("images") or {}
    if isinstance(images, dict):
        image = images.get(image_ref)
        return image if isinstance(image, dict) else None
    if isinstance(images, list):
        for image in images:
            if isinstance(image, dict) and str(image.get("image_ref") or image.get("ref") or "") == image_ref:
                return image
    return None


def image_payload(image: dict[str, Any]) -> tuple[str | bytes, str] | None:
    if image.get("image_ref"):
        from .image_store import read_image
        current = read_image(image["image_ref"])
        if current is None or current.get("unavailable_status"):
            return None
        return current["data"], current["mime"]
    if image.get("unavailable_status"):
        return None
    mime = str(image.get("mime") or image.get("mime_type") or "image/jpeg")
    data = image.get("data")
    if isinstance(data, bytes):
        return data, mime
    if isinstance(data, str) and data:
        return data, mime

    file_path = image.get("file_path")
    if file_path:
        from pathlib import Path
        p = Path(str(file_path))
        if p.is_file():
            try:
                return p.read_bytes(), mime
            except OSError:
                pass

    b64 = image.get("base64")
    if isinstance(b64, str) and b64:
        try:
            base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            return None
        return b64, mime

    return None


def image_unavailable_status(image: dict[str, Any]) -> str:
    if status := image.get("unavailable_status"):
        return str(status)
    for key in ("pending", "expired", "failed"):
        if image.get(key):
            return key
    b64 = image.get("base64")
    if isinstance(b64, str) and b64:
        try:
            base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            return "invalid_image_data"
    return "unavailable"


def image_bytes(image: dict[str, Any]) -> tuple[bytes, str] | None:
    """Decode the shared payload without transforming the original image bytes."""
    payload = image_payload(image)
    if payload is None:
        return None
    data, mime = payload
    if isinstance(data, bytes):
        return data, mime
    try:
        return base64.b64decode(data, validate=True), mime
    except (binascii.Error, ValueError):
        return None
