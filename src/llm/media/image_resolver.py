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
    """Resolve an ``image_ref`` using the same visibility order as the world view."""

    def __init__(
        self,
        session: Any,
        *,
        history_loader: HistoryLoader | None = None,
        browser_image_reader: BrowserImageReader | None = None,
    ) -> None:
        self.session = session
        self.history_loader = history_loader or load_history_window
        self.browser_image_reader = browser_image_reader or read_browser_image_file

    def resolve(self, image_ref: object) -> tuple[dict[str, Any], str] | None:
        """Return the first visible image and its source, or ``None``."""

        normalized_ref = normalize_image_ref(image_ref)
        if not normalized_ref:
            return None

        for entry in getattr(self.session, "context_messages", []) or []:
            if image := image_from_entry(entry, normalized_ref):
                return image, "chat"

        if getattr(self.session, "is_browsing_history", lambda: False)():
            view = getattr(self.session, "chat_window_view", {}) or {}
            top_db_id = view.get("top_db_id")
            if top_db_id:
                try:
                    page_size = int(view.get("page_size") or 10)
                    for entry in self.history_loader(self.session, int(top_db_id), page_size):
                        if image := image_from_entry(entry, normalized_ref):
                            return image, "history"
                except Exception:
                    logger.debug("[tools] view_image: 历史窗口查找失败", exc_info=True)

        for entry in visible_forward_entries(self.session):
            if image := image_from_entry(entry, normalized_ref):
                return image, "forward"

        try:
            browser_image = self.browser_image_reader(normalized_ref)
        except Exception:
            logger.debug("[tools] view_image: browser 图片查找失败", exc_info=True)
            browser_image = None
        if browser_image is not None:
            raw, mime = browser_image
            return {"data": raw, "mime": mime or "image/jpeg"}, "browser"

        return None

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
    mime = str(image.get("mime") or image.get("mime_type") or "image/jpeg")
    data = image.get("data")
    if isinstance(data, bytes):
        return data, mime
    if isinstance(data, str) and data:
        return data, mime

    b64 = image.get("base64")
    if isinstance(b64, str) and b64:
        try:
            base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            return None
        return b64, mime

    phash = image.get("phash")
    if phash:
        try:
            from llm.media.image_cache import read_image_b64

            cached = read_image_b64(str(phash))
        except Exception:
            logger.debug("[tools] view_image: cache 读取失败 phash=%s", phash, exc_info=True)
            cached = None
        if cached:
            return cached
    return None


def image_unavailable_status(image: dict[str, Any]) -> str:
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
