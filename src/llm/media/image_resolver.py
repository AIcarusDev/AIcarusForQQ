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

        # 收藏图片先检查内容完整性。
        from .sticker_collection import StickerCollectionError, get_sticker_image
        try:
            sticker = get_sticker_image(normalized_ref)
        except StickerCollectionError as exc:
            return {"unavailable_status": exc.code}, "sticker"
        if sticker is not None:
            return sticker, "sticker"

        # Visible entries own mutable status and vision metadata. Return the
        # original object so examine_image writes back to the active message.
        if self.session is not None:
            for entry in getattr(self.session, "context_messages", []) or []:
                if (image := image_from_entry(entry, normalized_ref)) is not None:
                    return self._visible_result(normalized_ref, image, "chat")
            if getattr(self.session, "is_browsing_history", lambda: False)():
                view = getattr(self.session, "chat_window_view", {}) or {}
                if top_db_id := view.get("top_db_id"):
                    try:
                        for entry in self.history_loader(self.session, int(top_db_id), int(view.get("page_size") or 10)):
                            if (image := image_from_entry(entry, normalized_ref)) is not None:
                                return self._visible_result(normalized_ref, image, "history")
                    except Exception:
                        logger.debug("[image_resolver] History lookup failed", exc_info=True)
            for entry in visible_forward_entries(self.session):
                if (image := image_from_entry(entry, normalized_ref)) is not None:
                    return self._visible_result(normalized_ref, image, "forward")

        # L1 跨会话缓存用于当前窗口之外的图片。
        try:
            from .media_cache import get_recent_image, cache_recent_image
            if cached := get_recent_image(normalized_ref):
                return cached
        except Exception:
            pass

        # 时序物理磁盘文件直读 (data/media/YYYY/MM/)
        try:
            from .media_storage import read_media_bytes
            if media_res := read_media_bytes(normalized_ref):
                raw_bytes, inferred_mime = media_res
                img_dict = {"data": raw_bytes, "mime": inferred_mime}
                from database import lookup_media_ref_sync
                record = lookup_media_ref_sync(normalized_ref)
                source = str(record.get("source_type") or "chat") if record else "chat"
                try:
                    from .media_cache import cache_recent_image
                    cache_recent_image(normalized_ref, img_dict, source=source)
                except Exception:
                    pass
                return img_dict, source
        except Exception:
            pass

        # 浏览器临时图片
        if include_browser:
            try:
                browser_image = self.browser_image_reader(normalized_ref)
            except Exception:
                browser_image = None
            if browser_image is not None:
                raw, mime = browser_image
                img_dict = {"data": raw, "mime": mime or "image/jpeg"}
                try:
                    from .media_cache import cache_recent_image
                    cache_recent_image(normalized_ref, img_dict, source="browser")
                except Exception:
                    pass
                return img_dict, "browser"

        # L2 SQLite media_registry 点查冷数据
        try:
            from database import lookup_media_ref_sync, load_chat_image_payload_sync
            record = lookup_media_ref_sync(normalized_ref)
            if record:
                stype = record.get("source_type")
                locator = record.get("locator", "")
                mime = record.get("mime", "image/jpeg")
                from pathlib import Path
                path = Path(locator)
                if path.is_file():
                    try:
                        raw = path.read_bytes()
                        img_dict = {"data": raw, "mime": mime}
                        try:
                            from .media_cache import cache_recent_image
                            cache_recent_image(normalized_ref, img_dict, source=stype)
                        except Exception:
                            pass
                        return img_dict, stype
                    except OSError:
                        logger.debug("[image_resolver] 读取文件失败: %s", locator, exc_info=True)
                elif stype == "chat":
                    parts = locator.split("::", 1)
                    if len(parts) == 2:
                        s_key, m_id = parts
                        payload = load_chat_image_payload_sync(s_key, m_id, normalized_ref)
                        if payload and isinstance(payload, dict):
                            try:
                                from .media_cache import cache_recent_image
                                cache_recent_image(normalized_ref, payload, source="chat")
                            except Exception:
                                pass
                            return payload, "chat"
        except Exception:
            logger.debug("[image_resolver] L2 media_registry 查询异常", exc_info=True)

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
