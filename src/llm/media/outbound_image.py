"""Normalize images before sending them to OpenAI-compatible vision APIs."""

from __future__ import annotations

import base64
import binascii
import io
import logging

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger("AICQ.llm.media.outbound_image")

def _clean_mime(mime: str) -> str:
    mime = (mime or "image/jpeg").split(";", 1)[0].strip().lower()
    return "image/jpeg" if mime == "image/jpg" else mime


def _decode_base64(b64: str) -> bytes | None:
    try:
        return base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.warning("[outbound_image] base64 解码失败: %s", exc)
        return None


def _open_image(raw: bytes) -> Image.Image | None:
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            copied = img.copy()
            copied.info.update(img.info)
            copied.format = img.format
            return copied
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("[outbound_image] 图片校验失败: %s", exc)
        return None


def _mime_from_format(fmt: str | None) -> str | None:
    fmt = (fmt or "").upper()
    if fmt in {"JPEG", "JPG"}:
        return "image/jpeg"
    if fmt == "PNG":
        return "image/png"
    return None


def _has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )


def normalize_for_openai_compatible(
    b64: str,
    mime: str,
) -> tuple[str, str] | None:
    """Return ``(base64, mime)`` safe for broad OpenAI-compatible vision APIs.

    JPEG and PNG are passed through after validation. Other decodable formats
    such as WebP/GIF are converted to PNG when they need alpha, otherwise JPEG.
    """
    raw = _decode_base64(b64)
    if raw is None:
        return None

    mime = _clean_mime(mime)
    img = _open_image(raw)
    if img is None:
        return None

    actual_mime = _mime_from_format(img.format)
    if actual_mime in {"image/jpeg", "image/png"}:
        return base64.b64encode(raw).decode("ascii"), actual_mime

    try:
        output = io.BytesIO()
        if _has_alpha(img):
            img.convert("RGBA").save(output, format="PNG")
            out_mime = "image/png"
        else:
            img.convert("RGB").save(output, format="JPEG", quality=92)
            out_mime = "image/jpeg"
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning(
            "[outbound_image] 图片格式转换失败 mime=%s: %s", mime, exc
        )
        return None

    return base64.b64encode(output.getvalue()).decode("ascii"), out_mime


def make_data_url(b64: str, mime: str) -> str | None:
    """Build a data URL after normalizing the image payload."""
    normalized = normalize_for_openai_compatible(b64, mime)
    if normalized is None:
        return None
    out_b64, out_mime = normalized
    return f"data:{out_mime};base64,{out_b64}"
