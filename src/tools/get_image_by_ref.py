"""Return a visible world image as a real multimodal attachment by ref."""

from __future__ import annotations

import base64
import binascii
import logging
import re
from typing import Any, Callable

from browser_adapter.session import read_browser_image_file
from llm.prompt.history_window import load_history_window

logger = logging.getLogger("AICQ.tools")

DECLARATION: dict = {
    "name": "get_image_by_ref",
    "description": (
        "在 <world> 中，为了节省上下文和注意力，并不是每一个图片多模态信息都会展现。有一些可能只可见 ref，而没有内嵌真正的图片内容。"
        "如果需要查看这些图片，可以使用这个工具，写入 ref，返回真实图片。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_ref": {
                "type": "string",
                "description": "目标图片的 ref，来自 <world> 中的 ref 标注。",
            },
        },
        "required": ["image_ref"],
    },
}

REQUIRES_CONTEXT: list[str] = ["session"]


def condition(config: dict) -> bool:
    return bool(config.get("vision", True))


def repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(args, dict):
        return args, []
    if args.get("image_ref") or "ref" not in args:
        return args, []
    repaired = dict(args)
    repaired["image_ref"] = repaired.pop("ref")
    return repaired, ["ref -> image_ref"]


def sanitize_semantic_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str], str | None]:
    image_ref = _normalize_image_ref(args.get("image_ref"))
    repaired = dict(args)
    changes: list[str] = []
    if image_ref != args.get("image_ref"):
        repaired["image_ref"] = image_ref
        changes.append("image_ref: normalized")
    if not image_ref:
        return repaired, changes, "image_ref is empty"
    return repaired, changes, None


def make_handler(session: Any) -> Callable:
    def handler(image_ref: str, **_: Any) -> dict:
        normalized_ref = _normalize_image_ref(image_ref)
        if not normalized_ref:
            return {"ok": False, "status": "invalid_ref", "image_ref": ""}

        found = _find_world_image(session, normalized_ref)
        if found is None:
            logger.info("[tools] get_image_by_ref: 未找到 ref=%s", normalized_ref)
            return {
                "ok": False,
                "status": "not_found",
                "image_ref": normalized_ref,
            }

        image, source = found
        payload = _image_payload(image)
        if payload is None:
            status = _image_unavailable_status(image)
            logger.info(
                "[tools] get_image_by_ref: 图片不可用 ref=%s source=%s status=%s",
                normalized_ref,
                source,
                status,
            )
            return {
                "ok": False,
                "status": status,
                "image_ref": normalized_ref,
                "source": source,
            }

        data, mime = payload
        logger.info(
            "[tools] get_image_by_ref: 返回图片 ref=%s source=%s mime=%s",
            normalized_ref,
            source,
            mime,
        )
        return {
            "ok": True,
            "image_ref": normalized_ref,
            "source": source,
            "mime_type": mime,
            "_multimodal_parts": [
                {
                    "data": data,
                    "mime_type": mime,
                    "display_name": f"{source}:{normalized_ref}",
                }
            ],
        }

    return handler


def _normalize_image_ref(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"\bref\s*=\s*['\"]([^'\"]+)['\"]", text)
    if match:
        text = match.group(1).strip()
    return text.strip("`'\"[] ")


def _find_world_image(session: Any, image_ref: str) -> tuple[dict, str] | None:
    for entry in getattr(session, "context_messages", []) or []:
        if image := _image_from_entry(entry, image_ref):
            return image, "chat"

    if getattr(session, "is_browsing_history", lambda: False)():
        view = getattr(session, "chat_window_view", {}) or {}
        top_db_id = view.get("top_db_id")
        if top_db_id:
            try:
                page_size = int(view.get("page_size") or 10)
                for entry in load_history_window(session, int(top_db_id), page_size):
                    if image := _image_from_entry(entry, image_ref):
                        return image, "history"
            except Exception:
                logger.debug("[tools] get_image_by_ref: 历史窗口查找失败", exc_info=True)

    for entry in _visible_forward_entries(session):
        if image := _image_from_entry(entry, image_ref):
            return image, "forward"

    try:
        browser_image = read_browser_image_file(image_ref)
    except Exception:
        logger.debug("[tools] get_image_by_ref: browser 图片查找失败", exc_info=True)
        browser_image = None
    if browser_image is not None:
        raw, mime = browser_image
        return {"data": raw, "mime": mime or "image/jpeg"}, "browser"

    return None


def _visible_forward_entries(session: Any) -> list[dict]:
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


def _image_from_entry(entry: dict, image_ref: str) -> dict | None:
    images = entry.get("images") or {}
    if isinstance(images, dict):
        image = images.get(image_ref)
        return image if isinstance(image, dict) else None
    if isinstance(images, list):
        for image in images:
            if isinstance(image, dict) and str(image.get("ref") or "") == image_ref:
                return image
    return None


def _image_payload(image: dict) -> tuple[str | bytes, str] | None:
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
            logger.debug("[tools] get_image_by_ref: cache 读取失败 phash=%s", phash, exc_info=True)
            cached = None
        if cached:
            return cached
    return None


def _image_unavailable_status(image: dict) -> str:
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
