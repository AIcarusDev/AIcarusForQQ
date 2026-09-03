"""Return a visible world image as a real multimodal attachment by image_ref."""

from __future__ import annotations

import logging
from typing import Any, Callable

from llm.media.image_resolver import ImageResolver, normalize_image_ref
from pydantic import Field

from tools.contract import ToolArgsModel, ToolContract

logger = logging.getLogger("AICQ.tools")

class ViewImageByRefArgs(ToolArgsModel):
    image_ref: str = Field(
        min_length=1,
        description="目标图片的 image_ref，来自 <world> 中的 image_ref 标注。",
    )


TOOL_CONTRACT = ToolContract(
    name="view_image_by_ref",
    description=(
        "在 <world> 中，为了节省上下文和注意力，部分图片即便加载完成，也可能只展现 image_ref，而没有真正的图片显示。"
        "如果需要查看这些图片，可以使用这个工具，写入 image_ref，返回真实图片。"
        "注意：当图片可见的存在于 world 中时，不需要用该工具查看。"
    ),
    args_model=ViewImageByRefArgs,
)

REQUIRES_CONTEXT: list[str] = ["session"]
PARALLEL_SAFE = True
PARALLEL_KEY = "session_read"


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
    image_ref = normalize_image_ref(args.get("image_ref"))
    repaired = dict(args)
    changes: list[str] = []
    if image_ref != args.get("image_ref"):
        repaired["image_ref"] = image_ref
        changes.append("image_ref: normalized")
    if not image_ref:
        return repaired, changes, "image_ref is empty"
    return repaired, changes, None


def make_handler(session: Any) -> Callable:
    resolver = ImageResolver(session)

    def handler(image_ref: str, **_: Any) -> dict:
        normalized_ref = normalize_image_ref(image_ref)
        if not normalized_ref:
            return {"ok": False, "status": "invalid_ref", "image_ref": ""}

        found = resolver.resolve(normalized_ref)
        if found is None:
            logger.info("[tools] view_image_by_ref: 未找到 image_ref=%s", normalized_ref)
            return {
                "ok": False,
                "status": "not_found",
                "image_ref": normalized_ref,
            }

        image, source = found
        payload = resolver.payload(image)
        if payload is None:
            status = resolver.unavailable_status(image)
            logger.info(
                "[tools] view_image_by_ref: 图片不可用 image_ref=%s source=%s status=%s",
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
            "[tools] view_image_by_ref: 返回图片 image_ref=%s source=%s mime=%s",
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
