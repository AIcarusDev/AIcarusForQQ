"""Return an image from the visible world or an Agent-home Linux path."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Union

from llm.media.image_resolver import (
    ImagePayloadError,
    ImageResolver,
    inspect_image_payload,
    normalize_image_ref,
)
from pydantic import Field, RootModel

from tools.contract import ToolArgsModel, ToolContract
from workspace.tools._common import run_on_main_loop

logger = logging.getLogger("AICQ.tools")

MAX_VIEW_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIEW_IMAGE_PIXELS = 100_000_000


class ViewImageRefArgs(ToolArgsModel):
    image_ref: str = Field(
        min_length=1,
        description="目标图片的 image_ref，来自 <world> 中的 image_ref 标注。",
    )


class ViewImagePathArgs(ToolArgsModel):
    path: str = Field(
        min_length=1,
        description="已有 Linux 图片路径，必须位于 /home/agent 内。",
    )


class ViewImageArgs(RootModel[Union[ViewImageRefArgs, ViewImagePathArgs]]):
    pass


TOOL_CONTRACT = ToolContract(
    name="view_image",
    description=(
        "查看图片。可传 image_ref 或 path，返回真实图片；两者选一个。"
        "示例：在 <world> 中需要查看的图片因节省上下文关系只有 image_ref，而省略了真实多模态信息 -> 填写 image_ref 即可查看。"
        "需要看已经保存到 Linux 目录 /home/agent 内的图片 -> 填写 path 即可查看。"
    ),
    args_model=ViewImageArgs,
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
    if args.get("image_ref") is None:
        return dict(args), [], None
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

    def view_by_ref(image_ref: str) -> dict[str, Any]:
        normalized_ref = normalize_image_ref(image_ref)
        if not normalized_ref:
            return {"ok": False, "status": "invalid_ref", "image_ref": ""}

        found = resolver.resolve(normalized_ref)
        if found is None:
            logger.info("[tools] view_image: 未找到 image_ref=%s", normalized_ref)
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
                "[tools] view_image: 图片不可用 image_ref=%s source=%s status=%s",
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
            "[tools] view_image: 返回图片 image_ref=%s source=%s mime=%s",
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

    async def view_by_path(workspace_service: Any, path: str) -> dict[str, Any]:
        async with workspace_service.stage_host_file(path) as staged:
            if staged.size > MAX_VIEW_IMAGE_BYTES:
                return {
                    "ok": False,
                    "status": "image_too_large",
                    "path": staged.workspace_path,
                    "size_bytes": staged.size,
                    "limit_bytes": MAX_VIEW_IMAGE_BYTES,
                }
            try:
                raw = await asyncio.to_thread(Path(staged.host_path).read_bytes)
            except OSError:
                logger.warning("[tools] view_image: Linux 暂存图片读取失败 path=%s", staged.workspace_path)
                return {"ok": False, "status": "read_failed", "path": staged.workspace_path}
            try:
                image_info = inspect_image_payload(
                    raw,
                    max_bytes=MAX_VIEW_IMAGE_BYTES,
                    max_pixels=MAX_VIEW_IMAGE_PIXELS,
                )
            except ImagePayloadError as exc:
                logger.info(
                    "[tools] view_image: Linux 路径不是可查看图片 path=%s status=%s",
                    staged.workspace_path,
                    exc.code,
                )
                return {
                    "ok": False,
                    "status": exc.code,
                    "path": staged.workspace_path,
                    **exc.details,
                }
            logger.info(
                "[tools] view_image: 返回 Linux 图片 path=%s mime=%s",
                staged.workspace_path,
                image_info.mime_type,
            )
            return {
                "ok": True,
                "path": staged.workspace_path,
                "source": "path",
                "mime_type": image_info.mime_type,
                "_multimodal_parts": [
                    {
                        "data": raw,
                        "mime_type": image_info.mime_type,
                        "display_name": staged.name,
                    }
                ],
            }

    def handler(
        image_ref: str | None = None,
        path: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if (image_ref is None) == (path is None):
            return {
                "ok": False,
                "status": "invalid_arguments",
                "error": "image_ref 与 path 必须且只能提供一个",
            }
        if image_ref is not None:
            return view_by_ref(image_ref)

        import app_state

        workspace_service = getattr(app_state, "workspace_service", None)
        if workspace_service is None:
            return {
                "ok": False,
                "code": "runtime_unavailable",
                "error": "Agent 电脑服务不可用",
            }
        return run_on_main_loop(
            view_by_path(workspace_service, str(path or "")),
            getattr(app_state, "main_loop", None),
        )

    return handler
