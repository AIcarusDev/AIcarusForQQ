"""Set the bot's QQ avatar from a visible image or Agent Linux file."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
from pathlib import Path
from typing import Any, Callable, Union

from llm.media.image_resolver import (
    ImagePayloadError,
    ImagePayloadInfo,
    ImageResolver,
    inspect_image_payload,
    normalize_image_ref,
)
from platforms.qq.adapter.conversation import format_adapter_error
from pydantic import Field, RootModel
from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, ToolContract
from workspace.errors import WorkspaceError


logger = logging.getLogger("AICQ.tools")

MAX_AVATAR_BYTES = 20 * 1024 * 1024
MAX_AVATAR_PIXELS = 100_000_000
_UPLOAD_TIMEOUT_SECONDS = 60.0


class SetAvatarRefArgs(ToolArgsModel):
    image_ref: str = Field(
        min_length=1,
        description="目标图片的 image_ref，来自当前可见的聊天、转发或浏览器图片。",
    )


class SetAvatarPathArgs(ToolArgsModel):
    path: str = Field(
        min_length=1,
        pattern=r"^/home/agent/.+",
        description="已有图片的 Agent Linux 路径，必须位于 /home/agent 内。",
    )


class SetAvatarArgs(RootModel[Union[SetAvatarRefArgs, SetAvatarPathArgs]]):
    pass


TOOL_CONTRACT = ToolContract(
    name="set_avatar",
    description=(
        "修改你自己的 QQ 头像。可传当前可见图片的 image_ref，或 Linux 中 /home/agent 下已有图片的 path；"
        "两者必须且只能提供一个。"
        "注意：图片会原样上传，不会自动裁剪、缩放或转码，若需要精确的头像范围，请先在 Linux 端将图片裁剪为预期范围的方形图。"
    ),
    args_model=SetAvatarArgs,
)

EXTERNALLY_PERCEPTIBLE: bool = True
TOOL_EFFECT: dict[str, str] = {"surface": "qq", "kind": "profile_write"}
REQUIRES_CONTEXT: list[str] = ["qq_client", "session"]


def repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(args, dict) or args.get("image_ref") or "ref" not in args:
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


def _failure(status: str, message: str, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": False, "status": status, "error": message}
    result.update(details)
    return result


def _decode_image_payload(payload: tuple[str | bytes, str]) -> bytes:
    value, _mime_type = payload
    if isinstance(value, bytes):
        return value
    encoded = value.strip()
    max_encoded_chars = ((MAX_AVATAR_BYTES + 2) // 3) * 4 + 4
    if len(encoded) > max_encoded_chars:
        raise ImagePayloadError(
            "image_too_large",
            details={"limit_bytes": MAX_AVATAR_BYTES},
        )
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImagePayloadError("invalid_image_data") from exc


def _prepare_upload(raw: bytes) -> tuple[str, ImagePayloadInfo]:
    image_info = inspect_image_payload(
        raw,
        max_bytes=MAX_AVATAR_BYTES,
        max_pixels=MAX_AVATAR_PIXELS,
    )
    encoded = base64.b64encode(raw).decode("ascii")
    return f"base64://{encoded}", image_info


def _image_error(exc: ImagePayloadError) -> dict[str, Any]:
    return _failure(exc.code, "图片内容无效", **exc.details)


def _adapter_result(
    response: dict[str, Any] | None,
    image_info: ImagePayloadInfo,
    size_bytes: int,
) -> dict[str, Any]:
    if response is None:
        return _failure("adapter_timeout", "修改头像超时或 QQ adapter 未连接")
    if not isinstance(response, dict):
        return _failure("adapter_error", "QQ adapter 返回了无效响应")
    if response.get("status") != "ok":
        return _failure(
            "adapter_error",
            format_adapter_error(
                {**response, "action": "set_qq_avatar"},
                "修改头像失败",
            ),
        )
    return {
        "ok": True,
        "success": True,
        "mime_type": image_info.mime_type,
        "width": image_info.width,
        "height": image_info.height,
        "size_bytes": size_bytes,
        "note": "头像已提交更新；工具未对原图做裁剪、缩放或转码。",
    }


def make_handler(qq_client: Any, session: Any) -> Callable:
    resolver = ImageResolver(session)

    async def upload(file_value: str) -> dict[str, Any] | None:
        return await qq_client.send_api_raw(
            "set_qq_avatar",
            {"file": file_value},
            timeout=_UPLOAD_TIMEOUT_SECONDS,
        )

    async def upload_path(workspace_service: Any, path: str) -> dict[str, Any]:
        try:
            async with workspace_service.stage_host_file(path) as staged:
                if staged.size > MAX_AVATAR_BYTES:
                    return _failure(
                        "image_too_large",
                        "图片超过大小限制",
                        size_bytes=staged.size,
                        limit_bytes=MAX_AVATAR_BYTES,
                    )
                raw = await asyncio.to_thread(Path(staged.host_path).read_bytes)
                try:
                    file_value, image_info = await asyncio.to_thread(_prepare_upload, raw)
                except ImagePayloadError as exc:
                    return _image_error(exc)
                response = await upload(file_value)
                return _adapter_result(response, image_info, len(raw))
        except WorkspaceError as exc:
            return _failure(exc.code.value, "无法读取指定的 Linux 图片")
        except OSError:
            return _failure("read_failed", "无法读取指定的 Linux 图片")

    def execute(
        image_ref: str | None = None,
        path: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if (image_ref is None) == (path is None):
            return _failure("invalid_arguments", "image_ref 与 path 必须且只能提供一个")
        if not qq_client or not getattr(qq_client, "connected", False):
            return _failure("adapter_disconnected", "QQ adapter 未连接，无法修改头像")
        loop: asyncio.AbstractEventLoop | None = getattr(qq_client, "_loop", None)
        if loop is None or not loop.is_running():
            return _failure("runtime_unavailable", "主事件循环不可用")

        if image_ref is not None:
            normalized_ref = normalize_image_ref(image_ref)
            found = resolver.resolve(normalized_ref)
            if found is None:
                return _failure("not_found", "未找到指定的 image_ref")
            image, _source = found
            payload = resolver.payload(image)
            if payload is None:
                status = resolver.unavailable_status(image)
                return _failure(status, "图片当前不可用")
            try:
                raw = _decode_image_payload(payload)
                file_value, image_info = _prepare_upload(raw)
            except ImagePayloadError as exc:
                return _image_error(exc)
            try:
                response = run_coroutine_sync(
                    upload(file_value),
                    loop,
                    timeout=_UPLOAD_TIMEOUT_SECONDS,
                )
            except Exception:
                logger.warning("[tools] set_avatar: image_ref 头像上传异常", exc_info=True)
                return _failure("upload_failed", "修改头像失败")
            return _adapter_result(response, image_info, len(raw))

        import app_state

        workspace_service = getattr(app_state, "workspace_service", None)
        if workspace_service is None:
            return _failure("runtime_unavailable", "Agent 电脑服务不可用")
        try:
            return run_coroutine_sync(
                upload_path(workspace_service, str(path or "")),
                loop,
                timeout=None,
            )
        except Exception:
            logger.warning("[tools] set_avatar: Linux 路径头像上传异常", exc_info=True)
            return _failure("upload_failed", "修改头像失败")

    return execute
