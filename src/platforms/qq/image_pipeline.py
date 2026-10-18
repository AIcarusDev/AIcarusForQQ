"""QQ 发送图片管线。

模型能用三种来源给一条消息附图：sticker 的 image_ref、Agent Linux 本地 path、
浏览器 <images> 里的 resource_ref。本模块把它们统一校验并固化成 adapter 可以
直接发送的 segment 字段，同时在发送前拦截需要高风险确认的浏览器原图。
"""

import asyncio
import base64
import copy
import logging
from pathlib import Path
from typing import Any

from llm.media.image_resolver import (
    ImagePayloadError,
    ImageResolver,
    image_bytes,
    inspect_image_payload,
    normalize_image_ref,
)
from llm.media.sticker_collection import MAX_STICKER_BYTES, valid_image_ref
from workspace.errors import WorkspaceError

logger = logging.getLogger("AICQ.tools")

_MAX_LOCAL_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_LOCAL_IMAGE_PIXELS = 100_000_000
_MAX_LOCAL_IMAGES_PER_CALL = 4
_MAX_BROWSER_IMAGES_PER_CALL = 4


def prepare_sticker_segment(segment: dict, session: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one sticker segment and resolve its bytes for the adapter."""
    image_ref = normalize_image_ref(segment.get("image_ref"))
    if "sticker_id" in segment or not valid_image_ref(image_ref):
        return None, "sticker segment 需要 image_ref，旧表情包编号已停用。"
    resolver = ImageResolver(session)
    found = resolver.resolve(image_ref, include_browser=False)
    if found is None:
        return None, f"未找到可见或已收藏的图片 image_ref={image_ref!r}。"
    image, source = found
    payload = image_bytes(image)
    if payload is None:
        return None, f"图片不可用 image_ref={image_ref!r}: {resolver.unavailable_status(image)}"
    raw, _mime = payload
    try:
        info = inspect_image_payload(raw, max_bytes=MAX_STICKER_BYTES, max_pixels=100_000_000)
    except ImagePayloadError as exc:
        return None, f"图片内容无效 image_ref={image_ref!r}: {exc.code}"
    return {
        "image_ref": str(image.get("image_ref") or image_ref),
        "_image_bytes": raw,
        "_image_mime": info.mime_type,
        "_image_source": source,
    }, None


async def materialize_local_image_paths(
    messages: list[dict[str, Any]],
    workspace_service: Any,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Validate Agent-local image paths and embed their bytes for the adapter."""

    prepared = copy.deepcopy(messages)
    local_segments: list[tuple[dict[str, Any], str]] = []
    for message in prepared:
        if not isinstance(message, dict):
            continue
        for segment in message.get("segments") or []:
            if not isinstance(segment, dict) or segment.get("command") != "image":
                continue
            sources = {
                "path": str(segment.get("path") or "").strip(),
                "image_ref": str(segment.get("image_ref") or "").strip(),
                "resource_ref": str(segment.get("resource_ref") or "").strip(),
            }
            selected = [(name, value) for name, value in sources.items() if value]
            if len(selected) != 1:
                return None, "image segment 的 path、image_ref、resource_ref 必须且只能提供一个。"
            if selected[0][0] == "path":
                local_segments.append((segment, selected[0][1]))

    if len(local_segments) > _MAX_LOCAL_IMAGES_PER_CALL:
        return None, f"单次发送最多选择 {_MAX_LOCAL_IMAGES_PER_CALL} 张 Agent Linux 本地图片。"
    if local_segments and workspace_service is None:
        return None, "Agent 电脑服务不可用，无法读取本地图片。"

    try:
        for segment, path in local_segments:
            async with workspace_service.stage_host_file(path) as staged:
                if staged.size > _MAX_LOCAL_IMAGE_BYTES:
                    return None, "本地图片超过 20 MiB 大小限制。"
                raw = await asyncio.to_thread(Path(staged.host_path).read_bytes)
                image_info = await asyncio.to_thread(
                    inspect_image_payload,
                    raw,
                    max_bytes=_MAX_LOCAL_IMAGE_BYTES,
                    max_pixels=_MAX_LOCAL_IMAGE_PIXELS,
                )
                from llm.media.image_store import register_image
                record = await asyncio.to_thread(register_image, raw, "workspace")
                segment.pop("path", None)
                segment["image_ref"] = record["image_ref"]
                segment["_local_image_ref"] = record["image_ref"]
                segment["_local_image_base64"] = base64.b64encode(raw).decode("ascii")
                segment["_local_image_mime"] = image_info.mime_type
    except WorkspaceError as exc:
        logger.info("[send_message] Agent Linux 图片暂存失败 code=%s", exc.code.value)
        return None, "无法读取指定的 Agent Linux 图片。"
    except ImagePayloadError as exc:
        logger.info("[send_message] Agent Linux 图片校验失败 status=%s", exc.code)
        return None, f"Agent Linux 图片内容无效（{exc.code}）。"
    except OSError:
        logger.warning("[send_message] Agent Linux 图片读取失败", exc_info=True)
        return None, "无法读取指定的 Agent Linux 图片。"

    return prepared, None


def materialize_browser_resources(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]], str | None]:
    selected: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        for segment in message.get("segments") or []:
            if not isinstance(segment, dict) or segment.get("command") != "image":
                continue
            image_ref = str(segment.get("image_ref") or "").strip()
            resource_ref = str(segment.get("resource_ref") or "").strip()
            if image_ref and resource_ref:
                return None, [], "image segment 不能同时包含 image_ref 和 resource_ref。"
            if resource_ref:
                selected.append(resource_ref)
    if not selected:
        return copy.deepcopy(messages), [], None
    if len(selected) > 4:
        return None, [], "单次发送最多选择 4 张浏览器图片。"

    try:
        from browser import materialize_browser_resources

        artifacts = materialize_browser_resources(selected)
    except Exception:
        logger.warning("[send_message] 浏览器原图固化失败", exc_info=True)
        return None, [], "浏览器原图固化失败，未发送且不会降级为截图。"
    if len(artifacts) != len(selected):
        return None, [], "浏览器原图固化结果数量不一致，未发送。"

    prepared = copy.deepcopy(messages)
    artifact_index = 0
    for message in prepared:
        if not isinstance(message, dict):
            continue
        for segment in message.get("segments") or []:
            if not isinstance(segment, dict) or segment.get("command") != "image":
                continue
            if not str(segment.get("resource_ref") or "").strip():
                continue
            artifact = artifacts[artifact_index]
            expected_resource = selected[artifact_index]
            artifact_index += 1
            if str(artifact.get("resource_ref") or "") != expected_resource:
                return None, [], "浏览器原图固化身份校验失败，未发送。"
            image_ref = str(artifact.get("image_ref") or "")
            if not image_ref:
                return None, [], "浏览器原图固化未生成 image_ref，未发送。"
            segment.pop("resource_ref", None)
            segment["image_ref"] = image_ref
    return prepared, artifacts, None


def unconfirmed_high_risk_image_error(
    messages: list[dict[str, Any]],
    session: Any,
    config: dict | None,
) -> str | None:
    from browser.config import browser_image_send_confirmation

    if browser_image_send_confirmation(config) != "high_risk":
        return None
    from browser import read_sendable_browser_image_file
    from browser.image_confirmation import current_pending

    pending = current_pending(session)
    pending_refs = {
        str(artifact.get("image_ref") or "")
        for artifact in pending.artifacts
    } if pending is not None else set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        for segment in message.get("segments") or []:
            if not isinstance(segment, dict) or segment.get("command") not in ("image", "sticker"):
                continue
            image_ref = str(segment.get("image_ref") or "").strip()
            if not image_ref:
                continue
            from llm.media.image_store import lookup_image
            record = lookup_image(image_ref)
            if record:
                image_ref = record["image_ref"]
            if image_ref in pending_refs:
                return "该原图属于待确认批次，必须使用 confirm_browser_image_send，不能直接发送。"
            artifact = read_sendable_browser_image_file(image_ref)
            if artifact is not None and artifact[2].get("confirmation_reasons"):
                return "该原图需要高风险确认，不能通过 image_ref 直接发送。"
    return None
