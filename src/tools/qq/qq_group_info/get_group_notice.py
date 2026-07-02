"""get_group_notice.py - list or read current group notices.

需要运行时上下文：qq_adapter_client、session。
仅群聊目标可执行；非群聊会话中由工具返回明确错误。

action="list" 返回公告摘要列表，token 安全。
action="read" 需要 index，返回指定公告完整正文和图片元数据。

注意：QQ adapter 的 _get_group_notice API 对图片只返回 {id, height, width}，
不含任何可用 URL；群公告图片存储于 groupboard.qpic.cn CDN，当前环境
TLS 握手失败，无法访问。因此本工具仅返回图片元数据，不下载图片内容。
"""

import asyncio
import html
from datetime import datetime
from typing import Any, Callable, Literal

from pydantic import Field, RootModel

from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, ToolContract

class GroupNoticeListArgs(ToolArgsModel):
    action: Literal["list"] = Field(description="列出公告摘要。")


class GroupNoticeReadArgs(ToolArgsModel):
    action: Literal["read"] = Field(description="读取指定 index 的完整公告。")
    index: int = Field(ge=0, description="公告在 list 结果中的序号，从 0 开始。")


class GetGroupNoticeArgs(RootModel[GroupNoticeListArgs | GroupNoticeReadArgs]):
    pass


TOOL_CONTRACT = ToolContract(
    name="get_group_notice",
    description=(
        "查看当前群公告（仅群聊会话中可用）。"
        "action=list 时返回公告摘要列表：index、发布者 QQ、发布时间、正文前 60 字预览、是否含图片。"
        "action=read 时必须传 index，返回对应公告完整正文及图片元数据。"
        "群公告图片目前只能返回 id/width/height，无法直接显示图片。"
    ),
    args_model=GetGroupNoticeArgs,
)

REQUIRES_CONTEXT: list[str] = ["qq_adapter_client", "session"]

_PREVIEW_LEN = 60
_FETCH_TIMEOUT_SECONDS = 15
_IMAGE_NOTE = (
    "QQ adapter 不提供群公告图片的可访问 URL（API 仅返回 id/width/height），"
    "如需查看图片请通过 QQ 客户端查看原始公告。"
)


def _format_publish_time(value: Any, *, with_seconds: bool) -> str:
    fmt = "%Y-%m-%d %H:%M:%S" if with_seconds else "%Y-%m-%d %H:%M"
    try:
        return datetime.fromtimestamp(value).strftime(fmt)
    except Exception:
        return str(value)


def _decode_notice_text(item: dict[str, Any]) -> str:
    msg = item.get("message", {}) or {}
    return html.unescape(msg.get("text", "") or "")


def _notice_images(item: dict[str, Any]) -> list[Any]:
    msg = item.get("message", {}) or {}
    images = msg.get("images", []) or []
    return images if isinstance(images, list) else []


def _notice_summary(index: int, item: dict[str, Any]) -> dict[str, Any]:
    text = _decode_notice_text(item)
    images = _notice_images(item)
    preview = text[:_PREVIEW_LEN] + ("..." if len(text) > _PREVIEW_LEN else "")

    return {
        "index": index,
        "sender_id": str(item.get("sender_id", "")),
        "publish_time": _format_publish_time(item.get("publish_time", 0), with_seconds=False),
        "preview": preview,
        "has_images": len(images) > 0,
    }


def _notice_detail(group_id: str, index: int, item: dict[str, Any]) -> dict[str, Any]:
    images = _notice_images(item)
    result: dict[str, Any] = {
        "mode": "read",
        "group_id": group_id,
        "index": index,
        "sender_id": str(item.get("sender_id", "")),
        "publish_time": _format_publish_time(item.get("publish_time", 0), with_seconds=True),
        "content": _decode_notice_text(item),
        "image_count": len(images),
    }

    if images:
        result["images"] = [
            {
                "id": img.get("id", ""),
                "width": img.get("width", ""),
                "height": img.get("height", ""),
            }
            for img in images
            if isinstance(img, dict)
        ]
        result["image_note"] = _IMAGE_NOTE

    return result


def make_handler(qq_adapter_client: Any, session: Any) -> Callable:
    def execute(**kwargs) -> dict:
        action = kwargs.get("action")
        if action not in {"list", "read"}:
            return {"error": "action 必须是 list 或 read"}
        if action == "list" and "index" in kwargs:
            return {"error": "action=list 时不能传 index"}
        if action == "read" and kwargs.get("index") is None:
            return {"error": "action=read 时必须传 index"}

        index: int | None = None
        if action == "read":
            raw_index = kwargs.get("index")
            if raw_index is None:
                return {"error": "action=read 时必须传 index"}
            try:
                if isinstance(raw_index, bool):
                    raise ValueError
                index = int(raw_index)
            except (TypeError, ValueError):
                return {"error": f"index 必须是整数，收到: {raw_index!r}"}
            if index < 0:
                return {"error": f"index 必须大于等于 0，收到: {index}"}

        if getattr(session, "conv_type", "") != "group":
            return {"error": "get_group_notice 仅能在群聊会话中使用"}
        group_id = str(getattr(session, "conv_id", "") or "").strip()
        if not group_id:
            return {"error": "当前群号未知，无法获取群公告"}
        if not qq_adapter_client or not qq_adapter_client.connected:
            return {"error": "QQ adapter 未连接，无法获取群公告"}

        loop: asyncio.AbstractEventLoop | None = qq_adapter_client._loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        try:
            raw: list[dict[str, Any]] | None = run_coroutine_sync(
                qq_adapter_client.send_api(
                    "_get_group_notice",
                    {"group_id": int(group_id)},
                ),
                loop,
                timeout=_FETCH_TIMEOUT_SECONDS,
            )
        except Exception as e:
            return {"error": f"获取群公告失败: {e}"}

        if raw is None:
            return {"error": "API 返回为空（可能群号有误或权限不足）"}

        if action == "list":
            if not raw:
                return {
                    "mode": "list",
                    "group_id": group_id,
                    "total": 0,
                    "notices": [],
                    "note": "该群暂无公告",
                }
            return {
                "mode": "list",
                "group_id": group_id,
                "total": len(raw),
                "notices": [_notice_summary(i, item) for i, item in enumerate(raw)],
            }

        if index is None:
            return {"error": "action=read 时必须传 index"}
        if not raw:
            return {"error": f"该群暂无公告，无法读取 index {index}"}
        if index >= len(raw):
            return {"error": f"index {index} 超出范围（共 {len(raw)} 条公告，合法范围 0~{len(raw)-1}）"}

        return _notice_detail(group_id, index, raw[index])

    return execute
