"""get_group_notice_detail.py — 获取指定群公告的完整内容

需要运行时上下文：napcat_client、group_id。
通过 index（来自 get_group_notice_list 的返回值）读取单条公告完整内容。

注意：NapCat 的 _get_group_notice API 对图片只返回 {id, height, width}，
不含任何可用 URL；群公告图片存储于 groupboard.qpic.cn CDN，当前环境
TLS 握手失败，无法访问。因此本工具仅返回图片元数据，不下载图片内容。
"""

import asyncio
import html
import logging
from datetime import datetime
from typing import Any, Callable

from tools._async_bridge import run_coroutine_sync

logger = logging.getLogger("AICQ.tools")

SCOPE: str = "group"  # 仅群聊会话可用
ALWAYS_AVAILABLE: bool = False

DECLARATION: dict = {
    "name": "get_group_notice_detail",
    "description": (
        "获取指定群公告的完整内容。"
        "需先通过 get_group_notice_list 工具获取公告列表，再凭其中的 index 调用本工具。"
        "返回公告完整正文及图片元数据（如有）。"
        "注意：NapCat API 似乎不提供群公告图片的可访问 URL，无法直接显示图片。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "index": {
                "type": "integer",
                "description": "公告在列表中的序号（从 0 开始），即 get_group_notice_list 返回的 index 字段。",
            },
            "motivation": {
                "type": "string",
            },
        },
        "required": ["index", "motivation"],
    },
}

REQUIRES_CONTEXT: list[str] = ["napcat_client", "group_id"]


def make_handler(napcat_client: Any, group_id: str) -> Callable:
    def execute(**kwargs) -> dict:
        raw_index = kwargs.get("index")
        if raw_index is None:
            return {"error": "缺少参数 index"}

        if not napcat_client or not napcat_client.connected:
            return {"error": "NapCat 未连接，无法获取群公告"}

        loop: asyncio.AbstractEventLoop | None = napcat_client._loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        try:
            raw: list[dict] | None = run_coroutine_sync(
                napcat_client.send_api(
                    "_get_group_notice",
                    {"group_id": int(group_id)},
                ),
                loop,
                timeout=15,
            )
        except Exception as e:
            return {"error": f"获取群公告失败: {e}"}

        if raw is None:
            return {"error": "API 返回为空（可能群号有误或权限不足）"}

        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            return {"error": f"index 必须是整数，收到: {raw_index!r}"}

        if index < 0 or index >= len(raw):
            return {"error": f"index {index} 超出范围（共 {len(raw)} 条公告，合法范围 0~{len(raw)-1}）"}

        item = raw[index]
        msg = item.get("message", {}) or {}
        raw_text = msg.get("text", "") or ""
        # 解码 HTML 实体（&#10; → \n，&nbsp; → 空格等）
        text = html.unescape(raw_text)
        images = msg.get("images", []) or []

        publish_time = item.get("publish_time", 0)
        try:
            time_str = datetime.fromtimestamp(publish_time).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            time_str = str(publish_time)

        result: dict = {
            "group_id": group_id,
            "index": index,
            "sender_id": str(item.get("sender_id", "")),
            "publish_time": time_str,
            "content": text,
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
            ]
            result["image_note"] = (
                "NapCat 不提供群公告图片的可访问 URL（API 仅返回 id/width/height），"
                "如需查看图片请通过 QQ 客户端查看原始公告。"
            )

        return result

    return execute
