"""get_group_notice_list.py — 获取群公告摘要列表

需要运行时上下文：napcat_client、group_id。
仅返回各条公告的摘要（发布者、时间、正文前 60 字预览、是否含图），
不含图片，token 安全。
若需查看某条公告的完整内容，请用 get_group_notice_detail 工具传入对应 index。
"""

import asyncio
import html
from datetime import datetime
from typing import Any, Callable

SCOPE: str = "group"  # 仅群聊会话可用
ALWAYS_AVAILABLE: bool = False

DECLARATION: dict = {
    "name": "get_group_notice_list",
    "description": (
        "获取当前群的公告摘要列表（仅群聊会话中可用）。"
        "每条公告只返回序号（index）、发布者 QQ、发布时间、正文前 60 字预览及是否含图片，"
        "若需要查看某条公告的完整内容，请再调用 get_group_notice_detail 工具并传入对应 index。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "motivation": {
                "type": "string",
            },
        },
        "required": ["motivation"],
    },
}

REQUIRES_CONTEXT: list[str] = ["napcat_client", "group_id"]

_PREVIEW_LEN = 60


def make_handler(napcat_client: Any, group_id: str) -> Callable:
    def execute(**kwargs) -> dict:
        if not napcat_client or not napcat_client.connected:
            return {"error": "NapCat 未连接，无法获取群公告"}

        loop: asyncio.AbstractEventLoop | None = napcat_client._loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        try:
            coro = napcat_client.send_api(
                "_get_group_notice",
                {"group_id": int(group_id)},
            )
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            raw: list[dict] | None = future.result(timeout=15)
        except Exception as e:
            return {"error": f"获取群公告失败: {e}"}

        if raw is None:
            return {"error": "API 返回为空（可能群号有误或权限不足）"}

        if not raw:
            return {"group_id": group_id, "total": 0, "notices": [], "note": "该群暂无公告"}

        notices = []
        for i, item in enumerate(raw):
            msg = item.get("message", {}) or {}
            raw_text = msg.get("text", "") or ""
            # 解码 HTML 实体后再截断（&#10; → \n，&nbsp; → 空格等）
            text = html.unescape(raw_text)
            images = msg.get("images", []) or []

            preview = text[:_PREVIEW_LEN] + ("..." if len(text) > _PREVIEW_LEN else "")

            publish_time = item.get("publish_time", 0)
            try:
                time_str = datetime.fromtimestamp(publish_time).strftime("%Y-%m-%d %H:%M")
            except Exception:
                time_str = str(publish_time)

            notices.append({
                "index": i,
                "sender_id": str(item.get("sender_id", "")),
                "publish_time": time_str,
                "preview": preview,
                "has_images": len(images) > 0,
            })

        return {
            "group_id": group_id,
            "total": len(notices),
            "notices": notices,
        }

    return execute
