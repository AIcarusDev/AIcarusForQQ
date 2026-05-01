"""plus_one.py — 复读目标消息（+1）

获取目标消息的内容并原样转发到当前群聊。
工具在 LLM 输出阶段之前执行，消息会立即发出。

仅适用于群聊（需要 napcat_client 和 group_id）。
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Callable

from tools._async_bridge import run_coroutine_sync

logger = logging.getLogger("AICQ.tools")

# 发送时过滤掉这些不适合复读的 segment 类型
_SKIP_TYPES: frozenset[str] = frozenset({"reply"})

SCOPE: str = "group"  # 仅群聊会话可用

DECLARATION: dict = {
    "name": "plus_one",
    "description": (
        "复读某条消息。获取目标消息的完整内容（文字、图片等），"
        "原样发送到当前会话，消息立即发出（会在你真正发送消息之前就发出）。"
        "仅用于那些非常经典、值得复读或有节目效果的他人消息。"
        "不要滥用。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message_id": {
                "type": "integer",
                "description": "要复读的目标消息 ID。",
            },
            "motivation": {
                "type": "string"
            },
        },
        "required": ["message_id", "motivation"],
    },
}

REQUIRES_CONTEXT: list[str] = ["napcat_client", "group_id", "session"]


def make_handler(napcat_client: Any, group_id: str, session: Any) -> Callable:
    def execute(message_id: int, **kwargs) -> dict:
        if not napcat_client or not napcat_client.connected:
            return {"error": "NapCat 未连接，无法复读消息"}

        loop: asyncio.AbstractEventLoop | None = napcat_client._loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        # ── 1. 获取目标消息内容 ──────────────────────────────────
        try:
            msg_data: dict | None = run_coroutine_sync(
                napcat_client.send_api(
                    "get_msg",
                    {"message_id": message_id},
                ),
                loop,
                timeout=15,
            )
        except Exception as e:
            return {"error": f"获取消息失败: {e}"}

        if not msg_data:
            return {"error": f"未找到消息 ID={message_id}，可能已过期或不存在"}

        raw_segments: list[dict] = msg_data.get("message") or []
        if not raw_segments:
            return {"error": "目标消息内容为空，无法复读"}

        # ── 2. 过滤不适合复读的 segment ──────────────────────────
        segments: list[dict] = [
            seg for seg in raw_segments
            if seg.get("type") not in _SKIP_TYPES
        ]
        if not segments:
            return {"error": "过滤后消息内容为空（例如纯引用消息），无法复读"}

        # ── 3. 发送到当前群聊 ────────────────────────────────────
        try:
            send_result: dict | None = run_coroutine_sync(
                napcat_client.send_api(
                    "send_msg",
                    {
                        "message_type": "group",
                        "group_id": int(group_id),
                        "message": segments,
                    },
                ),
                loop,
                timeout=15,
            )
        except Exception as e:
            return {"error": f"发送复读消息失败: {e}"}

        if not send_result:
            return {"error": "复读消息发送失败（NapCat 无响应）"}

        sent_id = send_result.get("message_id")
        logger.info(
            "[tools] plus_one: 复读成功 原消息=%d 新消息=%s group=%s",
            message_id, sent_id, group_id,
        )

        # ── 4. 录入 session 上下文 ──────────────────────────────
        from napcat.segments import napcat_segments_to_text, build_content_segments, _determine_content_type
        import app_state

        content_text = napcat_segments_to_text(
            segments,
            bot_id=session._qq_id,
            bot_display_name=session._qq_name,
        )
        content_segs = build_content_segments(
            segments,
            bot_id=session._qq_id,
            bot_display_name=session._qq_name,
        )
        content_type = _determine_content_type(segments)
        now_ts = datetime.now(app_state.TIMEZONE).isoformat()
        entry = {
            "role": "bot",
            "message_id": str(sent_id) if sent_id else f"qr_{uuid.uuid4().hex[:8]}",
            "sender_id": session._qq_id or "bot",
            "sender_name": session._qq_name or app_state.BOT_NAME,
            "sender_role": "",
            "timestamp": now_ts,
            "content": content_text,
            "content_type": content_type,
            "content_segments": content_segs,
        }
        session.add_to_context(entry)

        async def _persist() -> None:
            from database import save_chat_message
            conv_id = f"{session.conv_type}_{session.conv_id}"
            await save_chat_message(conv_id, entry)

        asyncio.run_coroutine_threadsafe(_persist(), loop)

        return {
            "success": True,
            "original_message_id": message_id,
            "sent_message_id": sent_id,
        }

    return execute
