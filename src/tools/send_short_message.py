"""send_short_message.py — 发送简短消息

直接发出一条简短的纯文本消息（如"嗯嗯"、"好的"、"啊？"等），
不走模拟打字延迟，适合即时、自然的短句反应。
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger("AICQ.tools")

DECLARATION: dict = {
    "name": "send_short_message",
    "description": (
        "立即发出一条简短的纯文本消息，无打字延迟，直接送达。"
        "适合快速表达语气、简单回应、随口一句等极短场合。"
        "仅支持纯文本，建议控制在 5 字以内。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "要发送的文本，应简短，通常不超过 5 个字。",
                "maxLength": 50,
            },
            "reply_to": {
                "type": ["string"],
                "description": "要引用/回复的消息 ID（可选）。",
            },
        },
        "required": ["text"],
    },
}

REQUIRES_CONTEXT: list[str] = ["napcat_client", "session"]

# 下一轮 prompt 中整条调用记录不出现：消息本身已进入聊天上下文，无需重复
RESULT_MAX_CHARS: int = -1


def make_handler(napcat_client: Any, session: Any) -> Callable:
    def execute(text: str, reply_to: str | None = None, **kwargs) -> dict:
        if not napcat_client or not napcat_client.connected:
            return {"error": "NapCat 未连接"}

        loop: asyncio.AbstractEventLoop | None = napcat_client._loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        # ── 构建消息段 ─────────────────────────────────────────────
        napcat_segs: list[dict] = []
        if reply_to:
            napcat_segs.append({"type": "reply", "data": {"id": str(reply_to)}})
        napcat_segs.append({"type": "text", "data": {"text": text}})

        # ── 构建 send_msg 参数 ─────────────────────────────────────
        params: dict = {"message": napcat_segs}
        if session.conv_type == "group":
            params["group_id"] = int(session.conv_id)
            params["message_type"] = "group"
        else:
            params["user_id"] = int(session.conv_id)
            params["message_type"] = "private"

        # ── 直接发送（绕过打字延迟）────────────────────────────────
        try:
            send_result: dict | None = asyncio.run_coroutine_threadsafe(
                napcat_client.send_api("send_msg", params), loop
            ).result(timeout=15)
        except Exception as e:
            return {"error": f"发送失败: {e}"}

        if send_result is None:
            return {"error": "发送失败（NapCat 无响应）"}

        message_id = str(
            send_result.get("message_id") or f"qr_{uuid.uuid4().hex[:8]}"
        )

        # ── 录入 session 上下文 ────────────────────────────────────
        import app_state
        bot_sender_id = session._qq_id or "bot"
        bot_sender_name = session._qq_name or app_state.BOT_NAME
        now_ts = datetime.now(app_state.TIMEZONE).isoformat()

        entry = {
            "role": "bot",
            "message_id": message_id,
            "sender_id": bot_sender_id,
            "sender_name": bot_sender_name,
            "sender_role": "",
            "timestamp": now_ts,
            "content": text,
            "content_type": "text",
            "content_segments": [{"type": "text", "text": text}],
        }
        session.add_to_context(entry)

        # ── 异步持久化（不阻塞工具返回）──────────────────────────
        async def _persist() -> None:
            from database import save_chat_message
            conv_id = f"{session.conv_type}_{session.conv_id}"
            await save_chat_message(conv_id, entry)

        asyncio.run_coroutine_threadsafe(_persist(), loop)

        return {"ok": True, "message_id": message_id}

    return execute
