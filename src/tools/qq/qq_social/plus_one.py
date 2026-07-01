"""plus_one.py — 复读目标消息（+1）

获取目标消息的内容并原样转发到当前会话。
工具在 LLM 输出阶段之前执行，消息会立即发出。

群聊、私聊和临时会话都可尝试；目标不匹配时由执行层返回明确错误。
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

DECLARATION: dict = {
    "name": "plus_one",
    "description": (
        "复读某条消息。获取目标消息的完整内容（文字、图片等），"
        "原样发送到当前会话。"
        "仅用于那些非常经典、值得复读或有节目效果的他人消息。"
        "不要滥用。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message_id": {
                "type": "string",
                "x-coerce-integer": True,
                "description": "要复读的目标消息 ID。",
            },
        },
        "required": ["message_id"],
    },
}

EXTERNALLY_PERCEPTIBLE: bool = True
TOOL_EFFECT: dict[str, str] = {"surface": "qq", "kind": "session_write"}

REQUIRES_CONTEXT: list[str] = ["qq_adapter_client", "session"]


def _resolve_send_target(session: Any) -> tuple[int | None, int | None, int | None, str | None]:
    conv_type = getattr(session, "conv_type", "")
    conv_id = getattr(session, "conv_id", "")
    try:
        if conv_type == "group":
            return int(conv_id), None, None, None
        if conv_type == "private":
            return None, int(conv_id), None, None
        if conv_type == "temp":
            source_group_id = str(getattr(session, "temp_source_group_id", "") or "").strip()
            if not source_group_id:
                return None, None, None, "临时会话缺少来源群，无法复读。请先从可用群聊打开该临时会话。"
            return None, int(conv_id), int(source_group_id), None
    except (ValueError, TypeError):
        return None, None, None, f"会话 ID 无效: {conv_id}"
    return None, None, None, f"当前会话类型不支持复读: {conv_type or 'unknown'}"


def make_handler(qq_adapter_client: Any, session: Any) -> Callable:
    def execute(message_id: str | int, **kwargs) -> dict:
        if not qq_adapter_client or not qq_adapter_client.connected:
            return {"error": "QQ adapter 未连接，无法复读消息"}

        loop: asyncio.AbstractEventLoop | None = qq_adapter_client._loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        raw_message_id = str(message_id).strip()
        try:
            adapter_message_id = int(raw_message_id)
        except (TypeError, ValueError):
            return {"error": f"消息 ID 无法用于 QQ adapter: {message_id}"}

        # ── 1. 获取目标消息内容 ──────────────────────────────────
        try:
            msg_data: dict | None = run_coroutine_sync(
                qq_adapter_client.send_api(
                    "get_msg",
                    {"message_id": adapter_message_id},
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

        # ── 3. 发送到当前会话 ────────────────────────────────────
        group_id, user_id, temp_source_group_id, target_error = _resolve_send_target(session)
        if target_error:
            return {"error": target_error}
        try:
            send_result: dict | None = run_coroutine_sync(
                qq_adapter_client.send_message(
                    group_id=group_id,
                    user_id=user_id,
                    temp_source_group_id=temp_source_group_id,
                    message=segments,
                    llm_elapsed=0.0,
                ),
                loop,
                timeout=15,
            )
        except Exception as e:
            return {"error": f"发送复读消息失败: {e}"}

        if not send_result:
            return {"error": "复读消息发送失败（QQ adapter 无响应）"}

        sent_id = send_result.get("message_id")
        logger.info(
            "[tools] plus_one: 复读成功 原消息=%s 新消息=%s conv=%s_%s",
            raw_message_id, sent_id, session.conv_type, session.conv_id,
        )

        # ── 4. 录入 session 上下文 ──────────────────────────────
        from qq_adapter.segments import qq_adapter_segments_to_text, build_content_segments, _determine_content_type
        import app_state

        content_text = qq_adapter_segments_to_text(
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
            "original_message_id": raw_message_id,
            "sent_message_id": sent_id,
        }

    return execute
