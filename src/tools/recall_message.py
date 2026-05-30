"""recall_message.py — 撤回一条已发送的消息

需要运行时上下文：session、qq_adapter_client。
调用 QQ adapter delete_msg 接口，通过消息 ID 撤回消息；可在撤回成功后补发纯文本。
"""

import asyncio
from datetime import datetime
from typing import Any, Callable

from tools._async_bridge import run_coroutine_sync
from qq_adapter.conversation import format_adapter_error

DECLARATION: dict = {
    "name": "recall_message",
    "description": (
        "撤回你之前已经发送的某条消息。"
        "需要提供消息 ID（message_id）。"
        "通常只能撤回自己发的消息，且只能撤回 2 分钟内发送的消息。"
        "如果当前会话是群聊，且你是管理员，则可以撤回其它普通成员发送的群消息，并且没有时间限制。"
        "若要撤回后补发纯文本，填写 edited_text 即可。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message_id": {
                "type": "integer",
                "description": "要撤回的消息 ID（整数）。",
            },
            "edited_text": {
                "type": "string",
                "description": "可选。如果填写，会在撤回成功后发送这条纯文本消息，视为对原内容的编辑后发送。",
            },
        },
        "required": ["message_id"],
    },
}

REQUIRES_CONTEXT: list[str] = ["session", "qq_adapter_client"]


def _build_send_msg_params(session: Any, text: str) -> dict[str, Any] | None:
    conv_type = getattr(session, "conv_type", "")
    conv_id = getattr(session, "conv_id", "")
    message = [{"type": "text", "data": {"text": text}}]
    try:
        if conv_type == "group":
            return {
                "message_type": "group",
                "group_id": int(conv_id),
                "message": message,
            }
        if conv_type == "private":
            return {
                "message_type": "private",
                "user_id": int(conv_id),
                "message": message,
            }
        if conv_type == "temp":
            source_group_id = str(getattr(session, "temp_source_group_id", "") or "").strip()
            if not source_group_id:
                return None
            return {
                "message_type": "private",
                "user_id": int(conv_id),
                "group_id": int(source_group_id),
                "message": message,
            }
    except (TypeError, ValueError):
        return None
    return None


def _record_edited_text_message(
    *,
    session: Any,
    loop: asyncio.AbstractEventLoop,
    message_id: str,
    edited_text: str,
) -> None:
    import app_state
    from database import save_chat_message
    from llm.core.round_context import get_current_inner_state
    from web.debug_server import broadcast_chat_event

    conv_type = getattr(session, "conv_type", "")
    conv_id = getattr(session, "conv_id", "")
    conversation_id = f"{conv_type}_{conv_id}"
    entry = {
        "role": "bot",
        "message_id": message_id,
        "sender_id": getattr(session, "_qq_id", "") or "bot",
        "sender_name": getattr(session, "_qq_name", "") or "",
        "sender_role": "",
        "timestamp": datetime.now(app_state.TIMEZONE).isoformat(),
        "content": edited_text,
        "content_type": "text",
        "content_segments": [{"type": "text", "text": edited_text}],
    }
    session.add_to_context(entry)
    asyncio.run_coroutine_threadsafe(save_chat_message(conversation_id, entry), loop)
    asyncio.run_coroutine_threadsafe(
        broadcast_chat_event({
            "type": "bot_turn",
            "conv_id": conversation_id,
            "conv_name": getattr(session, "conv_name", "") or conversation_id,
            "conv_type": conv_type or "unknown",
            "entries": [entry],
            "inner_state": get_current_inner_state(),
        }),
        loop,
    )


def make_handler(session: Any, qq_adapter_client: Any) -> Callable:
    def execute(message_id: int, edited_text: str | None = None, **kwargs) -> dict:
        if not qq_adapter_client or not qq_adapter_client.connected:
            return {"error": "QQ adapter 未连接，无法撤回消息"}

        loop: asyncio.AbstractEventLoop | None = qq_adapter_client._loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        replacement_text = edited_text.strip() if isinstance(edited_text, str) else ""
        send_params = None
        if replacement_text:
            send_params = _build_send_msg_params(session, replacement_text)
            if send_params is None:
                return {"error": "当前会话不支持撤回后补发编辑文本"}

        try:
            resp: dict | None = run_coroutine_sync(
                qq_adapter_client.send_api_raw(
                    "delete_msg",
                    {"message_id": message_id},
                ),
                loop,
                timeout=15,
            )
        except Exception as e:
            err_str = str(e)
            if "recallMsg" in err_str and ("Timeout" in err_str or "decode failed" in err_str):
                return {"error": "撤回失败：只能撤回 2 分钟内自己发送的消息。"}
            return {"error": f"撤回消息失败: {e}"}

        if resp is None:
            return {"error": "撤回消息超时或 QQ adapter 未连接"}
        if resp.get("status") != "ok":
            msg = resp.get("message") or resp.get("msg") or "未知错误"
            return {"error": f"撤回消息失败: {msg}"}

        result = {
            "success": True,
            "message_id": message_id,
            "note": "消息已撤回。",
            "edited_message_sent": False,
        }
        if not replacement_text:
            return result

        try:
            send_result: dict | None = run_coroutine_sync(
                qq_adapter_client.send_api(
                    "send_msg",
                    send_params,
                ),
                loop,
                timeout=15,
            )
        except Exception as e:
            result["edited_message_error"] = f"编辑文本发送失败: {e}"
            return result

        if not send_result or send_result.get("message_id") is None:
            result["edited_message_error"] = format_adapter_error(
                getattr(qq_adapter_client, "last_api_error", None),
                "编辑文本发送失败（QQ adapter 无响应）",
            )
            return result

        edited_message_id = str(send_result["message_id"])
        _record_edited_text_message(
            session=session,
            loop=loop,
            message_id=edited_message_id,
            edited_text=replacement_text,
        )
        result["edited_message_sent"] = True
        result["edited_message_id"] = edited_message_id
        return result

    return execute
