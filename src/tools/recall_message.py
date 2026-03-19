"""recall_message.py — 撤回一条已发送的消息

需要运行时上下文：napcat_client。
调用 NapCat delete_msg 接口，通过消息 ID 撤回消息。
"""

import asyncio
from typing import Any, Callable

DECLARATION: dict = {
    "max_calls_per_response": 3,
    "name": "recall_message",
    "description": (
        "撤回你之前已经发送的某条消息。"
        "需要提供消息 ID（message_id）。"
        "只能撤回自己发的消息，只能撤回 2 分钟内发送的消息。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message_id": {
                "type": "integer",
                "description": "要撤回的消息 ID（整数）。",
            },
            "motivation": {
                "type": "string",
                "description": "调用此工具的动机或原因。",
            },
        },
        "required": ["message_id"],
    },
}

REQUIRES_CONTEXT: list[str] = ["napcat_client"]


def make_handler(napcat_client: Any) -> Callable:
    def execute(message_id: int, **kwargs) -> dict:
        if not napcat_client or not napcat_client.connected:
            return {"error": "NapCat 未连接，无法撤回消息"}

        loop: asyncio.AbstractEventLoop | None = napcat_client._loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        try:
            coro = napcat_client.send_api_raw(
                "delete_msg",
                {"message_id": message_id},
            )
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            resp: dict | None = future.result(timeout=15)
        except Exception as e:
            err_str = str(e)
            if "recallMsg" in err_str and ("Timeout" in err_str or "decode failed" in err_str):
                return {"error": "撤回失败：只能撤回 2 分钟内自己发送的消息。"}
            return {"error": f"撤回消息失败: {e}"}

        if resp is None:
            return {"error": "撤回消息超时或 NapCat 未连接"}
        if resp.get("status") != "ok":
            msg = resp.get("message") or resp.get("msg") or "未知错误"
            return {"error": f"撤回消息失败: {msg}"}

        return {
            "success": True,
            "message_id": message_id,
            "note": "消息已撤回。",
        }

    return execute
