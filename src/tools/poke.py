"""poke.py — 发起 QQ 戳一戳

向群成员或私聊对象发起戳一戳操作。这是个没啥用的功能，实际意义不大。

工具在 LLM 输出阶段之前执行，戳一戳会立即发出。

群聊使用 group_poke，私聊使用 friend_poke。
"""

import asyncio
import logging
from typing import Any, Callable

logger = logging.getLogger("AICQ.tools")

DECLARATION: dict = {
    "name": "poke",
    "description": (
        "向他人发起 qq 戳一戳。这是个没啥用的功能，实际意义不大。"
        "戳一戳会立即发出（会在你调用后就发出）。"
        "不要滥用。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "integer",
                "description": "要戳一戳的目标用户 QQ 号。",
            },
            "motivation": {"type": "string"},
        },
        "required": ["user_id", "motivation"],
    },
}

REQUIRES_CONTEXT: list[str] = ["napcat_client", "session"]


def make_handler(napcat_client: Any, session: Any) -> Callable:
    def execute(user_id: int, **kwargs) -> dict:
        if not napcat_client or not napcat_client.connected:
            return {"error": "NapCat 未连接，无法发起戳一戳"}

        loop: asyncio.AbstractEventLoop | None = napcat_client._loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        # ── 根据会话类型选择接口 ──────────────────────────────────
        is_group = session.conv_type == "group"
        if is_group:
            api_action = "group_poke"
            api_params = {"group_id": int(session.conv_id), "user_id": int(user_id)}
        else:
            api_action = "friend_poke"
            api_params = {"user_id": int(user_id)}

        # ── 发起戳一戳 ────────────────────────────────────────────
        try:
            poke_coro = napcat_client.send_api_raw(api_action, api_params)
            poke_result: dict | None = asyncio.run_coroutine_threadsafe(
                poke_coro, loop
            ).result(timeout=15)
        except Exception as e:
            return {"error": f"戳一戳失败: {e}"}

        if not poke_result or poke_result.get("status") != "ok":
            return {
                "error": (
                    f"戳一戳失败（NapCat 响应异常）: "
                    f"{poke_result.get('message', '未知错误')}"
                    if poke_result
                    else "戳一戳失败（NapCat 无响应）"
                )
            }

        return {"success": True, "message": f"成功向 {user_id} 发起了戳一戳"}

    return execute
