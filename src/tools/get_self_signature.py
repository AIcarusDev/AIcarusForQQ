"""get_self_signature.py — 查看自己当前的 QQ 个性签名

需要运行时上下文：napcat_client。
调用 NapCat get_stranger_info 接口，取 bot 自身 QQ 的 sign 字段。
"""

import asyncio
from typing import Any, Callable

DECLARATION: dict = {
    "max_calls_per_response": 1,
    "name": "get_self_signature",
    "description": (
        "查看你自己当前的 QQ 个性签名。"
        "返回内容仅自己可见，若不主动透露则无法被他人知晓。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "motivation": {
                "type": "string",
                "description": "调用此工具的动机或原因。",
            },
        },
    },
}

REQUIRES_CONTEXT: list[str] = ["napcat_client"]


def make_handler(napcat_client: Any) -> Callable:
    def execute(**kwargs) -> dict:
        if not napcat_client or not napcat_client.connected:
            return {"error": "NapCat 未连接，无法查询签名"}

        bot_id = napcat_client.bot_id
        if not bot_id:
            return {"error": "bot_id 未初始化，无法查询签名"}

        loop: asyncio.AbstractEventLoop | None = napcat_client._loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        try:
            coro = napcat_client.send_api(
                "get_stranger_info",
                {"user_id": int(bot_id), "no_cache": True},
            )
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            data: dict | None = future.result(timeout=15)
        except Exception as e:
            return {"error": f"查询签名失败: {e}"}

        if data is None:
            return {"error": "API 返回为空，可能权限不足或 QQ 号有误"}

        # NapCat 返回的签名字段为 longNick（QQ 协议原始字段名），兼容 sign 作为回退
        signature = data.get("longNick") or data.get("sign") or ""
        return {
            "qq_number": bot_id,
            "signature": signature if signature else "（当前签名为空）",
        }

    return execute
