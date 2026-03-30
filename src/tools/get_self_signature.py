"""get_self_signature.py — 通过 QQ 号查询用户的个性签名

需要运行时上下文：napcat_client。
调用 NapCat get_stranger_info 接口，取目标用户的 longNick（签名）字段。
不传 user_id 时默认查询 bot 自身。
"""

import asyncio
from typing import Any, Callable

DECLARATION: dict = {
    "max_calls_per_response": 2,
    "name": "get_self_signature",
    "description": (
        "通过 QQ 号查询指定用户的个性签名，也可查询你自己的签名。"
        "不传 user_id 时默认查询你自身的签名。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "要查询签名的 QQ 号。不填则查询你自己的签名。",
            },
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

        # user_id 未传时默认查询 bot 自身
        raw_uid: str | None = kwargs.get("user_id")
        if raw_uid:
            target_id = str(raw_uid).strip()
        else:
            target_id = napcat_client.bot_id

        if not target_id:
            return {"error": "bot_id 未初始化且未传入 user_id，无法查询签名"}

        loop: asyncio.AbstractEventLoop | None = napcat_client._loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        try:
            coro = napcat_client.send_api(
                "get_stranger_info",
                {"user_id": int(target_id), "no_cache": True},
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
            "qq_number": target_id,
            "signature": signature if signature else "（当前签名为空）",
        }

    return execute
