"""set_self_qq_signature.py — 修改自己的 QQ 个性签名

需要运行时上下文：napcat_client。
调用 NapCat set_qq_profile 接口，仅覆盖 personal_note（签名）字段。
"""

import asyncio
from typing import Any, Callable

from tools._async_bridge import run_coroutine_sync

ALWAYS_AVAILABLE: bool = False

DECLARATION: dict = {
    "name": "set_self_qq_signature",
    "description": (
        "修改（覆盖）你自己的 QQ 个性签名。"
        "设置成功后新签名将立即生效。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "signature": {
                "type": "string",
                "description": "要设置的新签名内容，传空字符串可清空签名。",
            },
            "motivation": {"type": "string"},
        },
        "required": ["signature", "motivation"],
    },
}

REQUIRES_CONTEXT: list[str] = ["napcat_client"]


def make_handler(napcat_client: Any) -> Callable:
    def execute(signature: str = "", **kwargs) -> dict:
        if not napcat_client or not napcat_client.connected:
            return {"error": "NapCat 未连接，无法修改签名"}

        bot_id = napcat_client.bot_id
        if not bot_id:
            return {"error": "bot_id 未初始化，无法修改签名"}

        loop: asyncio.AbstractEventLoop | None = napcat_client._loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        # set_qq_profile 要求 nickname 必填，先取当前昵称再提交
        try:
            info: dict | None = run_coroutine_sync(
                napcat_client.send_api(
                    "get_stranger_info",
                    {"user_id": int(bot_id), "no_cache": False},
                ),
                loop,
                timeout=15,
            )
        except Exception as e:
            return {"error": f"获取当前昵称失败，无法修改签名: {e}"}

        if info is None:
            return {"error": "获取当前昵称失败，无法修改签名"}

        nickname = info.get("nickname", "")

        try:
            resp: dict | None = run_coroutine_sync(
                napcat_client.send_api_raw(
                    "set_qq_profile",
                    {"nickname": nickname, "personal_note": signature},
                ),
                loop,
                timeout=15,
            )
        except Exception as e:
            return {"error": f"修改签名失败: {e}"}

        if resp is None:
            return {"error": "修改签名超时或 NapCat 未连接"}
        if resp.get("status") != "ok":
            msg = resp.get("message") or resp.get("msg") or "未知错误"
            return {"error": f"修改签名失败: {msg}"}

        return {
            "success": True,
            "new_signature": signature if signature else "（已清空签名）",
            "note": "签名已更新，若未立即生效请稍候。",
        }

    return execute