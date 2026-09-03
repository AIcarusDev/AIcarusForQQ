"""get_qq_signature.py — 通过 QQ 号查询用户的个性签名

需要运行时上下文：qq_client。
调用 QQ adapter get_stranger_info 接口，取目标用户的 longNick（签名）字段。
不传 user_id 时默认查询 bot 自身。
"""

import asyncio
import logging
from typing import Any, Callable

from pydantic import Field

from platforms.qq.adapter.conversation import format_adapter_error
from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, ToolContract

logger = logging.getLogger("AICQ.tools")

class GetQQSignatureArgs(ToolArgsModel):
    user_id: str | None = Field(
        default=None,
        description="要查询 QQ 个性签名的 QQ 号。不填则查询你自己的签名。",
    )


TOOL_CONTRACT = ToolContract(
    name="get_qq_signature",
    description=(
        "通过 QQ 号查询指定用户的 QQ 个性签名，也可查询你自己的 QQ 个性签名。"
        "不传 user_id 时默认查询你自身的签名。"
    ),
    args_model=GetQQSignatureArgs,
)

REQUIRES_CONTEXT: list[str] = ["qq_client"]
PARALLEL_SAFE = True
PARALLEL_KEY = "qq_client_read"


def make_handler(qq_client: Any) -> Callable:
    def execute(**kwargs) -> dict:
        if not qq_client or not qq_client.connected:
            return {"error": "QQ adapter 未连接，无法查询签名"}

        raw_uid: str | None = kwargs.get("user_id")
        if raw_uid:
            target_id = str(raw_uid).strip()
        else:
            target_id = qq_client.bot_id

        if not target_id:
            return {"error": "bot_id 未初始化且未传入 user_id，无法查询签名"}

        loop: asyncio.AbstractEventLoop | None = qq_client._loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        try:
            data: dict | None = run_coroutine_sync(
                qq_client.send_api(
                    "get_stranger_info",
                    {"user_id": int(target_id), "no_cache": True},
                ),
                loop,
                timeout=15,
            )
        except Exception:
            logger.warning("[tools] get_qq_signature: 查询异常", exc_info=True)
            return {"error": "查询签名失败"}

        if data is None:
            return {
                "error": format_adapter_error(
                    getattr(qq_client, "last_api_error", None),
                    "API 返回为空，可能权限不足或 QQ 号有误",
                )
            }

        signature = data.get("longNick") or data.get("sign") or ""

        result: dict = {
            "qq_number": target_id,
            "signature": signature if signature else "（当前签名为空）",
        }

        return result

    return execute


