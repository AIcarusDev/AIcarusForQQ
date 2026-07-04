"""poke.py — 发起 QQ 戳一戳

向群成员或私聊对象发起戳一戳操作。这是个没啥用的功能，实际意义不大。

工具在 LLM 输出阶段之前执行，戳一戳会立即发出。

群聊使用 group_poke，私聊使用 friend_poke。
"""

import asyncio
import logging
from typing import Any, Callable

from pydantic import Field

from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, ToolContract

logger = logging.getLogger("AICQ.tools")

class PokeArgs(ToolArgsModel):
    user_id: str = Field(
        min_length=1,
        json_schema_extra={"x-coerce-integer": True},
        description="要戳一戳的目标用户 QQ 号。",
    )


TOOL_CONTRACT = ToolContract(
    name="poke",
    description="向他人发起 qq 戳一戳。",
    args_model=PokeArgs,
)

EXTERNALLY_PERCEPTIBLE: bool = True
TOOL_EFFECT: dict[str, str] = {"surface": "qq", "kind": "session_write"}

REQUIRES_CONTEXT: list[str] = ["qq_client", "session"]


def make_handler(qq_client: Any, session: Any) -> Callable:
    def execute(user_id: str, **kwargs) -> dict:
        if not qq_client or not qq_client.connected:
            return {"error": "QQ adapter 未连接，无法发起戳一戳"}

        loop: asyncio.AbstractEventLoop | None = qq_client._loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        target_user_id = str(user_id).strip()
        if not target_user_id.isdigit():
            return {"error": f"无效的 QQ 号：{user_id!r}，请传入纯数字字符串"}

        # ── 根据会话类型选择接口 ──────────────────────────────────
        is_group = session.conv_type == "group"
        if is_group:
            api_action = "group_poke"
            api_params = {"group_id": int(session.conv_id), "user_id": int(target_user_id)}
        else:
            api_action = "friend_poke"
            api_params = {"user_id": int(target_user_id)}

        # ── 发起戳一戳 ────────────────────────────────────────────
        try:
            poke_result: dict | None = run_coroutine_sync(
                qq_client.send_api_raw(api_action, api_params),
                loop,
                timeout=15,
            )
        except Exception as e:
            return {"error": f"戳一戳失败: {e}"}

        if not poke_result or poke_result.get("status") != "ok":
            return {
                "error": (
                    f"戳一戳失败（QQ adapter 响应异常）: "
                    f"{poke_result.get('message', '未知错误')}"
                    if poke_result
                    else "戳一戳失败（QQ adapter 无响应）"
                )
            }

        return {"success": True, "message": f"成功向 {target_user_id} 发起了戳一戳"}

    return execute


