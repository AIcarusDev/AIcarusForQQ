"""get_group_members.py — 获取群成员列表

需要运行时上下文：qq_adapter_client、session。
仅群聊目标可执行；非群聊会话中由工具返回明确错误。
"""

import asyncio
import logging
from typing import Any, Callable

from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, ToolContract

logger = logging.getLogger("AICQ.tools")

class GetGroupMembersArgs(ToolArgsModel):
    pass


TOOL_CONTRACT = ToolContract(
    name="get_group_members",
    description=(
        "获取当前群聊的成员列表。"
        "返回每位成员的 QQ 号（id）、QQ 昵称（name）和群名片（card）。"
        "最多返回前 20 条记录。"
    ),
    args_model=GetGroupMembersArgs,
)

# build_tools() 在发现此字段后，会检查 context 中是否存在对应键，
# 若任一键为 None / 缺失则自动跳过本工具。
REQUIRES_CONTEXT: list[str] = ["qq_adapter_client", "session"]


def make_handler(qq_adapter_client: Any, session: Any) -> Callable:
    """为特定群聊会话创建 get_group_members 处理函数。

    返回的函数是同步的，内部通过 run_coroutine_threadsafe 跨线程
    调用 QQ adapter 异步 API，适合在 asyncio.to_thread 的工作线程中使用。
    """
    def execute(**kwargs) -> dict:
        if getattr(session, "conv_type", "") != "group":
            return {"error": "get_group_members 仅能在群聊会话中使用"}
        group_id = str(getattr(session, "conv_id", "") or "").strip()
        if not group_id:
            return {"error": "当前群号未知，无法获取群成员列表"}
        if not qq_adapter_client or not qq_adapter_client.connected:
            logger.warning("[tools] get_group_members: QQ adapter 未连接 group_id=%s", group_id)
            return {"error": "QQ adapter 未连接，无法获取群成员列表"}
        loop: asyncio.AbstractEventLoop | None = qq_adapter_client._loop
        if loop is None or not loop.is_running():
            logger.warning("[tools] get_group_members: 事件循环不可用 group_id=%s", group_id)
            return {"error": "主事件循环不可用"}
        try:
            logger.info("[tools] get_group_members: 获取群成员列表开始 group_id=%s", group_id)
            raw: list[dict] | None = run_coroutine_sync(
                qq_adapter_client.send_api(
                    "get_group_member_list",
                    {"group_id": int(group_id)},
                ),
                loop,
                timeout=15,
            )
        except Exception as e:
            logger.warning("[tools] get_group_members: API 调用异常 group_id=%s — %s", group_id, e)
            return {"error": f"获取群成员列表失败: {e}"}

        if raw is None:
            logger.warning("[tools] get_group_members: API 返回为空 group_id=%s", group_id)
            return {"error": "API 返回为空（可能群号有误或权限不足）"}

        # 最多取前 20 条，防止 token 爆炸
        members_raw = raw[:20]
        members = []
        for m in members_raw:
            qq_id = str(m.get("user_id", ""))
            nickname = m.get("nickname", "")
            card = m.get("card", "") or nickname  # 群名片为空时回退到昵称
            members.append({"id": qq_id, "name": nickname, "card": card})

        logger.info("[tools] get_group_members: 获取完成 group_id=%s 成员数=%d", group_id, len(members))
        return {
            "group_id": group_id,
            "total_in_group": len(raw),
            "returned": len(members),
            "note": "最多返回前 20 条，超出部分已截断",
            "members": members,
        }

    return execute
