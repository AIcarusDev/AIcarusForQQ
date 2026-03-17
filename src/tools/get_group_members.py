"""get_group_members.py — 获取群成员列表

需要运行时上下文：napcat_client、group_id。
仅在群聊会话中被 build_tools() 纳入注册表。
"""

import asyncio
from typing import Any, Callable

DECLARATION: dict = {
    "max_calls_per_response": 1,
    "name": "get_group_members",
    "description": (
        "获取当前群聊的成员列表（仅群聊会话中可用）。"
        "返回每位成员的 QQ 号（id）、QQ 昵称（name）和群名片（card）。"
        "最多返回前 20 条记录。"
        "当你需要知道群里有哪些人、查找某人的 QQ 号或群名片时可以调用。"
        "返回内容仅自己可见。"
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

# build_tools() 在发现此字段后，会检查 context 中是否存在对应键，
# 若任一键为 None / 缺失则自动跳过本工具。
REQUIRES_CONTEXT: list[str] = ["napcat_client", "group_id"]


def make_handler(napcat_client: Any, group_id: str) -> Callable:
    """为特定群聊会话创建 get_group_members 处理函数。

    返回的函数是同步的，内部通过 run_coroutine_threadsafe 跨线程
    调用 NapCat 异步 API，适合在 asyncio.to_thread 的工作线程中使用。
    """
    def execute(**kwargs) -> dict:
        import logging
        logger = logging.getLogger("AICQ.tools")
        
        if not napcat_client or not napcat_client.connected:
            logger.warning("[tools] get_group_members: NapCat 未连接 group_id=%s", group_id)
            return {"error": "NapCat 未连接，无法获取群成员列表"}
        loop: asyncio.AbstractEventLoop | None = napcat_client._loop
        if loop is None or not loop.is_running():
            logger.warning("[tools] get_group_members: 事件循环不可用 group_id=%s", group_id)
            return {"error": "主事件循环不可用"}
        try:
            logger.info("[tools] get_group_members: 获取群成员列表开始 group_id=%s", group_id)
            coro = napcat_client.send_api(
                "get_group_member_list",
                {"group_id": int(group_id)},
            )
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            raw: list[dict] | None = future.result(timeout=15)
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
