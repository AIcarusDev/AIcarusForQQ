"""get_list.py — 获取机器人好友或群聊列表

需要运行时上下文：napcat_client、config。
返回当前平台（QQ）中，同时满足以下条件的条目：
  1. 在 NapCat 好友/群列表中（即机器人实际已加为好友/已加入的群）
  2. 在 config.napcat.whitelist 白名单中（白名单为空时不过滤）

返回字段：name（名称）和 qqid（QQ 号）。
"""

import asyncio
import logging
from typing import Any, Callable

logger = logging.getLogger("AICQ.tools")

WATCHER_ALLOW: bool = True  # watcher 模式下允许调用（用于切换目标会话）

DECLARATION: dict = {
    "max_calls_per_response": 1,
    "name": "get_list",
    "description": (
        "获取你的好友列表或群聊列表。"
        "返回在白名单中且存在的条目。"
        "可选参数 type 指定类型：friend（好友）或 group（群聊），"
        "不填则同时返回好友和群聊。"
        "每项包含 name（名称）和 qqid（QQ 号）。"
        "返回内容仅自己可见。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["friend", "group"],
                "description": "要查询的类型：friend（好友列表）或 group（群聊列表）。不填则同时返回两者。",
            },
            "motivation": {
                "type": "string",
            },
        },
        "required": ["motivation"],
    },
}

REQUIRES_CONTEXT: list[str] = ["napcat_client", "config"]


def make_handler(napcat_client: Any, config: dict) -> Callable:
    """创建 get_list 处理函数。"""

    def execute(**kwargs) -> dict:
        query_type: str | None = kwargs.get("type")  # "friend" / "group" / None

        if not napcat_client or not napcat_client.connected:
            return {"error": "NapCat 未连接，无法获取列表"}

        loop: asyncio.AbstractEventLoop | None = napcat_client._loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        whitelist_cfg: dict = (
            config.get("napcat", {}).get("whitelist", {})
        )
        private_whitelist: list[str] = [
            str(u) for u in whitelist_cfg.get("private_users", [])
        ]
        group_whitelist: list[str] = [
            str(g) for g in whitelist_cfg.get("group_ids", [])
        ]

        result: dict = {}

        # ── 好友列表 ────────────────────────────────────────────
        if query_type in (None, "friend"):
            try:
                future = asyncio.run_coroutine_threadsafe(
                    napcat_client.send_api("get_friend_list", {}),
                    loop,
                )
                raw_friends: list[dict] | None = future.result(timeout=15)
            except Exception as e:
                logger.warning("[tools] get_list: 获取好友列表异常 — %s", e)
                raw_friends = None

            if raw_friends is None:
                result["friend_error"] = "获取好友列表失败"
            else:
                friends = []
                for f in raw_friends:
                    if not isinstance(f, dict):
                        continue
                    uid = str(f.get("user_id", ""))
                    # 白名单过滤：为空则不限
                    if private_whitelist and uid not in private_whitelist:
                        continue
                    name = f.get("remark") or f.get("nickname") or uid
                    friends.append({"name": name, "qqid": uid})
                result["friends"] = friends

        # ── 群聊列表 ────────────────────────────────────────────
        if query_type in (None, "group"):
            try:
                future = asyncio.run_coroutine_threadsafe(
                    napcat_client.send_api("get_group_list", {}),
                    loop,
                )
                raw_groups: list[dict] | None = future.result(timeout=15)
            except Exception as e:
                logger.warning("[tools] get_list: 获取群聊列表异常 — %s", e)
                raw_groups = None

            if raw_groups is None:
                result["group_error"] = "获取群聊列表失败"
            else:
                groups = []
                for g in raw_groups:
                    if not isinstance(g, dict):
                        continue
                    gid = str(g.get("group_id", ""))
                    # 白名单过滤：为空则不限
                    if group_whitelist and gid not in group_whitelist:
                        continue
                    name = g.get("group_name") or gid
                    groups.append({"name": name, "qqid": gid})
                result["groups"] = groups

        logger.info(
            "[tools] get_list: 查询完成 type=%s friends=%s groups=%s",
            query_type,
            len(result.get("friends", [])) if "friends" in result else "N/A",
            len(result.get("groups", [])) if "groups" in result else "N/A",
        )
        return result

    return execute
