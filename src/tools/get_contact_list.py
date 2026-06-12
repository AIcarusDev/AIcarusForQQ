"""get_contact_list.py — 获取好友、群聊或已登记临时会话列表

需要运行时上下文：qq_adapter_client、config。
返回当前平台（QQ）中，同时满足以下条件的条目：
  1. 好友/群聊在 QQ adapter 好友/群列表中
  2. 白名单模式下在 config.qq_adapter.whitelist 中；自由模式下不过滤

返回字段：name（名称）和 qqid（QQ 号）。
临时会话只返回已打开/登记过的会话，不枚举潜在群成员。
"""

import asyncio
import logging
from typing import Any, Callable

from qq_adapter.access_control import is_session_allowed_by_config
from tools._async_bridge import run_coroutine_sync

logger = logging.getLogger("AICQ.tools")

ALWAYS_AVAILABLE: bool = False

DECLARATION: dict = {
    "name": "get_contact_list",
    "description": (
        "获取你的好友、群聊或已打开的临时会话列表。"
        "可选参数 type 指定类型：friend（好友）、group（群聊）或 temp（临时会话），"
        "不填则同时返回三类。临时会话只列已经打开/登记过的对象。"
        "每项包含 name（名称）和 qqid（QQ 号）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["friend", "group", "temp"],
            },
        },
        "required": [],
    },
}

REQUIRES_CONTEXT: list[str] = ["qq_adapter_client", "config"]


def _main_loop_fallback() -> asyncio.AbstractEventLoop | None:
    try:
        import app_state
        return getattr(app_state, "main_loop", None)
    except Exception:
        return None


def _collect_live_temp_sessions(qq_adapter_cfg: dict) -> dict[str, dict]:
    from llm.session import sessions

    result: dict[str, dict] = {}
    for session in sessions.values():
        if getattr(session, "conv_type", "") != "temp":
            continue
        uid = str(getattr(session, "conv_id", "") or "").strip()
        if not uid or not is_session_allowed_by_config(qq_adapter_cfg, "temp", uid):
            continue
        result[uid] = {
            "name": getattr(session, "conv_name", "") or uid,
            "qqid": uid,
            "source_group_id": str(getattr(session, "temp_source_group_id", "") or ""),
            "source_group_name": str(getattr(session, "temp_source_group_name", "") or ""),
        }
    return result


def make_handler(qq_adapter_client: Any, config: dict) -> Callable:
    """创建 get_contact_list 处理函数。"""

    def execute(**kwargs) -> dict:
        query_type: str | None = kwargs.get("type")  # "friend" / "group" / "temp" / None

        needs_adapter = query_type in (None, "friend", "group")
        if needs_adapter and (not qq_adapter_client or not qq_adapter_client.connected):
            return {"error": "QQ adapter 未连接，无法获取列表"}

        loop: asyncio.AbstractEventLoop | None = (
            getattr(qq_adapter_client, "_loop", None)
            if qq_adapter_client is not None
            else None
        ) or _main_loop_fallback()
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        qq_adapter_cfg: dict = config.get("qq_adapter", {})
        result: dict = {}

        # ── 好友列表 ────────────────────────────────────────────
        if query_type in (None, "friend"):
            try:
                raw_friends: list[dict] | None = run_coroutine_sync(
                    qq_adapter_client.send_api("get_friend_list", {}),
                    loop,
                    timeout=15,
                )
            except Exception as e:
                logger.warning("[tools] get_contact_list: 获取好友列表异常 — %s", e)
                raw_friends = None

            if raw_friends is None:
                result["friend_error"] = "获取好友列表失败"
            else:
                friends = []
                for f in raw_friends:
                    if not isinstance(f, dict):
                        continue
                    uid = str(f.get("user_id", ""))
                    if not is_session_allowed_by_config(qq_adapter_cfg, "private", uid):
                        continue
                    name = f.get("remark") or f.get("nickname") or uid
                    friends.append({"name": name, "qqid": uid})
                result["friends"] = friends

        # ── 群聊列表 ────────────────────────────────────────────
        if query_type in (None, "group"):
            try:
                raw_groups: list[dict] | None = run_coroutine_sync(
                    qq_adapter_client.send_api("get_group_list", {}),
                    loop,
                    timeout=15,
                )
            except Exception as e:
                logger.warning("[tools] get_contact_list: 获取群聊列表异常 — %s", e)
                raw_groups = None

            if raw_groups is None:
                result["group_error"] = "获取群聊列表失败"
            else:
                groups = []
                for g in raw_groups:
                    if not isinstance(g, dict):
                        continue
                    gid = str(g.get("group_id", ""))
                    if not is_session_allowed_by_config(qq_adapter_cfg, "group", gid):
                        continue
                    name = g.get("group_name") or gid
                    groups.append({"name": name, "qqid": gid})
                result["groups"] = groups

        # ── 已登记临时会话 ─────────────────────────────────────
        if query_type in (None, "temp"):
            try:
                from database import load_chat_sessions

                temp_by_uid: dict[str, dict] = {}
                persisted: list[dict] = run_coroutine_sync(
                    load_chat_sessions(),
                    loop,
                    timeout=15,
                )
                for row in persisted:
                    if str(row.get("conv_type", "")) != "temp":
                        continue
                    uid = str(row.get("conv_id", "") or "").strip()
                    if not uid or not is_session_allowed_by_config(qq_adapter_cfg, "temp", uid):
                        continue
                    temp_by_uid[uid] = {
                        "name": row.get("conv_name") or uid,
                        "qqid": uid,
                        "source_group_id": str(row.get("temp_source_group_id", "") or ""),
                        "source_group_name": str(row.get("temp_source_group_name", "") or ""),
                    }
                temp_by_uid.update(_collect_live_temp_sessions(qq_adapter_cfg))
                result["temps"] = list(temp_by_uid.values())
            except Exception as e:
                logger.warning("[tools] get_contact_list: 获取临时会话列表异常 — %s", e)
                result["temp_error"] = f"获取临时会话列表失败: {e}"

        logger.info(
            "[tools] get_contact_list: 查询完成 type=%s friends=%s groups=%s temps=%s",
            query_type,
            len(result.get("friends", [])) if "friends" in result else "N/A",
            len(result.get("groups", [])) if "groups" in result else "N/A",
            len(result.get("temps", [])) if "temps" in result else "N/A",
        )
        return result

    return execute
