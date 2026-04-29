"""shift.py — 切换会话工具

Handler 校验目标会话合法性后，**直接修改全局焦点**
``app_state.current_focus``。下一 round 主循环会自动从新焦点的 session
重建 system / user prompt。

返回 ok=True 时，附带新焦点的元信息供模型理解；ok=False 时模型可在下一 round
看到错误并改用其它工具。
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger("AICQ.tools")

DECLARATION: dict = {
    "name": "shift",
    "description": (
        "切换到另一个会话。目标必须在白名单内且是好友/已加入的群。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["private", "group"],
                "description": "目标会话类型。",
            },
            "id": {
                "type": "string",
                "x-coerce-integer": True,
                "description": "目标会话 ID（QQ 号或群号）。",
            },
            "motivation": {"type": "string"},
        },
        "required": ["type", "id", "motivation"],
    },
}


async def _list_shift_type_candidates(target_id: str) -> set[str]:
    """列出同时满足白名单与当前联系人/群列表的候选会话类型。"""
    import app_state

    whitelist_cfg = app_state.napcat_cfg.get("whitelist", {})
    private_whitelist = {str(u) for u in whitelist_cfg.get("private_users", [])}
    group_whitelist = {str(g) for g in whitelist_cfg.get("group_ids", [])}
    allow_private = not private_whitelist or target_id in private_whitelist
    allow_group = not group_whitelist or target_id in group_whitelist

    client = app_state.napcat_client
    if not client or not client.connected:
        return set()

    candidates: set[str] = set()
    if allow_private:
        friends = await client.send_api("get_friend_list", {}) or []
        if isinstance(friends, list):
            friend_ids = {str(f.get("user_id", "")) for f in friends if isinstance(f, dict)}
            if target_id in friend_ids:
                candidates.add("private")

    if allow_group:
        groups = await client.send_api("get_group_list", {}) or []
        if isinstance(groups, list):
            group_ids = {str(g.get("group_id", "")) for g in groups if isinstance(g, dict)}
            if target_id in group_ids:
                candidates.add("group")

    return candidates


def _infer_missing_shift_type(target_id: str) -> tuple[str | None, str | None]:
    """在 type 缺失时，按白名单和当前联系人列表推断唯一会话类型。"""
    import app_state

    loop = getattr(app_state, "main_loop", None)
    if loop is None or not loop.is_running():
        return None, "主事件循环不可用"

    try:
        candidates = asyncio.run_coroutine_threadsafe(
            _list_shift_type_candidates(target_id), loop
        ).result(timeout=15)
    except Exception as exc:
        logger.warning("[shift] 类型推断异常: %s", exc)
        return None, f"类型推断异常: {exc}"

    if len(candidates) == 1:
        return next(iter(candidates)), None
    if not candidates:
        return None, f"无法根据联系人列表和白名单推断会话类型：{target_id}"
    return None, f"会话 ID {target_id} 同时匹配好友和群，必须显式提供 type"


def repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """缺失 type 时按当前联系人列表推断唯一会话类型。"""
    if "type" in args:
        return args, []

    target_id = args.get("id")
    if not isinstance(target_id, str) or not target_id:
        return args, []

    inferred_type, infer_error = _infer_missing_shift_type(target_id)
    if not inferred_type:
        if infer_error:
            logger.debug("[shift] schema 修复未补全 type: %s", infer_error)
        return args, []

    repaired_args = dict(args)
    repaired_args["type"] = inferred_type
    return repaired_args, [f"inferred type={inferred_type!r} from id {target_id!r}"]


async def _validate_shift_target(target_type: str, target_id: str) -> str | None:
    """检查 shift 目标是否在白名单和 QQ 联系人列表中。返回 None=合法，str=失败原因。"""
    import app_state

    if target_type not in ("private", "group"):
        return f"未知会话类型 {target_type!r}"

    whitelist_cfg = app_state.napcat_cfg.get("whitelist", {})
    if target_type == "private":
        private_whitelist = [str(u) for u in whitelist_cfg.get("private_users", [])]
        if private_whitelist and target_id not in private_whitelist:
            return f"私聊用户 {target_id} 不在白名单中"
    else:
        group_whitelist = [str(g) for g in whitelist_cfg.get("group_ids", [])]
        if group_whitelist and target_id not in group_whitelist:
            return f"群聊 {target_id} 不在白名单中"

    client = app_state.napcat_client
    if client and client.connected:
        if target_type == "private":
            friends = await client.send_api("get_friend_list", {}) or []
            if isinstance(friends, list):
                friend_ids = {str(f.get("user_id", "")) for f in friends if isinstance(f, dict)}
                if target_id not in friend_ids:
                    return f"用户 {target_id} 不在好友列表中"
        else:
            groups = await client.send_api("get_group_list", {}) or []
            if isinstance(groups, list):
                group_ids = {str(g.get("group_id", "")) for g in groups if isinstance(g, dict)}
                if target_id not in group_ids:
                    return f"群 {target_id} 不在已加入的群列表中"

    return None


def execute(type: str, id: str, motivation: str, **kwargs) -> dict:
    import app_state
    from llm.session import get_or_create_session

    loop = getattr(app_state, "main_loop", None)
    if loop is None or not loop.is_running():
        return {"ok": False, "error": "主事件循环不可用"}

    try:
        err = asyncio.run_coroutine_threadsafe(
            _validate_shift_target(type, id), loop
        ).result(timeout=15)
    except Exception as e:
        logger.warning("[shift] 校验异常: %s", e)
        return {"ok": False, "error": f"校验异常: {e}"}

    if err:
        logger.warning("[shift] 目标校验失败: %s", err)
        return {"ok": False, "error": err}

    new_key = f"{type}_{id}"
    target = get_or_create_session(new_key)
    if not target.conv_type:
        target.set_conversation_meta(type, id)

    prev_key = app_state.current_focus
    app_state.current_focus = new_key
    target.last_wake_reason = (
        f"shift 自 {prev_key or '?'}（动机：{motivation}）"
        if prev_key and prev_key != new_key
        else f"shift（动机：{motivation}）"
    )

    logger.info("[shift] 焦点切换 %s → %s", prev_key, new_key)
    return {
        "ok": True,
        "now_focusing": {
            "type": type,
            "id": id,
            "name": target.conv_name or "",
        },
        "motivation": motivation,
    }
