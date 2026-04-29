"""shift.py — 切换会话工具

Handler 校验目标会话合法性后，**直接修改全局焦点**
``app_state.current_focus``。下一 round 主循环会自动从新焦点的 session
重建 system / user prompt。

返回 ok=True 时，附带新焦点的元信息供模型理解；ok=False 时模型可在下一 round
看到错误并改用其它工具。
"""

import asyncio
import logging

logger = logging.getLogger("AICQ.tools")

DECLARATION: dict = {
    "name": "shift",
    "description": (
        "切换到另一个会话。目标必须在白名单内且是好友/已加入的群。"
        "切换后下一轮思考的上下文（聊天记录、可用工具）即变为该会话。"
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
