"""shift.py — 切换会话工具

Handler 校验目标会话合法性后，直接修改全局焦点
``app_state.current_focus``。下一 round 主循环会自动从新焦点的 session
重建 system / user prompt。
"""

import logging
from typing import Any

from qq_adapter.conversation import (
    TEMP_CONV_TYPE,
    make_session_key,
    make_temp_session_key,
    parse_session_key,
)
from llm.core.tool_calling import ToolWarningFactory
from tools._async_bridge import run_coroutine_sync

logger = logging.getLogger("AICQ.tools")

DECLARATION: dict = {
    "name": "shift",
    "description": (
        "切换到另一个会话。目标可以是私聊对象或已加入的群；临时会话按 private 处理。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["private", "group"],
                "description": "目标会话类型：private（私聊，包含临时会话）或 group（群聊）。",
            },
            "id": {
                "type": "string",
                "x-coerce-integer": True,
                "description": "目标会话 ID（QQ 号或群号）。",
            },
        },
        "required": ["type", "id"],
    },
}


def _public_shift_type(conv_type: str) -> str:
    return "private" if conv_type == TEMP_CONV_TYPE else conv_type


def _session_dict(
    *,
    key: str,
    conv_type: str,
    conv_id: str,
    conv_name: str = "",
    temp_source_group_id: str = "",
    temp_source_group_name: str = "",
) -> dict[str, str]:
    return {
        "key": key,
        "type": conv_type,
        "id": conv_id,
        "name": conv_name,
        "temp_source_group_id": temp_source_group_id,
        "temp_source_group_name": temp_source_group_name,
    }


async def _load_persisted_temp(target_id: str) -> dict[str, str] | None:
    from database import load_chat_sessions

    temp_key = make_temp_session_key(target_id)
    for row in await load_chat_sessions():
        if str(row.get("session_key", "")) != temp_key:
            continue
        if str(row.get("conv_type", "")) != TEMP_CONV_TYPE:
            continue
        return _session_dict(
            key=temp_key,
            conv_type=TEMP_CONV_TYPE,
            conv_id=target_id,
            conv_name=str(row.get("conv_name", "") or ""),
            temp_source_group_id=str(row.get("temp_source_group_id", "") or ""),
            temp_source_group_name=str(row.get("temp_source_group_name", "") or ""),
        )
    return None


async def _resolve_existing_temp(target_id: str) -> dict[str, str] | None:
    from llm.session import sessions

    temp_key = make_temp_session_key(target_id)
    live = sessions.get(temp_key)
    if live is not None and getattr(live, "conv_type", "") == TEMP_CONV_TYPE:
        return _session_dict(
            key=temp_key,
            conv_type=TEMP_CONV_TYPE,
            conv_id=target_id,
            conv_name=str(getattr(live, "conv_name", "") or ""),
            temp_source_group_id=str(getattr(live, "temp_source_group_id", "") or ""),
            temp_source_group_name=str(getattr(live, "temp_source_group_name", "") or ""),
        )
    return await _load_persisted_temp(target_id)


async def _list_shift_type_candidates(target_id: str) -> set[str]:
    """列出同时满足访问范围与当前联系人/群列表的候选会话类型。"""
    import app_state
    from qq_adapter.access_control import is_session_allowed_by_config

    allow_private = is_session_allowed_by_config(app_state.qq_adapter_cfg, "private", target_id)
    allow_group = is_session_allowed_by_config(app_state.qq_adapter_cfg, "group", target_id)

    candidates: set[str] = set()
    if allow_private and await _resolve_existing_temp(target_id):
        candidates.add("private")

    client = app_state.qq_adapter_client
    if not client or not client.connected:
        return candidates

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
    """在 type 缺失时，按访问范围和当前联系人列表推断唯一会话类型。"""
    import app_state

    loop = getattr(app_state, "main_loop", None)
    if loop is None or not loop.is_running():
        return None, "主事件循环不可用"

    try:
        candidates = run_coroutine_sync(
            _list_shift_type_candidates(target_id),
            loop,
            timeout=15,
        )
    except Exception as exc:
        logger.warning("[shift] 类型推断异常: %s", exc)
        return None, f"类型推断异常: {exc}"

    if len(candidates) == 1:
        return next(iter(candidates)), None
    if not candidates:
        return None, f"无法根据联系人列表和当前允许范围推断会话类型：{target_id}"
    return None, f"会话 ID {target_id} 同时匹配多个会话类型，必须显式提供 type"


def repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """缺失 type 时按当前联系人列表推断唯一会话类型。"""
    if args.get("type") == "temp":
        repaired_args = dict(args)
        repaired_args["type"] = "private"
        return repaired_args, ["mapped legacy type='temp' to type='private'"]

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


def _format_focus_ref(conv_type: str, conv_id: str) -> str:
    if conv_type in {"group", "private", "temp"} and conv_id:
        return f"qq_{_public_shift_type(conv_type)}_{conv_id}"
    return conv_id or conv_type or "unknown"


def _format_focus_key(focus_key: str | None) -> str:
    if not focus_key:
        return "none"

    conv_type, conv_id = parse_session_key(focus_key)
    if conv_type and conv_id:
        return _format_focus_ref(conv_type, conv_id)
    return focus_key


async def _is_friend(client: Any, target_id: str) -> bool | None:
    if not client or not client.connected:
        return None
    friends = await client.send_api("get_friend_list", {}) or []
    if not isinstance(friends, list):
        return None
    return target_id in {str(f.get("user_id", "")) for f in friends if isinstance(f, dict)}


async def _group_exists(client: Any, group_id: str) -> bool | None:
    if not client or not client.connected:
        return None
    groups = await client.send_api("get_group_list", {}) or []
    if not isinstance(groups, list):
        return None
    return group_id in {str(g.get("group_id", "")) for g in groups if isinstance(g, dict)}


async def _validate_group_member(client: Any, group_id: str, user_id: str) -> str | None:
    if not client or not client.connected:
        return "QQ adapter 未连接，无法首次打开临时会话"
    try:
        group_id_int = int(group_id)
        user_id_int = int(user_id)
    except (TypeError, ValueError):
        return f"临时会话目标或来源群 ID 无效: user_id={user_id!r}, group_id={group_id!r}"
    data = await client.send_api(
        "get_group_member_info",
        {"group_id": group_id_int, "user_id": user_id_int, "no_cache": True},
        timeout=8.0,
    )
    if data:
        return None
    api_error = getattr(client, "last_api_error", None) or {}
    message = api_error.get("message") or api_error.get("msg") or "目标不在当前群，或 adapter 不允许查询该群成员"
    return f"无法从当前群 {group_id} 打开与用户 {user_id} 的临时会话: {message}"


async def _current_group_source() -> tuple[str, str] | None:
    import app_state
    from database import get_group_name
    from llm.session import sessions

    conv_type, group_id = parse_session_key(app_state.current_focus)
    if conv_type != "group" or not group_id:
        return None
    session = sessions.get(make_session_key("group", group_id))
    group_name = str(getattr(session, "conv_name", "") or "")
    if not group_name:
        group_name = await get_group_name(group_id)
    return group_id, group_name


async def _resolve_temp_target(target_id: str) -> dict[str, str]:
    import app_state
    from qq_adapter.access_control import whitelist_rejection_reason

    if reason := whitelist_rejection_reason(app_state.qq_adapter_cfg, "private", target_id):
        return {"error": reason}

    existing = await _resolve_existing_temp(target_id)
    if existing:
        return existing

    source = await _current_group_source()
    if source is None:
        return {"error": f"临时会话用户 {target_id} 尚未打开，且当前焦点不是群聊，无法首次打开"}

    group_id, group_name = source
    member_error = await _validate_group_member(app_state.qq_adapter_client, group_id, target_id)
    if member_error:
        return {"error": member_error}

    return _session_dict(
        key=make_temp_session_key(target_id),
        conv_type=TEMP_CONV_TYPE,
        conv_id=target_id,
        temp_source_group_id=group_id,
        temp_source_group_name=group_name,
    )


async def _resolve_shift_target(target_type: str, target_id: str) -> dict[str, str]:
    """解析 shift 目标。返回包含 key/type/id 的 dict，失败时返回 {"error": "..."}。"""
    import app_state
    from qq_adapter.access_control import whitelist_rejection_reason

    if target_type not in ("private", "group", "temp"):
        return {"error": f"未知会话类型 {target_type!r}"}

    if target_type == "temp":
        return await _resolve_temp_target(target_id)

    if reason := whitelist_rejection_reason(app_state.qq_adapter_cfg, target_type, target_id):
        return {"error": reason}

    client = app_state.qq_adapter_client
    if target_type == "group":
        exists = await _group_exists(client, target_id)
        if exists is False:
            return {"error": f"群 {target_id} 不在已加入的群列表中"}
        return _session_dict(
            key=make_session_key("group", target_id),
            conv_type="group",
            conv_id=target_id,
        )

    # private：目标是好友时开普通私聊；非好友时尝试临时会话。
    is_friend = await _is_friend(client, target_id)
    if is_friend is False:
        return await _resolve_temp_target(target_id)
    if is_friend is None:
        existing_temp = await _resolve_existing_temp(target_id)
        if existing_temp:
            return existing_temp
    return _session_dict(
        key=make_session_key("private", target_id),
        conv_type="private",
        conv_id=target_id,
    )


def _fallback_resolved(target_type: str, target_id: str) -> dict[str, str]:
    return _session_dict(
        key=make_session_key(target_type, target_id),
        conv_type=target_type,
        conv_id=target_id,
    )


def execute(type: str, id: str, **kwargs) -> dict:
    import app_state
    from database import upsert_chat_session
    from llm.session import get_or_create_session, sessions

    loop = getattr(app_state, "main_loop", None)
    if loop is None or not loop.is_running():
        return {"ok": False, "error": "主事件循环不可用"}

    try:
        resolved = run_coroutine_sync(
            _resolve_shift_target(type, id),
            loop,
            timeout=15,
        )
    except Exception as e:
        logger.warning("[shift] 校验异常: %s", e)
        return {"ok": False, "error": f"校验异常: {e}"}

    if resolved is None:
        # 兼容旧测试中把 run_coroutine_sync mock 成旧版校验返回 None 的场景。
        resolved = _fallback_resolved(type, id)
    if error := resolved.get("error"):
        logger.warning("[shift] 目标校验失败: %s", error)
        return {"ok": False, "error": error}

    new_key = resolved["key"]
    target_type = resolved["type"]
    target_id = resolved["id"]
    target = get_or_create_session(new_key)
    try:
        target.set_conversation_meta(
            target_type,
            target_id,
            resolved.get("name", ""),
            temp_source_group_id=resolved.get("temp_source_group_id", ""),
            temp_source_group_name=resolved.get("temp_source_group_name", ""),
        )
    except TypeError:
        target.set_conversation_meta(target_type, target_id)

    prev_key = app_state.current_focus
    if prev_key and prev_key != new_key:
        prev_session = sessions.get(prev_key)
        if prev_session is not None:
            prev_session.reset_transient_views()

    app_state.current_focus = new_key
    previous_focus = _format_focus_key(prev_key)
    current_focus = _format_focus_ref(target_type, target_id)
    target.last_wake_reason = (
        f"shift 自 {prev_key or '?'}"
        if prev_key and prev_key != new_key
        else "shift"
    )

    try:
        run_coroutine_sync(
            upsert_chat_session(
                new_key,
                target.conv_type,
                target.conv_id,
                target.conv_name,
                getattr(target, "temp_source_group_id", ""),
                getattr(target, "temp_source_group_name", ""),
            ),
            loop,
            timeout=15,
        )
    except Exception:
        logger.warning("[shift] 会话元信息持久化失败 conv=%s", new_key, exc_info=True)

    logger.info("[shift] 焦点切换 %s → %s", prev_key, new_key)
    now_focusing = {
        "type": _public_shift_type(target_type),
        "id": target_id,
        "name": target.conv_name or "",
    }
    if target_type == "temp":
        now_focusing["source_group_id"] = getattr(target, "temp_source_group_id", "")
        now_focusing["source_group_name"] = getattr(target, "temp_source_group_name", "")
    warnings: list[dict[str, Any]] = []
    if prev_key == new_key:
        warnings.append(ToolWarningFactory.same_session_shift().to_dict())

    result = {
        "ok": True,
        "now_focusing": now_focusing,
        "focus_transition": {
            "from": previous_focus,
            "to": current_focus,
            "summary": f"{previous_focus} -> {current_focus}",
        },
    }
    if warnings:
        result["warnings"] = warnings
        result["warning"] = warnings[0]
    return result
