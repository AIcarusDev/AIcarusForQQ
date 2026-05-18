"""set_self_group_card.py — 修改自己在当前群聊中的群名片."""

import asyncio
import logging
from typing import Any, Callable

from tools._async_bridge import run_coroutine_sync

logger = logging.getLogger("AICQ.tools")

SCOPE: str = "group"
ALWAYS_AVAILABLE: bool = False

DECLARATION: dict = {
    "name": "set_self_group_card",
    "description": (
        "修改你自己在当前群聊中的群名称（card）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "card": {
                "type": "string",
                "description": "要设置的新群名称（card）。传空字符串表示清空群名称。",
            },
            "motivation": {"type": "string"},
        },
        "required": ["card", "motivation"],
    },
}

REQUIRES_CONTEXT: list[str] = ["napcat_client", "session", "group_id"]

_POLL_ATTEMPTS = 3
_POLL_DELAY_SECONDS = 0.15


def _onebot_id(value: str) -> int | str:
    raw = str(value or "").strip()
    return int(raw) if raw.isdigit() else raw


def _normalize_card(value: Any) -> str:
    return str(value or "").strip()


async def _find_self_member(napcat_client: Any, group_id: str, bot_id: str) -> dict[str, Any] | None:
    member_list = await napcat_client.send_api(
        "get_group_member_list",
        {"group_id": _onebot_id(group_id), "no_cache": True},
        timeout=15,
    )
    if not isinstance(member_list, list):
        return None
    for member in member_list:
        if isinstance(member, dict) and str(member.get("user_id", "")) == str(bot_id):
            return member
    return None


def _update_session_meta(session: Any, group_id: str, group_name: str, member_count: int, card: str) -> None:
    if getattr(session, "conv_type", "") != "group" or str(getattr(session, "conv_id", "")) != str(group_id):
        return
    if hasattr(session, "set_conversation_meta"):
        session.set_conversation_meta("group", str(group_id), group_name, member_count)
    else:
        session.conv_type = "group"
        session.conv_id = str(group_id)
        if group_name:
            session.conv_name = group_name
        if member_count:
            session.conv_member_count = member_count
    session._qq_card = card


async def _sync_confirmed_card(
    *,
    session: Any,
    napcat_client: Any,
    group_id: str,
    bot_id: str,
    member: dict[str, Any],
    card: str,
) -> dict[str, Any]:
    from database import get_group_info, upsert_chat_session, upsert_group, upsert_membership
    from llm.session import sessions

    db_group_name, db_member_count, _old_card = await get_group_info(group_id)
    group_name = str(getattr(session, "conv_name", "") or db_group_name or "")
    member_count = int(getattr(session, "conv_member_count", 0) or db_member_count or 0)

    try:
        remote_group = await napcat_client.send_api(
            "get_group_info",
            {"group_id": _onebot_id(group_id), "no_cache": True},
            timeout=8,
        )
    except Exception:
        remote_group = None
        logger.debug("[set_self_group_card] get_group_info failed group=%s", group_id, exc_info=True)

    if isinstance(remote_group, dict):
        group_name = str(remote_group.get("group_name") or group_name)
        try:
            member_count = int(remote_group.get("member_count") or member_count or 0)
        except (TypeError, ValueError):
            member_count = int(member_count or 0)

    await upsert_group(group_id, group_name, card, member_count)
    await upsert_membership(
        "qq",
        bot_id,
        group_id,
        nickname=str(member.get("nickname", "") or ""),
        cardname=card,
        title=str(member.get("title", "") or ""),
        permission_level=str(member.get("role", "") or "member"),
    )
    await upsert_chat_session(f"group_{group_id}", "group", group_id, group_name)

    seen: set[int] = set()
    _update_session_meta(session, group_id, group_name, member_count, card)
    seen.add(id(session))
    for candidate in list(sessions.values()):
        if id(candidate) in seen:
            continue
        _update_session_meta(candidate, group_id, group_name, member_count, card)

    return {
        "group_name": group_name,
        "member_count": member_count,
        "nickname": str(member.get("nickname", "") or ""),
        "role": str(member.get("role", "") or "member"),
    }


async def _set_confirm_and_sync(
    *,
    session: Any,
    napcat_client: Any,
    group_id: str,
    bot_id: str,
    card: str,
) -> dict[str, Any]:
    resp = await napcat_client.send_api_raw(
        "set_group_card",
        {
            "group_id": _onebot_id(group_id),
            "user_id": _onebot_id(bot_id),
            "card": card,
        },
        timeout=15,
    )
    if resp is None:
        return {"error": "修改群名片超时或 NapCat 未连接", "synced": False}
    if resp.get("status") != "ok":
        msg = resp.get("message") or resp.get("msg") or "未知错误"
        return {"error": f"修改群名片失败: {msg}", "synced": False}

    observed_card = ""
    for attempt in range(_POLL_ATTEMPTS):
        member = await _find_self_member(napcat_client, group_id, bot_id)
        if member:
            observed_card = str(member.get("card", "") or "")
            if observed_card == card:
                synced = await _sync_confirmed_card(
                    session=session,
                    napcat_client=napcat_client,
                    group_id=group_id,
                    bot_id=bot_id,
                    member=member,
                    card=card,
                )
                return {
                    "success": True,
                    "verified": True,
                    "synced": True,
                    "group_id": group_id,
                    "new_card": card,
                    **synced,
                }
        if attempt < _POLL_ATTEMPTS - 1:
            await asyncio.sleep(_POLL_DELAY_SECONDS)

    return {
        "error": "群名片 API 已返回成功，但未确认生效，已跳过本地同步",
        "status": "unconfirmed",
        "synced": False,
        "group_id": group_id,
        "requested_card": card,
        "observed_card": observed_card,
    }


def make_handler(napcat_client: Any, session: Any, group_id: str) -> Callable:
    def execute(card: str | None = None, **kwargs) -> dict:
        if card is None:
            return {"error": "缺少 card 参数，无法修改群名片", "synced": False}
        new_card = _normalize_card(card)

        if getattr(session, "conv_type", "") != "group":
            return {"error": "set_self_group_card 仅能在群聊会话中使用", "synced": False}

        current_group_id = str(group_id or getattr(session, "conv_id", "") or "").strip()
        if not current_group_id:
            return {"error": "当前群号未知，无法修改群名片", "synced": False}

        if not napcat_client or not napcat_client.connected:
            return {"error": "NapCat 未连接，无法修改群名片", "synced": False}

        bot_id = str(getattr(napcat_client, "bot_id", "") or "").strip()
        if not bot_id:
            return {"error": "bot_id 未初始化，无法修改群名片", "synced": False}

        loop: asyncio.AbstractEventLoop | None = getattr(napcat_client, "_loop", None)
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用", "synced": False}

        try:
            return run_coroutine_sync(
                _set_confirm_and_sync(
                    session=session,
                    napcat_client=napcat_client,
                    group_id=current_group_id,
                    bot_id=bot_id,
                    card=new_card,
                ),
                loop,
                timeout=30,
            )
        except Exception as exc:
            logger.warning("[set_self_group_card] 修改群名片异常 group=%s", current_group_id, exc_info=True)
            return {"error": f"修改群名片失败: {exc}", "synced": False}

    return execute


def sanitize_semantic_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str], str | None]:
    card = args.get("card")
    if not isinstance(card, str):
        return args, [], "card must be a string"

    normalized = card.strip()
    if normalized == card:
        return args, [], None

    repaired = dict(args)
    repaired["card"] = normalized
    return repaired, ["card: trimmed surrounding whitespace"], None
