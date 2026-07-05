"""query_group_members.py - query current QQ group members in bounded modes."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any, Callable, Literal

from pydantic import Field, RootModel

from platforms.qq.session_context import NO_CURRENT_SESSION_ERROR, ensure_session_provider
from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, ToolContract

logger = logging.getLogger("AICQ.tools")

_PAGE_SIZE = 20
_SEARCH_LIMIT = 10
_FETCH_TIMEOUT_SECONDS = 15
_ADMIN_ROLES = {"owner", "admin"}
_VALID_ROLES = {"owner", "admin", "member"}


class GroupMembersListAdminsArgs(ToolArgsModel):
    action: Literal["list_admins"] = Field(description="列出当前群的群主和所有管理员。")


class GroupMembersListMembersArgs(ToolArgsModel):
    action: Literal["list_members"] = Field(description="按群成员顺序分页列出当前群成员。")
    page: int = Field(default=1, ge=1, le=200, description="范围 1~200，默认 1。每页最多 20 人。")


class GroupMembersSearchArgs(ToolArgsModel):
    action: Literal["search"] = Field(description="按昵称或群 card 搜索当前群成员。")
    query: str = Field(min_length=1, max_length=32, description="搜索字符串，匹配昵称或群 card。长度 1~32。")


class QueryGroupMembersArgs(
    RootModel[
        Annotated[
            GroupMembersListAdminsArgs | GroupMembersListMembersArgs | GroupMembersSearchArgs,
            Field(discriminator="action"),
        ]
    ]
):
    pass


TOOL_CONTRACT = ToolContract(
    name="query_group_members",
    description=(
        "查询当前群聊成员（仅群聊会话中可用）。"
        "action=list_admins 返回所有群主和管理员。"
        "action=list_members 按群成员顺序分页返回，每页最多 20 人，page 默认 1。"
        "action=search 在全体群成员的昵称和群 card 中搜索，最多返回 10 人。"
    ),
    args_model=QueryGroupMembersArgs,
)

REQUIRES_CONTEXT: list[str] = ["qq_client", "qq_session_provider"]
PARALLEL_SAFE = True
PARALLEL_KEY = "qq_client_read"


def _onebot_id(value: str) -> int | str:
    raw = str(value or "").strip()
    return int(raw) if raw.isdigit() else raw


def _member_role(member: dict[str, Any]) -> str:
    role = str(member.get("role", "") or "").strip()
    return role if role in _VALID_ROLES else "member"


def _format_member(member: dict[str, Any]) -> dict[str, str]:
    return {
        "account": str(member.get("user_id", "") or ""),
        "name": str(member.get("nickname", "") or ""),
        "card": str(member.get("card", "") or ""),
        "role": _member_role(member),
    }


def _matches_member(member: dict[str, Any], query: str) -> bool:
    needle = query.casefold()
    nickname = str(member.get("nickname", "") or "").casefold()
    card = str(member.get("card", "") or "").casefold()
    return needle in nickname or needle in card


def _validate_action_args(kwargs: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    action = kwargs.get("action")
    if action not in {"list_admins", "list_members", "search"}:
        return None, {"error": "action 必须是 list_admins / list_members / search 之一"}

    if action == "list_admins":
        extra = set(kwargs) - {"action"}
        if extra:
            return None, {"error": "action=list_admins 时不能传其他参数", "extra": sorted(extra)}
        return action, None

    if action == "list_members":
        extra = set(kwargs) - {"action", "page"}
        if extra:
            return None, {"error": "action=list_members 时只能传 page", "extra": sorted(extra)}
        raw_page = kwargs.get("page", 1)
        try:
            if isinstance(raw_page, bool):
                raise ValueError
            page = int(raw_page)
        except (TypeError, ValueError):
            return None, {"error": f"page 必须是整数，收到: {raw_page!r}"}
        if page < 1 or page > 200:
            return None, {"error": f"page 范围必须是 1~200，收到: {page}"}
        kwargs["page"] = page
        return action, None

    extra = set(kwargs) - {"action", "query"}
    if extra:
        return None, {"error": "action=search 时只能传 query", "extra": sorted(extra)}
    query = str(kwargs.get("query", "") or "").strip()
    if not query:
        return None, {"error": "query 不能为空"}
    if len(query) > 32:
        return None, {"error": f"query 长度必须是 1~32，收到 {len(query)}"}
    kwargs["query"] = query
    return action, None


def make_handler(qq_client: Any, qq_session_provider: Callable[[], Any | None]) -> Callable:
    qq_session_provider = ensure_session_provider(qq_session_provider)

    def execute(**kwargs) -> dict:
        action, error = _validate_action_args(kwargs)
        if error is not None:
            return error

        session = qq_session_provider()
        if session is None:
            return {"error": NO_CURRENT_SESSION_ERROR}
        if getattr(session, "conv_type", "") != "group":
            return {"error": "query_group_members 仅能在群聊会话中使用"}
        group_id = str(getattr(session, "conv_id", "") or "").strip()
        if not group_id:
            return {"error": "当前群号未知，无法查询群成员"}
        if not qq_client or not qq_client.connected:
            logger.warning("[tools] query_group_members: QQ adapter 未连接 group_id=%s", group_id)
            return {"error": "QQ adapter 未连接，无法查询群成员"}

        loop: asyncio.AbstractEventLoop | None = qq_client._loop
        if loop is None or not loop.is_running():
            logger.warning("[tools] query_group_members: 事件循环不可用 group_id=%s", group_id)
            return {"error": "主事件循环不可用"}

        try:
            logger.info("[tools] query_group_members: 获取群成员列表开始 group_id=%s action=%s", group_id, action)
            raw: list[dict[str, Any]] | None = run_coroutine_sync(
                qq_client.send_api(
                    "get_group_member_list",
                    {"group_id": _onebot_id(group_id)},
                ),
                loop,
                timeout=_FETCH_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("[tools] query_group_members: API 调用异常 group_id=%s - %s", group_id, exc)
            return {"error": f"查询群成员失败: {exc}"}

        if raw is None:
            logger.warning("[tools] query_group_members: API 返回为空 group_id=%s", group_id)
            return {"error": "API 返回为空（可能群号有误或权限不足）"}
        if not isinstance(raw, list):
            return {"error": f"API 返回格式异常，预期 list，收到 {type(raw).__name__}"}

        if action == "list_admins":
            members = [_format_member(member) for member in raw if _member_role(member) in _ADMIN_ROLES]
            return {
                "action": "list_admins",
                "group_id": group_id,
                "total_in_group": len(raw),
                "returned": len(members),
                "members": members,
            }

        if action == "list_members":
            page = int(kwargs.get("page", 1))
            start = (page - 1) * _PAGE_SIZE
            end = start + _PAGE_SIZE
            members = [_format_member(member) for member in raw[start:end]]
            return {
                "action": "list_members",
                "group_id": group_id,
                "total_in_group": len(raw),
                "page": page,
                "page_size": _PAGE_SIZE,
                "has_more": end < len(raw),
                "returned": len(members),
                "members": members,
            }

        if action == "search":
            query = str(kwargs.get("query", "") or "").strip()
            matches = [_format_member(member) for member in raw if _matches_member(member, query)]
            members = matches[:_SEARCH_LIMIT]
            return {
                "action": "search",
                "group_id": group_id,
                "query": query,
                "total_matches": len(matches),
                "returned": len(members),
                "truncated": len(matches) > _SEARCH_LIMIT,
                "members": members,
            }

        return {"error": f"未知 action: {action!r}"}

    return execute
