"""get_user_info.py - query QQ user profile information by QQ number.

需要运行时上下文：qq_client。
调用 QQ adapter get_stranger_info 接口，返回目标用户的基础资料字段。
不传 user_id 时默认查询 bot 自身。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from pydantic import Field

from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, ToolContract

logger = logging.getLogger("AICQ.tools")

class GetUserInfoArgs(ToolArgsModel):
    user_id: str = Field(
        default="",
        json_schema_extra={"x-coerce-integer": True},
        description="要查询资料的 QQ 号。不填则查询你自己的资料。",
    )


TOOL_CONTRACT = ToolContract(
    name="get_user_info",
    description=(
        "通过 QQ 号查询指定用户的基础资料，包括昵称、性别、年龄、QID、等级、登录天数和个性签名等。"
        "不传 user_id 时默认查询你自己的资料。"
    ),
    args_model=GetUserInfoArgs,
)

REQUIRES_CONTEXT: list[str] = ["qq_client"]


def _normalize_sex(value: Any) -> str:
    text = str(value or "").strip()
    return {
        "male": "男",
        "female": "女",
        "unknown": "未知",
    }.get(text.lower(), text)


def _put_if_present(result: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return
    result[key] = value


def _format_user_info(data: dict[str, Any], target_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "qq_number": str(data.get("user_id") or target_id),
    }

    _put_if_present(result, "nickname", data.get("nickname"))
    _put_if_present(result, "qid", data.get("qid"))

    sex = _normalize_sex(data.get("sex"))
    _put_if_present(result, "sex", sex)

    _put_if_present(result, "age", data.get("age"))
    _put_if_present(result, "level", data.get("level"))
    _put_if_present(result, "login_days", data.get("login_days"))

    signature = data.get("longNick") or data.get("sign")
    _put_if_present(result, "signature", signature)

    return result


def make_handler(qq_client: Any) -> Callable:
    def execute(**kwargs) -> dict:
        if not qq_client or not qq_client.connected:
            return {"error": "QQ adapter 未连接，无法查询用户资料"}

        raw_uid: Any = kwargs.get("user_id")
        target_id = str(raw_uid).strip() if raw_uid is not None and str(raw_uid).strip() else ""
        if not target_id:
            target_id = str(getattr(qq_client, "bot_id", "") or "").strip()

        if not target_id:
            return {"error": "bot_id 未初始化且未传入 user_id，无法查询用户资料"}
        if not target_id.isdigit():
            return {"error": f"无效的 QQ 号：{target_id!r}，请传入纯数字字符串"}

        loop: asyncio.AbstractEventLoop | None = getattr(qq_client, "_loop", None)
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        try:
            logger.info("[tools] get_user_info: 查询用户资料 user_id=%s", target_id)
            data: dict | None = run_coroutine_sync(
                qq_client.send_api(
                    "get_stranger_info",
                    {"user_id": int(target_id), "no_cache": True},
                ),
                loop,
                timeout=15,
            )
        except Exception as e:
            logger.warning("[tools] get_user_info: 查询失败 user_id=%s - %s", target_id, e)
            return {"error": f"查询用户资料失败: {e}"}

        if not data:
            last_error = getattr(qq_client, "last_api_error", None) or {}
            message = last_error.get("message") or "API 返回为空，可能权限不足或 QQ 号有误"
            return {"error": message}

        return _format_user_info(data, target_id)

    return execute


