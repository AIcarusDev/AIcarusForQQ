"""Enumerate local QQ files from the active storage backend."""

from __future__ import annotations

import asyncio
from typing import Any, Literal, Union

from pydantic import Field, RootModel

from platforms.qq.files.service import QQFileError, get_qq_file_service
from platforms.qq.session_context import ensure_session_provider
from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, ToolContract


class ListCurrentArgs(ToolArgsModel):
    scope: Literal["current"] = Field(default="current", description="递归枚举当前会话的文件目录。")
    limit: int = Field(default=50, ge=1, le=200, description="本页最多返回的文件数，默认 50，最大 200。")


class ListConversationArgs(ToolArgsModel):
    scope: Literal["conversation"] = Field(description="递归枚举指定 QQ 会话的文件目录。")
    conversation_type: Literal["private", "group"] = Field(description="会话类型。")
    conversation_id: str = Field(min_length=1, description="好友 QQ 号或群号。", json_schema_extra={"x-coerce-integer": True})
    limit: int = Field(default=50, ge=1, le=200, description="本页最多返回的文件数，默认 50，最大 200。")


class ListAllArgs(ToolArgsModel):
    scope: Literal["all"] = Field(description="递归枚举当前 QQ 账号的全部文件目录。")
    limit: int = Field(default=50, ge=1, le=200, description="本页最多返回的文件数，默认 50，最大 200。")


class ListContinueArgs(ToolArgsModel):
    cursor: str = Field(min_length=1, max_length=2048, description="上次 list_files 返回的 next_cursor；使用时不能传其他字段。")


class ListFilesArgs(RootModel[Union[ListCurrentArgs, ListConversationArgs, ListAllArgs, ListContinueArgs]]):
    pass


TOOL_CONTRACT = ToolContract(
    name="list_files",
    description="枚举当前 QQ 账号文件根目录中的本地普通文件。",
    args_model=ListFilesArgs,
)

REQUIRES_CONTEXT = ["qq_client", "qq_session_provider", "workspace_service"]


def make_handler(qq_client: Any, qq_session_provider: Any, workspace_service: Any):
    session_provider = ensure_session_provider(qq_session_provider)
    service = get_qq_file_service(qq_client, workspace_service)

    def execute(**kwargs: Any) -> dict[str, Any]:
        loop: asyncio.AbstractEventLoop | None = getattr(qq_client, "_loop", None)
        if loop is None or not loop.is_running():
            return {"ok": False, "error": {"code": "runtime_unavailable", "message": "主事件循环当前不可用", "retryable": True}}
        cursor = kwargs.get("cursor")
        scope_value = kwargs.get("scope")
        scope = None
        if not cursor:
            scope = {"type": scope_value or "current"}
            if scope_value == "conversation":
                scope.update(
                    conversation_type=kwargs.get("conversation_type"),
                    conversation_id=kwargs.get("conversation_id"),
                )
        try:
            return run_coroutine_sync(
                service.list_files(
                    scope=scope,
                    limit=int(kwargs.get("limit", 50)),
                    cursor=str(cursor) if cursor else None,
                    session=session_provider(),
                ),
                loop,
                timeout=30.0,
            )
        except QQFileError as exc:
            return {"ok": False, "error": {"code": exc.code, "message": str(exc), "retryable": exc.retryable}}
        except Exception:
            return {"ok": False, "error": {"code": "internal_error", "message": "QQ 文件枚举失败", "retryable": True}}

    return execute
