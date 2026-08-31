"""Search local QQ files or synchronized QQ file-message history."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Literal, Union

from pydantic import Field, RootModel, model_validator

from platforms.qq.files.service import QQFileError, get_qq_file_service
from platforms.qq.session_context import ensure_session_provider
from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, ToolContract


class CurrentSearchScope(ToolArgsModel):
    type: Literal["current"] = Field(description="搜索当前 QQ 会话。")


class ConversationSearchScope(ToolArgsModel):
    type: Literal["conversation"] = Field(description="搜索指定 QQ 会话。")
    conversation_type: Literal["private", "group"] = Field(description="会话类型。")
    conversation_id: str = Field(min_length=1, description="好友 QQ 号或群号。", json_schema_extra={"x-coerce-integer": True})


class AllSearchScope(ToolArgsModel):
    type: Literal["all"] = Field(description="搜索当前 QQ 账号的全部会话。")


SearchScope = Annotated[Union[CurrentSearchScope, ConversationSearchScope, AllSearchScope], Field(discriminator="type")]


class SearchStartArgs(ToolArgsModel):
    source: Literal["local", "history"] = Field(description="local 搜索实际存在的本机文件；history 搜索 AICQ 已同步的 QQ 文件消息。")
    query: str | None = Field(default=None, min_length=1, max_length=255, description="文件名文字，按 Unicode 大小写不敏感的字面子串匹配；与 file_types 至少提供一个。")
    file_types: set[Annotated[str, Field(min_length=1, max_length=32)]] | None = Field(default=None, min_length=1, max_length=20, description="可选无点扩展名列表，例如 pdf、docx；大小写不敏感，与 query 至少提供一个。")
    limit: int = Field(default=50, ge=1, le=200, description="本页最多返回的结果数，默认 50，最大 200。")
    scope: SearchScope = Field(default_factory=lambda: CurrentSearchScope(type="current"), description="搜索的会话范围；省略时使用当前会话。")

    @model_validator(mode="after")
    def require_filter(self):
        if not (self.query or "") and not self.file_types:
            raise ValueError("query 与 file_types 至少需要提供一个")
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        schema = handler(core_schema)
        properties = schema.get("properties", {})
        properties["query"] = {
            "description": "文件名文字，按 Unicode 大小写不敏感的字面子串匹配；与 file_types 至少提供一个。",
            "maxLength": 255,
            "minLength": 1,
            "type": "string",
        }
        properties["file_types"] = {
            "description": "可选无点扩展名列表，例如 pdf、docx；大小写不敏感，与 query 至少提供一个。",
            "items": {"maxLength": 32, "minLength": 1, "type": "string"},
            "maxItems": 20,
            "minItems": 1,
            "type": "array",
            "uniqueItems": True,
        }
        scope = properties.get("scope", {})
        scope.pop("anyOf", None)
        scope.pop("$ref", None)
        scope["default"] = {"type": "current"}
        schema["anyOf"] = [{"required": ["query"]}, {"required": ["file_types"]}]
        return schema


class SearchContinueArgs(ToolArgsModel):
    cursor: str = Field(min_length=1, max_length=2048, description="上次 search 返回的 next_cursor；使用时不能传其他字段。")


class SearchArgs(RootModel[Union[SearchStartArgs, SearchContinueArgs]]):
    pass


TOOL_CONTRACT = ToolContract(
    name="search",
    description="按文件名搜索本机 QQ 文件或 AICQ 已同步的 QQ 文件消息。",
    args_model=SearchArgs,
)

REQUIRES_CONTEXT = ["qq_client", "qq_session_provider", "workspace_service"]


def make_handler(qq_client: Any, qq_session_provider: Any, workspace_service: Any):
    session_provider = ensure_session_provider(qq_session_provider)
    service = get_qq_file_service(qq_client, workspace_service)

    def execute(**kwargs: Any) -> dict[str, Any]:
        loop: asyncio.AbstractEventLoop | None = getattr(qq_client, "_loop", None)
        if loop is None or not loop.is_running():
            return {"ok": False, "error": {"code": "runtime_unavailable", "message": "主事件循环当前不可用", "retryable": True}}
        raw_scope = kwargs.get("scope")
        if hasattr(raw_scope, "model_dump"):
            raw_scope = raw_scope.model_dump()
        try:
            return run_coroutine_sync(
                service.search(
                    source=kwargs.get("source"),
                    query=kwargs.get("query"),
                    file_types=kwargs.get("file_types"),
                    scope=raw_scope,
                    limit=int(kwargs.get("limit", 50)),
                    cursor=str(kwargs["cursor"]) if kwargs.get("cursor") else None,
                    session=session_provider(),
                ),
                loop,
                timeout=45.0,
            )
        except QQFileError as exc:
            return {"ok": False, "error": {"code": exc.code, "message": str(exc), "retryable": exc.retryable}}
        except Exception:
            return {"ok": False, "error": {"code": "internal_error", "message": "QQ 文件搜索失败", "retryable": True}}

    return execute
