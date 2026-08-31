"""Start and manage persistent QQ file download tasks."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Literal, Union

from pydantic import Field, RootModel

from platforms.qq.files.service import QQFileError, get_qq_file_service
from platforms.qq.session_context import ensure_session_provider
from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, ToolContract


DownloadStatus = Literal[
    "queued", "resolving", "downloading", "verifying", "completed", "failed", "stopped"
]


class DownloadStartArgs(ToolArgsModel):
    action: Literal["start"] = Field(description="启动当前 QQ 会话中指定文件消息的下载。")
    message_id: str = Field(
        min_length=1,
        description="当前 QQ 会话中的文件消息 ID。",
        json_schema_extra={"x-coerce-integer": True},
    )


class DownloadPollArgs(ToolArgsModel):
    action: Literal["poll"] = Field(description="立即查询一个下载任务的当前状态和进度。")
    download_id: str = Field(min_length=1, description="start 或 list 返回的下载任务 ID。")


class DownloadListArgs(ToolArgsModel):
    action: Literal["list"] = Field(description="列出当前 QQ 账号的跨会话下载任务。")
    statuses: list[DownloadStatus] | None = Field(
        default=None,
        description="可选状态过滤，同时作用于活跃任务和终态任务；省略时返回全部活跃任务和最近的终态任务。",
        json_schema_extra={"uniqueItems": True},
    )
    offset: int = Field(default=0, ge=0, description="终态任务历史的分页偏移，默认 0；活跃任务始终全部返回。")
    limit: int = Field(default=20, ge=1, le=100, description="终态任务历史最多返回数量，默认 20，最大 100；活跃任务不受此限制。")


class DownloadStopArgs(ToolArgsModel):
    action: Literal["stop"] = Field(description="显式停止一个仍在运行的下载任务。")
    download_id: str = Field(min_length=1, description="start 或 list 返回的下载任务 ID。")


class DownloadArgs(RootModel[Annotated[
    Union[DownloadStartArgs, DownloadPollArgs, DownloadListArgs, DownloadStopArgs],
    Field(discriminator="action"),
]]):
    pass


TOOL_CONTRACT = ToolContract(
    name="download",
    description="启动、查询、列出或停止 QQ 文件下载任务。",
    args_model=DownloadArgs,
)

REQUIRES_CONTEXT = ["qq_client", "qq_session_provider", "workspace_service"]


def _error(action: str, exc: QQFileError) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": exc.code,
        "message": str(exc),
        "retryable": exc.retryable,
    }
    if exc.details:
        error["details"] = exc.details
    return {"ok": False, "action": action, "error": error}


def make_handler(qq_client: Any, qq_session_provider: Any, workspace_service: Any):
    session_provider = ensure_session_provider(qq_session_provider)
    service = get_qq_file_service(qq_client, workspace_service)

    def execute(action: str, **kwargs: Any) -> dict[str, Any]:
        normalized = str(action or "")
        loop: asyncio.AbstractEventLoop | None = getattr(qq_client, "_loop", None)
        if loop is None or not loop.is_running():
            return _error(normalized, QQFileError("runtime_unavailable", "主事件循环当前不可用", retryable=True))
        try:
            if normalized == "start":
                coroutine = service.start(kwargs.get("message_id"), session_provider())
                timeout = 40.0
            elif normalized == "poll":
                coroutine = service.poll(kwargs.get("download_id"))
                timeout = 15.0
            elif normalized == "list":
                coroutine = service.list_downloads(
                    kwargs.get("statuses"), int(kwargs.get("offset", 0)), int(kwargs.get("limit", 20))
                )
                timeout = 15.0
            elif normalized == "stop":
                coroutine = service.stop(kwargs.get("download_id"))
                timeout = 20.0
            else:
                return _error(normalized, QQFileError("internal_error", "不支持的下载操作"))
            return run_coroutine_sync(coroutine, loop, timeout=timeout)
        except QQFileError as exc:
            return _error(normalized, exc)
        except Exception:
            return _error(normalized, QQFileError("internal_error", "QQ 文件操作失败", retryable=True))

    return execute
