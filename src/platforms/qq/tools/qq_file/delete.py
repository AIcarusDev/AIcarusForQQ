"""Permanently delete one exact local QQ file path."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import Field

from platforms.qq.files.service import QQFileError, get_qq_file_service
from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, ToolContract


class DeleteArgs(ToolArgsModel):
    path: str = Field(min_length=1, description="要永久删除的绝对 Linux 普通文件路径，必须位于当前 QQ 账号的 file 根目录内。")


TOOL_CONTRACT = ToolContract(
    name="delete",
    description="永久删除当前 QQ 账号文件根目录中的一个本地普通文件。",
    args_model=DeleteArgs,
)

REQUIRES_CONTEXT = ["qq_client", "workspace_service"]


def make_handler(qq_client: Any, workspace_service: Any):
    service = get_qq_file_service(qq_client, workspace_service)

    def execute(path: str, **_kwargs: Any) -> dict[str, Any]:
        loop: asyncio.AbstractEventLoop | None = getattr(qq_client, "_loop", None)
        if loop is None or not loop.is_running():
            return {"ok": False, "error": {"code": "runtime_unavailable", "message": "主事件循环当前不可用", "retryable": True, "blocking_download_id": None}}
        try:
            return run_coroutine_sync(service.delete(path), loop, timeout=30.0)
        except QQFileError as exc:
            return {
                "ok": False,
                "error": {
                    "code": exc.code,
                    "message": str(exc),
                    "retryable": exc.retryable,
                    "blocking_download_id": exc.details.get("blocking_download_id"),
                },
            }
        except Exception:
            return {"ok": False, "error": {"code": "internal_error", "message": "QQ 文件删除失败", "retryable": True, "blocking_download_id": None}}

    return execute
