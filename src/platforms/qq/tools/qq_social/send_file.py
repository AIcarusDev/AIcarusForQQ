"""Send one file from the Linux computer to the current QQ session."""

from __future__ import annotations

import asyncio
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from pydantic import Field

from platforms.qq.session_context import NO_CURRENT_SESSION_ERROR, ensure_session_provider
from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, ToolContract
from workspace.config import workspace_enabled


class SendFileArgs(ToolArgsModel):
    path: str = Field(
        min_length=1,
        description="要发送的 Linux 电脑文件路径，必须位于 /home/agent 内。",
    )


TOOL_CONTRACT = ToolContract(
    name="send_file",
    description="将 Linux 电脑 /home/agent 中的一个文件发送到当前 QQ 会话中。",
    args_model=SendFileArgs,
)

EXTERNALLY_PERCEPTIBLE: bool = True
TOOL_EFFECT: dict[str, str] = {"surface": "qq", "kind": "session_write"}
REQUIRES_CONTEXT: list[str] = [
    "qq_client",
    "qq_session_provider",
    "workspace_service",
    "main_loop",
]


def condition(config: dict) -> bool:
    return workspace_enabled(config)


def _adapter_error(action: str, response: dict[str, Any] | None) -> str:
    if not response:
        return f"QQ adapter 未响应: {action}"
    parts = [action, str(response.get("status") or "failed")]
    if response.get("retcode") is not None:
        parts.append(f"retcode={response['retcode']}")
    wording = str(response.get("wording") or response.get("message") or "").strip()
    if wording:
        parts.append(wording)
    return "QQ adapter 返回错误: " + " / ".join(parts)


def _file_transfer_config(qq_client: Any) -> tuple[str, str]:
    raw = getattr(qq_client, "file_transfer", None)
    config = raw if isinstance(raw, dict) else {}
    host_directory = str(config.get("host_directory") or "").strip()
    adapter_directory = str(config.get("adapter_directory") or "").strip()
    if bool(host_directory) != bool(adapter_directory):
        raise ValueError("QQ 文件共享目录配置不完整：宿主目录与 Adapter 目录必须同时填写")
    return host_directory, adapter_directory


def _adapter_file_path(prepared: Any, host_directory: str, adapter_directory: str) -> str:
    if not host_directory:
        return str(prepared.host_path)
    host_root = Path(host_directory).expanduser().resolve()
    host_path = Path(prepared.host_path).resolve()
    try:
        relative = host_path.relative_to(host_root)
    except ValueError as exc:
        raise ValueError("暂存文件不在配置的 QQ 文件共享目录内") from exc
    return str(PurePosixPath(adapter_directory).joinpath(*relative.parts))


def make_handler(
    qq_client: Any,
    qq_session_provider: Callable[[], Any | None],
    workspace_service: Any,
    main_loop: asyncio.AbstractEventLoop,
) -> Callable:
    qq_session_provider = ensure_session_provider(qq_session_provider)

    def execute(path: str, **_: Any) -> dict[str, Any]:
        if not qq_client or not qq_client.connected:
            return {"error": "QQ adapter 未连接"}
        if main_loop is None or not main_loop.is_running():
            return {"error": "主事件循环不可用"}

        session = qq_session_provider()
        if session is None:
            return {"error": NO_CURRENT_SESSION_ERROR}

        conv_type = str(getattr(session, "conv_type", "") or "")
        conv_id = str(getattr(session, "conv_id", "") or "")
        if conv_type == "group":
            action = "upload_group_file"
            target_key = "group_id"
        elif conv_type == "private":
            action = "upload_private_file"
            target_key = "user_id"
        elif conv_type == "temp":
            return {"error": "QQ 临时会话不支持独立文件上传，未发送文件。"}
        else:
            return {"error": f"当前会话类型不支持发送 QQ 文件: {conv_type or 'unknown'}"}
        try:
            target_id = int(conv_id)
        except (TypeError, ValueError):
            return {"error": f"会话 ID 无效: {conv_id}"}

        async def _upload() -> tuple[Any, dict[str, Any] | None]:
            host_directory, adapter_directory = _file_transfer_config(qq_client)
            async with workspace_service.stage_host_file(
                path,
                staging_root=host_directory or None,
            ) as prepared:
                file_value = _adapter_file_path(prepared, host_directory, adapter_directory)
                response = await qq_client.send_api_raw(
                    action,
                    {
                        target_key: target_id,
                        "file": file_value,
                        "name": prepared.name,
                    },
                    timeout=None,
                )
            return prepared, response

        try:
            prepared, response = run_coroutine_sync(_upload(), main_loop, timeout=None)
        except Exception as exc:
            return {"error": f"文件发送失败: {exc}"}

        if not response or response.get("status") != "ok":
            return {"error": _adapter_error(action, response)}
        return {
            "success": True,
            "path": prepared.workspace_path,
            "name": prepared.name,
            "size": prepared.size,
            "target": f"{conv_type}_{conv_id}",
        }

    return execute
