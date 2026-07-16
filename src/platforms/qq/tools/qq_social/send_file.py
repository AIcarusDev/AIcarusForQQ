"""Send one file from the Agent's computer to the current QQ session."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from pydantic import Field

from platforms.qq.session_context import NO_CURRENT_SESSION_ERROR, ensure_session_provider
from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, ToolContract
from workspace.config import workspace_enabled


class SendFileArgs(ToolArgsModel):
    path: str = Field(
        min_length=1,
        description="要发送的 Agent 电脑文件路径，必须位于 /home/agent 内。",
    )


TOOL_CONTRACT = ToolContract(
    name="send_file",
    description="将 Agent 电脑 /home/agent 中的一个文件发送到当前 QQ 会话中。",
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
            async with workspace_service.stage_host_file(path) as prepared:
                response = await qq_client.send_api_raw(
                    action,
                    {
                        target_key: target_id,
                        "file": prepared.host_path,
                        "name": prepared.name,
                    },
                    timeout=120.0,
                )
            return prepared, response

        try:
            prepared, response = run_coroutine_sync(_upload(), main_loop, timeout=130.0)
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
