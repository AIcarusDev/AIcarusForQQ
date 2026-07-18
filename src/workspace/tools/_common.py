from __future__ import annotations

from typing import Any, Coroutine

from tools._async_bridge import LoopStoppedError, run_coroutine_sync
from workspace.errors import WorkspaceError
from workspace.models import CommandResult, FileReadResult, TextListResult


def run_on_main_loop(coro: Coroutine[Any, Any, Any], main_loop) -> Any:
    if main_loop is None or not main_loop.is_running():
        if hasattr(coro, "close"):
            coro.close()
        return {"ok": False, "code": "runtime_unavailable", "error": "主事件循环不可用"}
    try:
        return run_coroutine_sync(coro, main_loop, timeout=None)
    except LoopStoppedError:
        return {"ok": False, "code": "runtime_unavailable", "error": "主事件循环已停止"}
    except WorkspaceError as exc:
        return {
            "ok": False,
            "error": {"code": exc.code.value, "message": exc.message},
        }
    except Exception as exc:
        return {"ok": False, "code": "computer_error", "error": str(exc)}


def command_result(result: CommandResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "command_id": result.command_id,
        "status": result.status,
        "cwd": result.cwd,
        "cursor": result.cursor,
        "has_more": result.has_more,
        "truncated": result.truncated,
        "content": result.content,
    }
    if result.exit_code is not None:
        payload["exit_code"] = result.exit_code
    if result.content_file is not None:
        payload["content_file"] = result.content_file
    if result.content_chars is not None:
        payload["content_chars"] = result.content_chars
    return payload


def read_result(result: FileReadResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "path": result.path,
        "start_line": result.start_line,
        "end_line": result.end_line,
        "total_lines": result.total_lines,
        "has_more": result.has_more,
        "content": result.content,
    }
    if result.next_line is not None:
        payload["next_line"] = result.next_line
    if result.truncated_lines:
        payload["truncated_lines"] = list(result.truncated_lines)
    return payload


def list_result(result: TextListResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "path": result.path,
        "count": result.count,
        "offset": result.offset,
        "has_more": result.has_more,
        "truncated": result.truncated,
        "content": result.content,
    }
    if result.next_offset is not None:
        payload["next_offset"] = result.next_offset
    return payload


async def acknowledge_command(runtime_event_hub, command_id: str) -> None:
    if runtime_event_hub is None:
        return
    await runtime_event_hub.acknowledge(
        event_type="workspace_command_finished",
        key="command_id",
        value=command_id,
    )
