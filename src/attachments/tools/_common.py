from __future__ import annotations

from typing import Any, Coroutine

from tools._async_bridge import LoopStoppedError, run_coroutine_sync


def run_on_main_loop(coro: Coroutine[Any, Any, Any], main_loop) -> Any:
    if main_loop is None or not main_loop.is_running():
        if hasattr(coro, "close"):
            coro.close()
        return {"ok": False, "error": {"code": "runtime_unavailable", "message": "主事件循环不可用"}}
    try:
        return run_coroutine_sync(coro, main_loop, timeout=None)
    except LoopStoppedError:
        return {"ok": False, "error": {"code": "runtime_unavailable", "message": "主事件循环已停止"}}
    except Exception as exc:
        return {"ok": False, "error": {"code": "attachment_error", "message": str(exc)}}


async def acknowledge(runtime_event_hub, task_id: str) -> None:
    if runtime_event_hub is not None:
        await runtime_event_hub.acknowledge(
            event_type="attachment_download_finished", key="task_id", value=task_id
        )
