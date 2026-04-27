"""wait.py — wait 工具实现

Handler 真正阻塞 ``timeout`` 秒，可被 ``early_trigger`` 提前唤醒。
"""

import asyncio
import logging
import time

from .prompt import DESCRIPTION

logger = logging.getLogger("AICQ.tools.wait")

DECLARATION: dict = {
    "name": "wait",
    "description": DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "timeout": {
                "type": "integer",
                "minimum": 1,
                "maximum": 600,
                "description": "最长等待秒数。",
            },
            "early_trigger": {
                "type": "object",
                "description": "提前唤醒条件。",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["session", "global"],
                        "description": "监听范围：session=仅当前会话有新消息时触发，global=任意会话的消息均可触发。",
                    },
                    "condition": {
                        "type": "string",
                        "enum": ["any_message", "mentioned"],
                        "description": "触发条件：any_message=有任何新消息，mentioned=被@或被回复。",
                    },
                },
                "required": ["scope", "condition"],
            },
            "motivation": {
                "type": "string"
            },
        },
        "required": ["timeout", "motivation", "early_trigger"],
    },
}


def execute(timeout: int, motivation: str, early_trigger: dict, **kwargs) -> dict:
    """阻塞 timeout 秒或被 early_trigger 命中，二者之一先到。"""
    import app_state
    from llm.session import sessions, get_or_create_session

    loop = app_state.main_loop
    if loop is None or not loop.is_running():
        return {"ok": False, "error": "主事件循环不可用"}

    focus_key = app_state.current_focus
    session = sessions.get(focus_key) if focus_key else None
    if session is None:
        return {"ok": False, "error": "无当前焦点会话"}

    started_at = time.time()
    timeout_secs = max(1, int(timeout))

    async def _wait_until_triggered() -> str:
        ev = asyncio.Event()
        session.wait_event = ev
        session.wait_early_trigger = early_trigger
        # 处理 race：消费 handler 启动前已积累的 pending 触发
        pending = session.pending_early_trigger
        session.pending_early_trigger = None
        if pending and isinstance(early_trigger, dict):
            scope = early_trigger.get("scope")
            cond = early_trigger.get("condition")
            if scope == "session" and (
                cond == "any_message"
                or (cond == "mentioned" and pending == "mentioned")
            ):
                ev.set()
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout_secs)
            return "triggered"
        except asyncio.TimeoutError:
            return "timeout"
        finally:
            if session.wait_event is ev:
                session.wait_event = None
            session.wait_early_trigger = None

    try:
        reason = asyncio.run_coroutine_threadsafe(_wait_until_triggered(), loop).result()
    except Exception as exc:
        logger.warning("[wait] 异常: %s", exc)
        return {"ok": False, "error": f"wait 异常: {exc}"}

    elapsed = round(time.time() - started_at, 1)
    trigger_from_key = session.wait_trigger_from
    session.wait_trigger_from = None

    trigger_from_meta = None
    if reason == "triggered" and trigger_from_key:
        src = get_or_create_session(trigger_from_key)
        if src.conv_type:
            trigger_from_meta = {
                "type": src.conv_type,
                "id": src.conv_id,
                "name": src.conv_name,
            }

    result: dict = {
        "ok": True,
        "resumed": reason,
        "trigger_kind": early_trigger if reason == "triggered" else None,
        "trigger_from": trigger_from_meta,
        "elapsed_seconds": elapsed,
    }
    logger.info(
        "[wait] 完成 elapsed=%ss reason=%s focus=%s",
        elapsed, reason, focus_key,
    )
    return result
