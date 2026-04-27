"""sleep.py — sleep 工具实现

Handler **真正执行休眠**：在主事件循环上挂一个 ``sleep_wake_event``，等
``duration`` 分钟或被外部 set 中的较早者。被 mention 等事件打断时立刻醒。

返回结构供模型在下一轮看到："睡了多久 / 因何醒来"。
"""

import asyncio
import logging
import time

from .prompt import DESCRIPTION

logger = logging.getLogger("AICQ.tools.sleep")

DECLARATION: dict = {
    "name": "sleep",
    "description": DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "duration": {
                "type": "integer",
                "minimum": 30,
                "maximum": 600,
                "description": "想睡多久？单位分钟，范围 30~600。",
            },
            "motivation": {
                "type": "string",
            },
        },
        "required": ["duration", "motivation"],
    },
}


def execute(duration: int, motivation: str, **kwargs) -> dict:
    """阻塞当前工具线程直到休眠到期或被外部唤醒。"""
    import app_state
    from llm.session import sessions

    loop = app_state.main_loop
    if loop is None or not loop.is_running():
        return {"ok": False, "error": "主事件循环不可用"}

    focus_key = app_state.current_focus
    session = sessions.get(focus_key) if focus_key else None
    if session is None:
        return {"ok": False, "error": "无当前焦点会话"}

    duration_secs = max(1, int(duration)) * 60
    started_at = time.time()

    async def _sleep_until_woken() -> str:
        ev = asyncio.Event()
        session.sleep_wake_event = ev
        # 处理 race：handler 启动前若已有未消费的唤醒标记，立刻消费
        if session.sleep_pending_wake:
            session.sleep_pending_wake = False
            ev.set()
        try:
            await asyncio.wait_for(ev.wait(), timeout=duration_secs)
            return "woken"
        except asyncio.TimeoutError:
            return "timeout"
        finally:
            if session.sleep_wake_event is ev:
                session.sleep_wake_event = None

    try:
        reason = asyncio.run_coroutine_threadsafe(_sleep_until_woken(), loop).result()
    except Exception as exc:
        logger.warning("[sleep] 异常: %s", exc)
        return {"ok": False, "error": f"sleep 异常: {exc}"}

    elapsed = round(time.time() - started_at)
    woke_from = session.sleep_wake_from
    session.sleep_wake_from = None
    wake_reason = (session.last_wake_reason or "").strip()
    session.last_wake_reason = ""

    result: dict = {
        "ok": True,
        "slept_seconds": elapsed,
        "resumed": reason,
        "current_session": {
            "type": session.conv_type,
            "id": session.conv_id,
            "name": session.conv_name,
        },
    }
    if reason == "woken":
        if wake_reason:
            result["woke_up_because"] = wake_reason
        if woke_from:
            result["woke_from"] = woke_from
    logger.info(
        "[sleep] 完成 elapsed=%ds reason=%s focus=%s",
        elapsed, reason, focus_key,
    )
    return result
