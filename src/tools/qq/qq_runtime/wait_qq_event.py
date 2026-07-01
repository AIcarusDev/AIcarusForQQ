"""wait_qq_event.py - QQ-owned social wait tool."""

import asyncio
import logging
import time
from typing import Any

from tools._async_bridge import LoopStoppedError, run_coroutine_sync

logger = logging.getLogger("AICQ.tools.wait_qq_event")

TOOL_KIND = "passive_wait"
SOCIAL_SCOPES = {"session", "platforms"}
VALID_CONDITIONS = {"any_change", "mentioned"}

DECLARATION: dict = {
    "name": "wait_qq_event",
    "description": (
        "等待 QQ 新消息或被提及事件。适合对话中停顿、等待对方继续说、"
        "或结束当前话题后等待其它 QQ 会话新动静。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 600,
                "description": "最长等待秒数。",
            },
            "early_trigger": {
                "type": "object",
                "description": "QQ 等待范围以及提前唤醒条件。",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["session", "platforms"],
                        "description": "session 表示当前 QQ 会话；platforms 表示任意 QQ 会话。",
                    },
                    "condition": {
                        "type": "string",
                        "enum": ["any_change", "mentioned"],
                        "description": "any_change 表示任意新消息；mentioned 表示私聊、@ 或回复等明确提及。",
                    },
                },
                "required": ["scope", "condition"],
            },
        },
        "required": ["seconds", "early_trigger"],
    },
}


def _normalize_trigger(raw_trigger: object) -> tuple[dict[str, str] | None, str | None]:
    if not isinstance(raw_trigger, dict):
        return None, "early_trigger must be an object"
    scope = str(raw_trigger.get("scope") or "").strip().lower()
    condition = str(raw_trigger.get("condition") or "").strip().lower()
    if scope == "global":
        scope = "platforms"
    if condition == "any_message":
        condition = "any_change"
    if scope not in SOCIAL_SCOPES:
        return None, f"invalid early_trigger.scope: {scope!r}"
    if condition not in VALID_CONDITIONS:
        return None, f"invalid early_trigger.condition: {condition!r}"
    return {"scope": scope, "condition": condition}, None


def repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(args, dict):
        return args, []
    repaired = dict(args)
    changes: list[str] = []
    if "seconds" not in repaired and "timeout" in repaired:
        repaired["seconds"] = repaired.pop("timeout")
        changes.append("timeout -> seconds")
    if isinstance(repaired.get("seconds"), str):
        stripped_seconds = str(repaired["seconds"]).strip()
        if stripped_seconds.isdigit():
            repaired["seconds"] = int(stripped_seconds)
            changes.append("seconds: string -> int")
    trigger = repaired.get("early_trigger")
    if isinstance(trigger, dict):
        trigger_repaired = dict(trigger)
        if trigger_repaired.get("scope") == "global":
            trigger_repaired["scope"] = "platforms"
            changes.append("early_trigger.scope: global -> platforms")
        if trigger_repaired.get("condition") == "any_message":
            trigger_repaired["condition"] = "any_change"
            changes.append("early_trigger.condition: any_message -> any_change")
        repaired["early_trigger"] = trigger_repaired
    return repaired, changes


def sanitize_semantic_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str], str | None]:
    trigger, error = _normalize_trigger(args.get("early_trigger"))
    if error is not None:
        return args, [], error
    if trigger == args.get("early_trigger"):
        return args, [], None
    repaired = dict(args)
    repaired["early_trigger"] = trigger
    return repaired, ["normalized early_trigger"], None


def _pending_trigger_matches(trigger: dict[str, str], pending: object) -> bool:
    pending_kind = str(pending or "")
    if not pending_kind:
        return False
    condition = trigger.get("condition")
    return condition == "any_change" or (condition == "mentioned" and pending_kind == "mentioned")


def execute(seconds: int | None = None, early_trigger: dict | None = None, **kwargs) -> dict:
    import app_state
    from llm.session import get_or_create_session, sessions

    if seconds is None and "timeout" in kwargs:
        seconds = kwargs.get("timeout")
    trigger, trigger_error = _normalize_trigger(early_trigger)
    if trigger_error is not None:
        return {"ok": False, "error": trigger_error}
    assert trigger is not None

    loop = app_state.main_loop
    if loop is None or not loop.is_running():
        return {"ok": False, "error": "主事件循环不可用"}

    focus_key = app_state.current_focus
    session = sessions.get(focus_key) if focus_key else None
    if session is None:
        return {"ok": False, "error": "无当前焦点会话"}

    started_at = time.time()
    timeout_secs = min(600, max(1, int(seconds if seconds is not None else 1)))

    async def _wait_until_triggered() -> str:
        ev = asyncio.Event()
        session.wait_event = ev
        session.wait_early_trigger = trigger
        pending = session.pending_early_trigger
        session.pending_early_trigger = None
        if _pending_trigger_matches(trigger, pending):
            ev.set()
        if ev.is_set():
            return "triggered"

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
        reason = run_coroutine_sync(_wait_until_triggered(), loop, timeout=None)
    except LoopStoppedError:
        logger.info("[wait_qq_event] 事件循环已停止，wait 提前中断")
        return {"ok": False, "error": "wait 中断：进程被外部关闭"}
    except Exception as exc:
        logger.warning("[wait_qq_event] 异常: %s", exc)
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
        "trigger_kind": trigger if reason == "triggered" else None,
        "trigger_surface": "qq" if reason == "triggered" else None,
        "trigger_from": trigger_from_meta,
        "elapsed_seconds": elapsed,
    }
    logger.info("[wait_qq_event] 完成 elapsed=%ss reason=%s focus=%s", elapsed, reason, focus_key)
    return result
