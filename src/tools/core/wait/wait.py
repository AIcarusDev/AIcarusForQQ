"""wait.py — world-level wait tool."""

import asyncio
import logging
import time
from typing import Any

from tools._async_bridge import LoopStoppedError, run_coroutine_sync

from .prompt import DESCRIPTION

logger = logging.getLogger("AICQ.tools.wait")

SOCIAL_SCOPES = {"session", "platforms", "world"}
BROWSER_SCOPES = {"browser", "world"}
VALID_SCOPES = {"session", "platforms", "browser", "world"}
VALID_CONDITIONS = {"any_change", "mentioned"}
POLL_INTERVAL_SECONDS = 0.5

DECLARATION: dict = {
    "name": "wait",
    "description": DESCRIPTION,
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
                "description": "范围以及提前唤醒条件。",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["session", "platforms", "browser", "world"],
                    },
                    "condition": {
                        "type": "string",
                        "enum": ["any_change", "mentioned"],
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
    if scope not in VALID_SCOPES:
        return None, f"invalid early_trigger.scope: {scope!r}"
    if condition not in VALID_CONDITIONS:
        return None, f"invalid early_trigger.condition: {condition!r}"
    if scope == "browser" and condition == "mentioned":
        return None, "early_trigger condition 'mentioned' is not valid for browser scope"
    return {"scope": scope, "condition": condition}, None


def repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Map retired wait argument names to the current prompt-facing schema."""
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
    if trigger.get("scope") not in SOCIAL_SCOPES:
        return False
    pending_kind = str(pending or "")
    if not pending_kind:
        return False
    condition = trigger.get("condition")
    return condition == "any_change" or (condition == "mentioned" and pending_kind == "mentioned")


def _read_browser_signature() -> dict[str, Any] | None:
    try:
        from browser.session import browser_world_signature

        return browser_world_signature()
    except Exception:
        logger.debug("[wait] 读取 browser world signature 失败", exc_info=True)
        return None


def _browser_signature_changed(before: dict[str, Any] | None, after: dict[str, Any] | None) -> bool:
    if before is None and after is None:
        return False
    if before is None or after is None:
        return True
    return str(before.get("hash") or "") != str(after.get("hash") or "")


def execute(seconds: int | None = None, early_trigger: dict | None = None, **kwargs) -> dict:
    """Block until the requested world surface changes or the timeout expires."""
    import app_state
    from llm.session import sessions, get_or_create_session

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
    watch_browser = trigger["scope"] in BROWSER_SCOPES and trigger["condition"] == "any_change"

    async def _wait_until_triggered() -> tuple[str, str | None]:
        ev = asyncio.Event()
        session.wait_event = ev
        session.wait_early_trigger = trigger
        # 处理 race：消费 handler 启动前已积累的 pending 触发
        pending = session.pending_early_trigger
        session.pending_early_trigger = None
        if _pending_trigger_matches(trigger, pending):
            ev.set()
        if ev.is_set():
            return "triggered", "social"

        baseline_browser = await asyncio.to_thread(_read_browser_signature) if watch_browser else None
        deadline = time.monotonic() + timeout_secs
        try:
            while True:
                if ev.is_set():
                    return "triggered", "social"
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return "timeout", None
                try:
                    await asyncio.wait_for(ev.wait(), timeout=min(POLL_INTERVAL_SECONDS, remaining))
                    return "triggered", "social"
                except asyncio.TimeoutError:
                    pass
                if watch_browser:
                    current_browser = await asyncio.to_thread(_read_browser_signature)
                    if _browser_signature_changed(baseline_browser, current_browser):
                        return "triggered", "browser"
        finally:
            if session.wait_event is ev:
                session.wait_event = None
            session.wait_early_trigger = None

    try:
        reason, trigger_surface = run_coroutine_sync(_wait_until_triggered(), loop, timeout=None)
    except LoopStoppedError:
        logger.info("[wait] 事件循环已停止，wait 提前中断")
        return {"ok": False, "error": "wait 中断：进程被外部关闭"}
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
        "trigger_kind": trigger if reason == "triggered" else None,
        "trigger_surface": trigger_surface if reason == "triggered" else None,
        "trigger_from": trigger_from_meta,
        "elapsed_seconds": elapsed,
    }
    logger.info(
        "[wait] 完成 elapsed=%ss reason=%s focus=%s",
        elapsed, reason, focus_key,
    )
    return result
