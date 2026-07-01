"""wait_browser_event.py - browser-owned page-change wait tool."""

import asyncio
import logging
import time
from typing import Any

from tools._async_bridge import LoopStoppedError, run_coroutine_sync

logger = logging.getLogger("AICQ.tools.wait_browser_event")

TOOL_KIND = "passive_wait"
POLL_INTERVAL_SECONDS = 0.5

DECLARATION: dict = {
    "name": "wait_browser_event",
    "description": "等待浏览器页面出现新变化。适合页面加载、图片生成、异步内容刷新或点击后等待结果。",
    "parameters": {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 60,
                "description": "最长等待秒数。",
            },
            "early_trigger": {
                "type": "object",
                "description": "浏览器等待范围以及提前唤醒条件。",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["browser"],
                        "description": "browser 表示浏览器发生语义变化。",
                    },
                    "condition": {
                        "type": "string",
                        "enum": ["any_change"],
                        "description": "浏览器第一版只支持页面语义变化。",
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
    if condition == "any_message":
        condition = "any_change"
    if scope != "browser":
        return None, f"invalid early_trigger.scope: {scope!r}"
    if condition != "any_change":
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


def _read_browser_signature() -> dict[str, Any] | None:
    try:
        from browser.session import browser_world_signature

        return browser_world_signature()
    except Exception:
        logger.debug("[wait_browser_event] 读取 browser world signature 失败", exc_info=True)
        return None


def _browser_signature_changed(before: dict[str, Any] | None, after: dict[str, Any] | None) -> bool:
    if before is None and after is None:
        return False
    if before is None or after is None:
        return True
    return str(before.get("hash") or "") != str(after.get("hash") or "")


def execute(seconds: int | None = None, early_trigger: dict | None = None, **kwargs) -> dict:
    import app_state

    if seconds is None and "timeout" in kwargs:
        seconds = kwargs.get("timeout")
    trigger, trigger_error = _normalize_trigger(early_trigger)
    if trigger_error is not None:
        return {"ok": False, "error": trigger_error}

    loop = app_state.main_loop
    if loop is None or not loop.is_running():
        return {"ok": False, "error": "主事件循环不可用"}

    started_at = time.time()
    timeout_secs = min(60, max(1, int(seconds if seconds is not None else 1)))

    async def _wait_until_changed() -> str:
        baseline_browser = await asyncio.to_thread(_read_browser_signature)
        deadline = time.monotonic() + timeout_secs
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout"
            await asyncio.sleep(min(POLL_INTERVAL_SECONDS, remaining))
            current_browser = await asyncio.to_thread(_read_browser_signature)
            if _browser_signature_changed(baseline_browser, current_browser):
                return "triggered"

    try:
        reason = run_coroutine_sync(_wait_until_changed(), loop, timeout=None)
    except LoopStoppedError:
        logger.info("[wait_browser_event] 事件循环已停止，wait 提前中断")
        return {"ok": False, "error": "wait 中断：进程被外部关闭"}
    except Exception as exc:
        logger.warning("[wait_browser_event] 异常: %s", exc)
        return {"ok": False, "error": f"wait 异常: {exc}"}

    elapsed = round(time.time() - started_at, 1)
    result: dict = {
        "ok": True,
        "resumed": reason,
        "trigger_kind": trigger if reason == "triggered" else None,
        "trigger_surface": "browser" if reason == "triggered" else None,
        "elapsed_seconds": elapsed,
    }
    logger.info("[wait_browser_event] 完成 elapsed=%ss reason=%s", elapsed, reason)
    return result
