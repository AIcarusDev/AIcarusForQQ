"""Core runtime state management tool."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated, Any, Literal

from pydantic import Field, RootModel

from tools._async_bridge import LoopStoppedError, run_coroutine_sync
from tools.contract import ToolArgsModel, ToolContract

logger = logging.getLogger("AICQ.tools.runtime_manage")

TOOL_KIND = "runtime_manage"


class WaitActionArgs(ToolArgsModel):
    action: Literal["wait"] = Field(
        description="等待一小段时间，然后进入下一轮观察。",
    )
    seconds: int | None = Field(
        default=10,
        ge=1,
        le=20,
        description="范围 1~20，单位秒，默认 10。",
    )


class IdleActionArgs(ToolArgsModel):
    action: Literal["idle"] = Field(
        description="闲置，当已经多次等待，或暂时没什么事情要做的时候可以先闲置。",
    )
    minutes: int | None = Field(
        default=30,
        ge=1,
        le=60,
        description="范围 1~60，单位分钟，默认 30。",
    )


class SleepActionArgs(ToolArgsModel):
    action: Literal["sleep"] = Field(
        description="睡觉，当觉得没有更多要做的事，且之后大概率也不会有什么事时可以睡觉。",
    )
    minutes: int | None = Field(
        default=480,
        ge=30,
        le=600,
        description="范围 30~600，单位分钟，默认 480。",
    )


class RuntimeManageArgs(
    RootModel[
        Annotated[
            WaitActionArgs | IdleActionArgs | SleepActionArgs,
            Field(discriminator="action"),
        ]
    ]
):
    pass


TOOL_CONTRACT = ToolContract(
    name="runtime_manage",
    description=(
        "核心的运行状态管理工具，实现等待、闲置（发呆）、休眠（睡觉）。"
        "idle/sleep 期间若收到需要注意的事件（例如群聊@/私聊消息）会唤醒。"
        "注意：idle/sleep 在一些情况下，有可能会错过你想反应的事件"
        "（例如群聊的连贯社交中，你的交互对象不一定会专门 @ 你），"
        "这类情况可以优先考虑 wait。"
    ),
    args_model=RuntimeManageArgs,
)


def _coerce_positive_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(minimum, min(maximum, parsed))


def _elapsed_and_remaining(request_started_at: Any, target_seconds: float) -> tuple[float, float]:
    now = time.time()
    try:
        started_at = float(request_started_at)
    except (TypeError, ValueError):
        started_at = now
    if started_at <= 0 or started_at > now + 1:
        started_at = now
    elapsed_since_request = max(0.0, now - started_at)
    remaining = max(0.0, float(target_seconds) - elapsed_since_request)
    return elapsed_since_request, remaining


def _session_meta(session) -> dict[str, Any]:
    return {
        "type": getattr(session, "conv_type", ""),
        "id": getattr(session, "conv_id", ""),
        "name": getattr(session, "conv_name", ""),
    }


def _consume_wake_metadata(session) -> tuple[str, str | None]:
    wake_reason = str(getattr(session, "last_wake_reason", "") or "").strip()
    wake_from = getattr(session, "sleep_wake_from", None)
    session.last_wake_reason = ""
    session.sleep_wake_from = None
    return wake_reason, wake_from


def _consume_pending_wake_if_current(session, pending_wake_after: Any = None) -> bool:
    if not getattr(session, "sleep_pending_wake", False):
        return False

    try:
        pending_at = float(getattr(session, "sleep_pending_wake_at", 0.0) or 0.0)
    except (TypeError, ValueError):
        pending_at = 0.0

    try:
        threshold = float(pending_wake_after)
    except (TypeError, ValueError):
        threshold = 0.0

    if threshold <= 0 or (pending_at > 0 and pending_at >= threshold):
        session.sleep_pending_wake = False
        session.sleep_pending_wake_at = 0.0
        return True

    session.sleep_pending_wake = False
    session.sleep_pending_wake_at = 0.0
    session.last_wake_reason = ""
    session.sleep_wake_from = None
    return False


async def wait_until_attention(session, duration_secs: float, *, pending_wake_after: Any = None) -> str:
    if duration_secs <= 0:
        if _consume_pending_wake_if_current(session, pending_wake_after):
            return "woken"
        return "timeout"

    ev = asyncio.Event()
    session.sleep_wake_event = ev
    session.sleep_arming = False
    if _consume_pending_wake_if_current(session, pending_wake_after):
        ev.set()
    try:
        await asyncio.wait_for(ev.wait(), timeout=duration_secs)
        return "woken"
    except asyncio.TimeoutError:
        return "timeout"
    finally:
        if session.sleep_wake_event is ev:
            session.sleep_wake_event = None
        session.sleep_arming = False


def build_runtime_result(
    session,
    *,
    action: str,
    requested_seconds: float,
    waited_seconds: float,
    elapsed_since_request: float,
    reason: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": True,
        "action": action,
        "resumed": reason,
        "requested_seconds": round(float(requested_seconds), 1),
        "waited_seconds": round(max(0.0, float(waited_seconds)), 1),
        "elapsed_seconds": round(max(0.0, float(elapsed_since_request)), 1),
    }
    if session is not None:
        result["current_session"] = _session_meta(session)
    if reason == "woken" and session is not None:
        wake_reason, wake_from = _consume_wake_metadata(session)
        if wake_reason:
            result["woke_up_because"] = wake_reason
        if wake_from:
            result["woke_from"] = wake_from
    return result


def repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(args, dict):
        return args, []
    repaired = dict(args)
    changes: list[str] = []
    action = str(repaired.get("action") or "").strip().lower()
    if action and repaired.get("action") != action:
        repaired["action"] = action
        changes.append("action: normalized")
    if action == "wait":
        if "seconds" not in repaired and "timeout" in repaired:
            repaired["seconds"] = repaired.pop("timeout")
            changes.append("timeout -> seconds")
        if isinstance(repaired.get("seconds"), str):
            stripped = str(repaired["seconds"]).strip()
            if stripped.isdigit():
                repaired["seconds"] = int(stripped)
                changes.append("seconds: string -> int")
    if action in {"idle", "sleep"}:
        if "minutes" not in repaired and "duration" in repaired:
            repaired["minutes"] = repaired.pop("duration")
            changes.append("duration -> minutes")
        if isinstance(repaired.get("minutes"), str):
            stripped = str(repaired["minutes"]).strip()
            if stripped.isdigit():
                repaired["minutes"] = int(stripped)
                changes.append("minutes: string -> int")
    return repaired, changes


def _execute_wait(seconds: object, request_started_at: object) -> dict[str, Any]:
    requested_seconds = _coerce_positive_int(seconds, 10, 1, 20)
    elapsed_before_wait, remaining = _elapsed_and_remaining(request_started_at, requested_seconds)
    started_wait = time.time()
    try:
        if remaining > 0:
            time.sleep(remaining)
    except KeyboardInterrupt:
        logger.info("[runtime_manage] wait 被外部中断")
        return {"ok": False, "error": "runtime_manage wait 中断：进程被外部关闭"}

    waited_seconds = time.time() - started_wait
    elapsed_total = elapsed_before_wait + waited_seconds
    logger.info(
        "[runtime_manage] wait 完成 requested=%ss waited=%.1fs elapsed=%.1fs",
        requested_seconds,
        waited_seconds,
        elapsed_total,
    )
    return build_runtime_result(
        None,
        action="wait",
        requested_seconds=requested_seconds,
        waited_seconds=waited_seconds,
        elapsed_since_request=elapsed_total,
        reason="timeout",
    )


def _execute_attention_sleep(action: str, minutes: object, request_started_at: object) -> dict[str, Any]:
    import app_state
    from llm.session import sessions

    loop = app_state.main_loop
    if loop is None or not loop.is_running():
        return {"ok": False, "error": "主事件循环不可用"}

    focus_key = app_state.current_focus
    session = sessions.get(focus_key) if focus_key else None
    if session is None:
        return {"ok": False, "error": "无当前焦点会话"}

    if action == "idle":
        requested_minutes = _coerce_positive_int(minutes, 30, 1, 60)
    else:
        requested_minutes = _coerce_positive_int(minutes, 480, 30, 600)
    requested_seconds = requested_minutes * 60
    elapsed_before_wait, remaining = _elapsed_and_remaining(request_started_at, requested_seconds)
    started_wait = time.time()

    try:
        session.sleep_arming = True
        reason = run_coroutine_sync(
            wait_until_attention(
                session,
                remaining,
                pending_wake_after=request_started_at,
            ),
            loop,
            timeout=None,
        )
    except LoopStoppedError:
        session.sleep_arming = False
        logger.info("[runtime_manage] %s 事件循环已停止，提前中断", action)
        return {"ok": False, "error": f"runtime_manage {action} 中断：进程被外部关闭"}
    except Exception as exc:
        session.sleep_arming = False
        logger.warning("[runtime_manage] %s 异常: %s", action, exc)
        return {"ok": False, "error": f"runtime_manage {action} 异常: {exc}"}

    waited_seconds = time.time() - started_wait
    elapsed_total = elapsed_before_wait + waited_seconds
    result = build_runtime_result(
        session,
        action=action,
        requested_seconds=requested_seconds,
        waited_seconds=waited_seconds,
        elapsed_since_request=elapsed_total,
        reason=reason,
    )
    logger.info(
        "[runtime_manage] %s 完成 requested=%ss waited=%.1fs reason=%s focus=%s",
        action,
        requested_seconds,
        waited_seconds,
        reason,
        focus_key,
    )
    return result


def execute(action: str, seconds: int | None = None, minutes: int | None = None, **kwargs) -> dict:
    action = str(action or "").strip().lower()
    request_started_at = kwargs.get("_request_started_at")
    if action == "wait":
        return _execute_wait(seconds if seconds is not None else 10, request_started_at)
    if action in {"idle", "sleep"}:
        return _execute_attention_sleep(action, minutes, request_started_at)
    return {"ok": False, "error": f"未知 runtime_manage.action: {action!r}"}
