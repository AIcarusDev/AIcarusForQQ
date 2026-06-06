"""Emergency runtime reset helpers.

This is intentionally narrower than a process restart: it clears the current
runtime consciousness state and parks the main loop in the same no-focus wait
state used on a cold start.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import asdict, dataclass

import app_state
from consciousness import ConsciousnessFlow, consciousness_main_loop
from database import save_adapter_contents
from llm.session import sessions

logger = logging.getLogger("AICQ.runtime.emergency_reset")


@dataclass
class EmergencyResetResult:
    reset_id: str
    epoch: int
    previous_focus: str | None
    cleared_flow_rounds: int
    cleared_compression_pending_jobs: int
    cleared_compression_inflight_job: bool
    woken_waits: int
    woken_sleeps: int
    main_loop_restarted: bool

    def to_dict(self) -> dict:
        return asdict(self)


def expected_confirmation() -> str:
    """Return the exact string required for the WebUI dangerous action."""
    bot_name = str(getattr(app_state, "BOT_NAME", "") or "").strip() or "AIcarus"
    return f"RESET {bot_name}"


def is_runtime_epoch_stale(epoch: int) -> bool:
    return int(getattr(app_state, "runtime_reset_epoch", 0)) != int(epoch)


def make_runtime_epoch_checker(epoch: int):
    return lambda: is_runtime_epoch_stale(epoch)


def mark_result_aborted_by_reset(result, epoch: int):
    result.failed = True
    result.aborted_by_runtime_reset = True
    result.runtime_reset_epoch = epoch
    return result


async def _cancel_task(task: asyncio.Task | None, *, timeout: float, label: str) -> bool:
    if task is None or task.done():
        return False
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=timeout)
    except asyncio.CancelledError:
        pass
    except asyncio.TimeoutError:
        logger.warning("[emergency_reset] %s cancel timed out after %.1fs", label, timeout)
    except Exception:
        logger.debug("[emergency_reset] %s cancel raised", label, exc_info=True)
    return True


def _wake_and_clear_session_waits() -> tuple[int, int]:
    woken_waits = 0
    woken_sleeps = 0
    for session in list(sessions.values()):
        wait_event = getattr(session, "wait_event", None)
        if wait_event is not None and not wait_event.is_set():
            wait_event.set()
            woken_waits += 1
        sleep_event = getattr(session, "sleep_wake_event", None)
        if sleep_event is not None and not sleep_event.is_set():
            sleep_event.set()
            woken_sleeps += 1

        if hasattr(session, "wait_early_trigger"):
            session.wait_early_trigger = None
        if hasattr(session, "pending_early_trigger"):
            session.pending_early_trigger = None
        if hasattr(session, "wait_trigger_from"):
            session.wait_trigger_from = None
        if hasattr(session, "sleep_pending_wake"):
            session.sleep_pending_wake = False
        if hasattr(session, "sleep_arming"):
            session.sleep_arming = False
        if hasattr(session, "sleep_wake_from"):
            session.sleep_wake_from = None
        if hasattr(session, "last_wake_reason"):
            session.last_wake_reason = ""
        reset_transient_views = getattr(session, "reset_transient_views", None)
        if callable(reset_transient_views):
            try:
                reset_transient_views()
            except Exception:
                logger.debug("[emergency_reset] reset_transient_views failed", exc_info=True)
    return woken_waits, woken_sleeps


async def perform_emergency_reset() -> EmergencyResetResult:
    """Clear current runtime state and park the bot in no-focus waiting mode."""
    async with app_state.runtime_reset_lock:
        reset_id = uuid.uuid4().hex
        previous_focus = app_state.current_focus
        old_flow = app_state.consciousness_flow
        cleared_flow_rounds = old_flow.round_count if old_flow is not None else 0

        app_state.runtime_reset_epoch = int(getattr(app_state, "runtime_reset_epoch", 0)) + 1
        epoch = app_state.runtime_reset_epoch
        logger.warning(
            "[emergency_reset] starting reset_id=%s epoch=%d previous_focus=%s",
            reset_id,
            epoch,
            previous_focus,
        )

        old_main_task = app_state.consciousness_main_task
        app_state.consciousness_main_task = None

        compression_task = app_state.cognition_compression_task
        app_state.cognition_compression_task = None
        cleared_compression_pending_jobs = len(
            getattr(app_state, "cognition_compression_pending_jobs", None) or []
        )
        cleared_compression_inflight_job = (
            getattr(app_state, "cognition_compression_inflight_job", None) is not None
        )
        app_state.cognition_compression_pending_jobs = []
        app_state.cognition_compression_inflight_job = None

        app_state.current_focus = None
        app_state.last_active_session = None
        app_state.first_input_event.clear()

        woken_waits, woken_sleeps = _wake_and_clear_session_waits()

        await _cancel_task(old_main_task, timeout=0.5, label="consciousness_main_loop")
        await _cancel_task(compression_task, timeout=0.5, label="cognition_compression")

        app_state.consciousness_flow = ConsciousnessFlow()
        contents, timestamps = app_state.consciousness_flow.dump()
        await save_adapter_contents("flow", contents, timestamps)

        app_state.shutdown_event.clear()
        app_state.first_input_event.clear()
        app_state.consciousness_main_task = asyncio.create_task(
            consciousness_main_loop(),
            name="consciousness_main_loop",
        )

        logger.warning(
            "[emergency_reset] completed reset_id=%s epoch=%d waits=%d sleeps=%d",
            reset_id,
            epoch,
            woken_waits,
            woken_sleeps,
        )
        return EmergencyResetResult(
            reset_id=reset_id,
            epoch=epoch,
            previous_focus=previous_focus,
            cleared_flow_rounds=cleared_flow_rounds,
            cleared_compression_pending_jobs=cleared_compression_pending_jobs,
            cleared_compression_inflight_job=cleared_compression_inflight_job,
            woken_waits=woken_waits,
            woken_sleeps=woken_sleeps,
            main_loop_restarted=True,
        )


__all__ = [
    "EmergencyResetResult",
    "expected_confirmation",
    "is_runtime_epoch_stale",
    "make_runtime_epoch_checker",
    "mark_result_aborted_by_reset",
    "perform_emergency_reset",
]
