"""Core restart request protocol.

The in-process tool only records an intent.  The main loop asks Hypercorn to
shut down after the current round has been persisted, and an outer supervisor
relaunches run.py when it exits with RESTART_EXIT_CODE.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("AICQ.runtime.core_restart")

BASE_DIR = Path(__file__).resolve().parents[2]
REQUEST_PATH = BASE_DIR / "data" / "core_restart_request.json"
RESTART_EXIT_CODE = int(os.environ.get("AICQ_CORE_RESTART_EXIT_CODE", "75"))
MAX_TEXT_CHARS = 300


def _clean_text(value: str | None, *, default: str = "") -> str:
    text = str(value or "").strip()
    if not text:
        return default
    if len(text) > MAX_TEXT_CHARS:
        return text[: MAX_TEXT_CHARS - 1] + "..."
    return text


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def read_pending_intent(path: Path | None = None) -> dict[str, Any] | None:
    request_path = path or REQUEST_PATH
    if not request_path.exists():
        return None
    try:
        data = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to read core restart intent: %s", request_path, exc_info=True)
        return None
    return data if isinstance(data, dict) else None


def consume_pending_intent(path: Path | None = None) -> dict[str, Any] | None:
    request_path = path or REQUEST_PATH
    intent = read_pending_intent(request_path)
    if intent is None:
        return None
    try:
        request_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Failed to remove consumed core restart intent: %s", request_path, exc_info=True)
    return intent


def request_restart(
    *,
    focus_key: str | None,
    requested_by: str,
) -> dict[str, Any]:
    """Persist a restart intent and mark the current process for restart."""
    import app_state

    if getattr(app_state, "core_restart_requested", False):
        existing = read_pending_intent() or {}
        return {
            "ok": True,
            "restart_scheduled": True,
            "already_requested": True,
            "exit_code": RESTART_EXIT_CODE,
            "focus_key": existing.get("focus_key", focus_key or ""),
        }

    now = time.time()
    payload: dict[str, Any] = {
        "version": 1,
        "requested_at": now,
        "requested_at_iso": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "requested_by": _clean_text(requested_by, default="unknown"),
        "focus_key": _clean_text(focus_key, default=""),
        "pid": os.getpid(),
        "exit_code": RESTART_EXIT_CODE,
    }
    _atomic_write_json(REQUEST_PATH, payload)

    app_state.core_restart_requested = True
    app_state.core_restart_exit_code = RESTART_EXIT_CODE
    logger.warning(
        "Core restart requested by %s; focus=%s",
        payload["requested_by"],
        payload["focus_key"] or "N/A",
    )
    return {
        "ok": True,
        "restart_scheduled": True,
        "already_requested": False,
        **payload,
    }


def apply_startup_intent(intent: dict[str, Any] | None) -> str | None:
    """Restore the requested focus and wake the main loop on startup."""
    if not intent:
        return None
    focus_key = _clean_text(str(intent.get("focus_key") or ""), default="")
    if not focus_key:
        return None

    import app_state

    app_state.current_focus = focus_key
    app_state.first_input_event.set()
    logger.info(
        "[startup] Core restart intent consumed; focus restored to %s",
        focus_key,
    )
    return focus_key


def build_restart_completed_tool_result(intent: dict[str, Any] | None, *, focus_key: str | None = None) -> dict[str, Any]:
    """Build the prompt-visible result for a deferred restart_self call."""
    now = time.time()
    requested_at = now
    if isinstance(intent, dict):
        try:
            requested_at = float(intent.get("requested_at") or now)
        except (TypeError, ValueError):
            requested_at = now
    restored_focus = _clean_text(focus_key, default="")
    if not restored_focus and isinstance(intent, dict):
        restored_focus = _clean_text(str(intent.get("focus_key") or ""), default="")
    return {
        "ok": True,
        "restarted": True,
        "restart_completed": True,
        "offline_seconds": max(0, round(now - requested_at, 1)),
        "focus_key": restored_focus,
        "pid": os.getpid(),
        "message": "我已完成重启，并已回到当前会话继续。",
    }


async def shutdown_after_round_if_requested() -> bool:
    """Ask the server to shut down after the current round is safely persisted."""
    import app_state

    if not getattr(app_state, "core_restart_requested", False):
        return False
    event = getattr(app_state, "server_shutdown_event", None)
    if event is None:
        logger.error(
            "Core restart requested, but run.py did not expose server_shutdown_event. "
            "The current process cannot shut down gracefully from inside the main loop."
        )
        return False

    logger.warning(
        "Core restart shutdown trigger set; exit_code=%s",
        getattr(app_state, "core_restart_exit_code", RESTART_EXIT_CODE),
    )
    event.set()
    return True


def reset_runtime_request_state() -> None:
    """Reset process-local flags. Mainly useful in tests."""
    import app_state

    app_state.core_restart_requested = False
    app_state.core_restart_exit_code = None
    app_state.launcher_switch_requested = False
