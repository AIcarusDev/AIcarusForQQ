"""Shared helpers for platform page tools."""

from __future__ import annotations

from typing import Any

from platforms.focus import FocusRef, current_focus_key, normalize_focus
from platforms.core.session_context import CLOSED_PLATFORM_FOCUS, is_closed_platform_focus


def platform_registry() -> Any | None:
    import app_state

    return getattr(app_state, "platform_registry", None)


def focus_summary(value: Any) -> dict[str, str]:
    focus = normalize_focus(value)
    if focus is None:
        return {"key": "", "platform": "", "type": "", "id": "", "name": ""}
    return {
        "key": current_focus_key(focus),
        "platform": focus.platform,
        "type": focus.target_type,
        "id": focus.target_id,
        "name": focus.target_name,
    }


def runtime_main_focus(runtime: Any) -> FocusRef | None:
    getter = getattr(runtime, "main_focus", None)
    if not callable(getter):
        return None
    focus = normalize_focus(getter())
    return focus


def current_page_platform() -> str:
    import app_state

    focus = normalize_focus(getattr(app_state, "current_focus", None))
    if focus is None or is_closed_platform_focus(focus):
        return ""
    return focus.platform


def closed_focus() -> FocusRef:
    return CLOSED_PLATFORM_FOCUS
