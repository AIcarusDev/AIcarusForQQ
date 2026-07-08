"""Core platform focus helpers."""

from __future__ import annotations

from typing import Any

from platforms.focus import FocusRef, normalize_focus

CORE_MAIN_FOCUS = FocusRef("core", "private", "guardian", "Core 聊天页面")
CLOSED_PLATFORM_FOCUS = FocusRef("core", "page", "none", "无平台页面")


def is_core_main_focus(value: Any) -> bool:
    focus = normalize_focus(value)
    return bool(
        focus
        and focus.platform == CORE_MAIN_FOCUS.platform
        and focus.target_type == CORE_MAIN_FOCUS.target_type
        and focus.target_id == CORE_MAIN_FOCUS.target_id
    )


def is_closed_platform_focus(value: Any) -> bool:
    focus = normalize_focus(value)
    return bool(
        focus
        and focus.platform == CLOSED_PLATFORM_FOCUS.platform
        and focus.target_type == CLOSED_PLATFORM_FOCUS.target_type
        and focus.target_id == CLOSED_PLATFORM_FOCUS.target_id
    )
