"""Core platform focus helpers."""

from __future__ import annotations

from typing import Any

from platforms.focus import FocusRef, normalize_focus

CORE_MAIN_FOCUS = FocusRef("core", "private", "guardian", "Core 聊天页面")
CLOSED_PLATFORM_FOCUS = FocusRef("core", "page", "none", "无平台页面")
NO_CURRENT_SESSION_ERROR = "当前没有打开 core 聊天页面，请先 enter_platform 进入 core。"


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


def core_surface_for_focus(value: Any) -> str:
    if is_core_main_focus(value):
        return "session"
    if is_closed_platform_focus(value):
        return "closed"
    return ""


def resolve_current_core_session() -> Any | None:
    import app_state
    from llm.session import get_or_create_session

    focus = normalize_focus(getattr(app_state, "current_focus", None))
    if not is_core_main_focus(focus):
        return None
    return get_or_create_session(focus)


def make_static_session_provider(session: Any) -> Any:
    def provider() -> Any | None:
        return session if is_core_main_focus(getattr(session, "focus", None)) else None

    return provider


def ensure_session_provider(value: Any) -> Any:
    if callable(value):
        return value
    return make_static_session_provider(value)
