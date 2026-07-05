"""QQ focus/surface helpers for runtime tool execution."""

from __future__ import annotations

from typing import Any, Callable

from platforms.focus import FocusRef, normalize_focus

HOME_TARGET_TYPE = "home"
HOME_TARGET_ID = "default"
HOME_FOCUS = FocusRef("qq", HOME_TARGET_TYPE, HOME_TARGET_ID, "QQ 主页面")

NO_CURRENT_SESSION_ERROR = "当前没有打开具体 QQ 会话，请先 enter_qq_session 进入目标会话。"


def is_qq_home_focus(value: Any) -> bool:
    focus = normalize_focus(value)
    return bool(
        focus
        and focus.platform == "qq"
        and focus.target_type == HOME_TARGET_TYPE
    )


def qq_surface_for_focus(value: Any) -> str:
    focus = normalize_focus(value)
    if focus and focus.platform == "qq" and focus.target_type == HOME_TARGET_TYPE:
        return "home"
    return "session"


def resolve_current_qq_session() -> Any | None:
    import app_state
    from llm.session import get_or_create_session

    focus = normalize_focus(getattr(app_state, "current_focus", None))
    if focus is None or focus.platform != "qq" or focus.target_type == HOME_TARGET_TYPE:
        return None
    return get_or_create_session(focus)


def make_static_session_provider(session: Any) -> Callable[[], Any | None]:
    def provider() -> Any | None:
        return session

    return provider


def ensure_session_provider(value: Any) -> Callable[[], Any | None]:
    if callable(value):
        return value
    return make_static_session_provider(value)
