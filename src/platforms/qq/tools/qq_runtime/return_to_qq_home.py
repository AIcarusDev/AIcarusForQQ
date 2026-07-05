"""Return QQ focus to the platform home/index view."""

from __future__ import annotations

from typing import Any

from tools.contract import ToolArgsModel, ToolContract
from platforms.focus import current_focus_key
from platforms.qq.session_context import HOME_FOCUS, is_qq_home_focus

TOOL_KIND = "focus_switch"
TOOL_EFFECT: dict[str, str] = {"surface": "qq", "kind": "focus_switch"}


class ReturnToQQHomeArgs(ToolArgsModel):
    pass


TOOL_CONTRACT = ToolContract(
    name="return_to_qq_home",
    description=(
        "关闭当前 QQ 会话窗口并返回 QQ 主页面/会话列表。"
        "适合结束当前会话处理、回到未读和最近会话总览。"
    ),
    args_model=ReturnToQQHomeArgs,
)


def _focus_summary(value: Any) -> dict[str, str]:
    key = current_focus_key(value)
    focus = getattr(value, "as_dict", lambda: {})()
    return {
        "key": key,
        "platform": str(focus.get("platform") or ""),
        "type": str(focus.get("target_type") or ""),
        "id": str(focus.get("target_id") or ""),
        "name": str(focus.get("target_name") or ""),
    }


def execute(**kwargs: Any) -> dict[str, Any]:
    import app_state
    from llm.session import get_or_create_session, sessions

    prev_focus = app_state.current_focus
    prev_key = current_focus_key(prev_focus)
    already_home = is_qq_home_focus(prev_focus)

    if prev_key and not already_home:
        prev_session = sessions.get(prev_key)
        if prev_session is not None:
            prev_session.reset_transient_views()

    app_state.current_focus = HOME_FOCUS
    home_session = get_or_create_session(HOME_FOCUS)
    home_session.last_wake_reason = "return_to_qq_home"

    return {
        "ok": True,
        "now_focusing": {
            "type": "home",
            "id": HOME_FOCUS.target_id,
            "name": HOME_FOCUS.target_name,
        },
        "focus_transition": {
            "from": _focus_summary(prev_focus),
            "to": _focus_summary(HOME_FOCUS),
            "summary": f"{prev_key or 'none'} -> {HOME_FOCUS.key()}",
        },
        "already_home": already_home,
    }

