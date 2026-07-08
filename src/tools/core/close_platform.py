"""Close the currently opened platform page."""

from __future__ import annotations

from typing import Any

from tools.contract import ToolArgsModel, ToolContract

from ._chat_notes import record_core_focus_transition
from ._platform_tools import closed_focus, focus_summary, is_closed_platform_focus

TOOL_KIND = "focus_switch"
TOOL_EFFECT: dict[str, str] = {"surface": "platform", "kind": "focus_switch"}


class ClosePlatformArgs(ToolArgsModel):
    pass


TOOL_CONTRACT = ToolContract(
    name="close_platform",
    description=(
        "关闭当前开启的 platform 视窗。"
    ),
    args_model=ClosePlatformArgs,
)


def execute(**_kwargs: Any) -> dict[str, Any]:
    import app_state
    from llm.session import get_or_create_session, sessions

    prev_focus = app_state.current_focus
    prev_summary = focus_summary(prev_focus)
    if not prev_summary["key"] or is_closed_platform_focus(prev_focus):
        return {
            "ok": True,
            "closed_platform": "",
            "now_focusing": focus_summary(closed_focus()),
            "focus_transition": {
                "from": prev_summary,
                "to": focus_summary(closed_focus()),
                "summary": "already closed",
            },
            "already_closed": True,
        }

    prev_session = sessions.get(prev_summary["key"])
    if prev_session is not None:
        prev_session.reset_transient_views()

    target_focus = closed_focus()
    app_state.current_focus = target_focus
    record_core_focus_transition(prev_focus, target_focus)
    target_session = get_or_create_session(target_focus)
    target_session.last_wake_reason = "close_platform"
    first_input_event = getattr(app_state, "first_input_event", None)
    if first_input_event is not None:
        first_input_event.set()

    return {
        "ok": True,
        "closed_platform": prev_summary["platform"],
        "now_focusing": focus_summary(target_focus),
        "focus_transition": {
            "from": prev_summary,
            "to": focus_summary(target_focus),
            "summary": f"{prev_summary['key']} -> {target_focus.key()}",
        },
        "already_closed": False,
    }
