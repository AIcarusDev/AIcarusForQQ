"""Enter a platform's main page."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from tools.contract import ToolArgsModel, ToolContract

from ._chat_notes import record_core_focus_transition
from ._platform_tools import focus_summary, platform_registry, runtime_main_focus

TOOL_KIND = "focus_switch"
TOOL_EFFECT: dict[str, str] = {"surface": "platform", "kind": "focus_switch"}


class EnterPlatformArgs(ToolArgsModel):
    name: str = Field(min_length=1, description="目标平台的 name。")


TOOL_CONTRACT = ToolContract(
    name="enter_platform",
    description=(
        "进入指定 platform 的主页面。"
    ),
    args_model=EnterPlatformArgs,
)


def _available(runtime: Any) -> bool:
    return bool(getattr(runtime, "enabled", True))


def execute(name: str = "", **_kwargs: Any) -> dict[str, Any]:
    import app_state
    from llm.session import get_or_create_session

    platform = str(name or "").strip().lower()
    registry = platform_registry()
    runtime = registry.get(platform) if registry is not None else None
    if runtime is None:
        return {
            "ok": False,
            "error": f"未知平台: {platform or name}",
            "platform": platform,
        }
    if not _available(runtime):
        return {
            "ok": False,
            "error": f"平台未启用: {platform}",
            "platform": platform,
        }

    target_focus = runtime_main_focus(runtime)
    if target_focus is None:
        return {
            "ok": False,
            "error": f"平台没有声明主页面: {platform}",
            "platform": platform,
        }

    prev_focus = app_state.current_focus
    already_open = focus_summary(prev_focus)["key"] == target_focus.key()
    if not already_open:
        prev_key = focus_summary(prev_focus)["key"]
        if prev_key:
            try:
                from llm.session import sessions

                prev_session = sessions.get(prev_key)
                if prev_session is not None:
                    prev_session.reset_transient_views()
            except Exception:
                pass
        app_state.current_focus = target_focus
        record_core_focus_transition(prev_focus, target_focus)
    target_session = get_or_create_session(target_focus)
    target_session.last_wake_reason = "enter_platform"
    first_input_event = getattr(app_state, "first_input_event", None)
    if first_input_event is not None:
        first_input_event.set()

    return {
        "ok": True,
        "platform": platform,
        "now_focusing": focus_summary(target_focus),
        "focus_transition": {
            "from": focus_summary(prev_focus),
            "to": focus_summary(target_focus),
            "summary": f"{focus_summary(prev_focus)['key'] or 'none'} -> {target_focus.key()}",
        },
        "already_open": already_open,
    }
