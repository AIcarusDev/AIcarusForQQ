"""List available platform pages."""

from __future__ import annotations

from typing import Any

from tools.contract import ToolArgsModel, ToolContract

from ._platform_tools import current_page_platform, focus_summary, platform_registry, runtime_main_focus


class ListPlatformsArgs(ToolArgsModel):
    pass


TOOL_CONTRACT = ToolContract(
    name="list_platforms",
    description=(
        "列出当前所有可用 platform"
    ),
    args_model=ListPlatformsArgs,
)


def execute(**_kwargs: Any) -> dict[str, Any]:
    import app_state
    from llm.session import sessions

    registry = platform_registry()
    current_platform = current_page_platform()
    current_focus = focus_summary(getattr(app_state, "current_focus", None))
    platforms: list[dict[str, Any]] = []
    runtimes = getattr(registry, "runtimes", {}) if registry is not None else {}
    for name, runtime in runtimes.items():
        main_focus = runtime_main_focus(runtime)
        main_summary = focus_summary(main_focus)
        session = sessions.get(main_summary["key"]) if main_summary["key"] else None
        unread = int(getattr(session, "unread_count", 0) or 0) if session is not None else 0
        account = getattr(runtime, "account", None)
        platforms.append({
            "name": str(name),
            "enabled": bool(getattr(runtime, "enabled", False)),
            "connected": bool(getattr(runtime, "connected", False)),
            "state": str(getattr(runtime, "state", "connecting") or "connecting"),
            "page_open": current_platform == str(name),
            "main": main_summary,
            "unread": unread,
            "account_id": str(getattr(account, "account_id", "") or ""),
            "account_name": str(getattr(account, "account_name", "") or ""),
        })

    return {
        "ok": True,
        "current_page": current_focus if current_platform else None,
        "platforms": platforms,
    }
