"""browser_control.py — viewport-world browser control tool."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, RootModel

from .prompt import DESCRIPTION
from browser.session import (
    close_browser_session,
    get_browser_session,
    record_browser_activity,
    run_in_browser_thread,
)
from llm.core.tool_calling import ToolWarningFactory
from tools.contract import ToolArgsModel, ToolContract


class BrowserOpenArgs(ToolArgsModel):
    action: Literal["open"] = Field(description="打开网页。")
    url: str | None = Field(
        default=None,
        description=(
            "action=open 时打开的 http/https/file URL；省略时打开 https://www.google.com/。"
            "Agent 电脑内 localhost Web 服务使用服务报告的同端口 URL。"
        ),
    )


class BrowserScrollArgs(ToolArgsModel):
    action: Literal["scroll"] = Field(description="滚动页面。")
    pixels: int = Field(default=700, description="action=scroll 时垂直滚动像素，正数向下，负数向上。默认 700。")


class BrowserScrollRegionArgs(ToolArgsModel):
    action: Literal["scroll_region"] = Field(description="滚动区域。")
    index: int = Field(description="action=scroll_region 时滚动当前 scroll_regions 中的第几个区域。")
    pixels: int = Field(default=700, description="垂直滚动像素，正数向下，负数向上。默认 700。")


class BrowserClickArgs(ToolArgsModel):
    action: Literal["click"] = Field(description="点击目标。")
    index: int = Field(description="action=click 时点击当前 click_targets 中的第几个目标。")


class BrowserMoveXYArgs(ToolArgsModel):
    action: Literal["move_xy"] = Field(description="移动坐标以校准点击位置。")
    x: float = Field(description="x 坐标，单位为当前浏览器视口 CSS 像素，左上角为 0,0。")
    y: float = Field(description="y 坐标，单位为当前浏览器视口 CSS 像素，左上角为 0,0。")


class BrowserConfirmClickArgs(ToolArgsModel):
    action: Literal["confirm_click"] = Field(description="确认上一次 move_xy 标定的坐标点击。")


class BrowserClickXYArgs(ToolArgsModel):
    action: Literal["click_xy"] = Field(description="点击坐标。")
    x: float = Field(description="x 坐标，单位为当前浏览器视口 CSS 像素，左上角为 0,0。")
    y: float = Field(description="y 坐标，单位为当前浏览器视口 CSS 像素，左上角为 0,0。")


class BrowserBackArgs(ToolArgsModel):
    action: Literal["back"] = Field(description="后退。")


class BrowserForwardArgs(ToolArgsModel):
    action: Literal["forward"] = Field(description="前进。")


class BrowserSwitchTabArgs(ToolArgsModel):
    action: Literal["switch_tab"] = Field(description="切换标签页。")
    index: int = Field(description="使用 <world><browser><tabs> 中 tab 的 index。")


class BrowserCloseTabArgs(ToolArgsModel):
    action: Literal["close_tab"] = Field(description="关闭标签页。")
    index: int | None = Field(default=None, description="使用 <world><browser><tabs> 中 tab 的 index；省略时关闭当前标签页。")


class BrowserCloseBrowserArgs(ToolArgsModel):
    action: Literal["close_browser"] = Field(description="关闭整个浏览器。")


class BrowserControlArgs(
    RootModel[
        BrowserOpenArgs
        | BrowserScrollArgs
        | BrowserScrollRegionArgs
        | BrowserClickArgs
        | BrowserMoveXYArgs
        | BrowserConfirmClickArgs
        | BrowserClickXYArgs
        | BrowserBackArgs
        | BrowserForwardArgs
        | BrowserSwitchTabArgs
        | BrowserCloseTabArgs
        | BrowserCloseBrowserArgs
    ]
):
    pass


TOOL_CONTRACT = ToolContract(
    name="browser_control",
    description=DESCRIPTION,
    args_model=BrowserControlArgs,
)

_CONTROL_ACTIONS = {
    "open",
    "scroll",
    "scroll_region",
    "click",
    "move_xy",
    "confirm_click",
    "click_xy",
    "back",
    "forward",
    "switch_tab",
    "close_tab",
    "close_browser",
}
_DEFAULT_OPEN_URL = "https://www.google.com/"


def _wait_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    seconds = kwargs.get("seconds")
    wait_ms_value = kwargs.get("wait_ms")
    if seconds is not None and wait_ms_value is None:
        try:
            wait_ms_value = min(max(float(seconds), 0.0), 30.0) * 1000
        except (TypeError, ValueError):
            wait_ms_value = None
    return {
        "wait_until": str(kwargs.get("wait_until") or "domcontentloaded"),
        "wait_ms": int(wait_ms_value if wait_ms_value is not None else 800),
        "visible_images": int(kwargs.get("visible_images") or 0),
        "selector": str(kwargs.get("selector") or ""),
        "timeout_ms": int(kwargs.get("timeout_ms") or 10_000),
    }


def _normalize_control_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(kwargs)
    action = str(normalized.get("action") or "").strip().lower()
    normalized["action"] = action
    return normalized


def _open_url(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("url") or "").strip() or _DEFAULT_OPEN_URL


def execute(**kwargs) -> dict:
    normalized = _normalize_control_kwargs(kwargs)
    action = str(normalized.get("action") or "")
    result = run_in_browser_thread(lambda: _execute_in_browser_thread(**normalized))
    record_browser_activity(action, result)
    return _compact_tool_result(action, result)


def _compact_clicked(clicked: Any) -> dict:
    if not isinstance(clicked, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in ("ok", "x", "y", "error", "count"):
        if key in clicked:
            compact[key] = clicked[key]
    target = clicked.get("target")
    if isinstance(target, dict):
        compact["target"] = {
            key: target[key]
            for key in ("index", "role", "name", "tag", "text", "href", "x", "y")
            if key in target and target[key]
        }
    return compact


def _compact_tool_result(action: str, result: Any) -> dict:
    if not isinstance(result, dict):
        return {"ok": False, "action": action, "error": "browser_control returned non-object result"}
    if result.get("error"):
        compact_error: dict[str, Any] = {"ok": False, "action": action, "error": str(result.get("error") or "")}
        for key in ("count", "limit", "max_tabs"):
            if key in result:
                compact_error[key] = result[key]
        if tabs := result.get("tabs"):
            compact_error["tabs"] = [
                {
                    key: tab[key]
                    for key in ("index", "active", "title", "url")
                    if isinstance(tab, dict) and key in tab and tab[key] not in ("", None)
                }
                for tab in tabs
                if isinstance(tab, dict)
            ][:12]
        return compact_error

    state = str(result.get("state") or "")
    closes_browser = action == "close_browser" or state in {"closed", "already_closed"}
    compact: dict[str, Any] = {
        "ok": True,
        "action": action,
        "world_updated": not closes_browser,
    }
    if url := result.get("url"):
        compact["url"] = url
    if title := result.get("title"):
        compact["title"] = title
    if tabs := result.get("tabs"):
        compact["tabs"] = [
            {
                key: tab[key]
                for key in ("index", "active", "title", "url")
                if isinstance(tab, dict) and key in tab and tab[key] not in ("", None)
            }
            for tab in tabs
            if isinstance(tab, dict)
        ][:12]
    if max_tabs := result.get("max_tabs"):
        compact["max_tabs"] = max_tabs

    if clicked := result.get("clicked"):
        compact["clicked"] = _compact_clicked(clicked)
    if scrolled_region := result.get("scrolled_region"):
        compact["scrolled_region"] = scrolled_region
    if pending := result.get("pending_click"):
        compact["pending_click"] = pending
    if message := result.get("message"):
        compact["message"] = message
    if state:
        compact["state"] = state
    if warnings := result.get("warnings"):
        compact["warnings"] = warnings
    if action == "close_browser":
        return compact
    return compact


def _execute_in_browser_thread(**kwargs) -> dict:
    kwargs = _normalize_control_kwargs(kwargs)
    action = str(kwargs.get("action") or "").strip().lower()
    if action not in _CONTROL_ACTIONS:
        return {"error": f"unknown action: {action!r}"}
    if action == "close_browser":
        closed = close_browser_session()
        if closed:
            return {"ok": True, "message": "browser session closed", "state": "closed"}
        return {
            "ok": True,
            "message": "browser session already closed",
            "state": "already_closed",
            "warnings": [ToolWarningFactory.no_browser_session().to_dict()],
        }

    session = get_browser_session()
    wait_kwargs = _wait_kwargs(kwargs)

    if action == "open":
        return session.open_new_page(_open_url(kwargs), **wait_kwargs)

    session.ensure()
    page = session.require_page()

    if action == "scroll":
        pixels_val = kwargs.get("pixels")
        pixels = int(pixels_val) if pixels_val is not None else 700
        page.mouse.wheel(0, pixels)
        events = session.wait_ready(**wait_kwargs)
        return session.result(events=[f"scroll={pixels}", *events])

    if action == "scroll_region":
        index = int(kwargs.get("index") or 0)
        pixels_val = kwargs.get("pixels")
        pixels = int(pixels_val) if pixels_val is not None else 700
        scrolled = session.scroll_region(index, pixels)
        if not scrolled.get("ok"):
            return {"error": scrolled.get("error") or "scroll_region failed"}
        events = session.wait_ready(**wait_kwargs)
        result = session.result(events=[f"scroll_region={index}:{pixels}", *events])
        result["scrolled_region"] = scrolled
        return result

    if action == "click":
        index = int(kwargs.get("index") or 0)
        clicked = session.click_target(index)
        href = ""
        if isinstance(clicked.get("target"), dict):
            href = str(clicked["target"].get("href") or "")
        if href and not href.startswith("#"):
            try:
                page.wait_for_url(href, wait_until=wait_kwargs["wait_until"], timeout=wait_kwargs["timeout_ms"])
            except Exception:
                pass
        events = session.wait_ready(**wait_kwargs)
        result = session.result(events=[f"click={index}", *events])
        result["clicked"] = clicked
        return result

    if action == "move_xy":
        try:
            x_val = kwargs.get("x")
            y_val = kwargs.get("y")
            if x_val is None or y_val is None:
                raise TypeError()
            x = float(x_val)
            y = float(y_val)
        except (TypeError, ValueError):
            return {"error": "move_xy requires numeric x and y"}
        pending = session.set_pending_click(x, y)
        result = session.result(events=[f"move_xy={x:.1f},{y:.1f}"])
        result["pending_click"] = pending
        return result

    if action == "confirm_click":
        clicked = session.confirm_pending_click()
        if not clicked.get("ok"):
            return {"error": clicked.get("error") or "confirm_click failed"}
        events = session.wait_ready(**wait_kwargs)
        result = session.result(events=[f"confirm_click={clicked['x']:.1f},{clicked['y']:.1f}", *events])
        result["clicked"] = clicked
        return result

    if action == "click_xy":
        try:
            x_val = kwargs.get("x")
            y_val = kwargs.get("y")
            if x_val is None or y_val is None:
                raise TypeError()
            x = float(x_val)
            y = float(y_val)
        except (TypeError, ValueError):
            return {"error": "click_xy requires numeric x and y"}
        page.mouse.click(x, y)
        events = session.wait_ready(**wait_kwargs)
        result = session.result(events=[f"click_xy={x:.1f},{y:.1f}", *events])
        result["clicked"] = {"ok": True, "x": x, "y": y}
        return result

    if action == "switch_tab":
        index = int(kwargs.get("index") or 0)
        switched = session.switch_tab(index)
        if not switched.get("ok"):
            return {"error": switched.get("error") or "switch_tab failed"}
        events = session.wait_ready(**wait_kwargs)
        result = session.result(events=[f"switch_tab={index}", *events])
        result["tabs"] = switched.get("tabs") or result.get("tabs") or []
        return result

    if action == "close_tab":
        raw_index = kwargs.get("index")
        tab_index = int(raw_index) if raw_index is not None else None
        closed = session.close_tab(tab_index)
        if not closed.get("ok"):
            return {"error": closed.get("error") or "close_tab failed"}
        if closed.get("last_tab"):
            close_browser_session()
            closed_index = closed.get("index")
            return {
                "ok": True,
                "message": "last tab closed; browser session closed",
                "state": "closed",
                "events": [f"close_tab={closed_index}", "closed_last_tab", "close_browser"],
            }
        events = session.wait_ready(**wait_kwargs)
        result = session.result(events=[f"close_tab={closed.get('index')}", *events])
        result["tabs"] = closed.get("tabs") or result.get("tabs") or []
        return result

    if action == "back":
        page.go_back(wait_until=wait_kwargs["wait_until"], timeout=wait_kwargs["timeout_ms"])
        events = session.wait_ready(**wait_kwargs)
        return session.result(events=["back", *events])

    if action == "forward":
        page.go_forward(wait_until=wait_kwargs["wait_until"], timeout=wait_kwargs["timeout_ms"])
        events = session.wait_ready(**wait_kwargs)
        return session.result(events=["forward", *events])

    return {"error": f"unknown action: {action!r}"}
