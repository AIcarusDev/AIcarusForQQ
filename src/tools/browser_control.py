"""browser_control.py — viewport-world browser control tool."""

from __future__ import annotations

from typing import Any

from browser.session import (
    close_browser_session,
    get_browser_session,
    record_browser_activity,
    run_in_browser_thread,
)
from llm.core.tool_calling import ToolWarningFactory

ALWAYS_AVAILABLE: bool = True

DECLARATION: dict = {
    "name": "browser_control",
    "description": (
        "浏览器控制工具。用于打开网页，并按 <world><browser> 里的可点击目标 index、可滚动区域 index、"
        "或 tabs 的 tab_index 进行滚动、点击、标签页切换/新建/关闭、坐标校准、后退/前进。"
        "这是便捷的轻量工具，如果需要按 DOM/CSS/ARIA locator 精确查找元素、填表输入文本、按键，读取元素文本或属性、"
        "统计 locator 匹配数量等进一步操作，则需要 browser_locator 工具。"
        "已经不需要再使用浏览器时，记得 close。"
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "open",
                    "scroll",
                    "scroll_region",
                    "click",
                    "move_xy",
                    "confirm_click",
                    "click_xy",
                    "new_tab",
                    "switch_tab",
                    "close_tab",
                    "back",
                    "forward",
                    "close",
                ],
                "description": "浏览器控制动作。",
            },
            "url": {
                "type": "string",
                "description": "action=open 时在当前标签页打开 http/https/file URL；action=new_tab 时可选填新标签页 URL。普通跳转优先用 open，只有需要并行保留页面时才用 new_tab。",
            },
            "pixels": {
                "type": "integer",
                "description": "action=scroll 时垂直滚动像素，正数向下，负数向上。默认 700。",
            },
            "index": {
                "type": "integer",
                "description": (
                    "action=click 时点击当前 click_targets 中的第几个目标；"
                    "action=scroll_region 时滚动当前 scroll_regions 中的第几个区域。"
                ),
            },
            "tab_index": {
                "type": "integer",
                "description": "action=switch_tab 或 close_tab 时切换/关闭 <world><browser><tabs> 中的第几个标签页。",
            },
            "x": {
                "type": "number",
                "description": "action=move_xy 或 click_xy 时的 x 坐标，单位为当前浏览器视口 CSS 像素，左上角为 0,0。",
            },
            "y": {
                "type": "number",
                "description": "action=move_xy 或 click_xy 时的 y 坐标，单位为当前浏览器视口 CSS 像素，左上角为 0,0。",
            },
        },
        "required": ["action"],
    },
}


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


def execute(**kwargs) -> dict:
    action = str(kwargs.get("action") or "").strip().lower()
    result = run_in_browser_thread(lambda: _execute_in_browser_thread(**kwargs))
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

    compact: dict[str, Any] = {
        "ok": True,
        "action": action,
        "world_updated": action != "close",
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
    if action == "close":
        compact["world_updated"] = False
        if message := result.get("message"):
            compact["message"] = message
        if state := result.get("state"):
            compact["state"] = state
        if warnings := result.get("warnings"):
            compact["warnings"] = warnings
        return compact
    return compact


def _execute_in_browser_thread(**kwargs) -> dict:
    action = str(kwargs.get("action") or "").strip().lower()
    if action == "close":
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
    session.ensure()

    wait_kwargs = _wait_kwargs(kwargs)
    page = session.require_page()

    if action == "open":
        url = str(kwargs.get("url") or "").strip()
        return session.open(url, **wait_kwargs)

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

    if action == "new_tab":
        url = str(kwargs.get("url") or "").strip()
        opened = session.new_tab(url)
        if not opened.get("ok"):
            return {
                "error": opened.get("error") or "new_tab failed",
                "count": opened.get("count"),
                "limit": opened.get("limit"),
                "max_tabs": opened.get("limit"),
                "tabs": opened.get("tabs") or [],
            }
        events = session.wait_ready(**wait_kwargs)
        result = session.result(events=[f"new_tab={opened.get('index')}", *events])
        result["tabs"] = opened.get("tabs") or result.get("tabs") or []
        return result

    if action == "switch_tab":
        tab_index = int(kwargs.get("tab_index") if kwargs.get("tab_index") is not None else kwargs.get("index") or 0)
        switched = session.switch_tab(tab_index)
        if not switched.get("ok"):
            return {"error": switched.get("error") or "switch_tab failed"}
        events = session.wait_ready(**wait_kwargs)
        result = session.result(events=[f"switch_tab={tab_index}", *events])
        result["tabs"] = switched.get("tabs") or result.get("tabs") or []
        return result

    if action == "close_tab":
        raw_index = kwargs.get("tab_index") if kwargs.get("tab_index") is not None else kwargs.get("index")
        tab_index = int(raw_index) if raw_index is not None else None
        closed = session.close_tab(tab_index)
        if not closed.get("ok"):
            return {"error": closed.get("error") or "close_tab failed"}
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
