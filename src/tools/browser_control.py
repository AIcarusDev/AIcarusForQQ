"""browser_control.py — viewport-world browser control tool."""

from __future__ import annotations

from typing import Any

from browser.session import (
    close_browser_session,
    get_browser_session,
    record_browser_activity,
    run_in_browser_thread,
)

ALWAYS_AVAILABLE: bool = True

DECLARATION: dict = {
    "name": "browser_control",
    "description": (
        "浏览器控制工具。用于打开网页，并按 <world><browser> 里的可点击目标 index、可滚动区域 index、"
        "或视口 CSS 坐标进行滚动、点击、坐标校准、后退/前进。"
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
                    "back",
                    "forward",
                    "close",
                ],
                "description": "浏览器控制动作。",
            },
            "url": {
                "type": "string",
                "description": "action=open 时要打开的 http/https/file URL。普通网页、图片站页面、详情页都用 URL 打开。",
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
        return {"ok": False, "action": action, "error": str(result.get("error") or "")}

    compact: dict[str, Any] = {
        "ok": True,
        "action": action,
        "world_updated": action != "close",
    }
    if url := result.get("url"):
        compact["url"] = url
    if title := result.get("title"):
        compact["title"] = title

    if clicked := result.get("clicked"):
        compact["clicked"] = _compact_clicked(clicked)
    if scrolled_region := result.get("scrolled_region"):
        compact["scrolled_region"] = scrolled_region
    if pending := result.get("pending_click"):
        compact["pending_click"] = pending
    if action == "close":
        compact["world_updated"] = False
    return compact


def _execute_in_browser_thread(**kwargs) -> dict:
    action = str(kwargs.get("action") or "").strip().lower()
    if action == "close":
        close_browser_session()
        return {"ok": True, "message": "browser session closed"}

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

    if action == "back":
        page.go_back(wait_until=wait_kwargs["wait_until"], timeout=wait_kwargs["timeout_ms"])
        events = session.wait_ready(**wait_kwargs)
        return session.result(events=["back", *events])

    if action == "forward":
        page.go_forward(wait_until=wait_kwargs["wait_until"], timeout=wait_kwargs["timeout_ms"])
        events = session.wait_ready(**wait_kwargs)
        return session.result(events=["forward", *events])

    return {"error": f"unknown action: {action!r}"}
