"""browser_control.py — Playwright-backed browser tool facade."""

from __future__ import annotations

from typing import Any

from tools.browser_session import close_browser_session, get_browser_session, run_in_browser_thread

ALWAYS_AVAILABLE: bool = False

DECLARATION: dict = {
    "name": "browser_control",
    "description": (
        "浏览器工具箱。使用项目内 Playwright 持久浏览器会话打开网页、查看当前视口截图、"
        "滚动、点击当前视口目标、后退/前进、等待页面稳定，并支持常用 Playwright locator 操作。"
        "返回结果会包含当前截图、url/title、scroll、click_targets、visible_images、cached_images。"
        "当用户给出网页 URL、图片站页面、搜索结果页、Pixiv/Pinterest 等无法直接发图的页面时，"
        "先用 action=open 打开；根据截图和 click_targets 决定 action=scroll 或 action=click；"
        "图片还没加载完时用 action=wait 等几秒或等 visible_images/selector；"
        "进入详情页后继续观察；从 cached_images 中选择合适的 brimg_xxx，再用 send_message 的 image_ref 发送。"
        "URL 只是导航入口；可发送图片优先使用 cached_images[].ref。任务完成、长时间不用或需要释放状态时用 action=close 关闭浏览器。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "open",
                    "screenshot",
                    "scroll",
                    "click",
                    "back",
                    "forward",
                    "wait",
                    "locator",
                    "close",
                ],
                "description": (
                    "浏览器动作。常用流程：open -> scroll/click/locator/back/forward -> "
                    "wait 等图片加载 -> 从 cached_images 选 image_ref -> send_message -> close。"
                ),
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
                    "action=click 时点击当前返回结果 click_targets 中的第几个目标。"
                    "点击后会等待并返回新截图；适合点进图片详情页或返回可见链接。"
                ),
            },
            "selector": {
                "type": "string",
                "description": "等待 selector；action=wait/open/scroll/click 后可用它等页面中某个元素出现。",
            },
            "wait_until": {
                "type": "string",
                "enum": ["commit", "domcontentloaded", "load", "networkidle"],
                "description": (
                    "导航/动作后的加载等待策略，默认 domcontentloaded。不要对 Pixiv、Pinterest、"
                    "无限滚动页面或其它 SPA 使用 networkidle；这类页面常有持续请求，容易超时。"
                    "需要等图片时优先设置 visible_images、selector 或 wait_ms。"
                ),
            },
            "wait_ms": {
                "type": "integer",
                "description": "额外固定等待毫秒数，默认 800。",
            },
            "seconds": {
                "type": "number",
                "description": "action=wait 时可用的等待秒数，语义类似主模型 wait；会转换为 wait_ms。",
            },
            "visible_images": {
                "type": "integer",
                "description": "等待当前视口至少出现 N 张已加载图片。",
            },
            "timeout_ms": {
                "type": "integer",
                "description": "操作超时毫秒数。",
            },
            "locator": {
                "type": "object",
                "description": (
                    "action=locator 的 Playwright locator 操作参数。用于更精确地按 css/text/role/"
                    "label/placeholder/test_id 定位，并执行 count/click/fill/press/text/attr/is_visible/wait。"
                    "如果 locator 匹配多个元素，click/fill/press 需要传 nth。"
                ),
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["css", "locator", "text", "role", "label", "placeholder", "test_id"],
                        "description": "定位策略。",
                    },
                    "value": {
                        "type": "string",
                        "description": "定位值。role 策略时为角色名，例如 button/link/img。",
                    },
                    "op": {
                        "type": "string",
                        "enum": ["count", "click", "fill", "press", "text", "attr", "is_visible", "wait"],
                        "description": "locator 操作。",
                    },
                    "nth": {
                        "type": "integer",
                        "description": "当 locator 匹配多个元素时选择第 n 个，0 起始。",
                    },
                    "text": {
                        "type": "string",
                        "description": "fill 的文本，或 press 的按键。",
                    },
                    "key": {
                        "type": "string",
                        "description": "press 的按键名，例如 Enter。",
                    },
                    "attr": {
                        "type": "string",
                        "description": "op=attr 时读取的属性名。",
                    },
                    "options": {
                        "type": "object",
                        "description": (
                            "locator 选项。text/label/placeholder 可用 exact；"
                            "role 可用 name/exact；wait 可用 state。"
                        ),
                    },
                },
                "required": ["strategy", "value", "op"],
            },
            "headful": {
                "type": "boolean",
                "description": "是否显示浏览器窗口。默认 false。",
            },
            "channel": {
                "type": "string",
                "enum": ["chrome", "msedge"],
                "description": "使用本机 Chrome 或 Edge channel；未传则使用 Playwright Chromium。",
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
    return run_in_browser_thread(lambda: _execute_in_browser_thread(**kwargs))


def _execute_in_browser_thread(**kwargs) -> dict:
    action = str(kwargs.get("action") or "").strip().lower()
    if action == "close":
        close_browser_session()
        return {"ok": True, "message": "browser session closed"}

    session = get_browser_session()
    session.ensure(
        headful=bool(kwargs.get("headful", False)),
        channel=str(kwargs.get("channel") or "") or None,
    )

    wait_kwargs = _wait_kwargs(kwargs)
    page = session.require_page()

    if action == "open":
        url = str(kwargs.get("url") or "").strip()
        return session.open(url, **wait_kwargs)

    if action == "screenshot":
        return session.result(events=["screenshot"])

    if action == "scroll":
        pixels = int(kwargs.get("pixels") if kwargs.get("pixels") is not None else 700)
        page.mouse.wheel(0, pixels)
        events = session.wait_ready(**wait_kwargs)
        return session.result(events=[f"scroll={pixels}", *events])

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

    if action == "back":
        page.go_back(wait_until=wait_kwargs["wait_until"], timeout=wait_kwargs["timeout_ms"])
        events = session.wait_ready(**wait_kwargs)
        return session.result(events=["back", *events])

    if action == "forward":
        page.go_forward(wait_until=wait_kwargs["wait_until"], timeout=wait_kwargs["timeout_ms"])
        events = session.wait_ready(**wait_kwargs)
        return session.result(events=["forward", *events])

    if action == "wait":
        events = session.wait_ready(**wait_kwargs)
        return session.result(events=events)

    if action == "locator":
        loc = kwargs.get("locator") or {}
        if not isinstance(loc, dict):
            return {"error": "locator must be an object"}
        op_result = session.locator_operation(
            strategy=str(loc.get("strategy") or "css"),
            value=str(loc.get("value") or loc.get("selector") or ""),
            op=str(loc.get("op") or "count"),
            nth=loc.get("nth") if loc.get("nth") is not None else None,
            text=str(loc.get("text") or ""),
            attr=str(loc.get("attr") or ""),
            key=str(loc.get("key") or ""),
            options=loc.get("options") if isinstance(loc.get("options"), dict) else {},
            timeout_ms=wait_kwargs["timeout_ms"],
            wait_kwargs=wait_kwargs,
        )
        result = session.result(
            events=["locator", *op_result["detail"].get("events", [])],
            include_screenshot=bool(op_result.get("changed", False)),
        )
        result["locator"] = op_result["detail"]
        return result

    return {"error": f"unknown action: {action!r}"}
