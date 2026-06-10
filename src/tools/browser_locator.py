"""browser_locator.py — precise DOM/ARIA browser escape hatch."""

from __future__ import annotations

from typing import Any

from tools.browser_session import (
    get_browser_session,
    record_browser_activity,
    run_in_browser_thread,
)

ALWAYS_AVAILABLE: bool = False

_OP_MAP = {
    "count": "count",
    "click": "click",
    "fill": "fill",
    "press": "press",
    "read_text": "text",
    "read_attribute": "attr",
    "is_visible": "is_visible",
}

_CHANGING_OPS = {"click", "fill", "press"}

DECLARATION: dict = {
    "name": "browser_locator",
    "description": (
        "浏览器高级定位工具。仅在 browser_control 无法完成时使用："
        "按 CSS/Playwright locator/text/role/label/placeholder/test_id 精确定位 DOM 或 ARIA 元素，"
        "进行填表输入、按键、精确点击、读取元素文本/属性、判断可见性或统计匹配数量。"
        "它不返回页面 snapshot，也不负责浏览当前视口；普通打开、滚动、按 <world><browser> index 点击、坐标校准、"
        "后退/前进和关闭浏览器都使用 browser_control。"
        "图片或页面仍在加载时使用全局 wait 等 browser/world 变化。"
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "strategy": {
                "type": "string",
                "enum": ["css", "locator", "text", "role", "label", "placeholder", "test_id"],
                "description": "定位策略。css/locator 使用 Playwright locator；role 使用 ARIA role；其余按可见文本、label、placeholder 或 test id 定位。",
            },
            "query": {
                "type": "string",
                "description": "定位查询。role 策略时填写角色名，例如 button、link、textbox、img。",
            },
            "op": {
                "type": "string",
                "enum": ["count", "click", "fill", "press", "read_text", "read_attribute", "is_visible"],
                "description": "对定位结果执行的操作。",
            },
            "nth": {
                "type": "integer",
                "description": "当定位结果匹配多个元素时选择第 n 个，0 起始。click/fill/press 遇到多匹配时应填写。",
            },
            "input_text": {
                "type": "string",
                "description": "op=fill 时填写的文本；op=press 且未传 key 时作为按键名使用。",
            },
            "key": {
                "type": "string",
                "description": "op=press 时的按键名，例如 Enter、Escape、ArrowDown。",
            },
            "attribute": {
                "type": "string",
                "description": "op=read_attribute 时读取的属性名，例如 href、src、aria-label。",
            },
            "options": {
                "type": "object",
                "description": "定位选项。text/label/placeholder 可用 exact；role 可用 name/exact。",
            },
        },
        "required": ["strategy", "query", "op"],
    },
}


def execute(**kwargs) -> dict:
    op = str(kwargs.get("op") or "").strip().lower()
    result = run_in_browser_thread(lambda: _execute_in_browser_thread(**kwargs))
    record_browser_activity(f"locator:{op}", result)
    return _compact_locator_result(op, result)


def _compact_locator_result(op: str, result: Any) -> dict:
    if not isinstance(result, dict):
        return {"ok": False, "op": op, "error": "browser_locator returned non-object result"}
    if result.get("error"):
        return {"ok": False, "op": op, "error": str(result.get("error") or "")}

    compact: dict[str, Any] = {
        "ok": True,
        "op": op,
        "world_updated": op in _CHANGING_OPS,
    }
    if url := result.get("url"):
        compact["url"] = url
    if title := result.get("title"):
        compact["title"] = title
    if locator := result.get("locator"):
        compact["locator"] = locator
    return compact


def _execute_in_browser_thread(**kwargs) -> dict:
    op = str(kwargs.get("op") or "").strip().lower()
    mapped_op = _OP_MAP.get(op)
    if mapped_op is None:
        return {"error": f"unknown locator op: {op!r}"}
    if op == "read_attribute" and not str(kwargs.get("attribute") or "").strip():
        return {"error": "read_attribute requires attribute"}

    session = get_browser_session()
    session.ensure()
    page = session.require_page()
    try:
        op_result = session.locator_operation(
            strategy=str(kwargs.get("strategy") or "css"),
            value=str(kwargs.get("query") or ""),
            op=mapped_op,
            nth=kwargs.get("nth") if kwargs.get("nth") is not None else None,
            text=str(kwargs.get("input_text") or ""),
            attr=str(kwargs.get("attribute") or ""),
            key=str(kwargs.get("key") or ""),
            options=kwargs.get("options") if isinstance(kwargs.get("options"), dict) else {},
            timeout_ms=10_000,
            wait_kwargs=None,
        )
    except Exception as exc:
        return {"error": str(exc)}

    detail = op_result["detail"]
    result = {
        "url": page.url,
        "title": page.title() or "",
        "locator": detail,
        "events": ["locator", *detail.get("events", [])],
    }
    return result
