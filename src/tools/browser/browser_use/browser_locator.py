"""browser_locator.py — precise DOM/ARIA browser escape hatch."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field, RootModel

from browser.session import (
    get_browser_session,
    record_browser_activity,
    run_in_browser_thread,
)
from tools.contract import ToolArgsModel, ToolContract

_OP_MAP = {
    "count": "count",
    "click": "click",
    "fill": "fill",
    "press": "press",
    "select_option": "select_option",
    "list_options": "list_options",
    "eval": "eval",
    "read_text": "text",
    "read_attribute": "attr",
    "is_visible": "is_visible",
}

_CHANGING_OPS = {"click", "fill", "press", "select_option", "eval"}

JsonValue = str | int | float | bool | None | list[Any] | dict[str, Any]


class BrowserLocatorMatchOptions(ToolArgsModel):
    exact: bool | None = Field(default=None, description="text/label/placeholder 可用 exact；role 可用 name/exact。")
    name: str | None = Field(default=None, description="role 策略可用。")


class BrowserLocatorSelectOptions(BrowserLocatorMatchOptions):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "anyOf": [
                {"required": ["value"]},
                {"required": ["label"]},
                {"required": ["index"]},
                {"required": ["values"]},
            ],
        },
    )

    value: str | None = Field(default=None, description="按 option value 选择。")
    label: str | None = Field(default=None, description="按 option label 选择。")
    index: int | None = Field(default=None, description="按 option 序号选择，0 起始。")
    values: list[str] | None = Field(default=None, min_length=1, description="一次选择多个 option value。")


class BrowserLocatorEvalOptions(BrowserLocatorMatchOptions):
    arg: JsonValue = Field(
        default=None,
        description="传给 element.evaluate(script, arg) 的任意 JSON 值。",
        json_schema_extra={"x-ts-type": "JsonValue"},
    )


class BrowserLocatorBaseArgs(ToolArgsModel):
    strategy: Literal["css", "locator", "text", "role", "label", "placeholder", "test_id"] = Field(
        description="定位策略。css/locator 使用 Playwright locator；role 使用 ARIA role；其余按可见文本、label、placeholder 或 test id 定位。"
    )
    query: str = Field(min_length=1, description="定位查询。role 策略时填写角色名，例如 button、link、textbox、img。")
    options: BrowserLocatorMatchOptions | None = Field(
        default=None,
        description="定位选项。text/label/placeholder 可用 exact；role 可用 name/exact。",
    )


class BrowserLocatorTargetArgs(BrowserLocatorBaseArgs):
    nth: int | None = Field(default=None, ge=0, description="当定位结果匹配多个元素时选择第 n 个，0 起始。")


class BrowserLocatorCountArgs(BrowserLocatorBaseArgs):
    op: Literal["count"] = Field(description="统计匹配元素数量。")


class BrowserLocatorClickArgs(BrowserLocatorTargetArgs):
    op: Literal["click"] = Field(description="点击匹配元素。多匹配时填写 nth。")


class BrowserLocatorFillArgs(BrowserLocatorTargetArgs):
    op: Literal["fill"] = Field(description="填写输入框。")
    input_text: str = Field(min_length=1, description="要填入的文本。")


class BrowserLocatorPressArgs(BrowserLocatorTargetArgs):
    op: Literal["press"] = Field(description="在匹配元素上按键。")
    key: str = Field(min_length=1, description="按键名，例如 Enter、Escape、ArrowDown。")


class BrowserLocatorSelectOptionArgs(BrowserLocatorTargetArgs):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "anyOf": [
                {"required": ["input_text"]},
                {"required": ["select_options"]},
            ],
        },
    )

    op: Literal["select_option"] = Field(description="选择 select 元素中的 option。")
    input_text: str | None = Field(default=None, min_length=1, description="简单按 option value 选择时填写。")
    select_options: BrowserLocatorSelectOptions | None = Field(
        default=None,
        description="选择参数，填写 value、label、index、values 之一；也可同时使用 exact/name 定位。",
    )


class BrowserLocatorListOptionsArgs(BrowserLocatorTargetArgs):
    op: Literal["list_options"] = Field(description="列出 select 元素的 options。多匹配时填写 nth。")


class BrowserLocatorEvalArgs(BrowserLocatorTargetArgs):
    op: Literal["eval"] = Field(description="在匹配元素上执行 element.evaluate。")
    input_text: str = Field(min_length=1, description="元素 evaluate 的 JavaScript，例如 el => el.textContent。")
    eval_options: BrowserLocatorEvalOptions | None = Field(
        default=None,
        description="eval 选项。可用 exact/name 定位；arg 会作为 element.evaluate(script, arg) 的第二参数。",
    )


class BrowserLocatorReadTextArgs(BrowserLocatorTargetArgs):
    op: Literal["read_text"] = Field(description="读取匹配元素的 innerText。")


class BrowserLocatorReadAttributeArgs(BrowserLocatorTargetArgs):
    op: Literal["read_attribute"] = Field(description="读取匹配元素属性。")
    attribute: str = Field(min_length=1, description="要读取的属性名，例如 href、src、aria-label。")


class BrowserLocatorIsVisibleArgs(BrowserLocatorTargetArgs):
    op: Literal["is_visible"] = Field(description="判断匹配元素是否可见。")


class BrowserLocatorArgs(
    RootModel[
        BrowserLocatorCountArgs
        | BrowserLocatorClickArgs
        | BrowserLocatorFillArgs
        | BrowserLocatorPressArgs
        | BrowserLocatorSelectOptionArgs
        | BrowserLocatorListOptionsArgs
        | BrowserLocatorEvalArgs
        | BrowserLocatorReadTextArgs
        | BrowserLocatorReadAttributeArgs
        | BrowserLocatorIsVisibleArgs
    ]
):
    pass


TOOL_CONTRACT = ToolContract(
    name="browser_locator",
    description=(
        "浏览器高级定位工具。在 browser_control 无法完成需求时可尝试使用："
        "按 CSS/Playwright locator/text/role/label/placeholder/test_id 精确定位 DOM 或 ARIA 元素，"
        "进行填表输入、按键、精确点击、读取元素文本/属性、判断可见性或统计匹配数量。"
        "普通打开、滚动、点击、坐标校准、后退/前进和关闭浏览器都使用 browser_control。"
    ),
    args_model=BrowserLocatorArgs,
)

PARALLEL_SAFE = True
PARALLEL_KEY = "browser_page"


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
            options=kwargs.get("select_options") or kwargs.get("eval_options") or (kwargs.get("options") if isinstance(kwargs.get("options"), dict) else {}),
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
