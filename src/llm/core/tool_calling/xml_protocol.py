"""XML text protocol for model tool calls.

The main chat loop uses this layer instead of provider-native function calling:
tool schemas are sent as a normal user message, and the model emits
``<tool_call>...</tool_call>`` blocks in its assistant text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html import escape
from types import SimpleNamespace
from typing import Any


XML_TOOL_CALL_ERROR_NAME = "__xml_tool_call_error__"

_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(?P<body>[\s\S]*?)\s*</tool_call>",
    re.IGNORECASE,
)
# 匹配有开头但缺少闭合标签的截断 tool_call（模型输出被截断时兜底）
_TOOL_CALL_TRUNCATED_RE = re.compile(
    r"<tool_call>\s*(?P<body>[\s\S]+)$",
    re.IGNORECASE,
)
_COGNITION_BLOCK_RE = re.compile(
    r"<cognition(?:\s[^>]*)?>\s*(?P<body>[\s\S]*?)\s*</cognition>",
    re.IGNORECASE,
)


@dataclass
class XmlToolCallParseResult:
    """Parsed XML tool calls from one model response."""

    tool_calls: list[SimpleNamespace] = field(default_factory=list)
    found_blocks: bool = False
    cognition: str = ""
    errors: list[str] = field(default_factory=list)


def strip_schema_extensions(obj: object) -> object:
    """Recursively remove custom JSON Schema extension keys before prompting."""
    if isinstance(obj, dict):
        return {
            key: strip_schema_extensions(value)
            for key, value in obj.items()
            if not str(key).startswith("x-")
        }
    if isinstance(obj, list):
        return [strip_schema_extensions(item) for item in obj]
    return obj


def build_tools_xml_message(
    declarations: list[dict[str, Any]],
    hidden_names: list[str] | None = None,
) -> str:
    """Render active schemas and hidden tool names as a persistent payload."""
    tools = [_normalize_declaration(declaration) for declaration in declarations]
    schemas_json = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    hidden_text = ",".join(
        escape(name, quote=False)
        for name in (hidden_names or [])
        if name
    )
    return (
        "<tools>\n"
        "调用格式：\n"
        "<tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}</tool_call>\n\n"
        "规则：\n"
        "- arguments 必须满足对应 parameters schema。\n"
        "- 如果需要连续使用多个工具，按执行顺序输出多个 <tool_call> 块。\n\n"
        "<activated>\n"
        "<schemas>\n"
        f"{schemas_json}\n"
        "</schemas>\n"
        "</activated>\n\n"
        f"<hidden>{hidden_text}</hidden>\n"
        "</tools>"
    )


def parse_xml_tool_calls(raw_text: str | None) -> XmlToolCallParseResult:
    """Extract tool calls from assistant text containing ``<tool_call>`` blocks."""
    text = raw_text or ""
    matches = list(_TOOL_CALL_BLOCK_RE.finditer(text))

    # 兜底：若无完整闭合标签，尝试匹配截断的 tool_call（缺少 </tool_call>）
    truncated = False
    if not matches:
        trunc_match = _TOOL_CALL_TRUNCATED_RE.search(text)
        if trunc_match:
            matches = [trunc_match]
            truncated = True

    result = XmlToolCallParseResult(
        found_blocks=bool(matches),
        cognition=extract_cognition_text(text),
    )
    for index, match in enumerate(matches, start=1):
        body = match.group("body").strip()
        if truncated:
            result.errors.append("tool_call 块缺少闭合标签，尝试解析截断内容")
        try:
            value = json.loads(body)
        except json.JSONDecodeError as exc:
            message = f"tool_call JSON 解析失败: {exc.msg}"
            result.errors.append(message)
            result.tool_calls.append(_make_protocol_error_call(index, message, body))
            continue

        calls = value if isinstance(value, list) else [value]
        for item_index, item in enumerate(calls, start=1):
            call_index = index if len(calls) == 1 else int(f"{index}{item_index}")
            call, error = _parse_tool_call_object(item, call_index)
            if error:
                result.errors.append(error)
                result.tool_calls.append(_make_protocol_error_call(call_index, error, body))
                continue
            assert call is not None
            result.tool_calls.append(call)
    return result


def extract_cognition_text(raw_text: str | None) -> str:
    """Extract natural-language ``<cognition>`` blocks from assistant text."""
    text = raw_text or ""
    parts = [
        match.group("body").strip()
        for match in _COGNITION_BLOCK_RE.finditer(text)
        if match.group("body").strip()
    ]
    return "\n\n".join(parts)


def _normalize_declaration(declaration: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": declaration.get("name", ""),
        "description": declaration.get("description", ""),
        "parameters": strip_schema_extensions(declaration.get("parameters", {})),
    }


def _parse_tool_call_object(
    item: object,
    index: int,
) -> tuple[SimpleNamespace | None, str | None]:
    if not isinstance(item, dict):
        return None, "tool_call 内容必须是 JSON object"

    function_obj = item.get("function") if isinstance(item.get("function"), dict) else None
    name = item.get("name") or item.get("tool")
    arguments = item.get("arguments") if "arguments" in item else item.get("args", {})
    if function_obj is not None:
        name = name or function_obj.get("name")
        if "arguments" in function_obj:
            arguments = function_obj.get("arguments")

    if not isinstance(name, str) or not name.strip():
        return None, "tool_call 缺少字符串字段 name"

    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        arguments_text = arguments
    else:
        arguments_text = json.dumps(arguments, ensure_ascii=False)

    call_id = f"call_{index}"

    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name.strip(), arguments=arguments_text),
    ), None


def _make_protocol_error_call(index: int, error: str, raw: str) -> SimpleNamespace:
    arguments = json.dumps({"error": error, "raw": raw}, ensure_ascii=False)
    return SimpleNamespace(
        id=f"call_{index}",
        function=SimpleNamespace(name=XML_TOOL_CALL_ERROR_NAME, arguments=arguments),
        protocol_error=error,
    )
