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


XML_TOOL_CALL_ERROR_NAME = "tool_call_error"

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
_TOOL_CALL_ENVELOPE_KEYS = frozenset({"id", "name", "tool", "function", "arguments", "args"})


@dataclass
class XmlToolCallParseResult:
    """Parsed XML tool calls from one model response."""

    tool_calls: list[SimpleNamespace] = field(default_factory=list)
    found_blocks: bool = False
    cognition: str = ""
    errors: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)


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
    next_call_index = 1
    for index, match in enumerate(matches, start=1):
        body = match.group("body").strip()
        values, errors, repairs = _load_tool_call_values(body, index, truncated)
        result.repairs.extend(repairs)
        if values is None:
            result.errors.extend(errors)
            message = errors[-1] if errors else "tool_call JSON 解析失败"
            result.tool_calls.append(_make_protocol_error_call(next_call_index, message, body))
            next_call_index += 1
            continue

        for item in values:
            call, error, repair = _parse_tool_call_object(item, next_call_index)
            if error:
                result.errors.append(error)
                result.tool_calls.append(_make_protocol_error_call(next_call_index, error, body))
                next_call_index += 1
                continue
            if repair:
                result.repairs.append(f"tool_call #{next_call_index}: {repair}")
            assert call is not None
            result.tool_calls.append(call)
            next_call_index += 1
    return result


def _load_tool_call_values(
    body: str,
    block_index: int,
    truncated: bool,
) -> tuple[list[Any] | None, list[str], list[str]]:
    errors: list[str] = []
    repairs: list[str] = []
    if truncated:
        errors.append("tool_call 块缺少闭合标签，尝试解析截断内容")

    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        value, repair_note = _repair_single_excess_closer_json(body)
        if repair_note:
            repairs.append(f"tool_call #{block_index}: {repair_note}")
        else:
            value, repair_note = _repair_missing_closers_json(body)
            if repair_note:
                repairs.append(f"tool_call #{block_index}: {repair_note}")

        if value is None:
            recovered = _recover_tool_call_json_objects(body)
            if recovered:
                note = f"tool_call #{block_index}: 从异常 tool_call 区域恢复 {len(recovered)} 个 JSON 工具调用对象"
                if truncated:
                    note += "，并忽略缺失的闭合标签"
                repairs.append(note)
                return recovered, [], repairs
            errors.append(f"tool_call JSON 解析失败: {exc.msg}")
            return None, errors, repairs

    values = value if isinstance(value, list) else [value]
    if truncated:
        repairs.append(f"tool_call #{block_index}: 补偿缺失的 tool_call 闭合标签")
    return values, [], repairs


def _recover_tool_call_json_objects(text: str) -> list[Any]:
    """Recover JSON tool-call objects from malformed XML wrapper text.

    The model sometimes emits valid JSON objects but corrupts the surrounding
    ``<tool_call>`` boundaries. Treat the XML tag as a hint and recover only
    JSON values that themselves look like tool-call objects.
    """
    recovered: list[Any] = []
    for candidate in _iter_balanced_json_object_slices(text):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if _looks_like_tool_call_object(value):
            recovered.append(value)
    return recovered


def _iter_balanced_json_object_slices(text: str):
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
                in_string = False
                escaped = False
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                yield text[start:index + 1]
                start = None


def _looks_like_tool_call_object(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if isinstance(value.get("name"), str) or isinstance(value.get("tool"), str):
        return True
    function_obj = value.get("function")
    return isinstance(function_obj, dict) and isinstance(function_obj.get("name"), str)


def _repair_single_excess_closer_json(body: str) -> tuple[Any | None, str | None]:
    """Recover JSON when the only problem is one extra structural ``}`` or ``]``."""
    repaired_values: dict[str, tuple[Any, str]] = {}
    for index in _iter_structural_closers(body):
        candidate = body[:index] + body[index + 1:]
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        value_key = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        repaired_values.setdefault(value_key, (value, candidate))

    if len(repaired_values) != 1:
        return None, None

    value, candidate = next(iter(repaired_values.values()))
    if candidate == body:
        return None, None
    return value, "移除了 1 个多余的 JSON 闭合符"


def _repair_missing_closers_json(body: str) -> tuple[Any | None, str | None]:
    """Recover JSON when structural closers were omitted before a valid closer."""
    candidate, closer_count = _insert_missing_json_closers(body)
    if not candidate or closer_count <= 0:
        return None, None

    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None, None

    return value, f"补齐了 {closer_count} 个缺失的 JSON 闭合符"


def _insert_missing_json_closers(text: str) -> tuple[str | None, int]:
    output: list[str] = []
    stack: list[str] = []
    in_string = False
    escaped = False
    inserted = 0

    for char in text.strip():
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            output.append(char)
            in_string = True
        elif char == "{":
            output.append(char)
            stack.append("}")
        elif char == "[":
            output.append(char)
            stack.append("]")
        elif char in "}]":
            if not stack:
                return None, 0
            while stack and stack[-1] != char:
                output.append(stack.pop())
                inserted += 1
            if not stack:
                return None, 0
            output.append(char)
            stack.pop()
        else:
            output.append(char)

    if in_string:
        return None, 0

    while stack:
        output.append(stack.pop())
        inserted += 1

    if inserted <= 0:
        return None, 0
    return "".join(output), inserted


def _iter_structural_closers(text: str):
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "}]":
            yield index


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
) -> tuple[SimpleNamespace | None, str | None, str | None]:
    if not isinstance(item, dict):
        return None, "tool_call 内容必须是 JSON object", None

    function_obj = item.get("function") if isinstance(item.get("function"), dict) else None
    name = item.get("name") or item.get("tool")
    arguments = item.get("arguments") if "arguments" in item else item.get("args", {})
    if function_obj is not None:
        name = name or function_obj.get("name")
        if "arguments" in function_obj:
            arguments = function_obj.get("arguments")

    if not isinstance(name, str) or not name.strip():
        return None, "tool_call 缺少字符串字段 name", None

    repair_note = None
    top_level_args = _extract_top_level_arguments(item, function_obj)
    if top_level_args and _arguments_are_empty(arguments):
        arguments = top_level_args
        repair_note = "顶层参数已移动到 arguments: " + ", ".join(top_level_args.keys())

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
    ), None, repair_note


def _arguments_are_empty(arguments: object) -> bool:
    if arguments is None:
        return True
    if arguments == {}:
        return True
    if isinstance(arguments, str):
        stripped = arguments.strip()
        return stripped == "" or stripped == "{}"
    return False


def _extract_top_level_arguments(
    item: dict[str, Any],
    function_obj: dict[str, Any] | None,
) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    for key, value in item.items():
        if key in _TOOL_CALL_ENVELOPE_KEYS:
            continue
        if function_obj is not None and key == "type" and value == "function":
            continue
        extracted[key] = value
    return extracted


def _make_protocol_error_call(index: int, error: str, raw: str) -> SimpleNamespace:
    arguments = json.dumps({"error": error, "raw": raw}, ensure_ascii=False)
    return SimpleNamespace(
        id=f"call_{index}",
        function=SimpleNamespace(name=XML_TOOL_CALL_ERROR_NAME, arguments=arguments),
        protocol_error=error,
    )
