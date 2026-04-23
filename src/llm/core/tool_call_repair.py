"""tool_call_repair.py — 工具调用参数解析与修复。

仅处理 OpenAI 兼容 tool/function calling 返回的 arguments 字符串。
目标不是通用 JSON 修复，而是用保守规则恢复“本应为参数对象”的内容，
并对已知工具的高频串台问题做静默修正。
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger("AICQ.tool_call_repair")

_INTEGER_LITERAL_RE = re.compile(r"^[+-]?\d+$")

_SEND_MESSAGE_TAIL_LEAK_RE = re.compile(
    r'^(?P<body>.*?)(?P<tail>(?:\s*[}\]]{2,}\s*,?\s*)+(?:"?(?P<key>motivation|messages|segments|quote|command|params|content)"?)\s*:.*)$',
    re.DOTALL,
)


def _strip_markdown_fence(text: str) -> str:
    """去掉模型错误塞进 arguments 的 Markdown 代码块包装。"""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_object_slice(text: str) -> str:
    """截取第一个 { 到最后一个 }，处理 arguments 前后多余杂质。"""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def _try_load_object(text: str) -> dict[str, Any] | None:
    """尝试将文本解析为 JSON object。"""
    try:
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _strip_tool_arg_tail_leak(text: str) -> tuple[str, bool]:
    """截断被错误吞进字符串里的后续 JSON 尾巴。"""
    match = _SEND_MESSAGE_TAIL_LEAK_RE.match(text)
    if not match:
        return text, False
    cleaned = match.group("body").rstrip()
    if not cleaned:
        return text, False
    return cleaned, True


def _sanitize_send_message_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """修复 send_message 中文本字段吞入后续键值的串台。"""
    touched_fields: list[str] = []

    def _walk(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return {
                key: _walk(nested, f"{path}.{key}" if path else str(key))
                for key, nested in value.items()
            }
        if isinstance(value, list):
            return [_walk(nested, f"{path}[{index}]") for index, nested in enumerate(value)]
        if isinstance(value, str):
            cleaned, changed = _strip_tool_arg_tail_leak(value)
            if changed:
                touched_fields.append(path or "<root>")
                return cleaned
        return value

    sanitized = _walk(args, "")
    if not isinstance(sanitized, dict):
        return args, touched_fields
    return sanitized, touched_fields


def _coerce_integer_value(value: Any) -> tuple[Any, bool]:
    """按保守规则将整数参数恢复为 int。"""
    if isinstance(value, bool):
        return value, False
    if isinstance(value, int):
        return value, False
    if isinstance(value, str):
        stripped = value.strip()
        if _INTEGER_LITERAL_RE.fullmatch(stripped):
            return int(stripped), True
    return value, False


def _repair_value_by_schema(
    value: Any,
    schema: dict[str, Any],
    path: str,
) -> tuple[Any, list[str]]:
    """按 JSON schema 的有限子集修复参数值。"""
    changes: list[str] = []
    schema_type = schema.get("type")

    if (schema_type == "object" or "properties" in schema) and isinstance(value, dict):
        props = schema.get("properties") or {}
        repaired = value
        for key, child_schema in props.items():
            if key not in value or not isinstance(child_schema, dict):
                continue
            child_path = f"{path}.{key}" if path else str(key)
            repaired_child, child_changes = _repair_value_by_schema(
                value[key],
                child_schema,
                child_path,
            )
            if child_changes:
                if repaired is value:
                    repaired = dict(value)
                repaired[key] = repaired_child
                changes.extend(child_changes)
        return repaired, changes

    if (schema_type == "array" or "items" in schema) and isinstance(value, list):
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return value, changes
        repaired = value
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]" if path else f"[{index}]"
            repaired_item, item_changes = _repair_value_by_schema(item, item_schema, item_path)
            if item_changes:
                if repaired is value:
                    repaired = list(value)
                repaired[index] = repaired_item
                changes.extend(item_changes)
        return repaired, changes

    if schema_type != "integer":
        return value, changes

    repaired_value, coerced = _coerce_integer_value(value)
    if coerced:
        changes.append(f"{path}: {value!r} -> {repaired_value!r} (int)")

    if isinstance(repaired_value, bool) or not isinstance(repaired_value, int):
        return repaired_value, changes

    clamped = repaired_value
    lo = schema.get("minimum")
    hi = schema.get("maximum")
    if lo is not None:
        clamped = max(clamped, lo)
    if hi is not None:
        clamped = min(clamped, hi)
    if clamped != repaired_value:
        lo_str = str(lo) if lo is not None else "-∞"
        hi_str = str(hi) if hi is not None else "+∞"
        changes.append(
            f"{path}: {repaired_value!r} -> {clamped!r} (range [{lo_str}, {hi_str}])"
        )
        repaired_value = clamped

    return repaired_value, changes


def repair_arguments_by_declaration(
    args: dict[str, Any],
    tool_declaration: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """按工具 declaration 修复 arguments。"""
    if not tool_declaration:
        return args, []

    parameters = tool_declaration.get("parameters")
    if not isinstance(parameters, dict):
        return args, []

    repaired, changes = _repair_value_by_schema(args, parameters, "")
    if not isinstance(repaired, dict):
        return args, changes
    return repaired, changes


def _sanitize_tool_arguments(fn_name: str, args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """按工具名执行窄范围语义修复。"""
    if fn_name == "send_message":
        return _sanitize_send_message_args(args)
    return args, []


def _parse_argument_object(raw_arguments: str) -> tuple[dict[str, Any] | None, bool, str | None]:
    """用少量 tool-call 专用策略把 arguments 恢复为 JSON object。"""
    candidates: list[str] = []

    def _push(candidate: str) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    raw = raw_arguments.strip()
    _push(raw)

    stripped = _strip_markdown_fence(raw_arguments)
    _push(stripped)
    _push(_extract_object_slice(raw))
    _push(_extract_object_slice(stripped))

    for candidate in candidates:
        parsed = _try_load_object(candidate)
        if parsed is not None:
            return parsed, candidate != raw_arguments, candidate
    return None, False, None


def parse_tool_arguments(
    raw_arguments: str | None,
    fn_name: str,
    provider_name: str,
    tool_declaration: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    """解析并修复工具调用参数。"""
    if not raw_arguments:
        return {}, True

    parsed_args, repaired_json, repaired_source = _parse_argument_object(raw_arguments)
    if parsed_args is None:
        logger.warning("[%s] 工具参数解析失败: %s", provider_name, fn_name)
        return {}, False

    repaired_args, schema_changes = repair_arguments_by_declaration(parsed_args, tool_declaration)
    sanitized_args, touched_fields = _sanitize_tool_arguments(fn_name, repaired_args)
    if repaired_json:
        logger.warning(
            "[%s] 工具参数已按 tool-call 规则自动恢复: %s",
            provider_name,
            fn_name,
        )
        if repaired_source is not None and repaired_source != raw_arguments:
            logger.debug(
                "[%s] 工具参数恢复前后不同: %s raw=%r repaired=%r",
                provider_name,
                fn_name,
                raw_arguments[:200],
                repaired_source[:200],
            )
    if schema_changes:
        logger.warning(
            "[%s] 工具参数已按 schema 自动修复: %s changes=%s",
            provider_name,
            fn_name,
            "; ".join(schema_changes),
        )
    if touched_fields:
        logger.warning(
            "[%s] 工具参数疑似边界串台，已静默修复: %s fields=%s",
            provider_name,
            fn_name,
            ", ".join(touched_fields),
        )
    return sanitized_args, True
