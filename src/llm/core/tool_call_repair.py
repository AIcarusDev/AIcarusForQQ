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
_NORMALIZED_WHITESPACE_RE = re.compile(r"\s+")

_SEND_MESSAGE_TAIL_LEAK_RE = re.compile(
    r'^(?P<body>.*?)(?P<tail>(?:\s*[}\]]{2,}\s*,?\s*)+(?:"?(?P<key>motivation|messages|segments|quote|command|params|content)"?)\s*:.*)$',
    re.DOTALL,
)

_SEND_MESSAGE_CONTENT_BOUNDARY_LEAK_RE = re.compile(
    r'("content"\s*:\s*")(?P<body>(?:\\.|[^"\\])*?)(?P<tail>(?:\s*[}\]]{2,}\s*,?\s*)+(?:"?(?:motivation|messages|segments|quote|command|params|content)"?)\s*:)'  # noqa: E501
    r'"\s*}}\s*,\s*{(?=\s*"(?:segments|quote)")',
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


def _normalize_repeat_text(text: str) -> str:
    """归一化文本，用于保守判断是否只是复读。"""
    return _NORMALIZED_WHITESPACE_RE.sub(" ", text).strip()


def _merge_motivation_texts(values: list[str]) -> tuple[str | None, bool]:
    """合并多个 motivation；纯复读时保留一个，否则按顺序拼接。"""
    unique_values: list[str] = []
    seen_markers: set[str] = set()

    for value in values:
        stripped = value.strip()
        if not stripped:
            continue
        marker = _normalize_repeat_text(stripped)
        if not marker or marker in seen_markers:
            continue
        seen_markers.add(marker)
        unique_values.append(stripped)

    if not unique_values:
        return None, False
    if len(unique_values) == 1:
        return unique_values[0], len(values) > 1
    return "\n\n".join(unique_values), True


def _build_motivation_merging_hook(changes: list[str]):
    """保留同层重复 motivation，避免 json.loads 直接覆盖前值。"""

    def _hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        motivation_values: list[Any] = []

        for key, value in pairs:
            if key == "motivation":
                motivation_values.append(value)
            result[key] = value

        if len(motivation_values) <= 1:
            return result

        if all(isinstance(value, str) for value in motivation_values):
            merged, changed = _merge_motivation_texts(motivation_values)
            if merged is not None:
                result["motivation"] = merged
                if changed:
                    changes.append(f"merged duplicate motivation ({len(motivation_values)} entries)")
                return result

        result["motivation"] = motivation_values[-1]
        return result

    return _hook


def _try_load_object(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    """尝试将文本解析为 JSON object，同时保留重复 motivation。"""
    parse_changes: list[str] = []
    try:
        value = json.loads(text, object_pairs_hook=_build_motivation_merging_hook(parse_changes))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, []
    return (value if isinstance(value, dict) else None), parse_changes


def _repair_send_message_raw_arguments(text: str) -> tuple[str, list[str]]:
    """修复 send_message 原始 arguments 中被 content 字符串吞掉的消息边界。"""
    repaired, count = _SEND_MESSAGE_CONTENT_BOUNDARY_LEAK_RE.subn(
        lambda match: match.group(1) + match.group("body") + '"}}]},{',
        text,
    )
    if not count:
        return text, []
    return repaired, [f"restored {count} leaked send_message content boundary"]


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
    """修复 send_message 的已知高频串台与 motivation 归位问题。"""
    repair_notes: list[str] = []

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
                repair_notes.append(f"trimmed leaked tail in {path or '<root>'}")
                return cleaned
        return value

    sanitized = _walk(args, "")
    if not isinstance(sanitized, dict):
        return args, repair_notes

    messages = sanitized.get("messages")
    if not isinstance(messages, list):
        return sanitized, repair_notes

    normalized_messages: list[Any] = []
    for index, item in enumerate(messages):
        if not isinstance(item, dict):
            normalized_messages.append(item)
            continue

        segments = item.get("segments")
        if not isinstance(segments, list):
            normalized_messages.append(item)
            continue

        current_segments: list[Any] = []
        leaked_messages: list[dict[str, Any]] = []
        for segment in segments:
            if (
                isinstance(segment, dict)
                and "segments" in segment
                and "command" not in segment
                and "params" not in segment
            ):
                leaked_messages.append(dict(segment))
                continue
            current_segments.append(segment)

        if not leaked_messages:
            normalized_messages.append(item)
            continue

        repaired_item = dict(item)
        repaired_item["segments"] = current_segments
        if current_segments:
            normalized_messages.append(repaired_item)
        normalized_messages.extend(leaked_messages)
        repair_notes.append(f"split leaked message objects from messages[{index}].segments")

    if normalized_messages != messages:
        sanitized = dict(sanitized)
        sanitized["messages"] = normalized_messages
        messages = normalized_messages

    collected_motivations: list[str] = []
    root_motivation = sanitized.get("motivation")
    if isinstance(root_motivation, str) and root_motivation.strip():
        collected_motivations.append(root_motivation)

    rewritten_messages = messages
    hoisted_fields: list[str] = []
    for index, item in enumerate(messages):
        if not isinstance(item, dict) or "motivation" not in item:
            continue
        if rewritten_messages is messages:
            rewritten_messages = list(messages)
        updated_item = dict(item)
        nested_motivation = updated_item.pop("motivation", None)
        rewritten_messages[index] = updated_item
        hoisted_fields.append(f"messages[{index}].motivation")
        if isinstance(nested_motivation, str) and nested_motivation.strip():
            collected_motivations.append(nested_motivation)

    if not hoisted_fields:
        return sanitized, repair_notes

    repaired_args = dict(sanitized)
    repaired_args["messages"] = rewritten_messages

    merged_motivation, _changed = _merge_motivation_texts(collected_motivations)
    if merged_motivation is not None:
        repaired_args["motivation"] = merged_motivation

    repair_notes.append(f"hoisted {', '.join(hoisted_fields)} -> motivation")
    return repaired_args, repair_notes


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


def _parse_argument_object(
    raw_arguments: str,
    fn_name: str,
) -> tuple[dict[str, Any] | None, bool, str | None, list[str]]:
    """用少量 tool-call 专用策略把 arguments 恢复为 JSON object。"""
    candidates: list[tuple[str, list[str]]] = []

    def _push(candidate: str, notes: list[str] | None = None) -> None:
        if not candidate:
            return
        if any(existing == candidate for existing, _existing_notes in candidates):
            return
        candidates.append((candidate, list(notes or [])))

    raw = raw_arguments.strip()
    _push(raw)

    stripped = _strip_markdown_fence(raw_arguments)
    _push(stripped)
    _push(_extract_object_slice(raw))
    _push(_extract_object_slice(stripped))

    if fn_name == "send_message":
        for candidate, notes in list(candidates):
            repaired_candidate, repair_notes = _repair_send_message_raw_arguments(candidate)
            if not repair_notes:
                continue
            merged_notes = [*notes, *repair_notes]
            _push(repaired_candidate, merged_notes)
            sliced_candidate = _extract_object_slice(repaired_candidate)
            if sliced_candidate != repaired_candidate:
                _push(sliced_candidate, merged_notes)

    for candidate, notes in candidates:
        parsed, parse_changes = _try_load_object(candidate)
        if parsed is not None:
            return parsed, candidate != raw_arguments, candidate, [*notes, *parse_changes]
    return None, False, None, []


def parse_tool_arguments(
    raw_arguments: str | None,
    fn_name: str,
    provider_name: str,
    tool_declaration: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    """解析并修复工具调用参数。"""
    if not raw_arguments:
        return {}, True

    parsed_args, repaired_json, repaired_source, parse_changes = _parse_argument_object(
        raw_arguments,
        fn_name,
    )
    if parsed_args is None:
        logger.warning("[%s] 工具参数解析失败: %s", provider_name, fn_name)
        return {}, False

    repaired_args, schema_changes = repair_arguments_by_declaration(parsed_args, tool_declaration)
    sanitized_args, tool_changes = _sanitize_tool_arguments(fn_name, repaired_args)
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
    if parse_changes:
        logger.warning(
            "[%s] 工具参数已在解析阶段静默修复: %s changes=%s",
            provider_name,
            fn_name,
            "; ".join(parse_changes),
        )
    if schema_changes:
        logger.warning(
            "[%s] 工具参数已按 schema 自动修复: %s changes=%s",
            provider_name,
            fn_name,
            "; ".join(schema_changes),
        )
    if tool_changes:
        logger.warning(
            "[%s] 工具参数已按工具语义静默修复: %s changes=%s",
            provider_name,
            fn_name,
            "; ".join(tool_changes),
        )
    return sanitized_args, True
