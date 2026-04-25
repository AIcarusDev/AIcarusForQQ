"""arguments 原始文本解析与结构恢复。"""

import json
import re
from typing import Any

from .common import merge_motivation_texts

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
            merged, changed = merge_motivation_texts(motivation_values)
            if merged is not None:
                result["motivation"] = merged
                if changed:
                    changes.append(
                        f"merged duplicate motivation ({len(motivation_values)} entries)"
                    )
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


def parse_argument_object(
    raw_arguments: str | None,
    fn_name: str,
) -> tuple[dict[str, Any] | None, bool, str | None, list[str]]:
    """用少量 tool-call 专用策略把 arguments 恢复为 JSON object。"""
    if raw_arguments is None:
        return {}, False, None, []

    raw_text = raw_arguments.strip()
    if not raw_text:
        return {}, False, raw_arguments, []

    candidates: list[tuple[str, list[str]]] = []

    def _push(candidate: str, notes: list[str] | None = None) -> None:
        if not candidate:
            return
        if any(existing == candidate for existing, _existing_notes in candidates):
            return
        candidates.append((candidate, list(notes or [])))

    _push(raw_text)

    stripped = _strip_markdown_fence(raw_arguments)
    _push(stripped)
    _push(_extract_object_slice(raw_text))
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