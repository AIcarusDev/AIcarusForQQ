"""arguments 原始文本解析与结构恢复。"""

import json
import re
from typing import Any

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


def _try_load_object(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    """尝试将文本解析为 JSON object。"""
    try:
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, []
    return (value if isinstance(value, dict) else None), []


def _repair_send_message_raw_arguments(text: str) -> tuple[str, list[str]]:
    """send_message 不再恢复旧 params 嵌套协议。"""
    return text, []


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
