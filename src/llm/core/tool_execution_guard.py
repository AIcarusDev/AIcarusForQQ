"""Pre-execution guard for externally perceptible tools.

The guard is intentionally not a tool/function call. It asks a configured
submodel for direct JSON and treats malformed or failed responses as allow.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("AICQ.llm.tool_execution_guard")


DEFAULT_GUARD_GEN: dict[str, Any] = {
    "temperature": 0.2,
    "max_output_tokens": 512,
    "enable_thinking": False,
}

SYSTEM_PROMPT = """
你的任务是动作执行前检查。

系统已经确认：当前 <world> 相比你刚做出本轮决策时的 <world> 发生了语义变化。
现在需要是判断：在新的 <world> 下，基于你的 cognition，被检查的函数工具调用是否仍然适合执行。
注意：道德并非你的判断标准，合理性才是。只需判断在当前 world 状态和 cognition 下，执行该函数工具是否合理。

你只能依据输入中的三部分判断：
- <cognition>：你做出工具调用时的认知。
- <tool_call_json>：即将执行的函数工具调用 JSON。
- <world>：当前最新世界状态。

输出要求：
- 只输出 JSON，不要 Markdown，不要解释文本。
- 推荐格式：{"execute": boolean }。
- execute=true 表示可以继续执行该函数工具。
- execute=false 表示需要阻止该函数工具执行。
- 信息不足时默认 execute=true。
"""


@dataclass(frozen=True)
class ToolExecutionGuardDecision:
    execute: bool
    reason: str
    checked: bool = False
    world_changed: bool = False
    raw_response: str = ""


def normalize_tool_execution_guard_config(cfg: dict | None) -> dict[str, Any]:
    raw = dict(cfg or {})
    generation = dict(DEFAULT_GUARD_GEN)
    if isinstance(raw.get("generation"), dict):
        generation.update(raw["generation"])
    return {
        "enabled": bool(raw.get("enabled", False)),
        "provider": str(raw.get("provider") or ""),
        "model": str(raw.get("model") or ""),
        "vision": bool(raw.get("vision", False)),
        "generation": generation,
    }


def _content_to_text(content: str | list | None) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return ""


_WORLD_OPEN_RE = re.compile(r"<world\b[^>]*>", flags=re.IGNORECASE)


def extract_world_text(content: str | list | None) -> str:
    text = _content_to_text(content)
    candidates = list(_WORLD_OPEN_RE.finditer(text))
    if not candidates:
        return text.strip()
    for match in candidates:
        end = text.find("</world>", match.end())
        if end < 0:
            continue
        candidate = text[match.start() : end + len("</world>")].strip()
        try:
            root = ET.fromstring(candidate)
        except ET.ParseError:
            continue
        if root.tag.lower() == "world":
            return candidate

    last = candidates[-1]
    end = text.rfind("</world>")
    if end < last.end():
        return text[last.start() :].strip()
    return text[last.start() : end + len("</world>")].strip()


_VOLATILE_WORLD_ATTRS = {"timestamp"}
_VOLATILE_WORLD_ATTR_RE = re.compile(
    r"\s+(?:timestamp)\s*=\s*(?:\"[^\"]*\"|'[^']*')",
    flags=re.IGNORECASE,
)


_CURRENT_TIME_RE = re.compile(
    r"<current_time\b[^>]*>.*?</current_time>",
    flags=re.DOTALL | re.IGNORECASE,
)

_SELF_PRIVATE_MESSAGE_RE = re.compile(
    r"<message\b(?=[^>]*\bfrom\s*=\s*['\"]self['\"])[^>]*>.*?</message>",
    flags=re.DOTALL | re.IGNORECASE,
)
_SELF_GROUP_MESSAGE_RE = re.compile(
    r"<message\b[^>]*>\s*"
    r"<sender\b(?=[^>]*\bid\s*=\s*['\"]self['\"])[^>]*(?:/>\s*|>\s*</sender>\s*)"
    r".*?</message>",
    flags=re.DOTALL | re.IGNORECASE,
)
_SELF_ID_RE = re.compile(
    r"<self\b[^>]*\bid\s*=\s*['\"]([^'\"]+)['\"]",
    flags=re.IGNORECASE,
)


def _collect_self_ids(root: ET.Element) -> set[str]:
    self_ids = {"self"}
    for element in root.iter("self"):
        self_id = str(element.attrib.get("id") or "").strip().lower()
        if self_id:
            self_ids.add(self_id)
    return self_ids


def _is_self_message_element(element: ET.Element, self_ids: set[str]) -> bool:
    if element.tag != "message":
        return False
    if str(element.attrib.get("from") or "").strip().lower() == "self":
        return True
    for child in list(element):
        if child.tag != "sender":
            continue
        sender_id = str(child.attrib.get("id") or "").strip().lower()
        if sender_id in self_ids:
            return True
        break
    return False


def _is_self_note_element(element: ET.Element, self_ids: set[str]) -> bool:
    if element.tag != "note":
        return False
    for child in list(element):
        if child.tag != "operator":
            continue
        operator_id = str(child.attrib.get("id") or "").strip().lower()
        return bool(operator_id and operator_id in self_ids)
    return False


def _drop_self_effects_from_element(parent: ET.Element, self_ids: set[str]) -> None:
    for child in list(parent):
        if _is_self_message_element(child, self_ids) or _is_self_note_element(child, self_ids):
            parent.remove(child)
            continue
        _drop_self_effects_from_element(child, self_ids)


def _drop_volatile_attrs_from_element(parent: ET.Element) -> None:
    for attr in list(parent.attrib):
        if attr.lower() in _VOLATILE_WORLD_ATTRS:
            del parent.attrib[attr]
    for child in list(parent):
        _drop_volatile_attrs_from_element(child)


def _drop_volatile_attrs_with_regex(world_text: str) -> str:
    return _VOLATILE_WORLD_ATTR_RE.sub("", world_text)


def _normalize_current_time_elements(parent: ET.Element) -> None:
    for element in parent.iter("current_time"):
        element.clear()


def _drop_self_effects_with_regex(world_text: str) -> str:
    self_ids = {"self"}
    self_ids.update(match.group(1).strip().lower() for match in _SELF_ID_RE.finditer(world_text))
    without_private = _SELF_PRIVATE_MESSAGE_RE.sub("", world_text)
    without_group = _SELF_GROUP_MESSAGE_RE.sub("", without_private)
    for self_id in sorted(self_ids, key=len, reverse=True):
        if not self_id:
            continue
        note_re = re.compile(
            r"<note\b[^>]*>\s*"
            r"<operator\b(?=[^>]*\bid\s*=\s*['\"]"
            + re.escape(self_id)
            + r"['\"])[^>]*(?:/>\s*|>\s*</operator>\s*)"
            r".*?</note>",
            flags=re.DOTALL | re.IGNORECASE,
        )
        without_group = note_re.sub("", without_group)
    return without_group


def _drop_self_messages_from_world_text(world_text: str) -> str:
    """Remove bot-authored chat effects before comparing external world changes."""
    stripped = world_text.strip()
    if not stripped:
        return stripped
    try:
        root = ET.fromstring(stripped)
    except ET.ParseError:
        return _drop_volatile_attrs_with_regex(_drop_self_effects_with_regex(world_text))
    _drop_self_effects_from_element(root, _collect_self_ids(root))
    _drop_volatile_attrs_from_element(root)
    _normalize_current_time_elements(root)
    return ET.tostring(root, encoding="unicode", short_empty_elements=True)


def _normalize_world_for_signature(world_text: str) -> str:
    without_self_messages = _drop_self_messages_from_world_text(world_text)
    without_time = _CURRENT_TIME_RE.sub("<current_time/>", without_self_messages)
    return re.sub(r"\s+", " ", without_time).strip()


def world_semantic_signature(content: str | list | None) -> str:
    world = extract_world_text(content)
    normalized = _normalize_world_for_signature(world)
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def _parsed_normalized_world_root(content: str | list | None) -> ET.Element | None:
    world_text = extract_world_text(content)
    try:
        root = ET.fromstring(world_text.strip())
    except ET.ParseError:
        return None
    _drop_volatile_attrs_from_element(root)
    _normalize_current_time_elements(root)
    return root


def _normalized_world_root(content: str | list | None) -> ET.Element | None:
    root = _parsed_normalized_world_root(content)
    if root is None:
        return None
    _drop_self_effects_from_element(root, _collect_self_ids(root))
    return root


def _element_signature(element: ET.Element) -> str:
    return re.sub(
        r"\s+",
        " ",
        ET.tostring(element, encoding="unicode", short_empty_elements=True),
    ).strip()


def _clear_chat_log_children(root: ET.Element) -> ET.Element:
    skeleton = copy.deepcopy(root)
    for chat_logs in skeleton.iter("chat_logs"):
        for child in list(chat_logs):
            chat_logs.remove(child)
    return skeleton


def _self_effect_signatures(root: ET.Element) -> set[str]:
    self_ids = _collect_self_ids(root)
    signatures: set[str] = set()

    def collect(parent: ET.Element) -> None:
        for child in list(parent):
            if _is_self_message_element(child, self_ids) or _is_self_note_element(child, self_ids):
                signatures.add(_element_signature(child))
                continue
            collect(child)

    collect(root)
    return signatures


def _chat_log_entry_maps(root: ET.Element) -> list[dict[tuple[str, str], str]]:
    maps: list[dict[tuple[str, str], str]] = []
    for chat_logs in root.iter("chat_logs"):
        entry_map: dict[tuple[str, str], str] = {}
        for index, child in enumerate(list(chat_logs)):
            if child.tag not in {"message", "note"}:
                continue
            signature = _element_signature(child)
            entry_id = str(child.attrib.get("id") or "")
            key = (child.tag, entry_id or f"@{index}:{signature}")
            entry_map[key] = signature
        maps.append(entry_map)
    return maps


def _only_chat_log_window_drift(
    decision_world: str | list | None,
    current_world: str | list | None,
) -> bool:
    decision_raw_root = _parsed_normalized_world_root(decision_world)
    current_raw_root = _parsed_normalized_world_root(current_world)
    if decision_raw_root is None or current_raw_root is None:
        return False
    new_self_effects = (
        _self_effect_signatures(current_raw_root)
        - _self_effect_signatures(decision_raw_root)
    )
    if not new_self_effects:
        return False

    decision_root = _normalized_world_root(decision_world)
    current_root = _normalized_world_root(current_world)
    if decision_root is None or current_root is None:
        return False

    decision_skeleton = _element_signature(_clear_chat_log_children(decision_root))
    current_skeleton = _element_signature(_clear_chat_log_children(current_root))
    if decision_skeleton != current_skeleton:
        return False

    decision_logs = _chat_log_entry_maps(decision_root)
    current_logs = _chat_log_entry_maps(current_root)
    if len(decision_logs) != len(current_logs):
        return False

    for decision_entries, current_entries in zip(decision_logs, current_logs):
        for key, current_signature in current_entries.items():
            if decision_entries.get(key) != current_signature:
                return False
    return True


def world_semantically_changed(
    decision_world: str | list | None,
    current_world: str | list | None,
) -> bool:
    if world_semantic_signature(decision_world) == world_semantic_signature(current_world):
        return False
    if _only_chat_log_window_drift(decision_world, current_world):
        return False
    return True


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_json_candidate(text: str) -> str:
    stripped = _strip_json_fence(text)
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered

    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start >= 0 and object_end > object_start:
        return stripped[object_start : object_end + 1]

    array_start = stripped.find("[")
    array_end = stripped.rfind("]")
    if array_start >= 0 and array_end > array_start:
        return stripped[array_start : array_end + 1]

    return stripped


def parse_guard_json(text: str | None) -> tuple[bool | None, str]:
    if not text:
        return None, "empty response"
    candidate = _extract_json_candidate(str(text))
    try:
        parsed = json.loads(candidate)
    except Exception as exc:
        return None, f"invalid JSON: {exc}"

    if isinstance(parsed, bool):
        return parsed, ""
    if not isinstance(parsed, dict):
        return None, "JSON is not a boolean or object"

    for key in (
        "execute",
        "can_execute",
        "allow",
        "allowed",
        "continue",
        "should_execute",
    ):
        if key in parsed:
            value = parsed[key]
            if isinstance(value, bool):
                return value, str(parsed.get("reason") or "")
            return None, f"{key} is not boolean"
    return None, "missing execute boolean"


def _build_user_prompt(
    *,
    cognition: str,
    tool_call_json: dict[str, Any],
    current_world: str,
) -> str:
    return "\n".join([
        "<cognition>",
        cognition.strip(),
        "</cognition>",
        "",
        "<tool_call_json>",
        json.dumps(tool_call_json, ensure_ascii=False, indent=2),
        "</tool_call_json>",
        "",
        current_world.strip(),
        "",
        '<final_instruction>只输出 JSON，例如 {"execute": true, "reason": "简短原因"}。</final_instruction>',
    ])


def decide_tool_execution(
    *,
    adapter: Any,
    cfg: dict | None,
    cognition: str,
    tool_call_json: dict[str, Any],
    current_world: str,
) -> ToolExecutionGuardDecision:
    normalized_cfg = normalize_tool_execution_guard_config(cfg)
    if not normalized_cfg["enabled"]:
        return ToolExecutionGuardDecision(
            execute=True,
            reason="tool_execution_guard disabled",
            checked=False,
            world_changed=True,
        )
    if adapter is None:
        return ToolExecutionGuardDecision(
            execute=True,
            reason="tool_execution_guard adapter not configured",
            checked=False,
            world_changed=True,
        )

    user_prompt = _build_user_prompt(
        cognition=cognition,
        tool_call_json=tool_call_json,
        current_world=current_world,
    )
    try:
        raw_response = adapter.call_simple_text(
            SYSTEM_PROMPT,
            user_prompt,
            normalized_cfg["generation"],
            log_tag="tool_execution_guard",
        )
    except Exception as exc:
        logger.warning("[tool_execution_guard] model call failed; allowing: %s", exc)
        return ToolExecutionGuardDecision(
            execute=True,
            reason=f"tool_execution_guard call failed: {exc}",
            checked=True,
            world_changed=True,
        )

    execute, reason = parse_guard_json(raw_response)
    if execute is None:
        logger.warning(
            "[tool_execution_guard] malformed JSON; allowing. reason=%s raw=%r",
            reason,
            raw_response,
        )
        return ToolExecutionGuardDecision(
            execute=True,
            reason=f"tool_execution_guard malformed JSON: {reason}",
            checked=True,
            world_changed=True,
            raw_response=str(raw_response or ""),
        )

    return ToolExecutionGuardDecision(
        execute=execute,
        reason=reason,
        checked=True,
        world_changed=True,
        raw_response=str(raw_response or ""),
    )


def evaluate_tool_execution_guard(
    *,
    decision_world: str | list | None,
    current_world_provider: Callable[[], str | list] | None,
    cognition: str,
    tool_call_json: dict[str, Any],
    adapter: Any = None,
    cfg: dict | None = None,
) -> ToolExecutionGuardDecision:
    if adapter is None or cfg is None:
        try:
            import app_state

            if adapter is None:
                adapter = getattr(app_state, "tool_execution_guard_adapter", None)
            if cfg is None:
                cfg = getattr(app_state, "tool_execution_guard_cfg", {})
        except Exception:
            adapter = adapter
            cfg = cfg or {}

    normalized_cfg = normalize_tool_execution_guard_config(cfg)
    if not normalized_cfg["enabled"]:
        return ToolExecutionGuardDecision(
            execute=True,
            reason="tool_execution_guard disabled",
        )

    if current_world_provider is None:
        return ToolExecutionGuardDecision(
            execute=True,
            reason="current_world_provider not configured",
        )

    try:
        current_content = current_world_provider()
    except Exception as exc:
        logger.warning("[tool_execution_guard] current world provider failed; allowing: %s", exc)
        return ToolExecutionGuardDecision(
            execute=True,
            reason=f"current world provider failed: {exc}",
        )

    if not world_semantically_changed(decision_world, current_content):
        return ToolExecutionGuardDecision(
            execute=True,
            reason="world unchanged since decision frame",
            checked=False,
            world_changed=False,
        )

    return decide_tool_execution(
        adapter=adapter,
        cfg=normalized_cfg,
        cognition=cognition,
        tool_call_json=tool_call_json,
        current_world=extract_world_text(current_content),
    )
