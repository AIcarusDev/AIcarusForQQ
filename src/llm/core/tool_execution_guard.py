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
你在一个 agent 系统中，你的任务的执行前检查。

你刚才基于 cognition 做出了一个动作决策，准备执行 tool_call_json 中的动作。
但是在你刚刚思考+做出行为决策期间，发生了有可能需要关注的新外部事件。
动作可能依然有效，与当前情况兼容，或不影响、不相干；也有可能因为情况的变化，使得动作与当前情况发生矛盾冲突，或会造成明显误导、不再合适。
现在需要以你自己的 cognition 为基础，看最新的 `<world>` 情况，判断刚才规划好的动作是否还适合继续执行。
注意：道德与具体语义、措词并非你的判断标准，你基于自己 cognition 的主观合理性才是。

你会依据输入中的四部分判断：

- cognition：你做出动作前的认知/思考。
- tool_call_json：即将执行的函数动作 JSON。
- world：在发生有可能需要关注的变化后的最新世界切片。

# 输出格式：

你会以json格式输出；先给出一个 "aware"，之后输出 "execute"。
具体 schema 以及其含义为：

```json
{
  "type": "object",
  "properties": {
    "aware": {
		"type": "string",
		"description": "你对现状的察觉，需要继承你 cognition 的语气，流畅的自然语言。允许轻度推理但不宜过长，例如从'我看到了新的情况...'开始，这部分察觉会同步到你之后的认知中。"
		},
    "execute": {
		"type": "boolean",
		"description": "最终的决定， ture 为可以继续执行，false 为需要重新决策。"
		},
	},
},
```

# output format

{
   "aware": "string",
   "execute": boolean
}
"""


@dataclass(frozen=True)
class ToolExecutionGuardDecision:
    execute: bool
    reason: str
    aware: str = ""
    checked: bool = False
    world_changed: bool = False
    raw_response: str = ""
    current_world: Any = None
    current_guard_snapshot: Any = None


@dataclass(frozen=True)
class GuardActivation:
    relevant: bool
    reason: str
    changes: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class QQGuardSnapshot:
    platform: str
    opened_focus_key: str
    session_key: str
    session_identity: tuple[str, ...]
    chat_log_mode: str
    external_entry_keys: tuple[tuple[str, str], ...] = ()
    external_entries: tuple[dict[str, Any], ...] = ()


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


def _extract_world_multimodal_content(content: str | list | None) -> str | list:
    if not isinstance(content, list):
        return extract_world_text(content)

    parts: list[dict[str, Any]] = []
    inside_world = False
    saw_world = False

    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            text = str(part.get("text") or "")
            cursor = 0
            while cursor < len(text):
                if not inside_world:
                    match = _WORLD_OPEN_RE.search(text, cursor)
                    if match is None:
                        break
                    inside_world = True
                    saw_world = True
                    cursor = match.start()

                close_index = text.find("</world>", cursor)
                if close_index < 0:
                    fragment = text[cursor:]
                    if fragment:
                        parts.append({"type": "text", "text": fragment})
                    break

                fragment = text[cursor : close_index + len("</world>")]
                if fragment:
                    parts.append({"type": "text", "text": fragment})
                inside_world = False
                cursor = close_index + len("</world>")
        elif inside_world and part_type == "image_url":
            parts.append(part)

    if not saw_world:
        return extract_world_text(content)
    if not parts:
        return ""
    if len(parts) == 1 and parts[0].get("type") == "text":
        return str(parts[0].get("text") or "")
    return parts


def _strip_image_parts(content: str | list) -> str | list:
    if not isinstance(content, list):
        return content
    text_parts = [
        part
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    if not text_parts:
        return ""
    return str(text_parts[0].get("text") or "") if len(text_parts) == 1 else text_parts


def _append_prompt_text(content: str | list, text: str) -> str | list:
    if isinstance(content, str):
        return content + text
    if content and isinstance(content[-1], dict) and content[-1].get("type") == "text":
        content[-1] = {**content[-1], "text": str(content[-1].get("text") or "") + text}
        return content
    return content + [{"type": "text", "text": text}]


def _build_multimodal_guard_prompt(
    *,
    cognition: str,
    tool_call_json: dict[str, Any],
    current_world: str | list | None,
    include_multimodal: bool,
) -> str | list:
    world_content = _extract_world_multimodal_content(current_world)
    if not include_multimodal:
        world_content = _strip_image_parts(world_content)

    prefix = "\n".join([
        "<cognition>",
        cognition.strip(),
        "</cognition>",
        "",
        "<tool_call_json>",
        json.dumps(tool_call_json, ensure_ascii=False, indent=2),
        "</tool_call_json>",
        "",
    ])
    suffix = "\n\n<final_instruction>只输出 JSON，例如 {\"aware\": \"我看到了新的情况...\", \"execute\": true}。</final_instruction>"

    if isinstance(world_content, list):
        return _append_prompt_text(
            [{"type": "text", "text": prefix}] + list(world_content),
            suffix,
        )
    return f"{prefix}{world_content}{suffix}"


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


_QQ_SURFACE_GUARD_KINDS = {"session_write", "message_mutation"}


def _effect_surface_kind(effect: Any) -> tuple[str, str]:
    if effect is None:
        return "", ""
    if isinstance(effect, dict):
        surface = effect.get("surface")
        kind = effect.get("kind")
    else:
        surface = getattr(effect, "surface", "")
        kind = getattr(effect, "kind", "")
    return (
        str(surface or "").strip().lower(),
        str(kind or "").strip().lower(),
    )


def _session_identity_from_session(session: Any) -> tuple[str, ...]:
    if session is None:
        return ()
    platform = str(
        getattr(getattr(session, "focus", None), "platform", "")
        or getattr(session, "get_platform_key", lambda: "")()
        or ""
    ).strip().lower()
    conv_type = str(getattr(session, "conv_type", "") or "").strip().lower()
    conv_id = str(getattr(session, "conv_id", "") or "").strip()
    if conv_type == "temp":
        source_group_id = str(getattr(session, "temp_source_group_id", "") or "").strip()
        return (platform, conv_type, conv_id, source_group_id)
    return (platform, conv_type, conv_id)


def _context_entry_signature(entry: dict[str, Any]) -> str:
    payload = {
        "role": entry.get("role"),
        "message_id": entry.get("message_id"),
        "sender_id": entry.get("sender_id"),
        "sender_name": entry.get("sender_name"),
        "content": entry.get("content"),
        "content_type": entry.get("content_type"),
        "delivery_state": entry.get("delivery_state"),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode(
            "utf-8",
            errors="replace",
        )
    ).hexdigest()


def _context_entry_key(index: int, entry: dict[str, Any]) -> tuple[str, str]:
    entry_id = str(entry.get("message_id") or entry.get("id") or "").strip()
    tag = "note" if entry.get("role") == "note" else "message"
    if entry_id:
        return (tag, entry_id)
    return (tag, f"@{index}:{_context_entry_signature(entry)}")


def _is_self_context_entry(entry: dict[str, Any], self_ids: set[str]) -> bool:
    role = str(entry.get("role") or "").strip().lower()
    if role in {"bot", "assistant"}:
        return True
    sender_id = str(
        entry.get("sender_id")
        or entry.get("operator_id")
        or ""
    ).strip().lower()
    if sender_id and sender_id in self_ids:
        return True
    if str(entry.get("from") or "").strip().lower() == "self":
        return True
    return False


def _summarize_context_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "tag": "note" if entry.get("role") == "note" else "message",
        "id": str(entry.get("message_id") or entry.get("id") or ""),
        "actor": str(
            entry.get("sender_id")
            or entry.get("sender_name")
            or entry.get("operator_id")
            or entry.get("operator_name")
            or ""
        ),
        "text": re.sub(r"\s+", " ", str(entry.get("content") or "")).strip()[:180],
    }


def _visible_external_context_entries(session: Any) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[dict[str, Any], ...],
]:
    if session is None:
        return (), ()
    self_ids = {"self", "bot"}
    qq_self_id = str(getattr(session, "_qq_id", "") or "").strip().lower()
    if qq_self_id:
        self_ids.add(qq_self_id)
    keys: list[tuple[str, str]] = []
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(list(getattr(session, "context_messages", []) or [])):
        if not isinstance(entry, dict):
            continue
        if _is_self_context_entry(entry, self_ids):
            continue
        key = _context_entry_key(index, entry)
        keys.append(key)
        entries.append(_summarize_context_entry(entry))
    return tuple(keys), tuple(entries)


def build_qq_guard_snapshot(session: Any, *, current_focus: Any = None) -> QQGuardSnapshot:
    try:
        from platforms.focus import current_focus_key
    except Exception:
        current_focus_key = lambda value: ""  # type: ignore[assignment]

    if current_focus is None:
        try:
            import app_state

            current_focus = getattr(app_state, "current_focus", None)
        except Exception:
            current_focus = None

    session_key = str(getattr(session, "key", "") or "").strip()
    opened_focus_key = current_focus_key(current_focus)
    platform = str(
        getattr(getattr(session, "focus", None), "platform", "")
        or getattr(session, "get_platform_key", lambda: "")()
        or "qq"
    ).strip().lower()
    chat_log_mode = (
        "history"
        if bool(getattr(session, "is_browsing_history", lambda: False)())
        else "current"
    )
    external_keys: tuple[tuple[str, str], ...] = ()
    external_entries: tuple[dict[str, Any], ...] = ()
    if chat_log_mode == "current":
        external_keys, external_entries = _visible_external_context_entries(session)
    return QQGuardSnapshot(
        platform=platform,
        opened_focus_key=opened_focus_key,
        session_key=session_key,
        session_identity=_session_identity_from_session(session),
        chat_log_mode=chat_log_mode,
        external_entry_keys=external_keys,
        external_entries=external_entries,
    )


def _qq_snapshot_guard_activation(
    *,
    decision_snapshot: Any,
    current_snapshot: Any,
    tool_effect: Any,
) -> GuardActivation | None:
    surface, kind = _effect_surface_kind(tool_effect)
    if surface != "qq" or kind not in _QQ_SURFACE_GUARD_KINDS:
        return None
    if not isinstance(decision_snapshot, QQGuardSnapshot) or not isinstance(
        current_snapshot,
        QQGuardSnapshot,
    ):
        return None
    if decision_snapshot.chat_log_mode != "current":
        return GuardActivation(
            relevant=False,
            reason=(
                "qq surface unchanged for action: decision frame was browsing "
                f"{decision_snapshot.chat_log_mode or 'unknown'} chat logs"
            ),
        )
    if current_snapshot.chat_log_mode != "current":
        return GuardActivation(
            relevant=False,
            reason=(
                "qq surface unchanged for action: current frame is not current "
                f"chat logs ({current_snapshot.chat_log_mode or 'unknown'})"
            ),
        )
    if decision_snapshot.opened_focus_key != current_snapshot.opened_focus_key:
        return GuardActivation(
            relevant=True,
            reason="qq opened session changed before action",
            changes=({
                "type": "opened_session_changed",
                "from": decision_snapshot.opened_focus_key,
                "to": current_snapshot.opened_focus_key,
            },),
        )
    if decision_snapshot.session_identity != current_snapshot.session_identity:
        return GuardActivation(
            relevant=True,
            reason="qq target session changed before action",
            changes=({
                "type": "session_changed",
                "from": list(decision_snapshot.session_identity),
                "to": list(current_snapshot.session_identity),
            },),
        )
    decision_keys = set(decision_snapshot.external_entry_keys)
    new_entries = [
        entry
        for key, entry in zip(
            current_snapshot.external_entry_keys,
            current_snapshot.external_entries,
        )
        if key not in decision_keys
    ]
    if new_entries:
        return GuardActivation(
            relevant=True,
            reason="qq current session has new visible external chat entries",
            changes=tuple(new_entries),
        )
    return GuardActivation(
        relevant=False,
        reason="qq surface unchanged for action: no new visible external chat entries",
    )


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
                return value, str(parsed.get("aware") or parsed.get("reason") or "")
            return None, f"{key} is not boolean"
    return None, "missing execute boolean"


def decide_tool_execution(
    *,
    adapter: Any,
    cfg: dict | None,
    cognition: str,
    tool_call_json: dict[str, Any],
    current_world: str | list | None,
    current_guard_snapshot: Any = None,
) -> ToolExecutionGuardDecision:
    normalized_cfg = normalize_tool_execution_guard_config(cfg)
    if not normalized_cfg["enabled"]:
        return ToolExecutionGuardDecision(
            execute=True,
            reason="tool_execution_guard disabled",
            checked=False,
            world_changed=True,
            current_guard_snapshot=current_guard_snapshot,
        )
    if adapter is None:
        return ToolExecutionGuardDecision(
            execute=True,
            reason="tool_execution_guard adapter not configured",
            checked=False,
            world_changed=True,
            current_guard_snapshot=current_guard_snapshot,
        )

    user_prompt = _build_multimodal_guard_prompt(
        cognition=cognition,
        tool_call_json=tool_call_json,
        current_world=current_world,
        include_multimodal=bool(normalized_cfg.get("vision", False)),
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
            current_world=current_world,
            current_guard_snapshot=current_guard_snapshot,
        )

    execute, aware = parse_guard_json(raw_response)
    if execute is None:
        logger.warning(
            "[tool_execution_guard] malformed JSON; allowing. reason=%s raw=%r",
            aware,
            raw_response,
        )
        return ToolExecutionGuardDecision(
            execute=True,
            reason=f"tool_execution_guard malformed JSON: {aware}",
            checked=True,
            world_changed=True,
            raw_response=str(raw_response or ""),
            current_world=current_world,
            current_guard_snapshot=current_guard_snapshot,
        )

    return ToolExecutionGuardDecision(
        execute=execute,
        reason=aware,
        aware=aware,
        checked=True,
        world_changed=True,
        raw_response=str(raw_response or ""),
        current_world=current_world,
        current_guard_snapshot=current_guard_snapshot,
    )


def evaluate_tool_execution_guard(
    *,
    decision_world: str | list | None,
    current_world_provider: Callable[[], str | list] | None,
    cognition: str,
    tool_call_json: dict[str, Any],
    tool_effect: Any = None,
    decision_guard_snapshot: Any = None,
    current_guard_snapshot_provider: Callable[[], Any] | None = None,
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

    current_guard_snapshot = None
    if current_guard_snapshot_provider is not None:
        try:
            current_guard_snapshot = current_guard_snapshot_provider()
        except Exception as exc:
            logger.warning(
                "[tool_execution_guard] current guard snapshot provider failed; allowing: %s",
                exc,
            )
            return ToolExecutionGuardDecision(
                execute=True,
                reason=f"current guard snapshot provider failed: {exc}",
            )

    activation = _qq_snapshot_guard_activation(
        decision_snapshot=decision_guard_snapshot,
        current_snapshot=current_guard_snapshot,
        tool_effect=tool_effect,
    )
    if activation is not None:
        if not activation.relevant:
            return ToolExecutionGuardDecision(
                execute=True,
                reason=activation.reason,
                checked=False,
                world_changed=False,
                current_world=current_content,
                current_guard_snapshot=current_guard_snapshot,
            )
        return decide_tool_execution(
            adapter=adapter,
            cfg=normalized_cfg,
            cognition=cognition,
            tool_call_json=tool_call_json,
            current_world=current_content,
            current_guard_snapshot=current_guard_snapshot,
        )

    if not world_semantically_changed(decision_world, current_content):
        return ToolExecutionGuardDecision(
            execute=True,
            reason="world unchanged since decision frame",
            checked=False,
            world_changed=False,
            current_world=current_content,
            current_guard_snapshot=current_guard_snapshot,
        )

    return decide_tool_execution(
        adapter=adapter,
        cfg=normalized_cfg,
        cognition=cognition,
        tool_call_json=tool_call_json,
        current_world=current_content,
        current_guard_snapshot=current_guard_snapshot,
    )
