"""Pre-execution guard for externally perceptible tools.

The guard is intentionally not a tool/function call. It asks a configured
submodel for direct JSON and treats malformed or failed responses as allow.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
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
现在需要是判断：在新的 <world> 下，基于你的 cognition，被检查的函数工具调用是否仍然可以，且适合执行。

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


def extract_world_text(content: str | list | None) -> str:
    text = _content_to_text(content)
    start = text.find("<world")
    if start < 0:
        return text.strip()
    end = text.rfind("</world>")
    if end < 0:
        return text[start:].strip()
    return text[start : end + len("</world>")].strip()


_CURRENT_TIME_RE = re.compile(
    r"<current_time\b[^>]*>.*?</current_time>",
    flags=re.DOTALL | re.IGNORECASE,
)


def _normalize_world_for_signature(world_text: str) -> str:
    without_time = _CURRENT_TIME_RE.sub("<current_time/>", world_text)
    return re.sub(r"\s+", " ", without_time).strip()


def world_semantic_signature(content: str | list | None) -> str:
    world = extract_world_text(content)
    normalized = _normalize_world_for_signature(world)
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def world_semantically_changed(
    decision_world: str | list | None,
    current_world: str | list | None,
) -> bool:
    return world_semantic_signature(decision_world) != world_semantic_signature(current_world)


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
