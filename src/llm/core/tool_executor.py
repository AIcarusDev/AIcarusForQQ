"""Local XML tool execution for one LLM round."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from consciousness.flow import ToolCall, ToolResponse
from hooks import emit_hook, hook_scope

from .decision_filter import normalize_send_messages
from .round_context import reset_current_inner_state, set_current_inner_state
from .tool_execution_guard import evaluate_tool_execution_guard
from .tool_calling.common import strip_legacy_motivation_fields
from .tool_calling import (
    attach_tool_result_warnings,
    build_tool_argument_error,
    process_tool_arguments,
)
from .tool_calling.xml_protocol import XML_TOOL_CALL_ERROR_NAME

logger = logging.getLogger("AICQ.llm.tool_executor")


class RuntimeResetAborted(Exception):
    """The current round was invalidated by a runtime reset."""


@dataclass
class ToolExecutionOutcome:
    tool_calls_log: list[dict] = field(default_factory=list)
    round_calls: list[ToolCall] = field(default_factory=list)
    round_responses: list[ToolResponse] = field(default_factory=list)


def _run_parallel_slots(
    parallel_slots: list[dict],
    executor,
    provider_name: str,
    stale_checker=None,
) -> None:
    """并行执行工具，并允许主线程在 Ctrl+C 时立刻停止等待。"""
    if not parallel_slots:
        return

    threads = [
        threading.Thread(
            target=executor,
            args=(slot,),
            name=f"tool-{provider_name}-{slot['fn_name']}",
            daemon=True,
        )
        for slot in parallel_slots
    ]

    for thread in threads:
        thread.start()

    try:
        while True:
            if stale_checker is not None and stale_checker():
                alive_tools = [
                    slot["fn_name"]
                    for slot, thread in zip(parallel_slots, threads)
                    if thread.is_alive()
                ]
                if alive_tools:
                    logger.warning(
                        "[%s] 运行时紧急恢复，停止等待中的工具: %s",
                        provider_name,
                        ", ".join(alive_tools),
                    )
                for slot, thread in zip(parallel_slots, threads):
                    if thread.is_alive() and slot.get("result") is None:
                        slot["result"] = {
                            "ok": False,
                            "error": "运行时已被紧急恢复，本工具结果已丢弃。",
                            "interrupted": True,
                            "aborted_by_runtime_reset": True,
                        }
                return
            any_alive = False
            for thread in threads:
                thread.join(timeout=0.1)
                if thread.is_alive():
                    any_alive = True
            if not any_alive:
                return
    except KeyboardInterrupt:
        alive_tools = [
            slot["fn_name"]
            for slot, thread in zip(parallel_slots, threads)
            if thread.is_alive()
        ]
        logger.warning(
            "[%s] 工具执行期间收到 Ctrl+C，停止等待中的工具: %s",
            provider_name,
            ", ".join(alive_tools) if alive_tools else "<none>",
        )
        raise


def _send_message_schema_kind(tool_collection: Any) -> str:
    spec = getattr(tool_collection, "active_specs", {}).get("send_message")
    declaration = getattr(spec, "declaration", None)
    if not isinstance(declaration, dict):
        return ""
    parameters = declaration.get("parameters")
    if not isinstance(parameters, dict):
        return ""
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return ""
    if "messages" in properties:
        return "array"
    if "segments" in properties:
        return "single"
    return ""


def _clone_send_message_slot(
    slot: dict,
    *,
    args: dict,
    call_id: str | None = None,
) -> dict:
    new_slot = dict(slot)
    new_slot["args"] = args
    if call_id is not None:
        new_slot["tc"] = SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(
                name=slot["fn_name"],
                arguments=json.dumps(args, ensure_ascii=False),
            ),
        )
    return new_slot


def _split_send_message_array_slots(slots: list[dict]) -> list[dict]:
    """Split array-shaped send_message calls into single-message executions."""
    expanded: list[dict] = []
    for slot in slots:
        if slot.get("fn_name") != "send_message" or slot.get("result") is not None:
            expanded.append(slot)
            continue

        args = slot.get("args")
        if not isinstance(args, dict):
            expanded.append(slot)
            continue
        messages = args.get("messages")
        if not isinstance(messages, list) or not messages:
            expanded.append(slot)
            continue

        single_args_list: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                single_args_list = []
                break
            single_args: dict[str, Any] = {"segments": message.get("segments", [])}
            if message.get("quote"):
                single_args["quote"] = message.get("quote")
            single_args_list.append(single_args)
        if not single_args_list:
            expanded.append(slot)
            continue

        original_id = str(getattr(slot["tc"], "id", "") or "call")
        if len(single_args_list) > 1:
            logger.info(
                "[send_message] array 形态已拆分为 %d 次独立工具执行 call_id=%s",
                len(single_args_list),
                original_id,
            )
        for index, single_args in enumerate(single_args_list, start=1):
            expanded.append(
                _clone_send_message_slot(
                    slot,
                    args=single_args,
                    call_id=None if index == 1 else f"{original_id}_split_{index}",
                )
            )

    return expanded


def _expanded_single_send_message_slots(slots: list[dict]) -> list[dict]:
    """Split a send_message containing multiple text segments into separate calls."""
    expanded: list[dict] = []
    for slot in slots:
        if slot.get("fn_name") != "send_message" or slot.get("result") is not None:
            expanded.append(slot)
            continue

        args = slot.get("args")
        if not isinstance(args, dict):
            expanded.append(slot)
            continue
        segments = args.get("segments")
        if not isinstance(segments, list):
            expanded.append(slot)
            continue
        if not all(isinstance(seg, dict) for seg in segments):
            expanded.append(slot)
            continue
        if sum(1 for seg in segments if seg.get("command") == "text") <= 1:
            expanded.append(slot)
            continue

        normalized_messages = normalize_send_messages([args])
        if len(normalized_messages) <= 1:
            expanded.append(slot)
            continue

        original_id = str(getattr(slot["tc"], "id", "") or "call")
        logger.warning(
            "[send_message] 多个 text segment 已规范化为 %d 次独立调用 call_id=%s",
            len(normalized_messages),
            original_id,
        )
        for index, normalized_args in enumerate(normalized_messages, start=1):
            expanded.append(
                _clone_send_message_slot(
                    slot,
                    args=normalized_args,
                    call_id=None if index == 1 else f"{original_id}_split_{index}",
                )
            )

    return expanded


def _tool_description(spec: Any) -> str:
    declaration = getattr(spec, "declaration", {}) or {}
    description = declaration.get("description", "")
    return str(description).strip() if description is not None else ""


def _namespace_name_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for item in value:
        name = str(item or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _namespace_tools_for_namespaces(
    namespaces: list[str],
    tool_collection,
) -> list[dict[str, Any]]:
    registry = getattr(tool_collection, "namespace_registry", None)
    all_specs = getattr(tool_collection, "all_specs", {}) or {}
    entries: list[dict[str, Any]] = []
    for namespace in namespaces:
        spec = registry.get(namespace) if registry is not None else None
        if spec is None:
            continue
        tools = [tool for tool in getattr(spec, "tools", ()) or () if tool in all_specs]
        if tools:
            entries.append({"namespace": namespace, "tools": tools})
    return entries


def _namespace_attached_tools_for_namespaces(
    namespaces: list[str],
    active_namespaces: list[str],
    tool_collection,
) -> list[dict[str, Any]]:
    registry = getattr(tool_collection, "namespace_registry", None)
    all_specs = getattr(tool_collection, "all_specs", {}) or {}
    active = set(active_namespaces)
    attached: list[dict[str, Any]] = []
    for namespace in namespaces:
        spec = registry.get(namespace) if registry is not None else None
        if spec is None:
            continue
        for attach in getattr(spec, "attach", ()) or ():
            if attach.namespace in active:
                continue
            if attach.tool not in all_specs:
                continue
            attached.append({
                "host_namespace": namespace,
                "source_namespace": attach.namespace,
                "tools": [attach.tool],
            })
    return attached


def _loaded_skills_for_namespaces(namespaces: list[str], registry) -> list[dict[str, str]]:
    try:
        from skills import load_skill_body
    except Exception:
        load_skill_body = None

    loaded: list[dict[str, str]] = []
    seen: set[str] = set()
    for namespace in namespaces:
        spec = registry.get(namespace) if registry is not None else None
        skill = str(getattr(spec, "skill", "") or "").strip()
        if not skill or skill in seen:
            continue
        if load_skill_body is not None and not load_skill_body(skill).strip():
            continue
        seen.add(skill)
        loaded.append({"namespace": namespace, "skill": skill})
    return loaded


def _set_non_empty(result: dict[str, Any], key: str, value: Any) -> None:
    if value not in (None, "", [], {}):
        result[key] = value


def _namespace_manage_result(args: dict, tool_collection) -> dict:
    registry = getattr(tool_collection, "namespace_registry", None)
    state = getattr(tool_collection, "namespace_state", None)
    if registry is None or state is None:
        return {"ok": False, "error": "namespace registry is unavailable"}

    result: dict[str, Any] = {"ok": True}
    opened_or_available: list[str] = []
    closed: list[str] = []
    already_closed: list[str] = []
    protected: list[str] = []
    not_found: list[str] = []

    for name in _namespace_name_list(args.get("open")):
        status = state.open(name, registry, getattr(tool_collection, "round_index", 0))
        if status in {"opened", "already_open"}:
            opened_or_available.append(name)
        else:
            not_found.append(name)

    for name in _namespace_name_list(args.get("close")):
        status = state.close(name, registry)
        if status == "closed":
            closed.append(name)
        elif status == "protected":
            protected.append(name)
        elif status == "already_closed":
            already_closed.append(name)
        else:
            not_found.append(name)

    previews: list[dict] = []
    preview_warnings: list[dict] = []
    for name in _namespace_name_list(args.get("preview")):
        preview = tool_collection.preview_namespace(name)
        if preview is None:
            preview_warnings.append({"name": name, "warning": "未找到 namespace。"})
        else:
            previews.append(preview)
    _set_non_empty(result, "closed", closed)
    _set_non_empty(result, "already_closed", already_closed)
    _set_non_empty(result, "protected", protected)
    _set_non_empty(result, "not_found", not_found)
    _set_non_empty(result, "preview", previews)
    _set_non_empty(result, "warnings", preview_warnings)

    search = args.get("search")
    if isinstance(search, str):
        _set_non_empty(result, "search", tool_collection.search_inactive_namespaces(search))

    active_namespaces = tool_collection.active_namespace_names()
    _set_non_empty(result, "tools", _namespace_tools_for_namespaces(opened_or_available, tool_collection))
    _set_non_empty(
        result,
        "attached_tools",
        _namespace_attached_tools_for_namespaces(opened_or_available, active_namespaces, tool_collection),
    )
    _set_non_empty(result, "skills", _loaded_skills_for_namespaces(opened_or_available, registry))
    return result


def _inactive_namespace_result(fn_name: str, namespace: str, tool_collection, *, reason: str) -> dict:
    registry = getattr(tool_collection, "namespace_registry", None)
    state = getattr(tool_collection, "namespace_state", None)
    if registry is not None and state is not None and reason == "inactive":
        state.open(namespace, registry, getattr(tool_collection, "round_index", 0))
    if reason == "opened_same_round":
        message = (
            f"The namespace `{namespace}` was opened in this same action, but its tool schema "
            "is only available from the next round. Use this tool on the next round."
        )
    elif reason == "closed_same_round":
        message = (
            f"The namespace `{namespace}` was closed earlier in this same action; "
            f"`{fn_name}` will not execute because the call order is inconsistent."
        )
    else:
        message = (
            f"The tool `{fn_name}` belongs to inactive namespace `{namespace}` and cannot be "
            "executed directly in this round. The namespace has been opened for the next round."
        )
    return {
        "ok": False,
        "error": message,
        "namespace": namespace,
    }


class ToolExecutor:
    """Parse processed XML tool calls into local handler executions."""

    _TERMINAL_CONTROL_TOOLS = frozenset({
        "restart",
    })

    def __init__(
        self,
        *,
        provider_name: str,
        tool_collection,
        flow=None,
        runtime_stale_checker=None,
        decision_world=None,
        current_world_provider=None,
        tool_execution_guard_adapter=None,
        tool_execution_guard_cfg=None,
    ) -> None:
        self.provider_name = provider_name
        self.tool_collection = tool_collection
        self.flow = flow
        self.runtime_stale_checker = runtime_stale_checker
        self.decision_world = decision_world
        self.current_world_provider = current_world_provider
        self.tool_execution_guard_adapter = tool_execution_guard_adapter
        self.tool_execution_guard_cfg = tool_execution_guard_cfg

    def _runtime_is_stale(self) -> bool:
        if self.runtime_stale_checker is None:
            return False
        try:
            return bool(self.runtime_stale_checker())
        except Exception:
            logger.debug("[%s] runtime_stale_checker 失败", self.provider_name, exc_info=True)
            return False

    def _abort_if_stale(self) -> None:
        if self._runtime_is_stale():
            raise RuntimeResetAborted()

    def _tool_hook_context(self, slot: dict, **extra: Any) -> dict[str, Any]:
        tool_call = slot.get("tc")
        context = {
            "provider": self.provider_name,
            "call_id": str(getattr(tool_call, "id", "") or ""),
            "module": str(slot.get("module_name") or ""),
        }
        context.update(extra)
        return context

    def _emit_tool_hook(
        self,
        point: str,
        slot: dict,
        *,
        result: Any = None,
        error: BaseException | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        emit_hook(
            namespace="tool",
            point=point,
            target=str(slot.get("fn_name") or ""),
            args=slot.get("args") if isinstance(slot.get("args"), dict) else {},
            result=result,
            error=error,
            context=context or self._tool_hook_context(slot),
        )

    def _build_slots(self, tool_calls: list) -> list[dict]:
        slots: list[dict] = []
        registry = getattr(self.tool_collection, "namespace_registry", None)
        state = getattr(self.tool_collection, "namespace_state", None)
        round_index = int(getattr(self.tool_collection, "round_index", 0) or 0)
        local_active_namespaces = set(self.tool_collection.active_namespace_names())
        opened_this_round: set[str] = set()
        closed_this_round: set[str] = set()
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            protocol_error = getattr(tool_call, "protocol_error", None)
            spec = self.tool_collection.get_active(fn_name)
            latent_spec = self.tool_collection.get_latent(fn_name) if spec is None else None
            origin_namespace = self.tool_collection.namespace_for_tool(fn_name)
            namespace = str(getattr(spec, "attached_to", "") or origin_namespace)
            handler = spec.handler if spec is not None else None
            processing = None
            args: dict = {}
            if protocol_error:
                try:
                    parsed_error_args = json.loads(tool_call.function.arguments or "{}")
                    args = parsed_error_args if isinstance(parsed_error_args, dict) else {}
                except Exception:
                    args = {"error": str(protocol_error)}
            elif spec is not None and handler is not None:
                processing = process_tool_arguments(
                    tool_call.function.arguments,
                    fn_name,
                    self.provider_name,
                    spec.declaration,
                    spec.schema_repairer,
                    spec.semantic_sanitizer,
                )
                args = processing.args

            args, _stripped_legacy_motivation = strip_legacy_motivation_fields(args)

            slot: dict = {
                "tc": tool_call,
                "fn_name": fn_name,
                "args": args,
                "fn": handler,
                "module_name": getattr(spec, "module_name", "") if spec is not None else "",
                "namespace": namespace,
                "externally_perceptible": (
                    bool(getattr(spec, "externally_perceptible", False))
                    if spec is not None
                    else False
                ),
                "result": None,
                "protocol_error": protocol_error,
            }
            if protocol_error:
                slot["result"] = {
                    "ok": False,
                    "error": f"工具调用格式错误: {protocol_error}",
                    "tool_not_executed": True,
                    "retryable": True,
                }
            elif fn_name == "namespace_manage" and processing is not None and processing.ok:
                slot["result"] = _namespace_manage_result(args, self.tool_collection)
                slot["_hook_executed"] = True
                for name in _namespace_name_list(args.get("open")):
                    if name not in local_active_namespaces:
                        opened_this_round.add(name)
                for name in slot["result"].get("closed", []):
                    local_active_namespaces.discard(name)
                    closed_this_round.add(name)
            elif handler is None:
                if namespace:
                    reason = (
                        "closed_same_round"
                        if namespace in closed_this_round
                        else "opened_same_round"
                        if namespace in opened_this_round
                        else "inactive"
                    )
                    slot["result"] = _inactive_namespace_result(
                        fn_name,
                        namespace,
                        self.tool_collection,
                        reason=reason,
                    )
                    if reason == "inactive":
                        opened_this_round.add(namespace)
                else:
                    slot["result"] = {"error": f"未知工具: {fn_name}"}
            elif namespace and namespace not in local_active_namespaces:
                reason = (
                    "closed_same_round"
                    if namespace in closed_this_round
                    else "opened_same_round"
                    if namespace in opened_this_round
                    else "inactive"
                )
                slot["result"] = _inactive_namespace_result(
                    fn_name,
                    namespace,
                    self.tool_collection,
                    reason=reason,
                )
                if reason == "inactive":
                    opened_this_round.add(namespace)
            elif processing is not None and not processing.ok:
                if namespace and registry is not None and state is not None:
                    state.mark_active(namespace, registry, round_index)
                slot["result"] = build_tool_argument_error(processing)
            elif namespace and registry is not None and state is not None:
                state.mark_active(namespace, registry, round_index)
            slots.append(slot)

        send_message_schema = _send_message_schema_kind(self.tool_collection)
        if send_message_schema == "array":
            slots = _split_send_message_array_slots(slots)
        elif send_message_schema == "single":
            slots = _expanded_single_send_message_slots(slots)
        return slots

    def _exec_one(self, slot: dict) -> None:
        fn_name = slot["fn_name"]
        logger.info("[%s] 执行工具开始: %s", self.provider_name, fn_name)
        started_at = time.perf_counter()
        error: BaseException | None = None
        slot["_hook_executed"] = True
        hook_context = self._tool_hook_context(slot)
        self._emit_tool_hook("before_call", slot, context=hook_context)
        try:
            with hook_scope(namespace="tool", target=fn_name, context=hook_context):
                slot["result"] = slot["fn"](**slot["args"])
            if isinstance(slot["result"], dict) and slot["result"].get("error"):
                logger.info(
                    "[%s] 执行工具完毕（失败）: %s — %s",
                    self.provider_name, fn_name, slot["result"]["error"],
                )
            else:
                logger.info("[%s] 执行工具完毕（成功）: %s", self.provider_name, fn_name)
            self._emit_tool_hook("after_call", slot, result=slot["result"], context=hook_context)
        except Exception as exc:
            error = exc
            logger.warning("[%s] 执行工具异常: %s — %s", self.provider_name, fn_name, exc)
            slot["result"] = {"error": str(exc)}
            self._emit_tool_hook(
                "on_error",
                slot,
                result=slot["result"],
                error=exc,
                context=hook_context,
            )
        finally:
            final_context = dict(hook_context)
            final_context["elapsed_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
            self._emit_tool_hook(
                "finally_call",
                slot,
                result=slot.get("result"),
                error=error,
                context=final_context,
            )

    def _tool_call_json(self, slot: dict) -> dict[str, Any]:
        return {
            "name": str(slot.get("fn_name") or ""),
            "arguments": slot.get("args") if isinstance(slot.get("args"), dict) else {},
        }

    def _build_world_changed_guard_result(self, reason: str) -> dict[str, Any]:
        result = {
            "ok": False,
            "error": "工具未执行：<world> 已变化，本次行动或许需要重新评估。",
            "tool_not_executed": True,
            "blocked_by": "tool_execution_guard",
            "block_reason": "world_changed_requires_redecision"
        }
        if reason:
            result["reason"] = reason
        return result

    def _build_prior_guard_blocked_result(self, blocked_slot: dict) -> dict[str, Any]:
        result = {
            "ok": False,
            "error": "工具未执行：较早的外界可感知工具已被阻止。",
            "tool_not_executed": True,
            "blocked_by": "tool_execution_guard",
            "block_reason": "prior_external_tool_requires_redecision"
        }
        prior_tool = str(blocked_slot.get("fn_name") or "")
        if prior_tool:
            result["prior_blocked_tool"] = prior_tool
        return result

    def _guard_external_effect_slot(self, slot: dict, inner_state: dict) -> bool:
        decision = evaluate_tool_execution_guard(
            decision_world=self.decision_world,
            current_world_provider=self.current_world_provider,
            cognition=str((inner_state or {}).get("cognition") or (inner_state or {}).get("think") or ""),
            tool_call_json=self._tool_call_json(slot),
            adapter=self.tool_execution_guard_adapter,
            cfg=self.tool_execution_guard_cfg,
        )
        if not decision.world_changed:
            return True
        event_result = {
            "ok": bool(decision.execute),
            "world_changed_since_decision": True,
            "checked": bool(decision.checked),
            "reason": decision.reason,
        }
        if decision.execute:
            self._emit_tool_hook("guard_allowed", slot, result=event_result)
            logger.info(
                "[%s] 工具执行前守门放行: %s checked=%s reason=%s",
                self.provider_name,
                slot["fn_name"],
                decision.checked,
                decision.reason,
            )
            return True

        slot["result"] = self._build_world_changed_guard_result(decision.reason)
        self._emit_tool_hook("guard_blocked", slot, result=slot["result"])
        logger.warning(
            "[%s] 工具执行前守门阻止: %s reason=%s",
            self.provider_name,
            slot["fn_name"],
            decision.reason,
        )
        return False

    def _block_later_external_effect_slots(
        self,
        external_effect_slots: list[dict],
        blocked_index: int,
        blocked_slot: dict,
    ) -> None:
        for later_slot in external_effect_slots[blocked_index + 1:]:
            if later_slot.get("result") is not None:
                continue
            later_slot["result"] = self._build_prior_guard_blocked_result(blocked_slot)
            self._emit_tool_hook("guard_blocked", later_slot, result=later_slot["result"])
            logger.warning(
                "[%s] 后续外界可感知工具跳过: %s prior_blocked=%s",
                self.provider_name,
                later_slot["fn_name"],
                blocked_slot["fn_name"],
            )

    def execute(self, tool_calls: list, *, inner_state: dict) -> ToolExecutionOutcome:
        self._abort_if_stale()
        slots = self._build_slots(tool_calls)
        pending_slots = [slot for slot in slots if slot["result"] is None]
        has_shift = any(slot["fn_name"] == "shift" for slot in pending_slots)
        external_effect_slots = [
            slot for slot in pending_slots
            if slot.get("externally_perceptible")
        ]
        if has_shift and external_effect_slots:
            for slot in external_effect_slots:
                slot["result"] = {
                    "ok": False,
                    "error": (
                        "本轮同时包含 shift 和外界可感知工具；系统暂没有兼容此种情况。"
                    ),
                    "tool_not_executed": True,
                    "incompatible_with": "shift",
                }
            for slot in pending_slots:
                if slot["fn_name"] == "shift" or slot.get("externally_perceptible"):
                    continue
                slot["result"] = {
                    "ok": False,
                    "error": "本轮同时包含 shift 和外界可感知工具；已只执行 shift，本工具跳过。",
                    "tool_not_executed": True,
                    "skipped_due_to": "shift_externally_perceptible_tool_conflict",
                    "interrupted": True,
                }
            pending_slots = [slot for slot in slots if slot["result"] is None]
            external_effect_slots = []

        non_external_effect_slots = [
            slot for slot in pending_slots
            if not slot.get("externally_perceptible")
        ]
        terminal_slots = [
            slot for slot in non_external_effect_slots
            if slot["fn_name"] in self._TERMINAL_CONTROL_TOOLS
        ]
        parallel_slots = [
            slot for slot in non_external_effect_slots
            if slot["fn_name"] not in self._TERMINAL_CONTROL_TOOLS
        ]

        inner_state_token = set_current_inner_state(inner_state)
        try:
            for index, slot in enumerate(external_effect_slots):
                if slot.get("result") is not None:
                    continue
                self._abort_if_stale()
                if not self._guard_external_effect_slot(slot, inner_state):
                    self._block_later_external_effect_slots(
                        external_effect_slots,
                        index,
                        slot,
                    )
                    self._abort_if_stale()
                    break
                self._exec_one(slot)
                self._abort_if_stale()
            restart_scheduled = False
            for slot in terminal_slots:
                self._abort_if_stale()
                self._exec_one(slot)
                self._abort_if_stale()
                slot_result = slot.get("result")
                if (
                    isinstance(slot_result, dict)
                    and slot_result.get("ok") is True
                    and slot_result.get("restart_scheduled") is True
                ):
                    restart_scheduled = True
            if restart_scheduled:
                for slot in parallel_slots:
                    slot["result"] = {
                        "ok": False,
                        "error": "自身重启已安排，本轮剩余工具跳过。",
                        "interrupted": True,
                    }
            elif parallel_slots:
                _run_parallel_slots(
                    parallel_slots,
                    self._exec_one,
                    self.provider_name,
                    stale_checker=self._runtime_is_stale,
                )
            self._abort_if_stale()
        finally:
            reset_current_inner_state(inner_state_token)

        return self._collect(slots)

    def _collect(self, slots: list[dict]) -> ToolExecutionOutcome:
        outcome = ToolExecutionOutcome()
        outcome.round_calls = [
            ToolCall(name=slot["fn_name"], args=slot["args"], call_id=slot["tc"].id)
            for slot in slots
            if not slot.get("protocol_error")
        ]

        for slot in slots:
            fn_name = slot["fn_name"]
            tool_call = slot["tc"]
            args = slot["args"]
            result_data = slot["result"]
            if not slot.get("_hook_executed"):
                self._emit_tool_hook("skipped", slot, result=result_data)

            if isinstance(result_data, dict):
                self.tool_collection.apply_lifecycle_after_tool(
                    fn_name,
                    args if isinstance(args, dict) else {},
                    result_data,
                )
                result_data.pop("_inject_tools", None)
                attach_tool_result_warnings(
                    tool_name=fn_name,
                    args=args if isinstance(args, dict) else {},
                    result=result_data,
                    flow=self.flow,
                )

            outcome.tool_calls_log.append({
                "function": fn_name,
                "arguments": args,
                "result": result_data,
            })

            raw_multimodal_parts: list = []
            if isinstance(result_data, dict) and "_multimodal_parts" in result_data:
                raw_multimodal_parts = result_data.pop("_multimodal_parts")

            outcome.round_responses.append(
                ToolResponse(
                    name=XML_TOOL_CALL_ERROR_NAME if slot.get("protocol_error") else fn_name,
                    response=result_data,
                    call_id=tool_call.id,
                    multimodal_parts=raw_multimodal_parts,
                )
            )

        return outcome
