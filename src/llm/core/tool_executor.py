"""Local AIC Action execution for one LLM round."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
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
from .tool_calling.aic_action import AIC_ACTION_ERROR_NAME

logger = logging.getLogger("AICQ.llm.tool_executor")


class RuntimeResetAborted(Exception):
    """The current round was invalidated by a runtime reset."""


@dataclass
class ToolExecutionOutcome:
    tool_calls_log: list[dict] = field(default_factory=list)
    round_calls: list[ToolCall] = field(default_factory=list)
    round_responses: list[ToolResponse] = field(default_factory=list)


@dataclass(frozen=True)
class ToolPathResolution:
    namespace: str = ""
    name: str = ""
    error: str = ""
    candidates: tuple[str, ...] = ()


def _send_message_schema_kind_for_spec(spec: Any) -> str:
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
    array_group: dict[str, Any] | None = None,
) -> dict:
    new_slot = dict(slot)
    new_slot["args"] = args
    if array_group is not None:
        new_slot["_send_message_array_group"] = array_group
    if call_id is not None:
        namespace = str(slot.get("namespace") or "")
        new_slot["tc"] = SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(
                name=slot["fn_name"],
                namespace=namespace,
                arguments=json.dumps(args, ensure_ascii=False),
            ),
        )
    return new_slot


def _split_send_message_array_slots(slots: list[dict]) -> list[dict]:
    """Split array-shaped send_message calls into single-message executions."""
    expanded: list[dict] = []
    for slot in slots:
        if (
            slot.get("fn_name") != "send_message"
            or slot.get("send_message_schema") != "array"
            or slot.get("result") is not None
        ):
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

        original_id = str(getattr(slot["tc"], "id", "") or "")
        split_id_base = original_id or f"send_message_array_{len(expanded) + 1}"
        group_key = f"{split_id_base}:{id(slot)}"
        original_args = dict(args)
        if len(single_args_list) > 1:
            logger.info(
                "[send_message] array 形态已拆分为 %d 次独立工具执行 call_id=%s",
                len(single_args_list),
                original_id or "<empty>",
            )
        for index, single_args in enumerate(single_args_list, start=1):
            array_group = {
                "call_id": original_id,
                "group_key": group_key,
                "args": original_args,
                "index": index - 1,
                "total": len(single_args_list),
            }
            expanded.append(
                _clone_send_message_slot(
                    slot,
                    args=single_args,
                    call_id=None if index == 1 else f"{split_id_base}_split_{index}",
                    array_group=array_group,
                )
            )

    return expanded


def _is_failed_tool_result(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("ok") is False:
        return True
    if result.get("error"):
        return True
    if result.get("tool_not_executed"):
        return True
    failed_count = result.get("failed_count")
    return isinstance(failed_count, int) and failed_count > 0


def _compact_send_message_array_item(index: int, result: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"index": index}
    if not isinstance(result, dict):
        item["ok"] = result is not None
        if result is None:
            item["error"] = "no_result"
        return item

    failed = _is_failed_tool_result(result)
    item["ok"] = not failed
    block_reason = result.get("block_reason")
    if block_reason:
        item["block_reason"] = block_reason
    if result.get("aware"):
        item["aware"] = result.get("aware")
    if result.get("reason"):
        item["reason"] = result.get("reason")

    error = result.get("error")
    failed_messages = result.get("failed_messages")
    if not block_reason:
        if isinstance(failed_messages, list) and failed_messages:
            first_failed = failed_messages[0]
            if isinstance(first_failed, dict) and first_failed.get("reason"):
                item["error"] = first_failed.get("reason")
        elif error:
            item["error"] = error

    sent_count = result.get("sent_count")
    if isinstance(sent_count, int) and sent_count not in (0, 1):
        item["sent_count"] = sent_count
    failed_count = result.get("failed_count")
    if isinstance(failed_count, int) and failed_count not in (0, 1):
        item["failed_count"] = failed_count
    return item


def _merge_send_message_array_results(slots: list[dict]) -> dict[str, Any]:
    sent_count = 0
    failed_count = 0
    new_messages_count = 0
    interrupted = False
    target: Any = None
    warnings: list[Any] = []
    item_results: list[dict[str, Any]] = []

    for fallback_index, slot in enumerate(slots):
        group = slot.get("_send_message_array_group")
        if isinstance(group, dict) and isinstance(group.get("index"), int):
            message_index: int = group["index"]
        else:
            message_index: int = fallback_index
        result = slot.get("result")
        item_results.append(_compact_send_message_array_item(message_index, result))

        if isinstance(result, dict):
            if target is None and result.get("to") is not None:
                target = result.get("to")
            if isinstance(result.get("sent_count"), int):
                sent_count += result["sent_count"]
            elif result.get("ok") is True:
                sent_count += 1

            result_failed_count = result.get("failed_count")
            if isinstance(result_failed_count, int):
                failed_count += result_failed_count
            elif _is_failed_tool_result(result):
                failed_count += 1

            if isinstance(result.get("new_messages_count"), int):
                new_messages_count += result["new_messages_count"]
            interrupted = interrupted or bool(result.get("interrupted"))

            result_warnings = result.get("warnings")
            if isinstance(result_warnings, list):
                warnings.extend(result_warnings)
            elif result.get("warning"):
                warnings.append(result.get("warning"))
        elif result is None:
            failed_count += 1

    total_count = len(slots)
    if sent_count + failed_count < total_count:
        failed_count = total_count - sent_count

    merged: dict[str, Any] = {
        "sent_count": sent_count,
        "failed_count": failed_count,
        "total_count": total_count,
        "interrupted": interrupted,
        "results": item_results,
    }
    if target is not None:
        merged["to"] = target
    if new_messages_count:
        merged["new_messages_count"] = new_messages_count
    if warnings:
        merged["warnings"] = warnings
        merged["warning"] = warnings[0]
    if failed_count:
        merged["error"] = "部分消息未发送。"
    return merged


def _expanded_single_send_message_slots(slots: list[dict]) -> list[dict]:
    """Split a send_message containing multiple text segments into separate calls."""
    expanded: list[dict] = []
    for slot in slots:
        if (
            slot.get("fn_name") != "send_message"
            or slot.get("send_message_schema") != "single"
            or slot.get("result") is not None
        ):
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


def _tool_call_label(namespace: str, name: str) -> str:
    namespace = str(namespace or "").strip()
    name = str(name or "").strip()
    return f"{namespace}.{name}" if namespace else name


def _spec_call_namespace(spec: Any) -> str:
    return str(getattr(spec, "call_namespace", "") or "").strip()


def _candidate_labels(candidates: list[Any]) -> tuple[str, ...]:
    labels: list[str] = []
    for spec in candidates:
        label = _tool_call_label(_spec_call_namespace(spec), str(getattr(spec, "name", "") or ""))
        if label and label not in labels:
            labels.append(label)
    return tuple(labels)


def _visible_candidates(candidates: list[Any]) -> list[Any]:
    visible: list[Any] = []
    for spec in candidates:
        if str(getattr(spec, "visibility", "visible") or "visible") != "internal":
            visible.append(spec)
            continue
        if (
            getattr(spec, "attached_to", "")
            or getattr(spec, "mounted_to", "")
            or getattr(spec, "visible_namespace", "")
        ):
            visible.append(spec)
    return visible


def _ambiguous_tool_result(name: str, candidates: tuple[str, ...]) -> dict[str, Any]:
    return {
        "ok": False,
        "error": (
            f"工具名 `{name}` 不明确；请在 tool_call 中同时填写 namespace 和 name。"
            f" 可选工具: {', '.join(candidates)}"
        ),
        "tool_not_executed": True,
        "candidates": list(candidates),
    }


def _canonical_tool_path(namespace: str, name: str, tool_collection) -> ToolPathResolution:
    """Resolve model output into the canonical namespace + short tool name.

    The canonical protocol uses separate ``namespace`` and ``name`` fields.
    Legacy bare names are accepted only when they resolve to one visible route.
    Legacy dotted names are accepted only when the prefix is a real namespace.
    """
    namespace = str(namespace or "").strip()
    name = str(name or "").strip()
    registry = getattr(tool_collection, "namespace_registry", None)

    if "." in name:
        prefix, suffix = name.split(".", 1)
        prefix = prefix.strip()
        suffix = suffix.strip()
        known_namespace = registry is not None and registry.get(prefix) is not None
        known_route = tool_collection.get_any(suffix, prefix) is not None
        if prefix and suffix and "." not in suffix and (known_namespace or known_route):
            if not namespace or namespace == prefix:
                namespace = prefix
                name = suffix

    if namespace:
        return ToolPathResolution(namespace=namespace, name=name)

    active_matches = _visible_candidates(tool_collection.matching_active(name))
    if len(active_matches) == 1:
        spec = active_matches[0]
        return ToolPathResolution(namespace=_spec_call_namespace(spec), name=str(getattr(spec, "name", "") or name))
    if len(active_matches) > 1:
        return ToolPathResolution(name=name, error="ambiguous", candidates=_candidate_labels(active_matches))

    latent_matches = _visible_candidates(tool_collection.matching_latent(name))
    if len(latent_matches) == 1:
        spec = latent_matches[0]
        return ToolPathResolution(namespace=_spec_call_namespace(spec), name=str(getattr(spec, "name", "") or name))
    if len(latent_matches) > 1:
        return ToolPathResolution(name=name, error="ambiguous", candidates=_candidate_labels(latent_matches))

    any_matches = _visible_candidates(tool_collection.matching_any(name))
    if len(any_matches) == 1:
        spec = any_matches[0]
        return ToolPathResolution(namespace=_spec_call_namespace(spec), name=str(getattr(spec, "name", "") or name))
    if len(any_matches) > 1:
        return ToolPathResolution(name=name, error="ambiguous", candidates=_candidate_labels(any_matches))

    return ToolPathResolution(name=name)

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
    """Parse processed AIC Action calls into local handler executions."""

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
        decision_guard_snapshot=None,
        current_guard_snapshot_provider=None,
        tool_execution_guard_adapter=None,
        tool_execution_guard_cfg=None,
        agent_run_id: str = "",
        request_started_at: float | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.tool_collection = tool_collection
        self.flow = flow
        self.runtime_stale_checker = runtime_stale_checker
        self.decision_world = decision_world
        self._guard_decision_world = decision_world
        self.current_world_provider = current_world_provider
        self.decision_guard_snapshot = decision_guard_snapshot
        self._guard_decision_snapshot = decision_guard_snapshot
        self.current_guard_snapshot_provider = current_guard_snapshot_provider
        self.tool_execution_guard_adapter = tool_execution_guard_adapter
        self.tool_execution_guard_cfg = tool_execution_guard_cfg
        self.agent_run_id = agent_run_id
        self.request_started_at = request_started_at

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
        if self.agent_run_id:
            context["round_id"] = self.agent_run_id
        context.update(extra)
        return context

    def _slot_label(self, slot: dict) -> str:
        return _tool_call_label(str(slot.get("namespace") or ""), str(slot.get("fn_name") or ""))

    def _emit_tool_hook(
        self,
        point: str,
        slot: dict,
        *,
        result: Any = None,
        error: BaseException | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        hook_context = context or self._tool_hook_context(slot)
        emit_hook(
            namespace="tool",
            point=point,
            target=self._slot_label(slot),
            args=slot.get("args") if isinstance(slot.get("args"), dict) else {},
            result=result,
            error=error,
            context=hook_context,
        )
        try:
            from agent_events import emit_agent_tool_hook

            emit_agent_tool_hook(
                point,
                target=self._slot_label(slot),
                args=slot.get("args") if isinstance(slot.get("args"), dict) else {},
                result=result,
                error=error,
                context=hook_context,
            )
        except Exception:
            logger.debug("[%s] agent tool event emit failed", self.provider_name, exc_info=True)

    def _build_slots(self, tool_calls: list) -> list[dict]:
        slots: list[dict] = []
        registry = getattr(self.tool_collection, "namespace_registry", None)
        state = getattr(self.tool_collection, "namespace_state", None)
        round_index = int(getattr(self.tool_collection, "round_index", 0) or 0)
        local_active_namespaces = set(self.tool_collection.active_namespace_names())
        opened_this_round: set[str] = set()
        closed_this_round: set[str] = set()
        for tool_call in tool_calls:
            original_fn_name = str(tool_call.function.name or "").strip()
            original_namespace = str(getattr(tool_call.function, "namespace", "") or "").strip()
            resolved_path = _canonical_tool_path(original_namespace, original_fn_name, self.tool_collection)
            fn_name = resolved_path.name
            call_namespace = resolved_path.namespace
            if call_namespace:
                tool_call.function.namespace = call_namespace
            tool_call.function.name = fn_name
            aic_action_error = getattr(tool_call, "aic_action_error", None)
            spec = self.tool_collection.get_active(fn_name, call_namespace)
            origin_namespace = (
                self.tool_collection.namespace_for_tool(fn_name, call_namespace)
                if spec is not None or call_namespace
                else ""
            )
            if spec is None and registry is not None:
                origin_spec = registry.get(origin_namespace)
                if origin_spec is not None and not getattr(origin_spec, "visible", True):
                    origin_namespace = ""
            namespace = str(
                getattr(spec, "mounted_to", "")
                or getattr(spec, "attached_to", "")
                or origin_namespace
            )
            handler = spec.handler if spec is not None else None
            processing = None
            args: dict = {}
            if aic_action_error:
                try:
                    parsed_error_args = json.loads(tool_call.function.arguments or "{}")
                    args = parsed_error_args if isinstance(parsed_error_args, dict) else {}
                except Exception:
                    args = {"error": str(aic_action_error)}
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
                "result_cdata": bool(getattr(spec, "result_cdata", False)) if spec is not None else False,
                "resolution_error": resolved_path.error,
                "resolution_candidates": resolved_path.candidates,
                "externally_perceptible": (
                    bool(getattr(spec, "externally_perceptible", False))
                    if spec is not None
                    else False
                ),
                "tool_kind": str(getattr(spec, "tool_kind", "") or "") if spec is not None else "",
                "effect": getattr(spec, "effect", None) if spec is not None else None,
                "execution": getattr(spec, "execution", None) if spec is not None else None,
                "send_message_schema": _send_message_schema_kind_for_spec(spec),
                "result": None,
                "aic_action_error": aic_action_error,
            }
            if aic_action_error:
                slot["result"] = {
                    "ok": False,
                    "error": f"AIC Action 格式错误: {aic_action_error}",
                    "tool_not_executed": True,
                    "retryable": True,
                }
            elif resolved_path.error == "ambiguous":
                slot["result"] = _ambiguous_tool_result(fn_name, resolved_path.candidates)
            elif handler is None:
                if not namespace:
                    slot["result"] = {"error": f"未知工具: {_tool_call_label(call_namespace, fn_name)}"}
            elif processing is not None and not processing.ok:
                if namespace and registry is not None and state is not None:
                    state.mark_active(namespace, registry, round_index)
                slot["result"] = build_tool_argument_error(processing)
            elif namespace and registry is not None and state is not None:
                state.mark_active(namespace, registry, round_index)
            slots.append(slot)

        slots = _split_send_message_array_slots(slots)
        slots = _expanded_single_send_message_slots(slots)
        return slots

    def _parallel_eligible(self, slot: dict) -> bool:
        execution = slot.get("execution")
        return (
            slot.get("result") is None
            and slot.get("fn") is not None
            and bool(getattr(execution, "parallel_safe", False))
            and not bool(slot.get("externally_perceptible"))
            and slot.get("effect") is None
            and slot.get("tool_kind") not in {"runtime_manage", "focus_switch"}
            and slot.get("fn_name") not in self._TERMINAL_CONTROL_TOOLS
            and not slot.get("_send_message_array_group")
        )

    def _parallel_key(self, slot: dict) -> str:
        execution = slot.get("execution")
        return str(getattr(execution, "parallel_key", "") or "")

    def _exec_one(self, slot: dict) -> None:
        fn_name = slot["fn_name"]
        tool_label = self._slot_label(slot)
        logger.info("[%s] 执行工具开始: %s", self.provider_name, tool_label)
        started_at = time.perf_counter()
        error: BaseException | None = None
        slot["_hook_executed"] = True
        hook_context = self._tool_hook_context(slot)
        self._emit_tool_hook("before_call", slot, context=hook_context)
        try:
            with hook_scope(namespace="tool", target=tool_label, context=hook_context):
                call_args = slot["args"]
                if slot.get("tool_kind") == "runtime_manage":
                    call_args = dict(call_args) if isinstance(call_args, dict) else {}
                    call_args["_request_started_at"] = self.request_started_at
                slot["result"] = slot["fn"](**call_args)
            if (
                slot.get("_world_change_aware")
                and isinstance(slot.get("result"), dict)
                and "aware" not in slot["result"]
            ):
                slot["result"]["aware"] = slot["_world_change_aware"]
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
            slot["elapsed_ms"] = final_context["elapsed_ms"]
            self._emit_tool_hook(
                "finally_call",
                slot,
                result=slot.get("result"),
                error=error,
                context=final_context,
            )

    def _tool_call_json(self, slot: dict) -> dict[str, Any]:
        namespace = str(slot.get("namespace") or "")
        payload: dict[str, Any] = {}
        if namespace:
            payload["namespace"] = namespace
        payload["name"] = str(slot.get("fn_name") or "")
        payload["arguments"] = slot.get("args") if isinstance(slot.get("args"), dict) else {}
        return payload

    def _build_world_changed_guard_result(self, reason: str, aware: str = "") -> dict[str, Any]:
        result = {
            "ok": False,
            "error": "工具未执行：我注意到 <world> 已变化，本次行动需要重新评估。",
            "tool_not_executed": True,
            "blocked_by": "self",
            "block_reason": "world_changed_requires_redecision"
        }
        if aware:
            result["aware"] = aware
        return result

    def _build_prior_guard_blocked_result(self, blocked_slot: dict) -> dict[str, Any]:
        result = {
            "ok": False,
            "error": "工具未执行：较早的外界可感知工具已被我中止，需要重新评估。",
            "tool_not_executed": True,
            "blocked_by": "self",
            "block_reason": "prior_external_tool_requires_redecision"
        }
        prior_tool = str(blocked_slot.get("fn_name") or "")
        if prior_tool:
            result["prior_blocked_tool"] = prior_tool
        aware = blocked_slot.get("_world_change_aware")
        if aware:
            result["aware"] = aware
        return result

    def _wait_current_focus_inbound_freshness(self) -> dict[str, Any]:
        try:
            import app_state
            from llm.session import sessions
            from platforms.focus import current_focus_key
        except Exception as exc:
            return {"ok": True, "reason": f"inbound freshness unavailable: {exc}"}

        focus_key = current_focus_key(getattr(app_state, "current_focus", None))
        if not focus_key:
            return {"ok": True, "reason": "no current focus"}
        session = sessions.get(focus_key)
        if session is None:
            return {"ok": True, "reason": "current focus session not loaded", "focus": focus_key}
        waiter = getattr(session, "wait_inbound_processed", None)
        if not callable(waiter):
            return {"ok": True, "reason": "session has no inbound freshness barrier", "focus": focus_key}
        state = waiter(timeout=0.75, quiet_ms=150.0)
        pending = bool(state.get("pending")) if isinstance(state, dict) else False
        return {
            "ok": not pending,
            "reason": "inbound pending" if pending else "inbound processed",
            "focus": focus_key,
            "state": state if isinstance(state, dict) else {},
        }

    def _build_pending_inbound_guard_result(self, freshness: dict[str, Any]) -> dict[str, Any]:
        state = freshness.get("state") if isinstance(freshness.get("state"), dict) else {}
        aware = (
            "我注意到当前会话还有刚到的新消息没有处理完成，"
            "如果现在继续发送，可能会基于过时上下文行动。"
        )
        return {
            "ok": False,
            "error": "工具未执行：当前会话仍有入站消息待处理，本次行动需要重新评估。",
            "tool_not_executed": True,
            "blocked_by": "self",
            "block_reason": "world_changed_requires_redecision",
            "aware": aware,
            "inbound_pending": {
                "focus": freshness.get("focus", ""),
                "received_seq": state.get("received_seq", 0),
                "processed_seq": state.get("processed_seq", 0),
            },
        }

    def _guard_external_effect_slot(self, slot: dict, inner_state: dict) -> bool:
        freshness = self._wait_current_focus_inbound_freshness()
        if not freshness.get("ok", True):
            slot["result"] = self._build_pending_inbound_guard_result(freshness)
            self._emit_tool_hook("guard_blocked", slot, result=slot["result"])
            logger.warning(
                "[%s] 工具执行前入站水位未追平，阻止: %s focus=%s state=%s",
                self.provider_name,
                slot["fn_name"],
                freshness.get("focus", ""),
                freshness.get("state", {}),
            )
            return False

        decision = evaluate_tool_execution_guard(
            decision_world=self._guard_decision_world,
            current_world_provider=self.current_world_provider,
            cognition=str((inner_state or {}).get("cognition") or (inner_state or {}).get("think") or ""),
            tool_call_json=self._tool_call_json(slot),
            tool_effect=slot.get("effect"),
            decision_guard_snapshot=self._guard_decision_snapshot,
            current_guard_snapshot_provider=self.current_guard_snapshot_provider,
            adapter=self.tool_execution_guard_adapter,
            cfg=self.tool_execution_guard_cfg,
        )
        if not decision.world_changed:
            return True
        if decision.aware:
            slot["_world_change_aware"] = decision.aware
        event_result = {
            "ok": bool(decision.execute),
            "world_changed_since_decision": True,
            "checked": bool(decision.checked),
            "reason": decision.reason,
        }
        if decision.aware:
            event_result["aware"] = decision.aware
        if decision.execute:
            if decision.current_world is not None:
                self._guard_decision_world = decision.current_world
            if decision.current_guard_snapshot is not None:
                self._guard_decision_snapshot = decision.current_guard_snapshot
            self._emit_tool_hook("guard_allowed", slot, result=event_result)
            logger.info(
                "[%s] 工具执行前守门放行: %s checked=%s reason=%s",
                self.provider_name,
                slot["fn_name"],
                decision.checked,
                decision.reason,
            )
            return True

        slot["result"] = self._build_world_changed_guard_result(decision.reason, decision.aware)
        self._emit_tool_hook("guard_blocked", slot, result=slot["result"])
        logger.warning(
            "[%s] 工具执行前守门阻止: %s reason=%s",
            self.provider_name,
            slot["fn_name"],
            decision.reason,
        )
        return False

    def _namespace_block_reason(
        self,
        namespace: str,
        opened_this_round: set[str],
        closed_this_round: set[str],
        local_active_namespaces: set[str],
    ) -> str:
        if namespace in closed_this_round:
            return "closed_same_round"
        if namespace in opened_this_round:
            return "opened_same_round"
        if namespace not in local_active_namespaces:
            return "inactive"
        return ""

    def _resolve_non_executable_slot(
        self,
        slot: dict,
        *,
        opened_this_round: set[str],
        closed_this_round: set[str],
        local_active_namespaces: set[str],
    ) -> bool:
        if slot.get("result") is not None:
            return True

        namespace = str(slot.get("namespace") or "")
        reason = (
            self._namespace_block_reason(
                namespace,
                opened_this_round,
                closed_this_round,
                local_active_namespaces,
            )
            if namespace
            else ""
        )
        if slot.get("fn") is None:
            if namespace:
                slot["result"] = _inactive_namespace_result(
                    str(slot.get("fn_name") or ""),
                    namespace,
                    self.tool_collection,
                    reason=reason or "inactive",
                )
                if reason in {"", "inactive"}:
                    opened_this_round.add(namespace)
                return True
            slot["result"] = {"error": f"未知工具: {slot.get('fn_name')}"}
            return True

        if reason:
            slot["result"] = _inactive_namespace_result(
                str(slot.get("fn_name") or ""),
                namespace,
                self.tool_collection,
                reason=reason,
            )
            if reason == "inactive":
                opened_this_round.add(namespace)
            return True
        return False

    def _apply_runtime_lifecycle(
        self,
        slot: dict,
        *,
        opened_this_round: set[str],
        closed_this_round: set[str],
        local_active_namespaces: set[str],
    ) -> None:
        result_data = slot.get("result")
        if not isinstance(result_data, dict):
            return
        lifecycle = result_data.get("_namespace_lifecycle")
        if not isinstance(lifecycle, dict):
            return
        for namespace in _namespace_name_list(lifecycle.get("opened")):
            if namespace not in local_active_namespaces:
                opened_this_round.add(namespace)
        for namespace in _namespace_name_list(lifecycle.get("closed")):
            local_active_namespaces.discard(namespace)
            closed_this_round.add(namespace)

    def _run_parallel_batch(
        self,
        batch: list[dict],
        *,
        opened_this_round: set[str],
        closed_this_round: set[str],
        local_active_namespaces: set[str],
    ) -> None:
        if not batch:
            return
        if len(batch) == 1:
            self._exec_one(batch[0])
        else:
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = [
                    executor.submit(copy_context().run, self._exec_one, slot)
                    for slot in batch
                ]
                for future in futures:
                    future.result()
        for slot in batch:
            self._abort_if_stale()
            self._apply_runtime_lifecycle(
                slot,
                opened_this_round=opened_this_round,
                closed_this_round=closed_this_round,
                local_active_namespaces=local_active_namespaces,
            )

    def execute(self, tool_calls: list, *, inner_state: dict) -> ToolExecutionOutcome:
        self._abort_if_stale()
        slots = self._build_slots(tool_calls)

        inner_state_token = set_current_inner_state(inner_state)
        try:
            restart_scheduled = False
            blocked_external_slot: dict | None = None
            parallel_batch: list[dict] = []
            parallel_keys: set[str] = set()
            local_active_namespaces = set(self.tool_collection.active_namespace_names())
            opened_this_round: set[str] = set()
            closed_this_round: set[str] = set()

            def flush_parallel_batch() -> None:
                nonlocal parallel_batch, parallel_keys
                self._run_parallel_batch(
                    parallel_batch,
                    opened_this_round=opened_this_round,
                    closed_this_round=closed_this_round,
                    local_active_namespaces=local_active_namespaces,
                )
                parallel_batch = []
                parallel_keys = set()

            for slot in slots:
                if parallel_batch:
                    parallel_key = self._parallel_key(slot) if self._parallel_eligible(slot) else ""
                    if not self._parallel_eligible(slot) or (
                        parallel_key and parallel_key in parallel_keys
                    ):
                        flush_parallel_batch()
                if self._resolve_non_executable_slot(
                    slot,
                    opened_this_round=opened_this_round,
                    closed_this_round=closed_this_round,
                    local_active_namespaces=local_active_namespaces,
                ):
                    continue
                self._abort_if_stale()
                if restart_scheduled:
                    flush_parallel_batch()
                    slot["result"] = {
                        "ok": False,
                        "error": "自身重启已安排，本轮剩余工具跳过。",
                        "interrupted": True,
                    }
                    continue
                if blocked_external_slot is not None and slot.get("externally_perceptible"):
                    flush_parallel_batch()
                    slot["result"] = self._build_prior_guard_blocked_result(blocked_external_slot)
                    self._emit_tool_hook("guard_blocked", slot, result=slot["result"])
                    logger.warning(
                        "[%s] 后续外界可感知工具跳过: %s prior_blocked=%s",
                        self.provider_name,
                        slot["fn_name"],
                        blocked_external_slot["fn_name"],
                    )
                    continue
                if slot.get("externally_perceptible") and not self._guard_external_effect_slot(
                    slot,
                    inner_state,
                ):
                    flush_parallel_batch()
                    blocked_external_slot = slot
                    self._abort_if_stale()
                    continue
                if self._parallel_eligible(slot):
                    parallel_key = self._parallel_key(slot)
                    if parallel_key and parallel_key in parallel_keys:
                        flush_parallel_batch()
                    parallel_batch.append(slot)
                    if parallel_key:
                        parallel_keys.add(parallel_key)
                    continue
                flush_parallel_batch()
                self._exec_one(slot)
                self._abort_if_stale()
                self._apply_runtime_lifecycle(
                    slot,
                    opened_this_round=opened_this_round,
                    closed_this_round=closed_this_round,
                    local_active_namespaces=local_active_namespaces,
                )
                slot_result = slot.get("result")
                if (
                    slot["fn_name"] in self._TERMINAL_CONTROL_TOOLS
                    and isinstance(slot_result, dict)
                    and slot_result.get("ok") is True
                    and slot_result.get("restart_scheduled") is True
                ):
                    restart_scheduled = True
            flush_parallel_batch()
            self._abort_if_stale()
        finally:
            reset_current_inner_state(inner_state_token)

        return self._collect(slots)

    def _collect(self, slots: list[dict]) -> ToolExecutionOutcome:
        outcome = ToolExecutionOutcome()

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
                result_data.pop("_namespace_lifecycle", None)
                result_data.pop("_inject_tools", None)
                attach_tool_result_warnings(
                    tool_name=fn_name,
                    args=args if isinstance(args, dict) else {},
                    result=result_data,
                    flow=self.flow,
                )

            tool_log = {
                "namespace": str(slot.get("namespace") or ""),
                "function": fn_name,
                "call_id": str(getattr(tool_call, "id", "") or ""),
                "arguments": args,
                "result": result_data,
            }
            if slot.get("elapsed_ms") is not None:
                tool_log["elapsed_ms"] = slot["elapsed_ms"]
            outcome.tool_calls_log.append(tool_log)

            raw_multimodal_parts: list = []
            if isinstance(result_data, dict) and "_multimodal_parts" in result_data:
                raw_multimodal_parts = result_data.pop("_multimodal_parts")

            slot["_round_multimodal_parts"] = raw_multimodal_parts

        index = 0
        while index < len(slots):
            slot = slots[index]
            group = slot.get("_send_message_array_group")
            if isinstance(group, dict) and slot.get("fn_name") == "send_message" and not slot.get("aic_action_error"):
                call_id = str(group.get("call_id") or getattr(slot["tc"], "id", "") or "")
                group_key = str(group.get("group_key") or call_id)
                grouped_slots: list[dict] = []
                while index < len(slots):
                    candidate = slots[index]
                    candidate_group = candidate.get("_send_message_array_group")
                    if (
                        not isinstance(candidate_group, dict)
                        or candidate.get("fn_name") != "send_message"
                        or str(candidate_group.get("group_key") or candidate_group.get("call_id") or "") != group_key
                    ):
                        break
                    grouped_slots.append(candidate)
                    index += 1
                _group_args = group.get("args")
                original_args: dict = _group_args if isinstance(_group_args, dict) else slot["args"]
                outcome.round_calls.append(
                    ToolCall(
                        name="send_message",
                        namespace=str(slot.get("namespace") or ""),
                        args=original_args,
                        call_id=call_id,
                    )
                )
                outcome.round_responses.append(
                    ToolResponse(
                        name="send_message",
                        namespace=str(slot.get("namespace") or ""),
                        response=_merge_send_message_array_results(grouped_slots),
                        call_id=call_id,
                    )
                )
                continue

            fn_name = slot["fn_name"]
            tool_call = slot["tc"]
            if not slot.get("aic_action_error"):
                outcome.round_calls.append(
                    ToolCall(
                        name=fn_name,
                        namespace=str(slot.get("namespace") or ""),
                        args=slot["args"],
                        call_id=tool_call.id,
                    )
                )
            outcome.round_responses.append(
                ToolResponse(
                    name=AIC_ACTION_ERROR_NAME if slot.get("aic_action_error") else fn_name,
                    namespace="" if slot.get("aic_action_error") else str(slot.get("namespace") or ""),
                    response=slot["result"],
                    call_id=tool_call.id,
                    result_cdata=bool(slot.get("result_cdata")),
                    multimodal_parts=slot.get("_round_multimodal_parts") or [],
                )
            )
            index += 1

        return outcome
