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
    array_group: dict[str, Any] | None = None,
) -> dict:
    new_slot = dict(slot)
    new_slot["args"] = args
    if array_group is not None:
        new_slot["_send_message_array_group"] = array_group
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
        if not getattr(spec, "visible", True) or not getattr(spec, "discoverable", True):
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
        if not getattr(spec, "visible", True):
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


def _namespace_matches_prefixed_tool(prefix: str, spec: Any) -> bool:
    return prefix in {
        str(getattr(spec, "namespace", "") or ""),
        str(getattr(spec, "visible_namespace", "") or ""),
        str(getattr(spec, "attached_to", "") or ""),
        str(getattr(spec, "mounted_to", "") or ""),
    }


def _canonical_prefixed_tool_name(fn_name: str, tool_collection) -> tuple[str, str]:
    """Accept accidental namespace-qualified tool names when unambiguous.

    The model-facing contract remains bare tool names inside each active
    namespace. This compatibility only strips a namespace prefix when the
    suffix is a known tool and the prefix matches either the tool's original
    namespace or its active visible/attached namespace.
    """
    name = str(fn_name or "").strip()
    if "." not in name:
        return name, ""
    prefix, suffix = name.split(".", 1)
    prefix = prefix.strip()
    suffix = suffix.strip()
    if not prefix or not suffix or "." in suffix:
        return name, ""

    registry = getattr(tool_collection, "namespace_registry", None)
    if registry is None or registry.get(prefix) is None:
        return name, ""

    spec = tool_collection.get_active(suffix)
    if spec is not None and _namespace_matches_prefixed_tool(prefix, spec):
        return suffix, f"normalized namespace-qualified tool name {name!r} -> {suffix!r}"

    spec = tool_collection.get_latent(suffix)
    if spec is not None and str(getattr(spec, "namespace", "") or "") == prefix:
        return suffix, f"normalized inactive namespace-qualified tool name {name!r} -> {suffix!r}"

    spec = tool_collection.get_any(suffix)
    if spec is not None and str(getattr(spec, "namespace", "") or "") == prefix:
        return suffix, f"normalized namespace-qualified tool name {name!r} -> {suffix!r}"

    return name, ""


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
        agent_run_id: str = "",
    ) -> None:
        self.provider_name = provider_name
        self.tool_collection = tool_collection
        self.flow = flow
        self.runtime_stale_checker = runtime_stale_checker
        self.decision_world = decision_world
        self.current_world_provider = current_world_provider
        self.tool_execution_guard_adapter = tool_execution_guard_adapter
        self.tool_execution_guard_cfg = tool_execution_guard_cfg
        self.agent_run_id = agent_run_id

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
            target=str(slot.get("fn_name") or ""),
            args=slot.get("args") if isinstance(slot.get("args"), dict) else {},
            result=result,
            error=error,
            context=hook_context,
        )
        try:
            from agent_events import emit_agent_tool_hook

            emit_agent_tool_hook(
                point,
                target=str(slot.get("fn_name") or ""),
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
            fn_name, name_repair = _canonical_prefixed_tool_name(original_fn_name, self.tool_collection)
            if name_repair:
                logger.warning("[%s] 工具名已按 namespace 兼容规则规范化: %s", self.provider_name, name_repair)
                tool_call.function.name = fn_name
            protocol_error = getattr(tool_call, "protocol_error", None)
            spec = self.tool_collection.get_active(fn_name)
            origin_namespace = self.tool_collection.namespace_for_tool(fn_name)
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
                "original_fn_name": original_fn_name if original_fn_name != fn_name else "",
                "name_repair": name_repair,
                "externally_perceptible": (
                    bool(getattr(spec, "externally_perceptible", False))
                    if spec is not None
                    else False
                ),
                "tool_kind": str(getattr(spec, "tool_kind", "") or "") if spec is not None else "",
                "effect": getattr(spec, "effect", None) if spec is not None else None,
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
            slot["elapsed_ms"] = final_context["elapsed_ms"]
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
            tool_effect=slot.get("effect"),
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
        focus_switch_slots = [
            slot for slot in pending_slots
            if slot.get("tool_kind") == "focus_switch"
        ]
        external_effect_slots = [
            slot for slot in pending_slots
            if slot.get("externally_perceptible")
        ]
        if focus_switch_slots and external_effect_slots:
            focus_switch_name = str(focus_switch_slots[0].get("fn_name") or "focus_switch")
            for slot in external_effect_slots:
                slot["result"] = {
                    "ok": False,
                    "error": (
                        "本轮同时包含焦点切换工具和外界可感知工具；系统暂没有兼容此种情况。"
                    ),
                    "tool_not_executed": True,
                    "incompatible_with": focus_switch_name,
                }
            for slot in pending_slots:
                if slot.get("tool_kind") == "focus_switch" or slot.get("externally_perceptible"):
                    continue
                slot["result"] = {
                    "ok": False,
                    "error": f"本轮同时包含焦点切换工具和外界可感知工具；已只执行 {focus_switch_name}，本工具跳过。",
                    "tool_not_executed": True,
                    "skipped_due_to": "focus_switch_externally_perceptible_tool_conflict",
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

            tool_log = {
                "function": fn_name,
                "call_id": str(getattr(tool_call, "id", "") or ""),
                "arguments": args,
                "result": result_data,
            }
            if slot.get("original_fn_name"):
                tool_log["original_function"] = str(slot.get("original_fn_name") or "")
            if slot.get("name_repair"):
                tool_log["repairs"] = [str(slot.get("name_repair") or "")]
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
            if isinstance(group, dict) and slot.get("fn_name") == "send_message" and not slot.get("protocol_error"):
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
                    ToolCall(name="send_message", args=original_args, call_id=call_id)
                )
                outcome.round_responses.append(
                    ToolResponse(
                        name="send_message",
                        response=_merge_send_message_array_results(grouped_slots),
                        call_id=call_id,
                    )
                )
                continue

            fn_name = slot["fn_name"]
            tool_call = slot["tc"]
            if not slot.get("protocol_error"):
                outcome.round_calls.append(
                    ToolCall(name=fn_name, args=slot["args"], call_id=tool_call.id)
                )
            outcome.round_responses.append(
                ToolResponse(
                    name=XML_TOOL_CALL_ERROR_NAME if slot.get("protocol_error") else fn_name,
                    response=slot["result"],
                    call_id=tool_call.id,
                    multimodal_parts=slot.get("_round_multimodal_parts") or [],
                )
            )
            index += 1

        return outcome
