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


def _send_message_uses_single_schema(tool_collection: Any) -> bool:
    spec = getattr(tool_collection, "active_specs", {}).get("send_message")
    declaration = getattr(spec, "declaration", None)
    if not isinstance(declaration, dict):
        return False
    parameters = declaration.get("parameters")
    if not isinstance(parameters, dict):
        return False
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return False
    return "segments" in properties and "messages" not in properties


def _expanded_send_message_slots(slots: list[dict]) -> list[dict]:
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
            new_slot = dict(slot)
            new_slot["args"] = normalized_args
            if index > 1:
                new_slot["tc"] = SimpleNamespace(
                    id=f"{original_id}_split_{index}",
                    function=SimpleNamespace(
                        name=slot["fn_name"],
                        arguments=json.dumps(normalized_args, ensure_ascii=False),
                    ),
                )
            expanded.append(new_slot)

    return expanded


def _build_latent_tool_activation_warning(fn_name: str) -> dict:
    return {
        "ok": False,
        "warning": (
            f"The tool `{fn_name}` is currently in a hidden, inactive state and cannot be executed directly."
            f"The system has precisely matched and activated the required tool based on the hidden tool name; `{fn_name}` is now ready for use."
        ),
        "tool_not_executed": True,
        "activation_deferred": True,
        "activated": [fn_name],
    }


def _tool_description(spec: Any) -> str:
    declaration = getattr(spec, "declaration", {}) or {}
    description = declaration.get("description", "")
    return str(description).strip() if description is not None else ""


def _preview_hidden_tools(requested: list[str], tool_collection) -> tuple[list[dict], list[dict]]:
    previews: list[dict] = []
    warnings: list[dict] = []
    for raw_name in requested:
        name = str(raw_name).strip()
        if not name:
            continue
        if tool_collection.get_active(name) is not None:
            warnings.append({
                "name": name,
                "warning": "该工具已经激活；preview 只作用于隐藏工具。",
            })
            continue
        latent_spec = tool_collection.get_latent(name)
        if latent_spec is None:
            warnings.append({
                "name": name,
                "warning": "未找到可预览的隐藏工具。",
            })
            continue
        previews.append({
            "name": name,
            "description": _tool_description(latent_spec),
        })
    return previews, warnings


def _search_hidden_tools(query: str, tool_collection) -> list[dict]:
    keyword = query.strip()
    if not keyword:
        return []

    matches: list[dict] = []
    for name in tool_collection.latent_names():
        spec = tool_collection.get_latent(name)
        if spec is None:
            continue
        description = _tool_description(spec)
        if keyword in description:
            matches.append({
                "name": name,
                "description": description,
            })
        if len(matches) >= 5:
            break
    return matches


def _annotate_tool_activation_result(result: dict, requested: list, tool_collection) -> dict:
    annotated = dict(result)

    already_active: list[str] = []
    newly_activated: list[str] = []
    for raw_name in requested:
        name = str(raw_name).strip()
        if not name:
            continue
        if tool_collection.get_active(name) is not None:
            already_active.append(name)
            continue
        if tool_collection.get_latent(name) is not None and tool_collection.activate(name) is not None:
            newly_activated.append(name)

    if not already_active and not newly_activated:
        return annotated

    activated = [
        name
        for name in annotated.get("activated", [])
        if isinstance(name, str) and name
    ]
    for name in [*already_active, *newly_activated]:
        if name not in activated:
            activated.append(name)
    annotated["activated"] = activated
    if already_active:
        annotated["already_active"] = already_active
        annotated["warning"] = (
            "重复激活；这些工具已经处于可用状态，现在可直接使用："
            + ", ".join(already_active)
            + "。无需再次 tools_manage.get。"
        )
    if newly_activated:
        annotated["newly_activated"] = newly_activated
    return annotated


def _annotate_tools_manage_result(result: object, args: dict, tool_collection) -> object:
    if not isinstance(result, dict):
        return result

    annotated = dict(result)

    preview = args.get("preview")
    if isinstance(preview, list):
        preview_items, warnings = _preview_hidden_tools(preview, tool_collection)
        annotated["preview"] = preview_items
        if warnings:
            annotated["warnings"] = [*annotated.get("warnings", []), *warnings]

    search = args.get("search")
    if isinstance(search, str):
        annotated["search"] = _search_hidden_tools(search, tool_collection)

    requested = args.get("get")
    if isinstance(requested, list):
        annotated = _annotate_tool_activation_result(annotated, requested, tool_collection)
    return annotated


class ToolExecutor:
    """Parse processed XML tool calls into local handler executions."""

    _TERMINAL_CONTROL_TOOLS = frozenset({
        "restart_self",
    })

    def __init__(
        self,
        *,
        provider_name: str,
        tool_collection,
        flow=None,
        runtime_stale_checker=None,
    ) -> None:
        self.provider_name = provider_name
        self.tool_collection = tool_collection
        self.flow = flow
        self.runtime_stale_checker = runtime_stale_checker

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
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            protocol_error = getattr(tool_call, "protocol_error", None)
            spec = self.tool_collection.get_active(fn_name)
            latent_spec = self.tool_collection.get_latent(fn_name) if spec is None else None
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
            elif handler is None:
                if latent_spec is not None:
                    slot["result"] = _build_latent_tool_activation_warning(fn_name)
                else:
                    slot["result"] = {"error": f"未知工具: {fn_name}"}
            elif processing is not None and not processing.ok:
                slot["result"] = build_tool_argument_error(processing)
            slots.append(slot)

        if _send_message_uses_single_schema(self.tool_collection):
            slots = _expanded_send_message_slots(slots)
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
                if fn_name == "tools_manage":
                    slot["result"] = _annotate_tools_manage_result(
                        slot["result"],
                        slot["args"],
                        self.tool_collection,
                    )
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
            for slot in external_effect_slots:
                self._abort_if_stale()
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
