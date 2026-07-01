"""One-round LLM orchestration on top of an OpenAI-compatible transport."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam

from consciousness.flow import ConsciousnessFlow, ToolCall, ToolResponse
from llm.compression.config import normalize_generation_config
from llm.discarded_response_log import (
    normalize_discarded_response_log_config,
    save_cognition_prefill_discard,
)
from llm.prompt_snapshot import normalize_prompt_snapshot_config, save_prompt_snapshot

from .duplicate_response_guard import (
    build_duplicate_model_response_error,
    choose_cognition_prefill,
    cognition_prefill_provider_supported,
    CognitionPrefillRetrySignal,
    CognitionRepeatStreamGuard,
    format_cognition_prefill,
    is_passive_duplicate_tool_set,
    normalize_duplicate_model_response_guard_config,
    normalize_response_text,
)
from .error_policy import classify_llm_exception
from .internal_tool import InternalToolSpec
from .prompt_diagnostics import log_prompt_prefix_comparison, serialize_prompt_prefix
from .tool_calling import parse_tool_arguments
from .tool_calling.xml_protocol import build_tools_xml_message, parse_xml_tool_calls
from .tool_execution_guard import extract_world_text
from .tool_executor import RuntimeResetAborted, ToolExecutor
from .transport import (
    OpenAICompatClient,
    add_enabled_sampling_kwargs,
    add_extra_generation_kwargs,
    prepare_streaming_create_kwargs,
)
from log_config import log_cognition, log_prompt, log_response
from llm_usage_recorder import parse_usage, record_llm_usage
from agent_events import AgentXmlStreamProjector, emit_agent_event, summarize_tool_payload

logger = logging.getLogger("AICQ.llm.provider")


class LLMCallFailed(Exception):
    """LLM 调用最终失败（预留给上层统一捕获）。"""


@dataclass
class RoundResult:
    """单轮 LLM 调用的产物。"""
    tool_calls_log: list[dict] = field(default_factory=list)
    system_prompt: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    cognition: str = ""
    inner_state: dict = field(default_factory=dict)
    prompt_snapshot_id: str = ""
    raw_response: str = ""
    duplicate_model_response: bool = False
    duplicate_model_response_count: int = 0
    duplicate_model_response_error: dict = field(default_factory=dict)
    cognition_prefill_retry: bool = False
    cognition_prefill_retry_error: dict = field(default_factory=dict)
    cognition_prefill: str = ""
    cognition_prefill_body: str = ""
    discarded_cognition: str = ""
    had_tool_call: bool = False
    # API 调用本身失败 / response.choices 为空时为 True
    failed: bool = False
    llm_error: dict = field(default_factory=dict)
    # WebUI 紧急恢复发生后，旧 round 只允许返回这个标记，不再执行工具/写 flow。
    aborted_by_runtime_reset: bool = False
    runtime_reset_epoch: int = 0
    agent_run_id: str = ""
    world_xml: str = ""


def _record_usage_event(
    *,
    provider: str,
    model: str,
    feature: str,
    subfeature: str = "",
    usage=None,
    status: str = "success",
) -> None:
    try:
        record_llm_usage(
            provider=provider,
            model=model,
            feature=feature,
            subfeature=subfeature,
            usage=usage,
            status=status,
        )
    except Exception:
        logger.debug("[%s] 记录 LLM token 用量失败", provider, exc_info=True)


def _finish_reason(response: Any) -> str:
    try:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        return str(getattr(choices[0], "finish_reason", "") or "")
    except Exception:
        return ""


def _log_finish_reason(tag: str, response: Any) -> str:
    reason = _finish_reason(response)
    if not reason:
        return ""
    if reason in {"length", "content_filter"}:
        logger.warning("[%s] finish_reason=%s", tag, reason)
    else:
        logger.debug("[%s] finish_reason=%s", tag, reason)
    return reason


def _simple_text_usage_scope(log_tag: str) -> tuple[str, str]:
    if log_tag.startswith("think_deeply/"):
        return "slow_thinking", log_tag.split("/", 1)[1]
    if log_tag == "cognition_compression":
        return "cognition_compression", ""
    return "simple_text", log_tag


def _forced_tool_usage_scope(log_tag: str) -> tuple[str, str]:
    if log_tag == "archiver":
        return "memory_archiver", ""
    return "forced_tool", log_tag


def _strip_images(user_content: str | list) -> str | list:
    """从多模态内容中剥除图片部分，仅保留文本。"""
    if not isinstance(user_content, list):
        return user_content
    if not (text_parts := [part for part in user_content if part.get("type") == "text"]):
        return ""
    return text_parts[0]["text"] if len(text_parts) == 1 else text_parts


def _message_content_to_text(raw_content: str | list | None) -> str:
    """将 OpenAI 兼容消息内容统一抽取为纯文本。"""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        return "\n".join(
            part.get("text", "")
            for part in raw_content
            if isinstance(part, dict) and "text" in part
        )
    return ""


def _inner_state_from_cognition(cognition: str) -> dict:
    cognition = cognition.strip()
    if not cognition:
        return {}
    return {"cognition": cognition, "think": cognition}


def _snapshot_create_kwargs(create_kwargs: dict) -> dict:
    return prepare_streaming_create_kwargs(create_kwargs)


class LLMRoundRunner:
    """Application-level LLM runner: prompts, XML tools, flow, and usage."""

    def __init__(self, cfg: dict):
        self.transport = OpenAICompatClient(cfg)
        self.provider = self.transport.provider
        self.model = self.transport.model
        self._vision_enabled = self.transport.vision_enabled
        self._assistant_prefill_supported = bool(
            getattr(self.transport, "assistant_prefill_supported", True)
        )
        self._prompt_snapshot_cfg = normalize_prompt_snapshot_config(
            cfg.get("prompt_snapshots")
        )
        self._discarded_response_log_cfg = normalize_discarded_response_log_config(
            cfg.get("discarded_response_logs")
        )

    def list_models(self) -> list[str]:
        return self._get_transport().list_models()

    def _get_transport(self) -> OpenAICompatClient:
        transport = getattr(self, "transport", None)
        if transport is not None:
            return transport

        # Unit tests sometimes construct the runner with object.__new__ to avoid
        # creating a real SDK client. Production construction always sets
        # self.transport in __init__.
        transport = object.__new__(OpenAICompatClient)
        transport.client = getattr(self, "client")
        transport.provider = getattr(self, "provider", "")
        transport.model = getattr(self, "model", "")
        transport.vision_enabled = bool(getattr(self, "_vision_enabled", True))
        transport._thinking_control = getattr(self, "_thinking_control", "enable_thinking")
        self.transport = transport
        return transport

    def _normalize_generation_for_transport(self, gen: dict) -> dict:
        transport = self._get_transport()
        transport.model = getattr(self, "model", transport.model)
        transport.provider = getattr(self, "provider", transport.provider)
        transport._thinking_control = getattr(
            self,
            "_thinking_control",
            getattr(transport, "_thinking_control", "enable_thinking"),
        )
        self._assistant_prefill_supported = bool(
            getattr(transport, "assistant_prefill_supported", True)
        )
        return transport.normalize_generation(gen)

    def _create_chat_completion(
        self,
        *,
        all_messages: list,
        create_kwargs: dict,
        on_text_delta=None,
        on_chunk=None,
    ) -> Any:
        return self._get_transport().create_chat_completion(
            all_messages=all_messages,
            create_kwargs=create_kwargs,
            on_text_delta=on_text_delta,
            on_chunk=on_chunk,
        )

    def call_one_round(
        self,
        system_prompt_builder,
        user_content: str | list,
        gen: dict,
        tool_collection,
        flow: ConsciousnessFlow | None = None,
        usage_feature: str = "main_round",
        usage_subfeature: str = "",
        prompt_snapshot_context: dict | None = None,
        runtime_stale_checker=None,
        current_world_provider=None,
        agent_run_id: str = "",
        agent_context: dict | None = None,
        assistant_prefill: str = "",
        prefill_exclusions: list[str] | tuple[str, ...] | None = None,
    ) -> RoundResult:
        """跑一轮 XML 文本工具协议：1 次 LLM 调用 + 本轮工具执行。"""
        assistant_prefill = str(assistant_prefill or "")
        if assistant_prefill:
            gen = dict(gen or {})
            gen["enable_thinking"] = False
        gen = self._normalize_generation_for_transport(gen)
        if tool_collection is None:
            from tools.specs import ToolCollection
            tool_collection = ToolCollection()

        def _runtime_is_stale() -> bool:
            if runtime_stale_checker is None:
                return False
            try:
                return bool(runtime_stale_checker())
            except Exception:
                logger.debug("[%s] runtime_stale_checker 失败", self.provider, exc_info=True)
                return False

        def _abort_for_runtime_reset() -> RoundResult:
            logger.warning("[%s] 运行时已被紧急恢复，本轮结果丢弃", self.provider)
            result.failed = True
            result.aborted_by_runtime_reset = True
            return result

        gen = normalize_generation_config(gen)
        max_rounds: int = gen["llm_contents_max_rounds"]
        duplicate_guard_cfg = normalize_duplicate_model_response_guard_config(
            (gen or {}).get("duplicate_model_response_guard")
        )
        prefill_cfg = duplicate_guard_cfg.get("prefill_guidance") or {}

        if not self._vision_enabled:
            user_content = _strip_images(user_content)

        full_system = system_prompt_builder(
            tool_collection.active_names(),
            tool_collection.latent_names(),
        )
        log_prompt(self.provider, full_system, user_content)

        user_msg: ChatCompletionMessageParam = {"role": "user", "content": user_content}
        system_msg: ChatCompletionMessageParam = {"role": "system", "content": full_system}

        namespace_blocks = tool_collection.namespace_prompt_blocks()
        create_kwargs: dict = {
            "model": self.model,
            "temperature": gen.get("temperature", 1.0),
            "max_tokens": gen.get("max_output_tokens", 10000),
        }
        add_extra_generation_kwargs(create_kwargs, gen)
        if extra_body := gen.get("extra_body"):
            create_kwargs["extra_body"] = extra_body
        add_enabled_sampling_kwargs(create_kwargs, gen)

        world_xml = extract_world_text(user_content)
        result = RoundResult(system_prompt=full_system, world_xml=world_xml)
        if agent_run_id:
            emit_agent_event(
                "world_frame",
                round_id=agent_run_id,
                provider=self.provider,
                model=self.model,
                world_xml=world_xml,
                world_chars=len(world_xml),
                **(agent_context or {}),
            )
            emit_agent_event(
                "model_request",
                round_id=agent_run_id,
                provider=self.provider,
                model=self.model,
                feature=usage_feature,
                subfeature=usage_subfeature,
                active_tools=tool_collection.active_names(),
                latent_tools=tool_collection.latent_names(),
                **(agent_context or {}),
            )
        if _runtime_is_stale():
            return _abort_for_runtime_reset()

        tools_messages: list[dict] = []
        if namespace_blocks:
            tools_messages.append({
                "role": "user",
                "content": build_tools_xml_message(
                    [],
                    namespace_blocks=namespace_blocks,
                ),
            })
        if flow:
            flow.promote_ready_compression_summary(max_rounds)
        flow_messages = flow.to_xml_messages() if flow else []
        all_messages = [system_msg] + tools_messages + flow_messages + [user_msg]
        if assistant_prefill:
            all_messages.append({"role": "assistant", "content": assistant_prefill})
        result.prompt_snapshot_id = save_prompt_snapshot(
            getattr(self, "_prompt_snapshot_cfg", {"enabled": False}),
            request_kind="main_round",
            provider=self.provider,
            model=self.model,
            messages=all_messages,
            create_kwargs=_snapshot_create_kwargs(create_kwargs),
            feature=usage_feature,
            subfeature=usage_subfeature,
            context=prompt_snapshot_context,
        )
        stable_prefix = serialize_prompt_prefix(cast(list[dict[str, Any]], [system_msg, *tools_messages]))
        previous_stable_prefix = getattr(self, "_last_main_stable_prompt_prefix", None)
        log_prompt_prefix_comparison(
            provider=self.provider,
            previous_prefix=previous_stable_prefix,
            current_prefix=stable_prefix,
        )
        self._last_main_stable_prompt_prefix = stable_prefix
        stream_projector = (
            AgentXmlStreamProjector(
                round_id=agent_run_id,
                provider=self.provider,
                model=self.model,
            )
            if agent_run_id
            else None
        )
        visible_cognitions: tuple[str, ...] = ()
        cognition_repeat_guard: CognitionRepeatStreamGuard | None = None
        prefill_supported = (
            bool(getattr(self, "_assistant_prefill_supported", True))
            and cognition_prefill_provider_supported(self.provider, self.model)
        )
        if (
            prefill_cfg.get("enabled")
            and flow is not None
            and prefill_supported
        ):
            visible_cognitions = tuple(
                flow.visible_cognitions(int(prefill_cfg.get("lookback_rounds") or 8))
            )
            if visible_cognitions:
                cognition_repeat_guard = CognitionRepeatStreamGuard(
                    visible_cognitions=visible_cognitions,
                    similarity_threshold=float(prefill_cfg.get("similarity_threshold") or 0.9),
                    min_chars=int(prefill_cfg.get("min_chars") or 80),
                )

        def _observe_text_delta(text: str) -> None:
            if stream_projector is not None:
                stream_projector.feed(text)
            if cognition_repeat_guard is not None:
                cognition_repeat_guard.feed(text)

        if assistant_prefill and (
            stream_projector is not None or cognition_repeat_guard is not None
        ):
            _observe_text_delta(assistant_prefill)

        try:
            response = self._create_chat_completion(
                all_messages=all_messages,
                create_kwargs=create_kwargs,
                on_text_delta=(
                    _observe_text_delta
                    if stream_projector is not None or cognition_repeat_guard is not None
                    else None
                ),
            )
        except CognitionPrefillRetrySignal as retry_exc:
            prefill_body = choose_cognition_prefill(
                visible_cognitions,
                used_prefills=tuple(prefill_exclusions or ()),
                seed_text=retry_exc.cognition,
            )
            result.cognition_prefill_retry = True
            result.discarded_cognition = retry_exc.cognition
            result.cognition = retry_exc.cognition
            result.cognition_prefill_body = prefill_body
            result.cognition_prefill = format_cognition_prefill(prefill_body)
            result.cognition_prefill_retry_error = {
                "error": "REPEATED_COGNITION_BLOCK",
                "message": "本轮 cognition 与当前可见意识流中的 cognition 高度重复，已在 action 前丢弃并准备预填充重调。",
                "tool_not_executed": True,
                "retryable": True,
                "similarity": round(retry_exc.similarity, 4),
                "matched_index": retry_exc.matched_index,
                "prefill": prefill_body,
            }
            discard_log_id = save_cognition_prefill_discard(
                getattr(self, "_discarded_response_log_cfg", None),
                provider=self.provider,
                model=self.model,
                feature=usage_feature,
                subfeature=usage_subfeature,
                prompt_snapshot_id=result.prompt_snapshot_id,
                agent_run_id=agent_run_id,
                context=prompt_snapshot_context,
                retry_attempt=len(tuple(prefill_exclusions or ())) + 1,
                similarity=retry_exc.similarity,
                matched_index=retry_exc.matched_index,
                discarded_cognition=retry_exc.cognition,
                matched_cognition=retry_exc.matched_cognition,
                chosen_prefill=prefill_body,
                visible_cognitions_count=len(visible_cognitions),
                prefill_exclusions=tuple(prefill_exclusions or ()),
                guard_config=prefill_cfg,
            )
            if discard_log_id:
                result.cognition_prefill_retry_error["discard_log_id"] = discard_log_id
            logger.warning(
                "[%s] repeated cognition detected similarity=%.4f; action stream discarded",
                self.provider,
                retry_exc.similarity,
            )
            if agent_run_id:
                event_payload = {
                    "round_id": agent_run_id,
                    "provider": self.provider,
                    "model": self.model,
                    "reason": "repeated_visible_cognition",
                    "similarity": round(retry_exc.similarity, 4),
                    "matched_index": retry_exc.matched_index,
                    "prefill_preview": prefill_body,
                }
                if discard_log_id:
                    event_payload["discard_log_id"] = discard_log_id
                emit_agent_event("cognition_discarded", **event_payload)
            return result
        except Exception as exc:
            error_decision = classify_llm_exception(exc)
            logger.warning(
                "[%s] LLM API 调用异常 category=%s status=%s action=%s cooldown=%.1fs: %s",
                self.provider,
                error_decision.category,
                error_decision.status_code,
                error_decision.action,
                error_decision.cooldown_seconds,
                error_decision.detail or exc,
            )
            _record_usage_event(
                provider=self.provider,
                model=self.model,
                feature=usage_feature,
                subfeature=usage_subfeature,
                usage=None,
                status="error",
            )
            try:
                if "image" in str(exc).lower() or "20015" in str(exc):
                    dump_dir = os.path.join(os.getcwd(), "logs", "failed_prompts")
                    os.makedirs(dump_dir, exist_ok=True)
                    dump_path = os.path.join(
                        dump_dir,
                        f"{time.strftime('%Y%m%d_%H%M%S')}_{self.provider}.json",
                    )
                    with open(dump_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {"error": str(exc), "model": getattr(self, "model", "?"), "messages": all_messages},
                            f, ensure_ascii=False, indent=2,
                        )
                    logger.warning("[%s] 已 dump 失败 prompt -> %s", self.provider, dump_path)
            except Exception as dump_exc:
                logger.debug("[%s] dump 失败 prompt 时出错: %s", self.provider, dump_exc)
            result.failed = True
            result.llm_error = error_decision.to_dict()
            if agent_run_id:
                emit_agent_event(
                    "round_error",
                    round_id=agent_run_id,
                    provider=self.provider,
                    model=self.model,
                    error=error_decision.summary,
                    category=error_decision.category,
                    status_code=error_decision.status_code,
                    retryable=error_decision.retryable,
                    cooldown_seconds=error_decision.cooldown_seconds,
                    stage="llm_call",
                )
            return result
        finally:
            if stream_projector is not None:
                stream_projector.finish()

        if _runtime_is_stale():
            return _abort_for_runtime_reset()

        if response is None:
            logger.warning("[%s] response 为 None", self.provider)
            _record_usage_event(
                provider=self.provider,
                model=self.model,
                feature=usage_feature,
                subfeature=usage_subfeature,
                usage=None,
                status="response_none",
            )
            result.failed = True
            if agent_run_id:
                emit_agent_event(
                    "round_error",
                    round_id=agent_run_id,
                    provider=self.provider,
                    model=self.model,
                    error="response_none",
                    stage="llm_call",
                )
            return result

        if _runtime_is_stale():
            return _abort_for_runtime_reset()

        usage = getattr(response, "usage", None)
        _record_usage_event(
            provider=self.provider,
            model=self.model,
            feature=usage_feature,
            subfeature=usage_subfeature,
            usage=usage,
            status="success" if response.choices else "empty_choices",
        )
        usage_counts = parse_usage(usage)
        if usage_counts["usage_available"]:
            result.prompt_tokens = usage_counts["input_tokens"]
            result.output_tokens = usage_counts["output_tokens"]
            logger.info(
                "[%s] token — 输入: %d, 输出: %d, 总计: %d",
                self.provider,
                result.prompt_tokens,
                result.output_tokens,
                usage_counts["total_tokens"],
            )

        if not response.choices:
            logger.warning("[%s] response.choices 为空", self.provider)
            result.failed = True
            if agent_run_id:
                emit_agent_event(
                    "round_error",
                    round_id=agent_run_id,
                    provider=self.provider,
                    model=self.model,
                    error="empty_choices",
                    stage="llm_call",
                )
            return result

        msg = response.choices[0].message
        raw_response_text = _message_content_to_text(getattr(msg, "content", None))
        if assistant_prefill:
            raw_stripped = raw_response_text.lstrip()
            if raw_response_text.startswith(assistant_prefill):
                pass
            elif raw_stripped.startswith("</cognition>") or not raw_stripped.lower().startswith("<cognition"):
                raw_response_text = assistant_prefill + raw_response_text
        result.raw_response = raw_response_text
        log_response(self.provider, raw_response_text)
        parsed_xml = parse_xml_tool_calls(raw_response_text)
        result.cognition = parsed_xml.cognition
        result.inner_state = _inner_state_from_cognition(parsed_xml.cognition)
        log_cognition(self.provider, result.cognition)
        if agent_run_id and result.cognition:
            emit_agent_event(
                "cognition_final",
                round_id=agent_run_id,
                cognition=result.cognition,
            )
        if parsed_xml.errors:
            logger.warning(
                "[%s] 工具调用协议错误: %s",
                self.provider,
                "; ".join(parsed_xml.errors),
            )
        if parsed_xml.repairs:
            logger.warning(
                "[%s] 工具调用已自动修复: %s",
                self.provider,
                "; ".join(parsed_xml.repairs),
            )
        tool_calls = parsed_xml.tool_calls
        if agent_run_id:
            for index, tc in enumerate(tool_calls, start=1):
                args: dict[str, Any] = {}
                try:
                    parsed_args = json.loads(tc.function.arguments or "{}")
                    if isinstance(parsed_args, dict):
                        args = parsed_args
                except Exception:
                    args = {}
                emit_agent_event(
                    "tool_planned",
                    round_id=agent_run_id,
                    call_id=tc.id,
                    tool_index=index,
                    tool_name=tc.function.name,
                    args=args,
                    args_preview=summarize_tool_payload(args),
                )

        tool_calls_count = len(tool_calls)
        logger.info(
            "[%s] 模型响应 — 工具调用数: %d",
            self.provider,
            tool_calls_count,
        )
        if tool_calls_count > 0:
            logger.info(
                "[%s] 模型请求的工具: %s",
                self.provider,
                ", ".join(tc.function.name for tc in tool_calls),
            )

        passive_duplicate_tools = is_passive_duplicate_tool_set(
            tuple(tc.function.name for tc in tool_calls)
        )
        if duplicate_guard_cfg["enabled"] and flow is not None and not passive_duplicate_tools:
            current_norm = normalize_response_text(
                raw_response_text,
                normalize_whitespace=duplicate_guard_cfg["normalize_whitespace"],
            )
            duplicate_count = 1
            for previous_raw in reversed(flow.recent_raw_responses(duplicate_guard_cfg["lookback_rounds"])):
                previous_norm = normalize_response_text(
                    previous_raw,
                    normalize_whitespace=duplicate_guard_cfg["normalize_whitespace"],
                )
                if previous_norm == current_norm and current_norm:
                    duplicate_count += 1
                else:
                    break
            if duplicate_count > 1:
                result.duplicate_model_response = True
                result.duplicate_model_response_count = duplicate_count
                result.duplicate_model_response_error = build_duplicate_model_response_error(
                    duplicate_count=duplicate_count,
                    max_retries=duplicate_guard_cfg["max_retries"],
                )
                logger.warning(
                    "[%s] duplicate model response detected count=%s max_retries=%s; tools not executed",
                    self.provider,
                    duplicate_count,
                    duplicate_guard_cfg["max_retries"],
                )
                duplicate_calls: list[ToolCall] = []
                duplicate_responses: list[ToolResponse] = []
                for tc in tool_calls:
                    call_args: dict = {}
                    try:
                        parsed_args = json.loads(tc.function.arguments or "{}")
                        if isinstance(parsed_args, dict):
                            call_args = parsed_args
                    except Exception:
                        call_args = {}
                    duplicate_calls.append(
                        ToolCall(
                            name=tc.function.name,
                            args=call_args,
                            call_id=tc.id,
                        )
                    )
                    duplicate_responses.append(
                        ToolResponse(
                            name=tc.function.name,
                            response=result.duplicate_model_response_error,
                            call_id=tc.id,
                        )
                    )
                flow.prune(max_rounds)
                flow.append_round(
                    duplicate_calls,
                    duplicate_responses,
                    cognition=result.cognition,
                    raw_response=result.raw_response,
                )
                return result

        if not tool_collection.has_active_tools():
            logger.error("[%s] 工具注册表为空，无法继续 XML 工具调用", self.provider)
            raise LLMCallFailed("工具注册表为空，无法继续 XML 工具调用")

        result.had_tool_call = bool(tool_calls)
        if not tool_calls:
            return result

        if _runtime_is_stale():
            return _abort_for_runtime_reset()

        executor = ToolExecutor(
            provider_name=self.provider,
            tool_collection=tool_collection,
            flow=flow,
            runtime_stale_checker=runtime_stale_checker,
            decision_world=user_content,
            current_world_provider=current_world_provider,
            agent_run_id=agent_run_id,
        )
        try:
            tool_outcome = executor.execute(tool_calls, inner_state=result.inner_state)
        except RuntimeResetAborted:
            return _abort_for_runtime_reset()

        result.tool_calls_log = tool_outcome.tool_calls_log
        if agent_run_id:
            emit_agent_event(
                "tools_collected",
                round_id=agent_run_id,
                tool_count=len(result.tool_calls_log),
                tools=[item.get("function") for item in result.tool_calls_log],
            )

        if _runtime_is_stale():
            return _abort_for_runtime_reset()

        if flow:
            flow.prune(max_rounds)
            flow.append_round(
                tool_outcome.round_calls,
                tool_outcome.round_responses,
                cognition=result.cognition,
                raw_response=result.raw_response,
            )

        return result

    def call_simple_text(
        self,
        system_prompt: str,
        user_content: str,
        gen: dict,
        log_tag: str = "slow_thinking",
    ) -> str | None:
        """纯文本生成（不带工具调用）。返回模型输出文本，失败返回 None。"""
        gen = self._normalize_generation_for_transport(gen)
        log_prompt(self.provider, system_prompt, user_content)
        extra_body = gen.get("extra_body") or {}
        feature, subfeature = _simple_text_usage_scope(log_tag)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": gen.get("temperature", 1.0),
            "max_tokens": gen.get("max_output_tokens", 10000),
            **(({"extra_body": extra_body}) if extra_body else {}),
        }
        add_extra_generation_kwargs(create_kwargs, gen)
        add_enabled_sampling_kwargs(create_kwargs, gen)
        save_prompt_snapshot(
            getattr(self, "_prompt_snapshot_cfg", {"enabled": False}),
            request_kind="simple_text",
            provider=self.provider,
            model=self.model,
            messages=messages,
            create_kwargs=create_kwargs,
            feature=feature,
            subfeature=subfeature or log_tag,
            context={"log_tag": log_tag},
        )
        try:
            response = self._create_chat_completion(
                all_messages=messages,
                create_kwargs=create_kwargs,
            )
        except Exception as exc:
            _record_usage_event(
                provider=self.provider,
                model=self.model,
                feature=feature,
                subfeature=subfeature,
                usage=None,
                status="error",
            )
            logger.warning("[%s/%s] 文本生成异常: %s", self.provider, log_tag, exc)
            return None

        if response is None:
            logger.warning("[%s/%s] response 为 None", self.provider, log_tag)
            return None

        tag = f"{self.provider}/{log_tag}"
        usage = getattr(response, "usage", None)
        _record_usage_event(
            provider=self.provider,
            model=self.model,
            feature=feature,
            subfeature=subfeature,
            usage=usage,
            status="success" if response.choices else "empty_choices",
        )
        usage_counts = parse_usage(usage)
        if usage_counts["usage_available"]:
            logger.info(
                "[%s] token — 输入: %d, 输出: %d",
                tag,
                usage_counts["input_tokens"],
                usage_counts["output_tokens"],
            )
        if not response.choices:
            logger.warning("[%s] response.choices 为空", tag)
            return None

        _log_finish_reason(tag, response)
        text = response.choices[0].message.content or ""
        log_response(self.provider, text)
        return text.strip() or None

    def _call_forced_tool(
        self,
        system_prompt: str,
        user_content: str | list,
        gen: dict,
        tool_decl: dict | InternalToolSpec,
        log_tag: str = "IS",
    ) -> dict | None:
        """单工具函数调用路径：依赖 prompt 引导工具调用，返回其参数 dict。失败返回 None。"""
        gen = self._normalize_generation_for_transport(gen)
        if not self._vision_enabled:
            user_content = _strip_images(user_content)

        if isinstance(tool_decl, InternalToolSpec):
            declaration = tool_decl.declaration
            schema_repairer = tool_decl.schema_repairer
            semantic_sanitizer = tool_decl.semantic_sanitizer
        else:
            declaration = tool_decl
            schema_repairer = None
            semantic_sanitizer = None

        log_prompt(self.provider, system_prompt, user_content)

        tool_name = declaration["name"]
        extra_body = gen.get("extra_body") or {}
        feature, subfeature = _forced_tool_usage_scope(log_tag)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": cast(Any, user_content)},
        ]
        tools = OpenAICompatClient.to_openai_tools([declaration])
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
            "temperature": gen.get("temperature", 0.3),
            "max_tokens": gen.get("max_output_tokens", 10000),
            **(({"extra_body": extra_body}) if extra_body else {}),
        }
        add_extra_generation_kwargs(create_kwargs, gen)
        add_enabled_sampling_kwargs(create_kwargs, gen)
        save_prompt_snapshot(
            getattr(self, "_prompt_snapshot_cfg", {"enabled": False}),
            request_kind="forced_tool",
            provider=self.provider,
            model=self.model,
            messages=messages,
            create_kwargs=create_kwargs,
            feature=feature,
            subfeature=subfeature or log_tag,
            context={"log_tag": log_tag, "tool_name": tool_name},
        )
        try:
            response = self._create_chat_completion(
                all_messages=messages,
                create_kwargs=create_kwargs,
            )
        except Exception:
            _record_usage_event(
                provider=self.provider,
                model=self.model,
                feature=feature,
                subfeature=subfeature,
                usage=None,
                status="error",
            )
            raise

        if response is None:
            logger.warning("[%s/%s] response 为 None", self.provider, log_tag)
            return None

        tag = f"{self.provider}/{log_tag}"
        usage = getattr(response, "usage", None)
        status = "success"
        if not response.choices:
            status = "empty_choices"
        elif not response.choices[0].message.tool_calls:
            status = "no_tool_call"
        _record_usage_event(
            provider=self.provider,
            model=self.model,
            feature=feature,
            subfeature=subfeature,
            usage=usage,
            status=status,
        )
        usage_counts = parse_usage(usage)
        if usage_counts["usage_available"]:
            logger.info(
                "[%s] token — 输入: %d, 输出: %d, 总计: %d",
                tag,
                usage_counts["input_tokens"],
                usage_counts["output_tokens"],
                usage_counts["total_tokens"],
            )

        if not response.choices:
            logger.warning("[%s] response.choices 为空", tag)
            return None

        msg = response.choices[0].message
        if not msg.tool_calls:
            logger.warning("[%s] 模型未返回函数调用", tag)
            return None

        args_json = msg.tool_calls[0].function.arguments  # type: ignore[union-attr]
        log_response(self.provider, args_json)
        parsed_args, ok = parse_tool_arguments(
            args_json,
            tool_name,
            tag,
            declaration,
            schema_repairer,
            semantic_sanitizer,
        )
        if ok:
            return parsed_args
        return None
