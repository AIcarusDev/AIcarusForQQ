"""provider.py — 基于 OpenAI 兼容接口的统一模型适配层。

每次 ``call_one_round()`` 仅完成一次 LLM 调用 + 本轮工具执行，
不再承担"循环到出口工具"的职责。多轮永动由 consciousness 主循环驱动。
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from consciousness.flow import ConsciousnessFlow, ToolCall, ToolResponse
from llm.compression.config import normalize_generation_config
from llm.prompt_snapshot import normalize_prompt_snapshot_config, save_prompt_snapshot

from .internal_tool import InternalToolSpec
from .round_context import reset_current_inner_state, set_current_inner_state
from .profiles import resolve_model_provider
from .tool_calling.common import strip_legacy_motivation_fields
from .tool_calling import build_tool_argument_error, parse_tool_arguments, process_tool_arguments
from .tool_calling.xml_protocol import (
    XML_TOOL_CALL_ERROR_NAME,
    build_tools_xml_message,
    parse_xml_tool_calls,
)
from log_config import log_cognition, log_prompt, log_response
from llm_usage_recorder import parse_usage, record_llm_usage
from tools.ordering import cacheable_tool_names

logger = logging.getLogger("AICQ.llm.provider")


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


def _simple_text_usage_scope(log_tag: str) -> tuple[str, str]:
    if log_tag.startswith("think_deeply/"):
        return "slow_thinking", log_tag.split("/", 1)[1]
    if log_tag == "cognition_compression":
        return "cognition_compression", ""
    return "simple_text", log_tag


def _forced_tool_usage_scope(log_tag: str) -> tuple[str, str]:
    if log_tag == "IS":
        return "interruption_sentinel", ""
    if log_tag == "archiver":
        return "memory_archiver", ""
    return "forced_tool", log_tag


def _is_stream_usage_option_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "stream_options" in text or "include_usage" in text


def _run_parallel_slots(parallel_slots: list[dict], executor, provider_name: str) -> None:
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


def _annotate_get_tools_result(result: object, args: dict, tool_collection) -> object:
    if not isinstance(result, dict):
        return result

    requested = args.get("tool_names")
    if not isinstance(requested, list):
        return result

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
        return result

    annotated = dict(result)
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
            + "。无需再次 get_tools。"
        )
    if newly_activated:
        annotated["newly_activated"] = newly_activated
    return annotated


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
    had_tool_call: bool = False
    # 第 1 轮思考结束时焦点会话出现新消息：调用方应丢弃本轮并立刻重调
    new_message_during_thinking: bool = False
    # API 调用本身失败 / response.choices 为空时为 True
    failed: bool = False


def _strip_images(user_content: "str | list") -> "str | list":
    """从多模态内容中剥除图片部分，仅保留文本。"""
    if not isinstance(user_content, list):
        return user_content
    if not (text_parts := [part for part in user_content if part.get("type") == "text"]):
        return ""
    return text_parts[0]["text"] if len(text_parts) == 1 else text_parts


def _message_content_to_text(raw_content: "str | list | None") -> str:
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


def _estimate_token_count(text: str) -> int:
    """Cheap mixed Chinese/ASCII token estimate for cache diagnostics."""
    ascii_chars = 0
    cjk_chars = 0
    other_chars = 0
    for char in text:
        codepoint = ord(char)
        if (
            0x4E00 <= codepoint <= 0x9FFF
            or 0x3400 <= codepoint <= 0x4DBF
            or 0x20000 <= codepoint <= 0x2A6DF
        ):
            cjk_chars += 1
        elif codepoint < 128:
            ascii_chars += 1
        else:
            other_chars += 1
    estimate = round((ascii_chars / 4) + cjk_chars + (other_chars / 2))
    return max(1, estimate)


def _build_prompt_cache_prefix(system_prompt: str, tool_collection) -> str:
    """Return the deterministic prompt prefix above the tool cache boundary."""
    cacheable_names = set(cacheable_tool_names())
    cacheable_declarations = [
        tool_collection.active_specs[name].declaration
        for name in tool_collection.active_names()
        if name in cacheable_names
    ]

    prefix_messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if cacheable_declarations:
        prefix_messages.append({
            "role": "user",
            "content": build_tools_xml_message(cacheable_declarations, []),
        })
    return json.dumps(prefix_messages, ensure_ascii=False, separators=(",", ":"))


def _snapshot_create_kwargs(
    create_kwargs: dict,
    *,
    streaming: bool,
    include_usage_requested: bool,
) -> dict:
    kwargs = dict(create_kwargs)
    if streaming:
        kwargs["stream"] = True
        if include_usage_requested:
            kwargs["stream_options"] = {"include_usage": True}
    return kwargs


class OpenAICompatAdapter:
    """使用 OpenAI SDK 调用 OpenAI 兼容端点。"""

    def __init__(self, cfg: dict):
        provider_name, provider_cfg, _providers = resolve_model_provider(cfg)
        model = (cfg.get("model") or "").strip()
        if not model:
            raise ValueError(f"模型供应商 {provider_name!r} 未绑定模型 ID")

        base_url = provider_cfg.get("base_url", "")
        env_key = provider_cfg.get("api_key_env", "")
        api_key = os.getenv(env_key, "") if env_key else ""
        if not api_key and not provider_cfg.get("requires_api_key", True):
            api_key = "openai-compat"

        proxy_url = os.getenv("OPENAI_PROXY", "").strip() or None
        client_kwargs: dict = {"api_key": api_key, "base_url": base_url}
        if proxy_url:
            client_kwargs["http_client"] = httpx.Client(proxy=proxy_url)

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.provider = provider_name
        self._vision_enabled: bool = bool(cfg.get("vision", True))
        self._stream_usage_unsupported: bool = False
        self._prompt_snapshot_cfg = normalize_prompt_snapshot_config(
            cfg.get("prompt_snapshots")
        )

    def list_models(self) -> list[str]:
        """返回该 provider 可用的模型 ID 列表。"""
        try:
            page = self.client.models.list()
            return sorted(model.id for model in page.data)
        except Exception:
            return []

    def _create_chat_completion(
        self,
        *,
        all_messages: list,
        create_kwargs: dict,
        new_message_checker=None,
    ):
        """发起 chat completion；需要可打断时改用 streaming 并主动 close。"""
        if new_message_checker is None:
            response = self.client.chat.completions.create(
                messages=all_messages,  # type: ignore
                **create_kwargs,
            )
            return response, False

        stream_kwargs = dict(create_kwargs)
        stream_kwargs["stream"] = True
        include_usage_requested = not self._stream_usage_unsupported
        if include_usage_requested:
            stream_kwargs["stream_options"] = {"include_usage": True}
        stream = None
        usage = None
        content_parts: list[str] = []

        try:
            try:
                stream = self.client.chat.completions.create(
                    messages=all_messages,  # type: ignore
                    **stream_kwargs,
                )
            except Exception as exc:
                if include_usage_requested and _is_stream_usage_option_error(exc):
                    self._stream_usage_unsupported = True
                    stream_kwargs.pop("stream_options", None)
                    logger.warning(
                        "[%s] streaming usage 选项不被兼容端点支持，降级重试: %s",
                        self.provider,
                        exc,
                    )
                    stream = self.client.chat.completions.create(
                        messages=all_messages,  # type: ignore
                        **stream_kwargs,
                    )
                else:
                    raise
            if new_message_checker():
                logger.info("[%s] 思考请求启动后检测到新消息，关闭 stream", self.provider)
                return None, True

            for chunk in stream:
                if chunk_usage := getattr(chunk, "usage", None):
                    usage = chunk_usage

                for choice in getattr(chunk, "choices", []) or []:
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    if content := getattr(delta, "content", None):
                        content_parts.append(content)

                if new_message_checker():
                    logger.info("[%s] 思考期间检测到新消息，关闭 stream", self.provider)
                    return None, True
        finally:
            if stream is not None and hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception:
                    logger.debug("[%s] 关闭 streaming response 失败", self.provider, exc_info=True)

        message = SimpleNamespace(
            content="".join(content_parts) if content_parts else None,
        )
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=message)]), False

    def call_one_round(
        self,
        system_prompt_builder,
        user_content: "str | list",
        gen: dict,
        tool_collection,
        flow: "ConsciousnessFlow | None" = None,
        new_message_checker=None,
        usage_feature: str = "main_round",
        usage_subfeature: str = "",
        prompt_snapshot_context: dict | None = None,
    ) -> RoundResult:
        """跑一轮 XML 文本工具协议：1 次 LLM 调用 + 本轮工具执行。

        - 不再尝试在内部往复多轮；外层（consciousness 主循环）负责持续调用。
        - 不识别"出口工具"：sleep/wait/shift 与其它工具完全等价，由它们的
          handler 自身阻塞或修改全局状态。
        - 工具执行结果与多模态附件统一写入 ``flow``（最近一轮）。
        - 若模型本轮一次工具都没调用，``had_tool_call=False``——调用方据此决定
          是重调还是硬注入兜底（参见 retry.py）。
        - 若 ``new_message_checker()`` 在 LLM 响应到达后返回 True，则丢弃本次
          响应、不写 flow、``new_message_during_thinking=True``——调用方应立刻
          重调一次。
        """
        if tool_collection is None:
            from tools.specs import ToolCollection
            tool_collection = ToolCollection()

        gen = normalize_generation_config(gen)
        max_rounds: int = gen["llm_contents_max_rounds"]

        if not self._vision_enabled:
            user_content = _strip_images(user_content)

        full_system = system_prompt_builder(
            tool_collection.active_names(),
            tool_collection.latent_names(),
        )
        log_prompt(self.provider, full_system, user_content)

        user_msg: ChatCompletionMessageParam = {"role": "user", "content": user_content}
        system_msg: ChatCompletionMessageParam = {"role": "system", "content": full_system}

        active_declarations = tool_collection.active_declarations()
        latent_names = tool_collection.latent_names()
        create_kwargs: dict = {
            "model": self.model,
            "temperature": gen.get("temperature", 1.0),
            "max_tokens": gen.get("max_output_tokens", 10000),
            "presence_penalty": gen.get("presence_penalty", 0.0),
            "frequency_penalty": gen.get("frequency_penalty", 0.0),
        }
        if extra_body := gen.get("extra_body"):
            create_kwargs["extra_body"] = extra_body

        # 写入思维链开关配置（默认开启）
        enable_thinking = gen.get("enable_thinking", True)
        extra_body = create_kwargs.setdefault("extra_body", {})
        extra_body["enable_thinking"] = enable_thinking

        result = RoundResult(system_prompt=full_system)

        tools_messages: list[dict] = []
        if active_declarations or latent_names:
            tools_messages.append({
                "role": "user",
                "content": build_tools_xml_message(active_declarations, latent_names),
            })
        if flow:
            flow.promote_ready_compression_summary(max_rounds)
        flow_messages = flow.to_xml_messages() if flow else []
        all_messages = [system_msg] + tools_messages + flow_messages + [user_msg]
        result.prompt_snapshot_id = save_prompt_snapshot(
            getattr(self, "_prompt_snapshot_cfg", {"enabled": False}),
            request_kind="main_round",
            provider=self.provider,
            model=self.model,
            messages=all_messages,
            create_kwargs=_snapshot_create_kwargs(
                create_kwargs,
                streaming=new_message_checker is not None,
                include_usage_requested=not getattr(
                    self, "_stream_usage_unsupported", False
                ),
            ),
            feature=usage_feature,
            subfeature=usage_subfeature,
            context=prompt_snapshot_context,
        )
        cache_prefix = _build_prompt_cache_prefix(full_system, tool_collection)
        previous_cache_prefix = getattr(self, "_last_main_cache_prefix", None)
        if previous_cache_prefix is not None and previous_cache_prefix == cache_prefix:
            logger.info(
                "[%s] prompt cache — 缓存有效，理论可命中约 %d token（预估）",
                self.provider,
                _estimate_token_count(cache_prefix),
            )
        else:
            logger.debug(
                "[%s] prompt cache — %s",
                self.provider,
                "首次记录缓存前缀" if previous_cache_prefix is None else "缓存前缀变化，本轮不标记有效",
            )
        self._last_main_cache_prefix = cache_prefix
        try:
            response, interrupted = self._create_chat_completion(
                all_messages=all_messages,
                create_kwargs=create_kwargs,
                new_message_checker=new_message_checker,
            )
        except Exception as exc:
            logger.warning("[%s] LLM API 调用异常: %s", self.provider, exc)
            _record_usage_event(
                provider=self.provider,
                model=self.model,
                feature=usage_feature,
                subfeature=usage_subfeature,
                usage=None,
                status="error",
            )
            # 失败时 dump 完整 messages 以便复现（仅当异常包含 image 相关字样时）
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
            return result

        if interrupted:
            _record_usage_event(
                provider=self.provider,
                model=self.model,
                feature=usage_feature,
                subfeature=usage_subfeature,
                usage=None,
                status="interrupted",
            )
            result.new_message_during_thinking = True
            return result

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
            return result

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
            return result

        msg = response.choices[0].message
        raw_response_text = _message_content_to_text(getattr(msg, "content", None))
        log_response(self.provider, raw_response_text)
        parsed_xml = parse_xml_tool_calls(raw_response_text)
        result.cognition = parsed_xml.cognition
        result.inner_state = _inner_state_from_cognition(parsed_xml.cognition)
        log_cognition(self.provider, result.cognition)
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

        # 思考期间焦点会话出现新消息：丢弃整个响应，调用方重调
        if new_message_checker is not None and new_message_checker():
            logger.info("[%s] 思考期间检测到新消息，丢弃本轮响应", self.provider)
            result.new_message_during_thinking = True
            return result

        if not tool_collection.has_active_tools():
            logger.error("[%s] 工具注册表为空，无法继续 XML 工具调用", self.provider)
            raise LLMCallFailed("工具注册表为空，无法继续 XML 工具调用")

        result.had_tool_call = bool(tool_calls)
        if not tool_calls:
            # 模型违规：一个工具都没调。不写 flow，留给调用方决策（重调 / 兜底 sleep）。
            return result

        # ── 解析、执行所有工具调用 ────────────────────────────────────────
        slots: list[dict] = []
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            protocol_error = getattr(tool_call, "protocol_error", None)
            spec = tool_collection.get_active(fn_name)
            latent_spec = tool_collection.get_latent(fn_name) if spec is None else None
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
                    self.provider,
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

        provider_name = self.provider

        def _exec_one(slot: dict) -> None:
            fn_name = slot["fn_name"]
            logger.info("[%s] 执行工具开始: %s", provider_name, fn_name)
            try:
                slot["result"] = slot["fn"](**slot["args"])
                if fn_name == "get_tools":
                    slot["result"] = _annotate_get_tools_result(
                        slot["result"],
                        slot["args"],
                        tool_collection,
                    )
                if isinstance(slot["result"], dict) and slot["result"].get("error"):
                    logger.info(
                        "[%s] 执行工具完毕（失败）: %s — %s",
                        provider_name, fn_name, slot["result"]["error"],
                    )
                else:
                    logger.info("[%s] 执行工具完毕（成功）: %s", provider_name, fn_name)
            except Exception as exc:
                logger.warning("[%s] 执行工具异常: %s — %s", provider_name, fn_name, exc)
                slot["result"] = {"error": str(exc)}

        # "输出类"工具（向用户发送内容）优先串行执行，之后再并行执行其余工具（含 sleep/wait/shift）。
        # 这样可以保证：当模型同时调用 send_message/send_voice_message/poke + wait 时，
        # 消息/语音/戳一戳先完成，wait 再开始计时等待，避免 early_trigger 在消息发出前就命中。
        _OUTPUT_FIRST_TOOLS = frozenset({
            "send_message",
            "send_voice_message",
            "recall_message",
            "poke",
        })
        _TERMINAL_CONTROL_TOOLS = frozenset({
            "restart_self",
        })
        pending_slots = [slot for slot in slots if slot["result"] is None]
        has_shift = any(slot["fn_name"] == "shift" for slot in pending_slots)
        output_slots = [slot for slot in pending_slots if slot["fn_name"] in _OUTPUT_FIRST_TOOLS]
        if has_shift and output_slots:
            for slot in output_slots:
                slot["result"] = {
                    "ok": False,
                    "error": (
                        "本轮同时包含 shift 和响应式发送工具；系统暂没有兼容此种情况。"
                    ),
                    "tool_not_executed": True,
                    "incompatible_with": "shift",
                }
            for slot in pending_slots:
                if slot["fn_name"] == "shift" or slot["fn_name"] in _OUTPUT_FIRST_TOOLS:
                    continue
                slot["result"] = {
                    "ok": False,
                    "error": "本轮同时包含 shift 和响应式发送工具；已只执行 shift，本工具跳过。",
                    "tool_not_executed": True,
                    "skipped_due_to": "shift_output_tool_conflict",
                    "interrupted": True,
                }
            pending_slots = [slot for slot in slots if slot["result"] is None]
            output_slots = []
        non_output_slots = [
            slot for slot in pending_slots
            if slot["fn_name"] not in _OUTPUT_FIRST_TOOLS
        ]
        terminal_slots = [
            slot for slot in non_output_slots
            if slot["fn_name"] in _TERMINAL_CONTROL_TOOLS
        ]
        parallel_slots = [
            slot for slot in non_output_slots
            if slot["fn_name"] not in _TERMINAL_CONTROL_TOOLS
        ]
        inner_state_token = set_current_inner_state(result.inner_state)
        try:
            for slot in output_slots:
                _exec_one(slot)
            restart_scheduled = False
            for slot in terminal_slots:
                _exec_one(slot)
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
                _run_parallel_slots(parallel_slots, _exec_one, provider_name)
        finally:
            reset_current_inner_state(inner_state_token)

        # ── 收集结果，写入 flow / log ─────────────────────────────────────
        round_calls: list[ToolCall] = [
            ToolCall(name=slot["fn_name"], args=slot["args"], call_id=slot["tc"].id)
            for slot in slots
            if not slot.get("protocol_error")
        ]
        round_responses: list[ToolResponse] = []

        for slot in slots:
            fn_name = slot["fn_name"]
            tool_call = slot["tc"]
            args = slot["args"]
            result_data = slot["result"]

            # _inject_tools 仅作清理：下一 round build_tools 会基于意识流自然恢复 latent 工具，
            # provider 不再承担显式 activate 职责。
            if isinstance(result_data, dict):
                result_data.pop("_inject_tools", None)

            result.tool_calls_log.append({
                "function": fn_name,
                "arguments": args,
                "result": result_data,
            })

            raw_multimodal_parts: list = []
            if isinstance(result_data, dict) and "_multimodal_parts" in result_data:
                raw_multimodal_parts = result_data.pop("_multimodal_parts")

            round_responses.append(
                ToolResponse(
                    name=XML_TOOL_CALL_ERROR_NAME if slot.get("protocol_error") else fn_name,
                    response=result_data,
                    call_id=tool_call.id,
                    multimodal_parts=raw_multimodal_parts,
                )
            )

        if flow:
            flow.prune(max_rounds)
            flow.append_round(round_calls, round_responses, cognition=result.cognition)

        return result

    # ── 纯文本生成路径（无工具调用，供 slow_thinking 等使用） ──────────

    def call_simple_text(
        self,
        system_prompt: str,
        user_content: str,
        gen: dict,
        log_tag: str = "slow_thinking",
    ) -> "str | None":
        """纯文本生成（不带工具调用）。返回模型输出文本，失败返回 None。"""
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
            response = self.client.chat.completions.create(
                messages=messages,
                **create_kwargs,
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

        text = response.choices[0].message.content or ""
        log_response(self.provider, text)
        return text.strip() or None

    # ── 兼容旧调用点（forced single tool 路径仍在 IS 中使用） ─────────

    def _call_forced_tool(
        self,
        system_prompt: str,
        user_content: "str | list",
        gen: dict,
        tool_decl: "dict | InternalToolSpec",
        log_tag: str = "IS",
    ) -> "dict | None":
        """单工具函数调用路径：依赖 prompt 引导工具调用，返回其参数 dict。失败返回 None。"""
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
        tools = self._to_openai_tools([declaration])  # type: ignore[arg-type]
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
            "temperature": gen.get("temperature", 0.3),
            "max_tokens": gen.get("max_output_tokens", 10000),
            **(({"extra_body": extra_body}) if extra_body else {}),
        }
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
            response = self.client.chat.completions.create(
                messages=messages,
                **create_kwargs,
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

    @staticmethod
    def _to_openai_tools(declarations: list[dict]) -> list[dict]:
        """将工具声明转为 OpenAI function calling 格式。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": declaration["name"],
                    "description": declaration.get("description", ""),
                    "parameters": OpenAICompatAdapter._strip_extensions(
                        declaration.get("parameters", {})
                    ),
                },
            }
            for declaration in declarations
        ]

    @staticmethod
    def _strip_extensions(obj: object) -> object:
        """递归去除 JSON Schema 中以 x- 开头的自定义扩展键，避免传入 LLM prompt。"""
        if isinstance(obj, dict):
            return {
                k: OpenAICompatAdapter._strip_extensions(v)
                for k, v in obj.items()
                if not k.startswith("x-")
            }
        if isinstance(obj, list):
            return [OpenAICompatAdapter._strip_extensions(item) for item in obj]
        return obj


def create_adapter(cfg: dict):
    """根据 config 中的 OpenAI 兼容模型供应商创建适配器。"""
    return OpenAICompatAdapter(cfg)


def _clean_model_text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _build_explicit_adapter_cfg(main_cfg: dict, model_cfg: dict, label: str) -> dict:
    provider = _clean_model_text(model_cfg.get("provider"))
    model = _clean_model_text(model_cfg.get("model"))
    if not provider or not model:
        raise ValueError(f"{label} 必须显式配置 provider 和 model")

    cfg = dict(main_cfg)
    cfg.pop("model_name", None)
    cfg.pop("profile", None)
    cfg.pop("base_url", None)
    cfg.pop("api_key_env", None)
    cfg["provider"] = provider
    cfg["model"] = model
    if "generation" in model_cfg:
        cfg["generation"] = model_cfg["generation"]
    if "vision" in model_cfg:
        cfg["vision"] = model_cfg["vision"]
    return cfg


def build_is_adapter_cfg(main_cfg: dict, is_cfg: dict) -> dict:
    """构建 IS（中断哨兵）专用的 adapter 配置。"""
    return _build_explicit_adapter_cfg(main_cfg, is_cfg, "IS 中断哨兵")


def build_slow_thinking_adapter_cfg(main_cfg: dict, st_cfg: dict) -> dict:
    """构建 slow_thinking 专用的 adapter 配置。"""
    return _build_explicit_adapter_cfg(main_cfg, st_cfg, "慢思考模型")


def build_archiver_adapter_cfg(main_cfg: dict, archiver_cfg: dict) -> dict:
    """构建记忆提取（archiver）专用的 adapter 配置。"""
    return _build_explicit_adapter_cfg(main_cfg, archiver_cfg, "记忆归档模型")


def build_compression_adapter_cfg(main_cfg: dict, compression_cfg: dict) -> dict:
    """构建上下文压缩专用的 adapter 配置。"""
    return _build_explicit_adapter_cfg(main_cfg, compression_cfg, "上下文压缩模型")
