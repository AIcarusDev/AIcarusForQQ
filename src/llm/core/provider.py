"""provider.py — 基于 OpenAI 兼容接口的统一模型适配层。"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from ..circuit_breaker import ToolRepeatBreaker
from .profiles import resolve_openai_profile
from .tool_calling import build_tool_argument_error, parse_tool_arguments, process_tool_arguments
from consciousness import ConsciousnessFlow, ToolCall, ToolResponse
from log_config import log_prompt, log_response

logger = logging.getLogger("AICQ.provider")

RETRY_ON_NEW_MESSAGE_ACTION = "_needs_retry"
RETRY_ON_EMPTY_TOOL_CALL_ACTION = "_needs_retry_no_tool_call"


class LLMCallFailed(Exception):
    """LLM 调用最终失败（预留给上层统一捕获）。"""


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


_EXIT_TOOLS: frozenset[str] = frozenset({"sleep", "wait", "shift"})


class OpenAICompatAdapter:
    """使用 OpenAI SDK 调用 OpenAI 兼容端点。"""

    def __init__(self, cfg: dict):
        profile_name, profile_cfg, _profiles = resolve_openai_profile(cfg)

        base_url = profile_cfg.get("base_url", "")
        env_key = profile_cfg.get("api_key_env", "")
        api_key = os.getenv(env_key, "") if env_key else ""
        if not api_key and not profile_cfg.get("requires_api_key", True):
            api_key = "openai-compat"

        proxy_url = os.getenv("OPENAI_PROXY", "").strip() or None
        client_kwargs: dict = {"api_key": api_key, "base_url": base_url}
        if proxy_url:
            client_kwargs["http_client"] = httpx.Client(proxy=proxy_url)

        self.client = OpenAI(**client_kwargs)
        self.model = cfg.get("model") or profile_cfg.get("default_model", "")
        self.profile = profile_name
        self.provider = profile_name
        self._vision_enabled: bool = bool(cfg.get("vision", True))

    def list_models(self) -> list[str]:
        """返回该 provider 可用的模型 ID 列表。"""
        try:
            page = self.client.models.list()
            return sorted(model.id for model in page.data)
        except Exception:
            return []

    def call(
        self,
        system_prompt_builder,
        user_content: "str | list",
        gen: dict,
        tool_collection=None,
        user_content_refresher=None,
        flow: "ConsciousnessFlow | None" = None,
        new_message_checker=None,
    ) -> "tuple[dict | None, list[dict], str]":
        """调用 OpenAI 兼容 API。"""
        return self._call_main_model(
            system_prompt_builder,
            user_content,
            gen,
            tool_collection,
            user_content_refresher,
            flow,
            new_message_checker,
        )

    def _call_main_model(
        self,
        system_prompt_builder,
        user_content: "str | list",
        gen: dict,
        tool_collection,
        user_content_refresher,
        flow: "ConsciousnessFlow | None",
        new_message_checker=None,
    ) -> "tuple[dict | None, list[dict], str]":
        """主模型纯 function calling 路径，通过 ConsciousnessFlow 管理意识流。"""
        if tool_collection is None:
            from tools.specs import ToolCollection

            tool_collection = ToolCollection()
        else:
            tool_collection = tool_collection.clone()

        breaker = ToolRepeatBreaker()
        tool_calls_log: list[dict] = []
        tool_round = 0
        max_rounds: int = gen.get("llm_contents_max_rounds", 15)

        if not self._vision_enabled:
            user_content = _strip_images(user_content)

        full_system = system_prompt_builder(
            tool_collection.active_names(),
            tool_collection.latent_names(),
        )
        log_prompt(self.provider, full_system, user_content)

        user_msg: ChatCompletionMessageParam = {"role": "user", "content": user_content}
        system_msg: ChatCompletionMessageParam = {"role": "system", "content": full_system}

        available_tools = self._to_openai_tools(tool_collection.active_declarations())
        create_kwargs: dict = {
            "model": self.model,
            "temperature": gen.get("temperature", 1.0),
            "max_tokens": gen.get("max_output_tokens", 8192),
            "presence_penalty": gen.get("presence_penalty", 0.0),
            "frequency_penalty": gen.get("frequency_penalty", 0.0),
        }
        if available_tools:
            create_kwargs["tools"] = available_tools
            create_kwargs["tool_choice"] = "auto"

        prompt_tokens = 0
        output_tokens = 0

        while True:
            all_messages = [system_msg] + (flow.to_openai_messages() if flow else []) + [user_msg]
            response = self.client.chat.completions.create(
                messages=all_messages,  # type: ignore
                **create_kwargs,
            )

            if usage := response.usage:
                prompt_tokens += usage.prompt_tokens or 0
                output_tokens += usage.completion_tokens or 0

            if not response.choices:
                logger.warning("[%s] response.choices 为空", self.provider)
                return None, tool_calls_log, full_system

            msg = response.choices[0].message
            tool_calls_count = len(msg.tool_calls) if msg.tool_calls else 0
            logger.info(
                "[%s] 第 %d 轮模型响应 — 工具调用数: %d",
                self.provider,
                tool_round + 1,
                tool_calls_count,
            )
            if tool_calls_count > 0:
                logger.info(
                    "[%s] 模型请求的工具: %s",
                    self.provider,
                    ", ".join(tool_call.function.name for tool_call in msg.tool_calls),
                )

            if tool_round == 0 and new_message_checker is not None and new_message_checker():
                logger.info("[%s] 第 1 轮响应后检测到新消息，丢弃本次结果触发重调", self.provider)
                return {"action": RETRY_ON_NEW_MESSAGE_ACTION}, [], full_system

            if not tool_collection.has_active_tools():
                logger.error("[%s] 工具注册表为空，无法继续 function calling", self.provider)
                logger.info(
                    "[%s] Token 用量（全轮累计）— 输入: %d, 输出: %d, 总计: %d",
                    self.provider,
                    prompt_tokens,
                    output_tokens,
                    prompt_tokens + output_tokens,
                )
                raise LLMCallFailed("工具注册表为空，无法继续 function calling")

            if not msg.tool_calls:
                logger.warning("[%s] 模型未调用任何工具，本轮结果作废并请求上层重调", self.provider)
                logger.info(
                    "[%s] Token 用量（全轮累计）— 输入: %d, 输出: %d, 总计: %d",
                    self.provider,
                    prompt_tokens,
                    output_tokens,
                    prompt_tokens + output_tokens,
                )
                return {"action": RETRY_ON_EMPTY_TOOL_CALL_ACTION}, [], full_system

            tool_round += 1
            breaker.begin_round(tool_round)

            round_calls: list[ToolCall] = []
            round_responses: list[ToolResponse] = []
            pending_injections: list[str] = []
            exit_action: dict | None = None

            slots: list[dict] = []
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                spec = tool_collection.get_active(fn_name)
                handler = spec.handler if spec is not None else None
                processing = None
                args: dict = {}
                if spec is not None and handler is not None:
                    processing = process_tool_arguments(
                        tool_call.function.arguments,
                        fn_name,
                        self.provider,
                        spec.declaration,
                        spec.schema_repairer,
                        spec.semantic_sanitizer,
                    )
                    args = processing.args

                logger.debug(
                    "[%s] tool call 原文: %s(%s)",
                    self.provider,
                    fn_name,
                    tool_call.function.arguments or "{}",
                )
                slot: dict = {
                    "tc": tool_call,
                    "fn_name": fn_name,
                    "args": args,
                    "fn": handler,
                    "result": None,
                    "circuit_broken": False,
                }
                if breaker.check_and_record(fn_name, args):
                    slot["result"] = {
                        "error": (
                            "CIRCUIT_BREAKER_TRIPPED: "
                            f"tool='{fn_name}' consecutive_calls={breaker.max_streak} "
                            f"threshold={breaker.max_streak}. Tool call REJECTED and tool "
                            "REMOVED from registry. You MUST stop calling this tool and call "
                            "sleep/wait/shift to end activation."
                        )
                    }
                    logger.warning(
                        "[%s] 熔断触发: 工具 %s 连续 %d 轮相同调用，已拦截并移除",
                        self.provider,
                        fn_name,
                        breaker.max_streak,
                    )
                    tool_collection.remove_active(fn_name)
                    slot["circuit_broken"] = True
                elif handler is None:
                    slot["result"] = {"error": f"未知工具: {fn_name}"}
                elif processing is not None and not processing.ok:
                    slot["result"] = build_tool_argument_error(processing)
                slots.append(slot)

            round_calls = [
                ToolCall(name=slot["fn_name"], args=slot["args"], call_id=slot["tc"].id)
                for slot in slots
            ]

            provider_name = self.provider

            def _exec_openai(slot: dict) -> None:
                fn_name = slot["fn_name"]
                logger.info("[%s] 执行工具开始: %s", provider_name, fn_name)
                try:
                    slot["result"] = slot["fn"](**slot["args"])
                    if isinstance(slot["result"], dict) and slot["result"].get("error"):
                        logger.info(
                            "[%s] 执行工具完毕（失败）: %s — %s",
                            provider_name,
                            fn_name,
                            slot["result"]["error"],
                        )
                    else:
                        logger.info("[%s] 执行工具完毕（成功）: %s", provider_name, fn_name)
                except Exception as exc:
                    logger.warning("[%s] 执行工具异常: %s — %s", provider_name, fn_name, exc)
                    slot["result"] = {"error": str(exc)}

            pending_slots = [slot for slot in slots if slot["result"] is None]
            send_msg_slots = [slot for slot in pending_slots if slot["fn_name"] == "send_message"]
            parallel_slots = [slot for slot in pending_slots if slot["fn_name"] != "send_message"]
            if parallel_slots:
                with ThreadPoolExecutor(max_workers=len(parallel_slots)) as pool:
                    list(pool.map(_exec_openai, parallel_slots))
            for slot in send_msg_slots:
                _exec_openai(slot)

            for slot in slots:
                fn_name = slot["fn_name"]
                tool_call = slot["tc"]
                args = slot["args"]
                result_data = slot["result"]
                circuit_broken = slot["circuit_broken"]

                if not circuit_broken:
                    if isinstance(result_data, dict) and "_inject_tools" in result_data:
                        pending_injections.extend(result_data.pop("_inject_tools") or [])

                tool_calls_log.append(
                    {
                        "round": tool_round,
                        "function": fn_name,
                        "arguments": args,
                        "result": result_data,
                        "circuit_broken": circuit_broken,
                    }
                )

                if not circuit_broken and fn_name in _EXIT_TOOLS:
                    if fn_name == "shift":
                        if isinstance(result_data, dict) and result_data.get("ok"):
                            exit_action = {
                                "action": "shift",
                                "type": result_data.get("type"),
                                "id": result_data.get("id"),
                                "motivation": result_data.get("motivation", ""),
                            }
                    else:
                        exit_action = {"action": fn_name, **args}

                raw_multimodal_parts: list = []
                if (
                    not circuit_broken
                    and isinstance(result_data, dict)
                    and "_multimodal_parts" in result_data
                ):
                    raw_multimodal_parts = result_data.pop("_multimodal_parts")

                round_responses.append(
                    ToolResponse(
                        name=fn_name,
                        response=result_data,
                        call_id=tool_call.id,
                        multimodal_parts=raw_multimodal_parts,
                    )
                )

            for inj_name in pending_injections:
                injected_spec = tool_collection.activate(inj_name)
                if injected_spec is not None:
                    logger.info("[%s] 注入潜伏工具: %s", self.provider, inj_name)
                else:
                    logger.warning(
                        "[%s] 无法注入工具 %s：不在潜伏工具列表中或已激活",
                        self.provider,
                        inj_name,
                    )

            if flow:
                flow.prune(max_rounds)
                flow.append_round(round_calls, round_responses)

            if exit_action is not None:
                if tool_calls_log:
                    call_counts: dict[str, int] = {}
                    for entry in tool_calls_log:
                        call_counts[entry["function"]] = call_counts.get(entry["function"], 0) + 1
                    summary = ", ".join(f"{name}×{count}" for name, count in call_counts.items())
                    logger.info(
                        "[%s] 工具调用共 %d 轮 %d 次: %s",
                        self.provider,
                        tool_round,
                        len(tool_calls_log),
                        summary,
                    )
                logger.info(
                    "[%s] Token 用量（全轮累计）— 输入: %d, 输出: %d, 总计: %d",
                    self.provider,
                    prompt_tokens,
                    output_tokens,
                    prompt_tokens + output_tokens,
                )
                return exit_action, tool_calls_log, full_system

            available_tools = self._to_openai_tools(tool_collection.active_declarations())
            if available_tools:
                create_kwargs["tools"] = available_tools
            else:
                create_kwargs.pop("tools", None)
                create_kwargs.pop("tool_choice", None)

            updated_system = system_prompt_builder(
                tool_collection.active_names(),
                tool_collection.latent_names(),
            )
            system_msg = {"role": "system", "content": updated_system}

            if user_content_refresher is not None:
                fresh = user_content_refresher()
                if not self._vision_enabled:
                    fresh = _strip_images(fresh)
                user_msg = {"role": "user", "content": fresh}
                logger.info("[%s] 工具调用第 %d 轮后已刷新 user prompt", self.provider, tool_round)

    def _call_forced_tool(
        self,
        system_prompt: str,
        user_content: "str | list",
        gen: dict,
        tool_decl: dict,
        log_tag: str = "IS",
    ) -> "dict | None":
        """单工具函数调用路径：依赖 prompt 引导工具调用，返回其参数 dict。失败返回 None。"""
        if not self._vision_enabled:
            user_content = _strip_images(user_content)

        log_prompt(self.provider, system_prompt, user_content)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            tools=self._to_openai_tools([tool_decl]),  # type: ignore[arg-type]
            temperature=gen.get("temperature", 0.3),
            max_tokens=gen.get("max_output_tokens", 300),
        )

        tag = f"{self.provider}/{log_tag}"
        if usage := response.usage:
            logger.info(
                "[%s] token — 输入: %d, 输出: %d, 总计: %d",
                tag,
                usage.prompt_tokens or 0,
                usage.completion_tokens or 0,
                (usage.prompt_tokens or 0) + (usage.completion_tokens or 0),
            )

        if not response.choices:
            logger.warning("[%s] response.choices 为空", tag)
            return None

        msg = response.choices[0].message
        if not msg.tool_calls:
            logger.warning("[%s] 模型未返回函数调用", tag)
            return None

        tool_name = tool_decl["name"]
        args_json = msg.tool_calls[0].function.arguments  # type: ignore[union-attr]
        log_response(self.provider, args_json)
        parsed_args, ok = parse_tool_arguments(args_json, tool_name, tag, tool_decl)
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
                    "parameters": declaration.get("parameters", {}),
                },
            }
            for declaration in declarations
        ]


def create_adapter(cfg: dict):
    """根据 config 中的 OpenAI 兼容 profile 创建适配器。"""
    return OpenAICompatAdapter(cfg)


def build_is_adapter_cfg(main_cfg: dict, is_cfg: dict) -> dict:
    """构建 IS（中断哨兵）专用的 adapter 配置。"""
    cfg = dict(main_cfg)
    if "profile" in is_cfg:
        cfg["profile"] = is_cfg["profile"]
    elif "provider" in is_cfg:
        cfg["profile"] = is_cfg["provider"]
    if "base_url" in is_cfg:
        cfg["base_url"] = is_cfg["base_url"]
    cfg["model"] = is_cfg.get("model", main_cfg.get("model"))
    cfg["model_name"] = is_cfg.get("model_name", cfg["model"])
    if "generation" in is_cfg:
        cfg["generation"] = is_cfg["generation"]
    if "thinking" in is_cfg:
        cfg["thinking"] = is_cfg["thinking"]
    if "vision" in is_cfg:
        cfg["vision"] = is_cfg["vision"]
    return cfg