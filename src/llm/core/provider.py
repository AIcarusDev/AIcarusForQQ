"""provider.py — 基于 OpenAI 兼容接口的统一模型适配层。"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import httpx
from jsonschema import ValidationError, validate
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from ..circuit_breaker import ToolRepeatBreaker
from .json_repair import clean_and_parse
from .profiles import resolve_openai_profile
from consciousness import ConsciousnessFlow, ToolCall, ToolResponse
from log_config import log_prompt, log_response

logger = logging.getLogger("AICQ.provider")


class LLMCallFailed(Exception):
    """LLM 调用最终失败（预留给上层统一捕获）。"""


def _schema_to_prompt(schema: dict, *, with_tools: bool = False) -> str:
    """将 JSON Schema 转为 system prompt 中的格式约束说明。"""
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)

    if with_tools:
        return (
            "## 严格输出格式\n"
            "你可以选择调用上述工具来获取信息，或直接生成回复。\n"
            "**如果选择不调用工具，你的回复必须是且仅是一个合法的 JSON 对象。**\n"
            "对象结构必须严格遵循下述 JSON Schema，不得包含任何 Markdown 代码块标记或额外文字。\n"
            "严格遵循以下 JSON Schema：\n"
            f"{schema_json}"
        )

    return (
        "## 严格输出格式\n"
        "你的回复必须是且仅是一个合法的 JSON 对象，尤其注意对象结构的正确性，"
        "不得包含任何 Markdown 代码块标记或额外文字。\n"
        "严格遵循以下 JSON Schema：\n"
        f"{schema_json}"
    )


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


_EXIT_TOOLS: frozenset[str] = frozenset({"idle", "wait", "shift"})


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
        self._supports_response_format: bool = bool(
            profile_cfg.get("supports_response_format", True)
        )
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
        schema: dict | None = None,
        tool_declarations: list | None = None,
        tool_registry: dict | None = None,
        latent_registry: dict | None = None,
        user_content_refresher=None,
        log_tag: str = "IS",
        flow: "ConsciousnessFlow | None" = None,
    ) -> "tuple[dict | None, list[dict], str]":
        """调用 OpenAI 兼容 API。"""
        if schema is None:
            return self._call_main_model(
                system_prompt_builder,
                user_content,
                gen,
                tool_declarations,
                tool_registry,
                latent_registry,
                user_content_refresher,
                flow,
            )
        return self._call_structured_output(
            system_prompt_builder,
            user_content,
            gen,
            schema,
            log_tag=log_tag,
        )

    def _call_main_model(
        self,
        system_prompt_builder,
        user_content: "str | list",
        gen: dict,
        tool_declarations: list | None,
        tool_registry: dict | None,
        latent_registry: dict | None,
        user_content_refresher,
        flow: "ConsciousnessFlow | None",
    ) -> "tuple[dict | None, list[dict], str]":
        """主模型纯 function calling 路径，通过 ConsciousnessFlow 管理意识流。"""
        tool_declarations = list(tool_declarations or [])
        tool_registry = dict(tool_registry or {})
        latent_registry = dict(latent_registry or {})

        breaker = ToolRepeatBreaker()
        tool_calls_log: list[dict] = []
        tool_round = 0
        max_rounds: int = gen.get("llm_contents_max_rounds", 15)

        if not self._vision_enabled:
            user_content = _strip_images(user_content)

        full_system = system_prompt_builder(
            [decl.get("name", "") for decl in tool_declarations],
            list(latent_registry.keys()),
        )
        log_prompt(self.provider, full_system, user_content)

        user_msg: ChatCompletionMessageParam = {"role": "user", "content": user_content}
        system_msg: ChatCompletionMessageParam = {"role": "system", "content": full_system}

        available_tools = self._to_openai_tools(tool_declarations)
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

            if not msg.tool_calls or not tool_registry:
                logger.warning("[%s] 模型未调用任何工具，隐式 idle", self.provider)
                logger.info(
                    "[%s] Token 用量（全轮累计）— 输入: %d, 输出: %d, 总计: %d",
                    self.provider,
                    prompt_tokens,
                    output_tokens,
                    prompt_tokens + output_tokens,
                )
                return {"action": "idle", "motivation": ""}, tool_calls_log, full_system

            tool_round += 1
            breaker.begin_round(tool_round)

            round_calls: list[ToolCall] = []
            round_responses: list[ToolResponse] = []
            pending_injections: list[str] = []
            exit_action: dict | None = None

            slots: list[dict] = []
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                handler = tool_registry.get(fn_name)
                try:
                    args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except (ValueError, json.JSONDecodeError):
                    args = {}

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
                            "idle/wait/shift to end activation."
                        )
                    }
                    logger.warning(
                        "[%s] 熔断触发: 工具 %s 连续 %d 轮相同调用，已拦截并移除",
                        self.provider,
                        fn_name,
                        breaker.max_streak,
                    )
                    tool_registry.pop(fn_name, None)
                    tool_declarations[:] = [
                        decl for decl in tool_declarations if decl.get("name") != fn_name
                    ]
                    slot["circuit_broken"] = True
                elif handler is None:
                    slot["result"] = {"error": f"未知工具: {fn_name}"}
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
                if inj_name in latent_registry:
                    inj_decl, inj_handler = latent_registry.pop(inj_name)
                    tool_declarations.append(inj_decl)
                    tool_registry[inj_name] = inj_handler
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

            available_tools = self._to_openai_tools(tool_declarations)
            if available_tools:
                create_kwargs["tools"] = available_tools
            else:
                create_kwargs.pop("tools", None)
                create_kwargs.pop("tool_choice", None)

            updated_system = system_prompt_builder(
                [decl.get("name", "") for decl in tool_declarations],
                list(latent_registry.keys()),
            )
            system_msg = {"role": "system", "content": updated_system}

            if user_content_refresher is not None:
                fresh = user_content_refresher()
                if not self._vision_enabled:
                    fresh = _strip_images(fresh)
                user_msg = {"role": "user", "content": fresh}
                logger.info("[%s] 工具调用第 %d 轮后已刷新 user prompt", self.provider, tool_round)

    def _call_structured_output(
        self,
        system_prompt_builder,
        user_content: "str | list",
        gen: dict,
        schema: dict,
        log_tag: str = "IS",
    ) -> "tuple[dict | None, list[dict], str]":
        """结构化 JSON 输出路径，使用局部 messages（不持久化）。"""
        tool_calls_log: list[dict] = []

        if not self._vision_enabled:
            user_content = _strip_images(user_content)

        base_system = system_prompt_builder([], [])
        full_system = base_system + "\n\n" + _schema_to_prompt(schema, with_tools=False)
        log_prompt(self.provider, full_system, user_content)

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_content},
        ]

        create_kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": gen.get("temperature", 1.0),
            "max_tokens": gen.get("max_output_tokens", 8192),
            "presence_penalty": gen.get("presence_penalty", 0.0),
            "frequency_penalty": gen.get("frequency_penalty", 0.0),
        }
        if self._supports_response_format:
            create_kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**create_kwargs)

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
            return None, tool_calls_log, full_system

        text = _message_content_to_text(response.choices[0].message.content)
        log_response(self.provider, text)

        if not text:
            logger.warning("[%s] response.content 为空", tag)
            return None, tool_calls_log, full_system

        result, _repaired = self._parse_and_validate_json(text, schema, gen)
        return result, tool_calls_log, full_system

    def _parse_and_validate_json(self, text: str, schema: dict, gen: dict) -> tuple[dict, bool]:
        """解析并校验 JSON，支持错误时自动修复。"""
        max_repair = gen.get("json_self_repair_retries", 1)
        try:
            result, repaired = clean_and_parse(text, f"[{self.provider}]")
            validate(instance=result, schema=schema)
            status = "已修复" if repaired else "原始"
            logger.info("[%s] JSON 解析成功（%s） — %d 个顶层字段", self.provider, status, len(result))
            return result, repaired
        except (json.JSONDecodeError, ValidationError) as parse_error:
            if max_repair <= 0:
                raise

            error_msg = f"{type(parse_error).__name__}: {parse_error}"
            logger.warning(
                "[%s] JSON 解析或校验失败 (%s)，启动 LLM 自修复（最大 %d 次）",
                self.provider,
                error_msg,
                max_repair,
            )

            last_err = parse_error
            for attempt in range(1, max_repair + 1):
                logger.info("[%s] JSON 自修复第 %d/%d 次", self.provider, attempt, max_repair)
                try:
                    error_detail = ""
                    if isinstance(last_err, ValidationError):
                        error_detail = f"Schema Validation Failed: {last_err.message}"
                    elif isinstance(last_err, json.JSONDecodeError):
                        error_detail = f"JSON Parse Error: {last_err.msg}"

                    repaired_raw = self._call_json_repair(text, schema, error_detail)
                    result, _ = clean_and_parse(
                        repaired_raw,
                        f"[{self.provider}][self_repair#{attempt}]",
                    )
                    validate(instance=result, schema=schema)
                    logger.info("[%s] JSON 自修复第 %d 次成功", self.provider, attempt)
                    return result, True
                except (json.JSONDecodeError, ValidationError) as exc:
                    last_err = exc
                    logger.warning(
                        "[%s] JSON 自修复第 %d/%d 次仍失败: %s",
                        self.provider,
                        attempt,
                        max_repair,
                        exc,
                    )

            logger.error("[%s] JSON 自修复 %d 次全部失败，放弃", self.provider, max_repair)
            raise last_err

    def _call_json_repair(self, raw_text: str, schema: dict, error_detail: str = "") -> str:
        """LLM 自修复：将无法解析的原始输出发给模型，要求转化为合法 JSON。"""
        schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
        repair_prompt = (
            "以下文本应为合法 JSON 对象，但解析或校验失败。请将其修复为符合下述 JSON Schema 的合法 JSON，"
            "只输出 JSON，不得包含任何 Markdown 标记或额外文字。\n\n"
        )
        if error_detail:
            repair_prompt += f"错误详情：\n{error_detail}\n\n"

        repair_prompt += f"JSON Schema：\n{schema_json}\n\n原始文本：\n{raw_text}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": repair_prompt}],
            temperature=0.0,
            max_tokens=8192,
        )
        return _message_content_to_text(response.choices[0].message.content) if response.choices else ""

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