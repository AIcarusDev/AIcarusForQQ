"""provider.py — 基于 OpenAI 兼容接口的统一模型适配层。

每次 ``call_one_round()`` 仅完成一次 LLM 调用 + 本轮工具执行，
不再承担"循环到出口工具"的职责。多轮永动由 consciousness 主循环驱动。
"""

import logging
import os
import threading
from dataclasses import dataclass, field

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from consciousness.flow import ConsciousnessFlow, ToolCall, ToolResponse

from .internal_tool import InternalToolSpec
from .profiles import resolve_model_provider
from .tool_calling import build_tool_argument_error, parse_tool_arguments, process_tool_arguments
from log_config import log_prompt, log_response

logger = logging.getLogger("AICQ.llm.provider")


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


class LLMCallFailed(Exception):
    """LLM 调用最终失败（预留给上层统一捕获）。"""


@dataclass
class RoundResult:
    """单轮 LLM 调用的产物。"""
    tool_calls_log: list[dict] = field(default_factory=list)
    system_prompt: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
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

    def list_models(self) -> list[str]:
        """返回该 provider 可用的模型 ID 列表。"""
        try:
            page = self.client.models.list()
            return sorted(model.id for model in page.data)
        except Exception:
            return []

    def call_one_round(
        self,
        system_prompt_builder,
        user_content: "str | list",
        gen: dict,
        tool_collection,
        flow: "ConsciousnessFlow | None" = None,
        new_message_checker=None,
    ) -> RoundResult:
        """跑 *一轮* function calling：1 次 LLM 调用 + 本轮工具执行。

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

        max_rounds: int = gen.get("llm_contents_max_rounds", 10)

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
            "max_tokens": gen.get("max_output_tokens", 10000),
            "presence_penalty": gen.get("presence_penalty", 0.0),
            "frequency_penalty": gen.get("frequency_penalty", 0.0),
        }
        if available_tools:
            create_kwargs["tools"] = available_tools
            create_kwargs["tool_choice"] = "auto"

        result = RoundResult(system_prompt=full_system)

        all_messages = [system_msg] + (flow.to_openai_messages() if flow else []) + [user_msg]
        try:
            response = self.client.chat.completions.create(
                messages=all_messages,  # type: ignore
                **create_kwargs,
            )
        except Exception as exc:
            logger.warning("[%s] LLM API 调用异常: %s", self.provider, exc)
            result.failed = True
            return result

        if usage := response.usage:
            result.prompt_tokens = usage.prompt_tokens or 0
            result.output_tokens = usage.completion_tokens or 0

        if not response.choices:
            logger.warning("[%s] response.choices 为空", self.provider)
            result.failed = True
            return result

        msg = response.choices[0].message
        tool_calls_count = len(msg.tool_calls) if msg.tool_calls else 0
        logger.info(
            "[%s] 模型响应 — 工具调用数: %d",
            self.provider,
            tool_calls_count,
        )
        if tool_calls_count > 0:
            logger.info(
                "[%s] 模型请求的工具: %s",
                self.provider,
                ", ".join(tc.function.name for tc in msg.tool_calls),
            )

        # 思考期间焦点会话出现新消息：丢弃整个响应，调用方重调
        if new_message_checker is not None and new_message_checker():
            logger.info("[%s] 思考期间检测到新消息，丢弃本轮响应", self.provider)
            result.new_message_during_thinking = True
            return result

        if not tool_collection.has_active_tools():
            logger.error("[%s] 工具注册表为空，无法继续 function calling", self.provider)
            raise LLMCallFailed("工具注册表为空，无法继续 function calling")

        result.had_tool_call = bool(msg.tool_calls)
        if not msg.tool_calls:
            # 模型违规：一个工具都没调。不写 flow，留给调用方决策（重调 / 兜底 sleep）。
            return result

        # ── 解析、执行所有工具调用 ────────────────────────────────────────
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
                self.provider, fn_name, tool_call.function.arguments or "{}",
            )
            slot: dict = {
                "tc": tool_call,
                "fn_name": fn_name,
                "args": args,
                "fn": handler,
                "result": None,
            }
            if handler is None:
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

        # send_message 串行；其余工具并行（含 sleep/wait/shift —— 它们就是普通慢工具）。
        pending_slots = [slot for slot in slots if slot["result"] is None]
        send_msg_slots = [slot for slot in pending_slots if slot["fn_name"] == "send_message"]
        parallel_slots = [slot for slot in pending_slots if slot["fn_name"] != "send_message"]
        if parallel_slots:
            _run_parallel_slots(parallel_slots, _exec_one, provider_name)
        for slot in send_msg_slots:
            _exec_one(slot)

        # ── 收集结果，写入 flow / log ─────────────────────────────────────
        round_calls: list[ToolCall] = [
            ToolCall(name=slot["fn_name"], args=slot["args"], call_id=slot["tc"].id)
            for slot in slots
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
                    name=fn_name,
                    response=result_data,
                    call_id=tool_call.id,
                    multimodal_parts=raw_multimodal_parts,
                )
            )

        if flow:
            flow.prune(max_rounds)
            flow.append_round(round_calls, round_responses)

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
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=gen.get("temperature", 1.0),
                max_tokens=gen.get("max_output_tokens", 10000),
            )
        except Exception as exc:
            logger.warning("[%s/%s] 文本生成异常: %s", self.provider, log_tag, exc)
            return None

        tag = f"{self.provider}/{log_tag}"
        if usage := response.usage:
            logger.info(
                "[%s] token — 输入: %d, 输出: %d",
                tag,
                usage.prompt_tokens or 0,
                usage.completion_tokens or 0,
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            tools=self._to_openai_tools([declaration]),  # type: ignore[arg-type]
            temperature=gen.get("temperature", 0.3),
            max_tokens=gen.get("max_output_tokens", 10000),
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

        tool_name = declaration["name"]
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
