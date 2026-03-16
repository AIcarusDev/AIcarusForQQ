"""provider.py — Gemini 原生 & OpenAI 兼容 双适配层

Gemini 系列模型使用 google-genai SDK 原生 API 调用：
  - SDK 自动管理思考签名（无需手动提取 extra_content）
  - 原生 thinking_level 配置
  - 原生 JSON 结构化输出

其他 provider 仍通过 OpenAI SDK 的兼容端点调用：
  - siliconflow : 硅基流动  （环境变量: SILICONFLOW_API_KEY）
  - bigmodel    : 智谱 AI   （环境变量: BIGMODEL_API_KEY）

通过 config.yaml 的 `provider` 字段一键切换后端。
"""

import base64
import json
import logging
import os
import time

import httpx

from json_repair import clean_and_parse
from log_config import log_prompt, log_response

logger = logging.getLogger("AICQ.provider")


# ── Provider 默认配置 ───────────────────────────────────────────────────────────

_PROVIDER_DEFAULTS: dict[str, dict] = {
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "env_key": "SILICONFLOW_API_KEY",
        "default_model": "THUDM/GLM-4-32B-0414",
    },
    "bigmodel": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "env_key": "BIGMODEL_API_KEY",
        "default_model": "glm-4-plus",
    },
}


# ── 工具配额管理 ─────────────────────────────────────────────────────────────────

class ToolBudgetManager:
    """按工具粒度追踪单次回复内的调用配额。

    从 TOOL_DECLARATIONS 中读取每个工具的 max_calls_per_response，
    在工具调用循环中实时扣减，并可：
      - 过滤掉已耗尽配额的工具声明
      - 生成供 dashboard 显示的配额字典
    """

    def __init__(self, tool_declarations: list[dict]):
        # {函数名: {"total": N, "remaining": N, "description": "..."}}
        self._budgets: dict[str, dict] = {}
        for decl in tool_declarations:
            name = decl.get("name", "")
            if not name:
                continue
            max_calls = decl.get("max_calls_per_response", 1)
            self._budgets[name] = {
                "total": max_calls,
                "remaining": max_calls,
                "description": decl.get("description", ""),
            }

    def consume(self, tool_name: str) -> None:
        """消耗一次指定工具的配额。"""
        if tool_name in self._budgets:
            self._budgets[tool_name]["remaining"] = max(
                0, self._budgets[tool_name]["remaining"] - 1
            )

    def is_available(self, tool_name: str) -> bool:
        """检查指定工具是否还有剩余配额。"""
        info = self._budgets.get(tool_name)
        return info is not None and info["remaining"] > 0

    def any_available(self) -> bool:
        """是否还有任何工具可用。"""
        return any(info["remaining"] > 0 for info in self._budgets.values())

    def filter_declarations(self, tool_declarations: list[dict]) -> list[dict]:
        """过滤掉已耗尽配额的工具声明，返回仍可用的干净声明。"""
        result = []
        for decl in tool_declarations:
            name = decl.get("name", "")
            if self.is_available(name):
                clean = {k: v for k, v in decl.items() if k != "max_calls_per_response"}
                result.append(clean)
        return result

    def get_budget_dict(self) -> dict[str, dict]:
        """返回完整配额字典，供 prompt dashboard 显示。"""
        return dict(self._budgets)


# ── 工具函数 ────────────────────────────────────────────────────────────────────

def _schema_to_prompt(schema: dict) -> str:
    """将 JSON Schema 转为 system prompt 中的格式约束说明。"""
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
    return (
        "## 严格输出格式\n"
        "你的回复必须是且仅是一个合法的 JSON 对象，"
        "不得包含任何 Markdown 代码块标记或额外文字。\n"
        "遵循以下 JSON Schema：\n"
        f"{schema_json}"
    )


# ══════════════════════════════════════════════════════════════════
#  Gemini 原生适配器（google-genai SDK）
# ══════════════════════════════════════════════════════════════════

class GeminiAdapter:
    """使用 google-genai SDK 的 Gemini 原生适配器。

    相比 OpenAI 兼容端点的优势：
    - SDK 自动管理思考签名（无需手动提取 extra_content）
    - 原生 thinking_level 配置（无需映射）
    - Gemini 3 支持函数调用 + 结构化输出同时使用
    """

    def __init__(self, cfg: dict):
        from google import genai

        api_key = os.getenv(_PROVIDER_DEFAULTS["gemini"]["env_key"], "")
        self.client = genai.Client(api_key=api_key)
        self.model = cfg.get("model", _PROVIDER_DEFAULTS["gemini"]["default_model"])
        self.provider = "gemini"
        self.thinking_level = cfg.get("thinking", {}).get("level")

    def list_models(self) -> list[str]:
        """返回该 provider 可用的模型 ID 列表。"""
        try:
            return sorted(
                m.name.split("/")[-1]
                for m in self.client.models.list()
                if m.name
            )
        except Exception:
            return []

    def call(
        self,
        system_prompt_builder,
        user_content: "str | list",
        gen: dict,
        schema: dict,
        tool_declarations: list | None = None,
        tool_registry: dict | None = None,
    ) -> tuple[dict | None, dict | None, bool, list[dict]]:
        """调用 Gemini 原生 API，支持工具调用循环。

        system_prompt_builder: 接受 tool_budget 字典参数的可调用对象，
                               返回构建好的 system prompt 字符串。
        tool_declarations:     原生格式的工具声明列表（含自定义 max_calls_per_response）
        tool_registry:         {函数名: callable} 字典
        返回 (result_dict, grounding_dict, repaired, tool_calls_log)。
        """
        from google.genai import types

        budget_mgr = ToolBudgetManager(tool_declarations or [])

        # 构建 system instruction（仅含工具配额，schema 通过原生 response_schema 参数传递）
        full_system = system_prompt_builder(budget_mgr.get_budget_dict())

        # 构建 user content（文本或多模态 Parts）
        user_parts = self._convert_user_content(user_content)
        contents = [types.Content(role="user", parts=user_parts)]

        log_prompt("gemini", full_system, user_content)

        # 构建配置
        available_decls = budget_mgr.filter_declarations(tool_declarations or [])

        config_kwargs: dict = {
            "system_instruction": full_system,
            "temperature": gen.get("temperature", 1.0),
            "max_output_tokens": gen.get("max_output_tokens", 8192),
            "response_mime_type": "application/json",
            "response_json_schema": schema,
        }

        if available_decls:
            config_kwargs["tools"] = [types.Tool(function_declarations=available_decls)]  # type: ignore[arg-type]
            config_kwargs["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )

        if self.thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.thinking_level,
            )

        config = types.GenerateContentConfig(**config_kwargs)

        # ── 工具调用循环 ──
        tool_calls_log: list[dict] = []
        tool_round = 0
        max_absolute_rounds = gen.get("max_tool_rounds", 5)

        while True:
            response = self._generate_with_retry(
                contents, config, max_retries=3, base_delay=2.0,
            )

            if not response.candidates:
                logger.warning("[gemini] response.candidates 为空")
                return None, None, False, tool_calls_log

            function_calls = response.function_calls

            if (
                function_calls
                and tool_registry
                and budget_mgr.any_available()
                and tool_round < max_absolute_rounds
            ):
                tool_round += 1

                # 将模型的完整回复（含思考签名）加入历史
                # SDK 自动保留 thought_signature，无需手动提取
                candidate_content = response.candidates[0].content
                if candidate_content is not None:
                    contents.append(candidate_content)

                # 执行函数并收集结果
                fn_response_parts = []
                for fc in function_calls:
                    fn_name = fc.name
                    if not fn_name:
                        continue
                    fn = tool_registry.get(fn_name)
                    args = dict(fc.args) if fc.args else {}

                    if not budget_mgr.is_available(fn_name):
                        result_data = {
                            "error": f"工具 {fn_name} 本轮调用次数已耗尽，请直接生成回复。"
                        }
                        logger.info("[gemini] 工具 %s 配额已耗尽", fn_name)
                    elif fn is None:
                        result_data = {"error": f"未知工具: {fn_name}"}
                    else:
                        try:
                            call_args = {k: v for k, v in args.items() if k != "motivation"}
                            result_data = fn(**call_args)
                        except Exception as e:
                            result_data = {"error": str(e)}

                    budget_mgr.consume(fn_name)

                    tool_calls_log.append({
                        "round": tool_round,
                        "function": fn_name,
                        "arguments": args,
                        "motivation": args.get("motivation", ""),
                        "result": result_data,
                    })

                    # 提取多模态附件（如图片），构建 Gemini 3 多模态函数响应
                    multimodal_extras = []
                    if isinstance(result_data, dict) and "_multimodal_parts" in result_data:
                        for mp in result_data.pop("_multimodal_parts"):  # type: ignore[union-attr]
                            mp: dict  # type: ignore[no-redef]
                            multimodal_extras.append(
                                types.FunctionResponsePart(
                                    inline_data=types.FunctionResponseBlob(
                                        mime_type=mp["mime_type"],
                                        display_name=mp["display_name"],
                                        data=mp["data"],
                                    )
                                )
                            )

                    fr_kwargs: dict = {
                        "name": fn_name,
                        "response": result_data if multimodal_extras else {"result": result_data},
                    }
                    if multimodal_extras:
                        fr_kwargs["parts"] = multimodal_extras

                    fn_response_parts.append(
                        types.Part.from_function_response(**fr_kwargs)
                    )

                # 将工具执行结果加入历史
                contents.append(types.Content(role="user", parts=fn_response_parts))

                # 更新工具声明：移除已耗尽配额的工具
                available_decls = budget_mgr.filter_declarations(tool_declarations or [])
                if available_decls:
                    config_kwargs["tools"] = [
                        types.Tool(function_declarations=available_decls)  # type: ignore[arg-type]
                    ]
                else:
                    config_kwargs.pop("tools", None)
                    config_kwargs.pop("automatic_function_calling", None)
                    logger.info("[gemini] 所有工具配额已耗尽，移除工具声明")

                # 更新 system prompt 中的配额显示
                config_kwargs["system_instruction"] = system_prompt_builder(budget_mgr.get_budget_dict())
                config = types.GenerateContentConfig(**config_kwargs)
            else:
                if function_calls and not budget_mgr.any_available():
                    logger.info("[gemini] 模型仍尝试调用工具但配额已耗尽")
                break

        if tool_calls_log:
            call_counts: dict[str, int] = {}
            for entry in tool_calls_log:
                name = entry["function"]
                call_counts[name] = call_counts.get(name, 0) + 1
            summary = ", ".join(f"{name}×{count}" for name, count in call_counts.items())
            logger.info(
                "[gemini] 工具调用共 %d 轮 %d 次: %s",
                tool_round, len(tool_calls_log), summary,
            )

        text = response.text
        log_response("gemini", text)
        if not text:
            logger.warning("[gemini] response.text 为空")
            return None, None, False, tool_calls_log

        result, repaired = clean_and_parse(text, "[gemini]")
        return result, None, repaired, tool_calls_log

    # ── 网络瞬态错误重试 ──

    _TRANSIENT_EXCEPTIONS = (
        httpx.ReadError,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
        httpx.WriteError,
        ConnectionResetError,
        ConnectionAbortedError,
        OSError,
    )

    def _generate_with_retry(self, contents, config, *, max_retries: int = 3, base_delay: float = 2.0):
        """带重试的 generate_content，捕获网络瞬态异常（如远程主机强制关闭连接）。"""
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=contents, config=config,  # type: ignore[arg-type]
                )
            except self._TRANSIENT_EXCEPTIONS as e:
                last_exc = e
                if attempt >= max_retries:
                    logger.error("[gemini] 网络错误，已重试 %d 次仍失败: %s", max_retries, e)
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning("[gemini] 网络错误 (%s)，%0.1fs 后重试 (%d/%d)", e, delay, attempt + 1, max_retries)
                time.sleep(delay)
        raise RuntimeError("重试耗尽") from last_exc  # 不应到达

    @staticmethod
    def _convert_user_content(user_content: "str | list") -> list:
        """将聊天记录格式化后的内容转为 google-genai Parts。

        build_multimodal_content 可能返回：
          - str: 纯文本 XML
          - list[dict]: OpenAI 多模态 parts 格式（text / image_url）
        统一转为 google.genai.types.Part 列表。
        """
        from google.genai import types

        if isinstance(user_content, str):
            return [types.Part(text=user_content)]

        parts = []
        for item in user_content:
            if item.get("type") == "text":
                parts.append(types.Part(text=item["text"]))
            elif item.get("type") == "image_url":
                url = item["image_url"]["url"]
                # data:image/jpeg;base64,/9j/4AAQ...
                header, b64data = url.split(",", 1)
                mime = header.split(":")[1].split(";")[0]
                parts.append(types.Part(
                    inline_data=types.Blob(
                        mime_type=mime,
                        data=base64.b64decode(b64data),
                    )
                ))
        return parts


# ══════════════════════════════════════════════════════════════════
#  OpenAI 兼容适配器（siliconflow / bigmodel 等）
# ══════════════════════════════════════════════════════════════════

class OpenAICompatAdapter:
    """使用 OpenAI SDK 的兼容适配器，用于非 Gemini 的 provider。"""

    def __init__(self, cfg: dict):
        from openai import OpenAI

        provider = cfg.get("provider", "siliconflow")
        defaults = _PROVIDER_DEFAULTS.get(provider)
        if defaults is None:
            raise ValueError(
                f"未知的 provider: {provider!r}，"
                f"可选值: {' / '.join(_PROVIDER_DEFAULTS)}"
            )

        base_url = cfg.get("base_url", defaults.get("base_url", ""))
        api_key = os.getenv(defaults["env_key"], "")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = cfg.get("model", defaults["default_model"])
        self.provider = provider

    def list_models(self) -> list[str]:
        """返回该 provider 可用的模型 ID 列表。"""
        try:
            page = self.client.models.list()
            return sorted(m.id for m in page.data)
        except Exception:
            return []

    def call(
        self,
        system_prompt_builder,
        user_content: "str | list",
        gen: dict,
        schema: dict,
        tool_declarations: list | None = None,
        tool_registry: dict | None = None,
    ) -> tuple[dict | None, dict | None, bool, list[dict]]:
        """调用 OpenAI 兼容 API，支持工具调用循环。"""
        budget_mgr = ToolBudgetManager(tool_declarations or [])
        full_system = (
            system_prompt_builder(budget_mgr.get_budget_dict())
            + "\n\n" + _schema_to_prompt(schema)
        )

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_content},
        ]

        log_prompt(self.provider, full_system, user_content)

        available_tools = self._to_openai_tools(
            budget_mgr.filter_declarations(tool_declarations or [])
        )

        create_kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": gen.get("temperature", 1.0),
            "max_tokens": gen.get("max_output_tokens", 8192),
            "presence_penalty": gen.get("presence_penalty", 0.0),
            "frequency_penalty": gen.get("frequency_penalty", 0.0),
        }

        if available_tools:
            create_kwargs["tools"] = available_tools
            create_kwargs["tool_choice"] = "auto"

        tool_calls_log: list[dict] = []
        tool_round = 0
        max_absolute_rounds = gen.get("max_tool_rounds", 5)

        while True:
            response = self.client.chat.completions.create(**create_kwargs)

            if not response.choices:
                logger.warning("[%s] response.choices 为空", self.provider)
                return None, None, False, tool_calls_log

            msg = response.choices[0].message

            if (
                msg.tool_calls
                and tool_registry
                and budget_mgr.any_available()
                and tool_round < max_absolute_rounds
            ):
                tool_round += 1
                assistant_msg: dict = {"role": "assistant", "content": msg.content}
                tc_list = []
                for tc in msg.tool_calls:
                    tc_list.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })
                assistant_msg["tool_calls"] = tc_list
                messages.append(assistant_msg)

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn = tool_registry.get(fn_name)
                    args = {}

                    if not budget_mgr.is_available(fn_name):
                        result_data = {
                            "error": f"工具 {fn_name} 本轮调用次数已耗尽，请直接生成回复。"
                        }
                        logger.info(
                            "[%s] 工具 %s 配额已耗尽", self.provider, fn_name,
                        )
                    elif fn is None:
                        result_data = {"error": f"未知工具: {fn_name}"}
                    else:
                        try:
                            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                            call_args = {k: v for k, v in args.items() if k != "motivation"}
                            result_data = fn(**call_args)
                        except Exception as e:
                            result_data = {"error": str(e)}

                    budget_mgr.consume(fn_name)

                    tool_calls_log.append({
                        "round": tool_round,
                        "tool_call_id": tc.id,
                        "function": fn_name,
                        "arguments": args,
                        "motivation": args.get("motivation", ""),
                        "result": result_data,
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result_data, ensure_ascii=False),
                    })

                available_tools = self._to_openai_tools(
                    budget_mgr.filter_declarations(tool_declarations or [])
                )
                if available_tools:
                    create_kwargs["tools"] = available_tools
                else:
                    create_kwargs.pop("tools", None)
                    create_kwargs.pop("tool_choice", None)
                    logger.info(
                        "[%s] 所有工具配额已耗尽，移除工具声明",
                        self.provider,
                    )

                updated_system = (
                    system_prompt_builder(budget_mgr.get_budget_dict())
                    + "\n\n" + _schema_to_prompt(schema)
                )
                messages[0]["content"] = updated_system
            else:
                if msg.tool_calls and not budget_mgr.any_available():
                    logger.info(
                        "[%s] 模型仍尝试调用工具但配额已耗尽",
                        self.provider,
                    )
                break

        if tool_calls_log:
            call_counts: dict[str, int] = {}
            for entry in tool_calls_log:
                name = entry["function"]
                call_counts[name] = call_counts.get(name, 0) + 1
            summary = ", ".join(f"{name}×{count}" for name, count in call_counts.items())
            logger.info(
                "[%s] 工具调用共 %d 轮 %d 次: %s",
                self.provider, tool_round, len(tool_calls_log), summary,
            )

        text = msg.content
        log_response(self.provider, text)
        if not text:
            logger.warning("[%s] response.content 为空", self.provider)
            return None, None, False, tool_calls_log

        result, repaired = clean_and_parse(text, f"[{self.provider}]")
        return result, None, repaired, tool_calls_log

    @staticmethod
    def _to_openai_tools(declarations: list[dict]) -> list[dict]:
        """将工具声明转为 OpenAI function calling 格式。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": d["name"],
                    "description": d.get("description", ""),
                    "parameters": d.get("parameters", {}),
                },
            }
            for d in declarations
        ]


# ── 工厂函数 ────────────────────────────────────────────────────────────────────

def create_adapter(cfg: dict):
    """根据 config.yaml 中的 provider 字段创建适配器。"""
    provider = cfg.get("provider", "gemini")
    if provider not in _PROVIDER_DEFAULTS:
        raise ValueError(
            f"未知的 provider: {provider!r}，"
            f"可选值: {' / '.join(_PROVIDER_DEFAULTS)}"
        )
    if provider == "gemini":
        return GeminiAdapter(cfg)
    return OpenAICompatAdapter(cfg)
