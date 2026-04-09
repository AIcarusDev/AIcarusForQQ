"""provider.py — Gemini 原生 & OpenAI 兼容 双适配层

Gemini 系列模型使用 google-genai SDK 原生 API 调用：
  - SDK 自动管理思考签名（无需手动提取 extra_content）
  - 原生 thinking_level 配置
  - 原生 JSON 结构化输出

其他 provider 仍通过 OpenAI SDK 的兼容端点调用：
    - siliconflow : 硅基流动  （环境变量: SILICONFLOW_API_KEY）
    - bigmodel    : 智谱 AI   （环境变量: BIGMODEL_API_KEY）
    - dashscope   : 阿里云百炼（环境变量: DASHSCOPE_API_KEY）

通过 config.yaml 的 `provider` 字段一键切换后端。
"""

import base64
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from google import genai
from google.genai import types
import httpx
from google.genai import errors as genai_errors
from jsonschema import validate, ValidationError

from ..circuit_breaker import ToolRepeatBreaker
from .json_repair import clean_and_parse
from .decision_filter import hoist_decision_motivation, fill_missing_motivations, remove_additional_properties_key
from log_config import log_prompt, log_response

logger = logging.getLogger("AICQ.provider")


class LLMCallFailed(Exception):
    """LLM 调用最终失败（已重试完毕或不可重试的配额/账单问题），属于预期内的软失败。
    上层捕获后只需打 warning，无需打印完整 traceback。
    """


# ── Provider 默认配置 ───────────────────────────────────────────────────────────

_PROVIDER_DEFAULTS: dict[str, dict] = {
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "default_model": "qwen3.5-flash",
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
        """消耗一次指定工具的配额（已禁用限制，空操作）。"""

    def is_available(self, tool_name: str) -> bool:
        """检查指定工具是否还有剩余配额（已禁用限制，始终返回 True）。"""
        return True

    def any_available(self) -> bool:
        """是否还有任何工具可用（已禁用限制，始终返回 True）。"""
        return True

    def add_tool(self, name: str, max_calls: int = 1) -> None:
        """动态添加一个新工具到配额管理器（运行时注入用）。"""
        self._budgets[name] = {
            "total": max_calls,
            "remaining": max_calls,
            "description": "",
        }

    def filter_declarations(self, tool_declarations: list[dict]) -> list[dict]:
        """返回可用的工具声明（已禁用限制，仅剥离自定义字段）。"""
        return [
            {k: v for k, v in decl.items() if k != "max_calls_per_response"}
            for decl in tool_declarations
        ]

    def get_budget_dict(self) -> dict[str, dict]:
        """返回完整配额字典，供 prompt dashboard 显示。"""
        return dict(self._budgets)


# ── 工具函数 ────────────────────────────────────────────────────────────────────

def _schema_to_prompt(schema: dict, *, with_tools: bool = False) -> str:
    """将 JSON Schema 转为 system prompt 中的格式约束说明。
    
    Args:
        schema: JSON Schema 定义
        with_tools: 是否有工具可用（影响约束措辞）
    """
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
    
    if with_tools:
        # 有工具时：给予灵活性，但如果不调用工具就必须返回 JSON
        return (
            "## 严格输出格式\n"
            "你可以选择调用上述工具来获取信息，或直接生成回复。\n"
            "**如果选择不调用工具，你的回复必须是且仅是一个合法的 JSON 对象。**\n"
            "对象结构必须严格遵循下述 JSON Schema，不得包含任何 Markdown 代码块标记或额外文字。\n"
            "严格遵循以下 JSON Schema：\n"
            f"{schema_json}"
        )
    else:
        # 无工具时：必须返回 JSON
        return (
            "## 严格输出格式\n"
            "你的回复必须是且仅是一个合法的 JSON 对象，尤其注意对象结构的正确性，"
            "不得包含任何 Markdown 代码块标记或额外文字。\n"
            "严格遵循以下 JSON Schema：\n"
            f"{schema_json}"
        )


def _strip_images(user_content: "str | list") -> "str | list":
    """从多模态内容中剥除图片部分，仅保留文本。

    vision: false 时使用：过滤掉所有 image_url 类型的 part，
    避免发送给不支持视觉的模型导致 400 错误。
    若剥除后只剩一条文本，返回纯字符串以保持最大兼容性。
    """
    if not isinstance(user_content, list):
        return user_content
    if not (text_parts := [p for p in user_content if p.get("type") == "text"]):
        return ""
    return text_parts[0]["text"] if len(text_parts) == 1 else text_parts


# ── Gemini 内置工具已禁用（统一使用自定义工具管理） ──────────────────────────────

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

        api_key = os.getenv(_PROVIDER_DEFAULTS["gemini"]["env_key"], "")

        # 代理配置：直接从环境变量读取（GEMINI_PROXY）
        # google-genai SDK 不直接支持传入 httpx.Client；通过设置 HTTP(S)_PROXY 环境变量生效
        proxy_url = os.getenv("GEMINI_PROXY", "").strip() or None
        if proxy_url:
            os.environ.setdefault("HTTPS_PROXY", proxy_url)
            os.environ.setdefault("HTTP_PROXY", proxy_url)

        self.client = genai.Client(api_key=api_key)
        self.model = cfg.get("model", _PROVIDER_DEFAULTS["gemini"]["default_model"])
        self.provider = "gemini"
        self.thinking_level = cfg.get("thinking", {}).get("level")
        self._vision_enabled: bool = bool(cfg.get("vision", True))

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
        latent_registry: dict | None = None,
        user_content_refresher=None,
    ) -> tuple[dict | None, dict | None, bool, list[dict], str]:
        """调用 Gemini 原生 API，支持工具调用循环。

        system_prompt_builder:  接受 (activated_names, latent_names) 的可调用对象，
                                返回构建好的 system prompt 字符串。
        tool_declarations:      原生格式的工具声明列表（含自定义 max_calls_per_response）
        tool_registry:          {函数名: callable} 字典
        latent_registry:        {函数名: (declaration, handler)} 的潜伏工具字典
        user_content_refresher: 无参可调用对象，每次工具调用后调用它重新构建聊天记录
                                并替换 contents[0]（含最新时间/新消息）；为 None 时不刷新。
        返回 (result_dict, grounding_dict, repaired, tool_calls_log, initial_system_prompt)。
        """
        tool_declarations = list(tool_declarations or [])
        tool_registry = dict(tool_registry or {})
        latent_registry = dict(latent_registry or {})

        budget_mgr = ToolBudgetManager(tool_declarations)
        breaker = ToolRepeatBreaker(name_only_tools={"short_wait"})

        # ── 工具调用循环 ──
        tool_calls_log: list[dict] = []
        tool_round = 0

        # 构建 system instruction（仅含工具配额，schema 通过原生 response_schema 参数传递）
        full_system = system_prompt_builder(list(budget_mgr.get_budget_dict().keys()), list(latent_registry.keys()))

        # 构建 user content（文本或多模态 Parts；未启用视觉时过滤图片）
        user_parts = self._convert_user_content(
            user_content if self._vision_enabled else _strip_images(user_content)
        )
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
            config_kwargs["tools"] = [types.Tool(
                function_declarations=available_decls,  # type: ignore[arg-type]
            )]
            config_kwargs["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )
            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.VALIDATED)
            )

        if self.thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.thinking_level,
            )

        config = types.GenerateContentConfig(**config_kwargs)

        _tok_prompt = 0
        _tok_output = 0
        _tok_thoughts = 0
        _round_num = 0

        while True:
            _round_num += 1
            response = self._generate_with_retry(
                contents, config, max_retries=3, base_delay=2.0,
            )

            if _u := response.usage_metadata:
                _r_prompt   = _u.prompt_token_count or 0
                _r_output   = _u.candidates_token_count or 0
                _r_thoughts = _u.thoughts_token_count or 0
                logger.info(
                    "[gemini] 第 %d 轮 token — 输入: %d, 输出: %d, 思维链: %s, total: %s",
                    _round_num, _r_prompt, _r_output,
                    _u.thoughts_token_count,  # 原始值，区分 None 和 0
                    _u.total_token_count,
                )
                _tok_prompt   += _r_prompt
                _tok_output   += _r_output
                _tok_thoughts += _r_thoughts
            else:
                logger.warning("[gemini] 第 %d 轮 usage_metadata 为 None!", _round_num)

            if not response.candidates:
                logger.warning("[gemini] response.candidates 为空")
                return None, None, False, tool_calls_log, full_system

            function_calls = response.function_calls
            func_count = len(function_calls) if function_calls else 0
            
            # 记录当轮响应的元信息（用 _round_num 保证与 token 日志对齐）
            logger.info(
                "[gemini] 第 %d 轮模型响应 — 工具调用数: %d, 候选项数: %d",
                _round_num, func_count, len(response.candidates)
            )
            if func_count > 0 and function_calls:
                func_names = [fc.name for fc in function_calls if fc.name]
                logger.info("[gemini] 模型请求的工具: %s", ", ".join(func_names))

            if (
                function_calls
                and tool_registry
                and budget_mgr.any_available()
            ):
                tool_round += 1
                breaker.begin_round(tool_round)

                # 将模型的完整回复（含思考签名）加入历史
                # SDK 自动保留 thought_signature，无需手动提取
                candidate_content = response.candidates[0].content
                if candidate_content is not None:
                    contents.append(candidate_content)

                # 执行函数并收集结果（三阶段：预检 → 并行执行 → 后处理）
                pending_injections: list[str] = []
                fn_response_parts = []

                # 阶段 1：顺序预检（熔断/配额/注册，保护共享状态）
                # slot 键: fc, fn_name, args, fn, result(None=待执行), circuit_broken
                _slots: list[dict] = []
                for fc in function_calls:
                    fn_name = fc.name
                    if not fn_name:
                        continue
                    _fn = tool_registry.get(fn_name)
                    args = dict(fc.args) if fc.args else {}
                    slot: dict = {
                        "fc": fc, "fn_name": fn_name, "args": args,
                        "fn": _fn, "result": None, "circuit_broken": False,
                    }
                    if breaker.check_and_record(fn_name, args):
                        slot["result"] = {"error": f"CIRCUIT_BREAKER_TRIPPED: tool='{fn_name}' consecutive_calls={breaker.max_streak} threshold={breaker.max_streak}. Tool call REJECTED and tool REMOVED from registry. You MUST stop calling this tool and output final schema immediately."}
                        logger.warning("[gemini] 熔断触发: 工具 %s 连续 %d 轮相同调用，已拦截并移除", fn_name, breaker.max_streak)
                        tool_registry.pop(fn_name, None)
                        tool_declarations[:] = [d for d in tool_declarations if d.get("name") != fn_name]
                        slot["circuit_broken"] = True
                    elif not budget_mgr.is_available(fn_name):
                        slot["result"] = {"error": f"工具 {fn_name} 本轮调用次数已耗尽，请直接生成回复。"}
                        logger.info("[gemini] 工具 %s 配额已耗尽", fn_name)
                    elif _fn is None:
                        slot["result"] = {"error": f"未知工具: {fn_name}"}
                    _slots.append(slot)

                # 阶段 2：并行执行需要实际调用的工具
                def _exec_gemini(slot: dict) -> None:
                    _fn_name = slot["fn_name"]
                    logger.info("[gemini] 执行工具开始: %s", _fn_name)
                    try:
                        slot["result"] = slot["fn"](**slot["args"])
                        if isinstance(slot["result"], dict):
                            if slot["result"].get("error"):
                                logger.info("[gemini] 执行工具完毕（失败）: %s — %s", _fn_name, slot["result"]["error"])
                            else:
                                logger.info("[gemini] 执行工具完毕（成功）: %s 返回字段: %s", _fn_name, list(slot["result"].keys()))
                        else:
                            logger.info("[gemini] 执行工具完毕（成功）: %s", _fn_name)
                    except Exception as e:
                        logger.warning("[gemini] 执行工具异常: %s — %s", _fn_name, e)
                        slot["result"] = {"error": str(e)}

                _pending_slots = [s for s in _slots if s["result"] is None]
                if _pending_slots:
                    with ThreadPoolExecutor(max_workers=len(_pending_slots)) as _pool:
                        list(_pool.map(_exec_gemini, _pending_slots))

                # 阶段 3：顺序后处理（消耗配额、收集注入、构建响应，保持原始顺序）
                for slot in _slots:
                    fn_name = slot["fn_name"]
                    fc = slot["fc"]
                    args = slot["args"]
                    result_data = slot["result"]
                    circuit_broken = slot["circuit_broken"]

                    if not circuit_broken:
                        budget_mgr.consume(fn_name)

                        # 提取注入信号（内部协议，不返回给模型）
                        if isinstance(result_data, dict) and "_inject_tools" in result_data:
                            pending_injections.extend(result_data.pop("_inject_tools") or [])

                    tool_calls_log.append({
                        "round": tool_round,
                        "function": fn_name,
                        "arguments": args,
                        "result": result_data,
                        "circuit_broken": circuit_broken,
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
                    # 并行调用同名工具时，必须回传 id 让 API 匹配 call ↔ response
                    if fc.id:
                        fr_kwargs["id"] = fc.id
                    if multimodal_extras:
                        fr_kwargs["parts"] = multimodal_extras

                    fn_response_parts.append(
                        types.Part(function_response=types.FunctionResponse(**fr_kwargs))
                    )

                # 将工具执行结果加入历史
                contents.append(types.Content(role="user", parts=fn_response_parts))

                # 注入通过 get_tools 激活的潜伏工具
                for inj_name in pending_injections:
                    if inj_name in latent_registry:
                        inj_decl, inj_handler = latent_registry.pop(inj_name)
                        tool_declarations.append(inj_decl)
                        tool_registry[inj_name] = inj_handler
                        budget_mgr.add_tool(inj_name, inj_decl.get("max_calls_per_response", 1))
                        logger.info("[gemini] 注入潜伏工具: %s", inj_name)
                    else:
                        logger.warning("[gemini] 无法注入工具 %s：不在潜伏工具列表中或已激活", inj_name)

                # 更新工具声明：移除已耗尽配额的工具
                if available_decls := budget_mgr.filter_declarations(tool_declarations):
                    config_kwargs["tools"] = [types.Tool(
                        function_declarations=available_decls,  # type: ignore[arg-type]
                    )]
                else:
                    config_kwargs.pop("tools", None)
                    config_kwargs.pop("automatic_function_calling", None)
                    config_kwargs.pop("tool_config", None)
                    logger.info("[gemini] 所有自定义工具配额已耗尽")

                # 更新 system prompt
                config_kwargs["system_instruction"] = system_prompt_builder(list(budget_mgr.get_budget_dict().keys()), list(latent_registry.keys()))

                # 刷新 user prompt（含最新聊天记录/时间/新消息）
                if user_content_refresher is not None:
                    fresh = user_content_refresher()
                    fresh_parts = self._convert_user_content(
                        fresh if self._vision_enabled else _strip_images(fresh)
                    )
                    contents[0] = types.Content(role="user", parts=fresh_parts)
                    logger.info("[gemini] 工具调用第 %d 轮后已刷新 user prompt", tool_round)

                config = types.GenerateContentConfig(**config_kwargs)
            else:
                if function_calls:
                    logger.info("[gemini] 模型仍尝试调用工具（配额已耗尽），移除所有工具声明后继续")
                    # 该 function_call 尚未入历史，移除所有工具声明，下一轮只能输出文本
                    config_kwargs.pop("tools", None)
                    config_kwargs.pop("automatic_function_calling", None)
                    config_kwargs.pop("tool_config", None)
                    config = types.GenerateContentConfig(**config_kwargs)
                    continue
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

        # 手动拼接 text parts，避免 response.text 在同时含 function_call 时触发 SDK 警告
        try:
            content = response.candidates[0].content
            if content and hasattr(content, 'parts') and content.parts:
                text = "".join(p.text for p in content.parts if hasattr(p, "text") and p.text)
            else:
                text = response.text
        except Exception:
            text = response.text
        log_response("gemini", text)
        
        # 记录原始响应摘要
        if text:
            text_preview = text[:200].replace('\n', ' ')
            logger.info("[gemini] 原始响应 — 长度: %d 字节, 摘要: %s...", len(text), text_preview)
        else:
            logger.warning("[gemini] 原始响应为空")

        logger.info(
            "[gemini] Token 用量（全轮累计）— 输入: %d, 输出: %d, 思维链: %d, 总计: %d",
            _tok_prompt,
            _tok_output,
            _tok_thoughts,
            _tok_prompt + _tok_output + _tok_thoughts,
        )

        if not text:
            logger.warning("[gemini] response.text 为空")
            return None, None, False, tool_calls_log, full_system

        result, repaired = clean_and_parse(text, "[gemini]")
        return result, None, repaired, tool_calls_log, full_system

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

    # 服务端繁忙 / 限流，值得重试的 HTTP 状态码
    _RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def _generate_with_retry(self, contents, config, *, max_retries: int = 3, base_delay: float = 2.0):
        """带重试的 generate_content，捕获网络瞬态异常及 API 限流/服务繁忙错误。"""
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=contents, config=config,  # type: ignore[arg-type]
                )
            except (genai_errors.ServerError, genai_errors.ClientError) as e:
                status = getattr(e, 'status_code', None) or getattr(e, 'code', None)
                if status not in self._RETRYABLE_STATUS_CODES:
                    raise
                # RESOURCE_EXHAUSTED (账单/配额耗尽) 无法靠重试解决，直接放弃
                err_str = str(e)
                if 'RESOURCE_EXHAUSTED' in err_str or 'spending cap' in err_str.lower():
                    msg = f"[gemini] 账单/配额已耗尽 (HTTP {status})，无法继续调用"
                    logger.warning(msg)
                    raise LLMCallFailed(msg) from e
                last_exc = e
                if attempt >= max_retries:
                    msg = f"[gemini] API 繁忙/限流 (HTTP {status})，已重试 {max_retries} 次仍失败，跳过本次调用"
                    logger.warning(msg)
                    raise LLMCallFailed(msg) from e
                delay = base_delay * (2 ** attempt)
                logger.warning("[gemini] API 繁忙/限流 (HTTP %s)，%0.1fs 后重试 (%d/%d)", status, delay, attempt + 1, max_retries)
                time.sleep(delay)
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

        provider = cfg.get("provider", "siliconflow")
        defaults = _PROVIDER_DEFAULTS.get(provider)
        if defaults is None:
            raise ValueError(
                f"未知的 provider: {provider!r}，"
                f"可选值: {' / '.join(_PROVIDER_DEFAULTS)}"
            )

        base_url = cfg.get("base_url", defaults.get("base_url", ""))
        api_key = os.getenv(defaults["env_key"], "")

        # 代理配置：直接从环境变量读取（OPENAI_PROXY）
        proxy_url = os.getenv("OPENAI_PROXY", "").strip() or None

        if proxy_url:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=httpx.Client(proxy=proxy_url),
            )
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = cfg.get("model", defaults["default_model"])
        self.provider = provider
        self._vision_enabled: bool = bool(cfg.get("vision", True))

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
        latent_registry: dict | None = None,
        user_content_refresher=None,
    ) -> tuple[dict | None, dict | None, bool, list[dict], str]:
        """调用 OpenAI 兼容 API，支持工具调用循环。

        user_content_refresher: 无参可调用对象，每次工具调用后调用它重新构建聊天记录
                                并替换 messages[1]（含最新时间/新消息）；为 None 时不刷新。
        """
        tool_declarations = list(tool_declarations or [])
        tool_registry = dict(tool_registry or {})
        latent_registry = dict(latent_registry or {})

        budget_mgr = ToolBudgetManager(tool_declarations)
        breaker = ToolRepeatBreaker(name_only_tools={"short_wait"})

        # 【修复 Qwen 工具调用问题】：优先检查是否有可用工具
        # 如果有工具可用，避免添加强制 JSON 格式约束，让 tool_choice="auto" 优先生效
        available_tools = self._to_openai_tools(
            budget_mgr.filter_declarations(tool_declarations)
        )

        # ── 构建系统提示 ──
        base_system = system_prompt_builder(list(budget_mgr.get_budget_dict().keys()), list(latent_registry.keys()))
        if available_tools:
            # 有工具时：添加灵活的约束 —— 可以调用工具或返回 JSON，但必须是其中之一
            # 这样既保留了工具调用的灵活性，也确保了不调用工具时的输出格式
            full_system = base_system + "\n\n" + _schema_to_prompt(schema, with_tools=True)
            logger.debug(
                "[%s] 有 %d 个可用工具，添加灵活约束（工具调用或 JSON）",
                self.provider, len(available_tools)
            )
        else:
            # 无工具时：添加强制 JSON 约束，确保返回结构化 JSON 响应
            full_system = base_system + "\n\n" + _schema_to_prompt(schema, with_tools=False)
            logger.debug("[%s] 无可用工具，添加强制 JSON 约束", self.provider)

        # 非 VLM 配置：剥除图片内容，仅保留文本
        if not self._vision_enabled:
            user_content = _strip_images(user_content)

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_content},
        ]

        log_prompt(self.provider, full_system, user_content)

        logger.debug(
            "[%s] 工具声明数量: 原始=%d, 过滤后=%d",
            self.provider,
            len(tool_declarations),
            len(available_tools or [])
        )
        if available_tools:
            tool_names = [t['function']['name'] for t in available_tools]
            logger.debug("[%s] 可用工具: %s", self.provider, ", ".join(tool_names))

        create_kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": gen.get("temperature", 1.0),
            "max_tokens": gen.get("max_output_tokens", 8192),
            "presence_penalty": gen.get("presence_penalty", 0.0),
            "frequency_penalty": gen.get("frequency_penalty", 0.0),
        }

        if available_tools:
            # 有工具可用时不设置 response_format：
            # 部分 provider（如硅基流动）不支持 tools + response_format 同时使用，
            # 会导致工具调用后模型返回 content=null；系统提示中已移除强制 JSON 约束。
            create_kwargs["tools"] = available_tools
            create_kwargs["tool_choice"] = "auto"
            logger.debug("[%s] 已设置工具: tools=%d个, tool_choice=auto", self.provider, len(available_tools))
        else:
            create_kwargs["response_format"] = {"type": "json_object"}
            logger.debug("[%s] 无可用工具，设置 response_format=json_object", self.provider)

        tool_calls_log: list[dict] = []
        tool_round = 0
        _tok_prompt = 0
        _tok_output = 0

        while True:
            response = self.client.chat.completions.create(**create_kwargs)

            if _u := response.usage:
                _tok_prompt += _u.prompt_tokens or 0
                _tok_output += _u.completion_tokens or 0

            if not response.choices:
                logger.warning("[%s] response.choices 为空", self.provider)
                return None, None, False, tool_calls_log, full_system

            msg = response.choices[0].message
            
            # ── 调试：详细记录响应结构 ──
            logger.debug("[%s] 原始 message 对象类型: %s", self.provider, type(msg).__name__)
            logger.debug(
                "[%s] message.tool_calls: %s | 类型: %s | 长度: %s",
                self.provider, 
                repr(msg.tool_calls),
                type(msg.tool_calls).__name__ if msg.tool_calls else 'None',
                len(msg.tool_calls) if msg.tool_calls else 0
            )
            
            # 检查是否有其他可能的工具调用字段
            for attr_name in ['function_call', 'tool_use', 'function', 'tools', 'finish_reason']:
                if hasattr(msg, attr_name):
                    try:
                        val = getattr(msg, attr_name)
                        logger.debug("[%s] message.%s: %s", self.provider, attr_name, repr(val))
                    except Exception as e:
                        logger.debug("[%s] message.%s: (读取失败) %s", self.provider, attr_name, e)
            
            # 记录当轮响应的元信息
            tool_calls_count = len(msg.tool_calls) if msg.tool_calls else 0
            logger.info(
                "[%s] 第 %d 轮模型响应 — 工具调用数: %d, 有内容: %s",
                self.provider, tool_round + 1, tool_calls_count, bool(msg.content)
            )
            if tool_calls_count > 0:
                func_names = [tc.function.name for tc in msg.tool_calls]
                logger.info("[%s] 模型请求的工具: %s", self.provider, ", ".join(func_names))
            else:
                # ── 调试：当没有工具调用时，更详细地记录 ──
                logger.debug(
                    "[%s] 模型未请求工具调用。finish_reason: %s",
                    self.provider, msg.finish_reason if hasattr(msg, 'finish_reason') else 'N/A'
                )
                if msg.content:
                    if isinstance(msg.content, str):
                        content_preview = msg.content[:200].replace('\n', ' ')
                    elif isinstance(msg.content, list):
                        texts = [p.get("text", "") for p in msg.content if isinstance(p, dict) and "text" in p]
                        content_preview = " ".join(texts)[:200].replace('\n', ' ')
                    else:
                        content_preview = str(msg.content)[:200].replace('\n', ' ')
                    logger.debug("[%s] 响应内容摘要: %s...", self.provider, content_preview)

            if (
                msg.tool_calls
                and tool_registry
                and budget_mgr.any_available()
            ):
                tool_round += 1
                breaker.begin_round(tool_round)
                assistant_msg: dict = {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
                messages.append(assistant_msg)

                pending_injections: list[str] = []

                # 阶段 1：顺序预检（熔断/配额/注册，保护共享状态）
                # slot 键: tc, fn_name, args, fn, result(None=待执行), circuit_broken
                _slots: list[dict] = []
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    _fn = tool_registry.get(fn_name)
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except (ValueError, json.JSONDecodeError):
                        args = {}
                    slot: dict = {
                        "tc": tc, "fn_name": fn_name, "args": args,
                        "fn": _fn, "result": None, "circuit_broken": False,
                    }
                    if breaker.check_and_record(fn_name, args):
                        slot["result"] = {"error": f"CIRCUIT_BREAKER_TRIPPED: tool='{fn_name}' consecutive_calls={breaker.max_streak} threshold={breaker.max_streak}. Tool call REJECTED and tool REMOVED from registry. You MUST stop calling this tool and output final schema immediately."}
                        logger.warning("[%s] 熔断触发: 工具 %s 连续 %d 轮相同调用，已拦截并移除", self.provider, fn_name, breaker.max_streak)
                        tool_registry.pop(fn_name, None)
                        tool_declarations[:] = [d for d in tool_declarations if d.get("name") != fn_name]
                        slot["circuit_broken"] = True
                    elif not budget_mgr.is_available(fn_name):
                        slot["result"] = {"error": f"工具 {fn_name} 本轮调用次数已耗尽，请直接生成回复。"}
                        logger.info("[%s] 工具 %s 配额已耗尽", self.provider, fn_name)
                    elif _fn is None:
                        slot["result"] = {"error": f"未知工具: {fn_name}"}
                    _slots.append(slot)

                # 阶段 2：并行执行需要实际调用的工具
                _provider = self.provider

                def _exec_openai(slot: dict) -> None:
                    _fn_name = slot["fn_name"]
                    logger.info("[%s] 执行工具开始: %s", _provider, _fn_name)
                    try:
                        slot["result"] = slot["fn"](**slot["args"])
                        if isinstance(slot["result"], dict):
                            if err := slot["result"].get("error"):
                                logger.info("[%s] 执行工具完毕（失败）: %s — %s", _provider, _fn_name, err)
                            else:
                                logger.info("[%s] 执行工具完毕（成功）: %s 返回字段: %s", _provider, _fn_name, list(slot["result"].keys()))
                        else:
                            logger.info("[%s] 执行工具完毕（成功）: %s", _provider, _fn_name)
                    except Exception as e:
                        logger.warning("[%s] 执行工具异常: %s — %s", _provider, _fn_name, e)
                        slot["result"] = {"error": str(e)}

                _pending_slots = [s for s in _slots if s["result"] is None]
                if _pending_slots:
                    with ThreadPoolExecutor(max_workers=len(_pending_slots)) as _pool:
                        list(_pool.map(_exec_openai, _pending_slots))

                # 阶段 3：顺序后处理（消耗配额、收集注入、记录日志、追加消息，保持原始顺序）
                for slot in _slots:
                    fn_name = slot["fn_name"]
                    tc = slot["tc"]
                    args = slot["args"]
                    result_data = slot["result"]
                    circuit_broken = slot["circuit_broken"]

                    if not circuit_broken:
                        budget_mgr.consume(fn_name)

                        # 提取注入信号（内部协议，不返回给模型）
                        if isinstance(result_data, dict) and "_inject_tools" in result_data:
                            pending_injections.extend(result_data.pop("_inject_tools") or [])

                    tool_calls_log.append({
                        "round": tool_round,
                        "function": fn_name,
                        "arguments": args,
                        "result": result_data,
                        "circuit_broken": circuit_broken,
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result_data, ensure_ascii=False),
                    })

                # 注入通过 get_tools 激活的潜伏工具
                for inj_name in pending_injections:
                    if inj_name in latent_registry:
                        inj_decl, inj_handler = latent_registry.pop(inj_name)
                        tool_declarations.append(inj_decl)
                        tool_registry[inj_name] = inj_handler
                        budget_mgr.add_tool(inj_name, inj_decl.get("max_calls_per_response", 1))
                        logger.info("[%s] 注入潜伏工具: %s", self.provider, inj_name)
                    else:
                        logger.warning("[%s] 无法注入工具 %s：不在潜伏工具列表中或已激活", self.provider, inj_name)

                if available_tools := self._to_openai_tools(
                    budget_mgr.filter_declarations(tool_declarations)
                ):
                    create_kwargs["tools"] = available_tools
                else:
                    create_kwargs.pop("tools", None)
                    create_kwargs.pop("tool_choice", None)
                    # 所有工具配额耗尽后恢复 response_format，确保最终回复为合法 JSON
                    create_kwargs["response_format"] = {"type": "json_object"}
                    logger.info(
                        "[%s] 所有工具配额已耗尽，移除工具声明并恢复 response_format",
                        self.provider,
                    )

                # 【统一约束逻辑】与首次请求保持一致：
                # - 有工具时：灵活约束（工具调用或 JSON）
                # - 无工具时：强制 JSON 约束
                base_system_updated = system_prompt_builder(list(budget_mgr.get_budget_dict().keys()), list(latent_registry.keys()))
                if available_tools:
                    updated_system = base_system_updated + "\n\n" + _schema_to_prompt(schema, with_tools=True)
                else:
                    updated_system = base_system_updated + "\n\n" + _schema_to_prompt(schema, with_tools=False)
                messages[0]["content"] = updated_system

                # 刷新 user prompt（含最新聊天记录/时间/新消息）
                if user_content_refresher is not None:
                    fresh = user_content_refresher()
                    if not self._vision_enabled:
                        fresh = _strip_images(fresh)
                    messages[1]["content"] = fresh
                    logger.info("[%s] 工具调用第 %d 轮后已刷新 user prompt", self.provider, tool_round)
            else:
                if msg.tool_calls:
                    logger.info("[%s] 模型仍尝试调用工具（配额已耗尽），但已停止处理", self.provider)
                else:
                    logger.info("[%s] 模型无工具调用，响应内容已准备就绪", self.provider)
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

        raw_content = msg.content
        if isinstance(raw_content, str):
            text = raw_content
        elif isinstance(raw_content, list):
            text = "\n".join(
                p.get("text", "") for p in raw_content
                if isinstance(p, dict) and "text" in p
            )
        else:
            text = ""
        log_response(self.provider, text)

        # 记录原始响应摘要
        if text:
            text_preview = text[:200].replace('\n', ' ')
            logger.info("[%s] 原始响应 — 长度: %d 字节, 摘要: %s...", self.provider, len(text), text_preview)
        else:
            logger.warning("[%s] 原始响应为空", self.provider)

        logger.info(
            "[%s] Token 用量（全轮累计）— 输入: %d, 输出: %d, 总计: %d",
            self.provider,
            _tok_prompt,
            _tok_output,
            _tok_prompt + _tok_output,
        )

        if not text:
            logger.warning("[%s] response.content 为空", self.provider)
            return None, None, False, tool_calls_log, full_system

        result, repaired = self._parse_and_validate_json(text, schema, gen)
        return result, None, repaired, tool_calls_log, full_system

    def _parse_and_validate_json(self, text: str, schema: dict, gen: dict) -> tuple[dict, bool]:
        """解析并校验 JSON，支持错误时自动修复。"""
        max_repair = gen.get("json_self_repair_retries", 1)
        try:
            result, repaired = clean_and_parse(text, f"[{self.provider}]")
            result, struct_repaired = hoist_decision_motivation(result)
            if struct_repaired:
                logger.warning(
                    "[%s] decision.motivation 误放在 action 子对象中，已自动提升",
                    self.provider,
                )
                repaired = True
            result, fill_repaired = fill_missing_motivations(result)
            if fill_repaired:
                logger.warning(
                    "[%s] motivation 字段缺失，已自动填入占位文本",
                    self.provider,
                )
                repaired = True
            result, add_prop_repaired = remove_additional_properties_key(result)
            if add_prop_repaired:
                logger.warning(
                    "[%s] 模型错误地输出了 additionalProperties 键，已自动移除",
                    self.provider,
                )
                repaired = True
            validate(instance=result, schema=schema)
            status = "已修复" if repaired else "原始"
            logger.info("[%s] JSON 解析成功（%s） — %d 个顶层字段", self.provider, status, len(result))
            return result, repaired
        except (json.JSONDecodeError, ValidationError) as _parse_err:
            if max_repair <= 0:
                raise

            if isinstance(_parse_err, ValidationError):
                error_msg = f"ValidationError: {_parse_err.message}"
            else:
                error_msg = f"JSONDecodeError: {_parse_err.msg}"
            logger.warning(
                "[%s] JSON 解析或校验失败 (%s)，启动 LLM 自修复（最大 %d 次）",
                self.provider, error_msg, max_repair
            )

            last_err = _parse_err
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
                        repaired_raw, f"[{self.provider}][self_repair#{attempt}]"
                    )
                    validate(instance=result, schema=schema)

                    logger.info("[%s] JSON 自修复第 %d 次成功", self.provider, attempt)
                    return result, True
                except (json.JSONDecodeError, ValidationError) as e2:
                    last_err = e2
                    e2_msg = e2.message if isinstance(e2, ValidationError) else str(e2)
                    logger.warning(
                        "[%s] JSON 自修复第 %d/%d 次仍失败: %s",
                        self.provider, attempt, max_repair, e2_msg
                    )

            logger.error(
                "[%s] JSON 自修复 %d 次全部失败，放弃", self.provider, max_repair
            )
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

        repair_prompt += (
            f"JSON Schema：\n{schema_json}\n\n"
            f"原始文本：\n{raw_text}"
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": repair_prompt}],
            temperature=0.0,
            max_tokens=8192,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or "" if response.choices else ""

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
    return GeminiAdapter(cfg) if provider == "gemini" else OpenAICompatAdapter(cfg)


def build_watcher_adapter_cfg(main_cfg: dict, watcher_cfg: dict) -> dict:
    """构建 watcher 专用的 adapter 配置。

    watcher 有自己的字段则覆盖，否则沿用主模型配置；不需要 thinking。
    """
    cfg = dict(main_cfg)
    if "provider" in watcher_cfg:
        cfg["provider"] = watcher_cfg["provider"]
    if "base_url" in watcher_cfg:
        cfg["base_url"] = watcher_cfg["base_url"]
    cfg["model"] = watcher_cfg.get("model", main_cfg.get("model"))
    cfg["model_name"] = watcher_cfg.get("model_name", cfg["model"])
    if "generation" in watcher_cfg:
        cfg["generation"] = watcher_cfg["generation"]
    cfg.pop("thinking", None)  # watcher 不需要 thinking
    return cfg


def build_is_adapter_cfg(main_cfg: dict, is_cfg: dict) -> dict:
    """构建 IS（中断哨兵）专用的 adapter 配置。

    is_cfg 有自己的字段则覆盖，否则沿用主模型配置。
    - thinking: is_cfg 中有配置则使用，否则保留主模型配置（不强制关闭）。
    - vision:   is_cfg 中有配置则使用，否则保留主模型配置。
    """
    cfg = dict(main_cfg)
    if "provider" in is_cfg:
        cfg["provider"] = is_cfg["provider"]
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
