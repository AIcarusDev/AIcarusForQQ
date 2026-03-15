"""provider.py — 统一 OpenAI 兼容模型适配层

所有 provider 均通过 OpenAI SDK 调用，包括 Gemini（使用其 OpenAI 兼容端点）。

通过 config.yaml 的 `provider` 字段一键切换后端：
  - gemini      : Google Gemini（环境变量: GEMINI_API_KEY）
  - siliconflow : 硅基流动  （环境变量: SILICONFLOW_API_KEY）
  - bigmodel    : 智谱 AI   （环境变量: BIGMODEL_API_KEY）
"""

import json
import logging
import os

from openai import OpenAI

from json_repair import clean_and_parse

logger = logging.getLogger("mita.provider")


# ── Provider 默认配置 ───────────────────────────────────────────────────────────

_PROVIDER_DEFAULTS: dict[str, dict] = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
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

# Gemini thinking level → OpenAI reasoning_effort 映射
_THINKING_LEVEL_MAP: dict[str, str] = {
    "MINIMAL": "low",
    "LOW": "low",
    "MEDIUM": "medium",
    "HIGH": "high",
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
            func = decl.get("function", {})
            name = func.get("name", "")
            if not name:
                continue
            max_calls = decl.get("max_calls_per_response", 1)
            self._budgets[name] = {
                "total": max_calls,
                "remaining": max_calls,
                "description": func.get("description", ""),
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
        """过滤掉已耗尽配额的工具声明，返回仍可用的声明列表。

        同时移除自定义的 max_calls_per_response 字段，
        只保留 OpenAI 标准格式。
        """
        result = []
        for decl in tool_declarations:
            name = decl.get("function", {}).get("name", "")
            if self.is_available(name):
                # 移除自定义字段，保留标准 OpenAI 格式
                clean = {k: v for k, v in decl.items() if k != "max_calls_per_response"}
                result.append(clean)
        return result

    def get_budget_dict(self) -> dict[str, dict]:
        """返回完整配额字典，供 prompt dashboard 显示。"""
        return dict(self._budgets)


# ── 统一适配器 ──────────────────────────────────────────────────────────────────

class Adapter:
    """统一的 OpenAI 兼容适配器，支持所有 provider。"""

    def __init__(self, cfg: dict):
        provider = cfg.get("provider", "gemini")
        defaults = _PROVIDER_DEFAULTS.get(provider)
        if defaults is None:
            raise ValueError(
                f"未知的 provider: {provider!r}，"
                f"可选值: {' / '.join(_PROVIDER_DEFAULTS)}"
            )

        base_url = cfg.get("base_url", defaults["base_url"])
        api_key = os.getenv(defaults["env_key"], "")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = cfg.get("model", defaults["default_model"])
        self.provider = provider

        # Gemini 特有：thinking level → reasoning_effort
        thinking_level = cfg.get("thinking", {}).get("level")
        self.reasoning_effort = (
            _THINKING_LEVEL_MAP.get(thinking_level.upper())
            if thinking_level else None
        )

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
        user_content: str,
        gen: dict,
        schema: dict,
        tool_declarations: list | None = None,
        tool_registry: dict | None = None,
    ) -> tuple[dict | None, dict | None, bool, list[dict]]:
        """调用模型并解析 JSON 结果，支持按工具粒度的配额调用循环。

        system_prompt_builder: 接受 tool_budget 字典参数的可调用对象，
                               返回构建好的 system prompt 字符串。
                               签名: (tool_budget: dict) -> str
        tool_declarations:     OpenAI 格式的 tools 列表（含自定义 max_calls_per_response）
        tool_registry:         {函数名: callable} 字典
        返回 (result_dict, grounding_dict, repaired, tool_calls_log)，
        失败时返回 (None, None, False, [])。
        """
        # 初始化工具配额管理器
        budget_mgr = ToolBudgetManager(tool_declarations or [])

        # 构建初始 system prompt（含完整工具配额）
        full_system = system_prompt_builder(budget_mgr.get_budget_dict()) + "\n\n" + _schema_to_prompt(schema)

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_content},
        ]

        # 过滤掉自定义字段后的干净声明
        available_decls = budget_mgr.filter_declarations(tool_declarations or [])

        create_kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": gen.get("temperature", 1.0),
            "max_tokens": gen.get("max_output_tokens", 8192),
        }

        # Gemini OpenAI 兼容端点不支持 presence_penalty / frequency_penalty
        if self.provider != "gemini":
            create_kwargs["presence_penalty"] = gen.get("presence_penalty", 0.0)
            create_kwargs["frequency_penalty"] = gen.get("frequency_penalty", 0.0)

        if available_decls:
            create_kwargs["tools"] = available_decls
            create_kwargs["tool_choice"] = "auto"

        if self.reasoning_effort and self.provider == "gemini":
            create_kwargs["reasoning_effort"] = self.reasoning_effort

        # 工具调用循环
        tool_calls_log: list[dict] = []
        tool_round = 0
        # 绝对上限：防止意外死循环（即使配额逻辑有 bug 也不会无限循环）
        max_absolute_rounds = gen.get("max_tool_rounds", 5)

        while True:
            # 使用 with_raw_response 获取原始 HTTP JSON，
            # 以可靠提取 Gemini 3 的 thought_signature（extra_content 字段）
            # OpenAI SDK 的 Pydantic 模型会丢弃非标准字段，必须从原始响应读取
            raw_resp = self.client.chat.completions.with_raw_response.create(**create_kwargs)
            response = raw_resp.parse()

            # 从原始 JSON 构建 id -> extra_content 的映射
            extra_content_map: dict = {}
            try:
                raw_dict = json.loads(raw_resp.content)
                for tc_raw in (
                    raw_dict.get("choices", [{}])[0]
                    .get("message", {})
                    .get("tool_calls") or []
                ):
                    if tc_raw.get("id") and tc_raw.get("extra_content"):
                        extra_content_map[tc_raw["id"]] = tc_raw["extra_content"]
                if extra_content_map:
                    logger.debug(
                        "[%s] 从原始响应提取到 %d 个 thought_signature",
                        self.provider, len(extra_content_map),
                    )
                elif response.choices and response.choices[0].message.tool_calls:
                    logger.warning(
                        "[%s] 本轮工具调用未携带 thought_signature，"
                        "模型将丢失推理上下文（可能导致重复调用）",
                        self.provider,
                    )
            except Exception as e:
                logger.debug("[%s] 解析原始响应 extra_content 失败: %s", self.provider, e)

            if not response.choices:
                logger.warning("[%s] response.choices 为空", self.provider)
                return None, None, False, tool_calls_log

            msg = response.choices[0].message

            # 检查是否有工具调用，以及是否还有可用配额
            if msg.tool_calls and tool_registry and budget_mgr.any_available() and tool_round < max_absolute_rounds:
                tool_round += 1
                # 把 assistant 的 tool_calls 消息加入上下文，必须携带 thought_signature
                assistant_msg: dict = {"role": "assistant", "content": msg.content}
                tc_list = []
                for tc in msg.tool_calls:
                    tc_dict: dict = {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    # 优先从原始 JSON 取（最可靠），其次回退到 Pydantic model_extra
                    extra = extra_content_map.get(tc.id)
                    if extra is None:
                        extra = getattr(tc, "extra_content", None)
                    if extra is None and hasattr(tc, "model_extra"):
                        extra = (tc.model_extra or {}).get("extra_content")
                    if extra:
                        tc_dict["extra_content"] = extra
                    tc_list.append(tc_dict)
                assistant_msg["tool_calls"] = tc_list
                messages.append(assistant_msg)

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn = tool_registry.get(fn_name)
                    args = {}

                    # 检查该工具是否还有配额
                    if not budget_mgr.is_available(fn_name):
                        result_data = {
                            "error": f"工具 {fn_name} 本轮调用次数已耗尽，请直接生成回复。"
                        }
                        logger.info(
                            "[%s] 工具 %s 配额已耗尽，返回错误提示",
                            self.provider, fn_name,
                        )
                    elif fn is None:
                        result_data = {"error": f"未知工具: {fn_name}"}
                    else:
                        try:
                            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                            # motivation 仅供记录，不传给实际工具函数
                            call_args = {k: v for k, v in args.items() if k != "motivation"}
                            result_data = fn(**call_args)
                        except Exception as e:
                            result_data = {"error": str(e)}

                    # 消耗配额（即使出错也算消耗，避免重试浪费）
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

                # 更新下一轮的工具声明：移除已耗尽配额的工具
                available_decls = budget_mgr.filter_declarations(tool_declarations or [])
                if available_decls:
                    create_kwargs["tools"] = available_decls
                else:
                    # 所有工具配额耗尽，移除工具声明强制文本回复
                    create_kwargs.pop("tools", None)
                    create_kwargs.pop("tool_choice", None)
                    logger.info(
                        "[%s] 所有工具配额已耗尽，移除工具声明以强制文本回复",
                        self.provider,
                    )

                # 更新 system prompt 中的配额显示
                updated_system = system_prompt_builder(budget_mgr.get_budget_dict()) + "\n\n" + _schema_to_prompt(schema)
                messages[0]["content"] = updated_system
            else:
                if msg.tool_calls and not budget_mgr.any_available():
                    logger.info(
                        "[%s] 模型仍尝试调用工具但所有配额已耗尽，强制返回文本",
                        self.provider,
                    )
                break

        if tool_calls_log:
            # 统计各工具调用次数
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
        if not text:
            logger.warning("[%s] response.content 为空", self.provider)
            return None, None, False, tool_calls_log

        result, repaired = clean_and_parse(text, f"[{self.provider}]")
        return result, None, repaired, tool_calls_log


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


def create_adapter(cfg: dict) -> Adapter:
    """根据 config.yaml 中的 provider 字段创建适配器。"""
    return Adapter(cfg)
