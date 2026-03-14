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
        system_prompt: str,
        user_content: str,
        gen: dict,
        schema: dict,
        tool_declarations: list | None = None,
        tool_registry: dict | None = None,
    ) -> tuple[dict | None, dict | None, bool]:
        """调用模型并解析 JSON 结果，支持工具调用循环。

        tool_declarations: OpenAI 格式的 tools 列表
        tool_registry:     {函数名: callable} 字典
        返回 (result_dict, grounding_dict, repaired)，失败时返回 (None, None, False)。
        """
        full_system = system_prompt + "\n\n" + _schema_to_prompt(schema)

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_content},
        ]

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

        if tool_declarations:
            create_kwargs["tools"] = tool_declarations
            create_kwargs["tool_choice"] = "auto"

        if self.reasoning_effort and self.provider == "gemini":
            create_kwargs["reasoning_effort"] = self.reasoning_effort

        # 工具调用循环
        while True:
            response = self.client.chat.completions.create(**create_kwargs)

            if not response.choices:
                logger.warning("[%s] response.choices 为空", self.provider)
                return None, None, False

            msg = response.choices[0].message

            if msg.tool_calls and tool_registry:
                # 把 assistant 的 tool_calls 消息加入上下文
                # 需要保留 extra_content（含 Gemini 3 的 thought_signature）
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
                    extra = getattr(tc, "extra_content", None)
                    if extra is None and hasattr(tc, "model_extra"):
                        extra = tc.model_extra.get("extra_content")
                    if extra:
                        tc_dict["extra_content"] = extra
                    tc_list.append(tc_dict)
                assistant_msg["tool_calls"] = tc_list
                messages.append(assistant_msg)

                for tc in msg.tool_calls:
                    fn = tool_registry.get(tc.function.name)
                    if fn is None:
                        result_data = {"error": f"未知工具: {tc.function.name}"}
                    else:
                        try:
                            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                            result_data = fn(**args)
                        except Exception as e:
                            result_data = {"error": str(e)}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result_data, ensure_ascii=False),
                    })
            else:
                break

        text = msg.content
        if not text:
            logger.warning("[%s] response.content 为空", self.provider)
            return None, None, False

        result, repaired = clean_and_parse(text, f"[{self.provider}]")
        return result, None, repaired


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
