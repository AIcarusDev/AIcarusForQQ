"""provider.py — 模型适配层

通过 config.yaml 的 `provider` 字段一键切换后端：
  - gemini      : Google Gemini（环境变量: GEMINI_API_KEY）
  - siliconflow : 硅基流动  （环境变量: SILICONFLOW_API_KEY）
  - bigmodel    : 智谱 AI   （环境变量: BIGMODEL_API_KEY）
"""

import json
import logging
import os
from abc import ABC, abstractmethod

from json_repair import clean_and_parse

logger = logging.getLogger("mita.provider")


class BaseAdapter(ABC):
    @abstractmethod
    def call(
        self,
        system_prompt: str,
        user_content: str,
        gen: dict,
        schema: dict,
    ) -> tuple[dict | None, dict | None, bool]:
        """调用模型并解析 JSON 结果。

        返回 (result_dict, grounding_dict, repaired)，失败时返回 (None, None, False)。
        repaired=True 表示原始输出经过了 json_repair 清洗才能解析。
        """
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """返回该 provider 可用的模型 ID 列表。"""
        ...


# ── Gemini 适配器 ──────────────────────────────────────────────────────────────

class GeminiAdapter(BaseAdapter):
    def __init__(self, cfg: dict):
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
        self.model = cfg.get("model", "gemini-2.0-flash")

        safety_threshold = cfg.get("safety", {}).get("threshold", "OFF")
        self.safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory[cat],
                threshold=types.HarmBlockThreshold[safety_threshold],
            )
            for cat in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ]
        ]

        thinking_level = cfg.get("thinking", {}).get("level", None)
        self.thinking_config = (
            types.ThinkingConfig(thinking_level=types.ThinkingLevel[thinking_level])
            if thinking_level else None
        )

    def call(self, system_prompt, user_content, gen, schema):
        types = self._types
        google_search_tool = types.Tool(google_search=types.GoogleSearch())

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=gen.get("temperature", 1.0),
                top_p=gen.get("top_p", 0.95),
                top_k=gen.get("top_k", 40),
                max_output_tokens=gen.get("max_output_tokens", 8192),
                presence_penalty=gen.get("presence_penalty", 0.0),
                frequency_penalty=gen.get("frequency_penalty", 0.0),
                response_mime_type="application/json",
                response_json_schema=schema,
                tools=[google_search_tool],
                safety_settings=self.safety_settings,
                thinking_config=self.thinking_config,
            ),
        )

        if not response.text:
            logger.warning("[Gemini] response.text 为空，模型可能被安全过滤拦截")
            return None, None, False

        result, repaired = clean_and_parse(response.text, "[Gemini]")
        return result, self._extract_grounding(response), repaired

    def list_models(self) -> list[str]:
        try:
            result = []
            for m in self.client.models.list():
                name = getattr(m, "name", "") or ""
                if name.startswith("models/"):
                    name = name[len("models/"):]
                if name:
                    result.append(name)
            return sorted(set(result))
        except Exception:
            return []

    def _extract_grounding(self, response) -> dict | None:
        try:
            meta = (
                response.candidates[0].grounding_metadata
                if response.candidates else None
            )
            if meta:
                g = {}
                if meta.web_search_queries:
                    g["search_queries"] = list(meta.web_search_queries)
                if meta.grounding_chunks:
                    g["sources"] = [
                        {
                            "title": getattr(c.web, "title", ""),
                            "uri": getattr(c.web, "uri", ""),
                        }
                        for c in meta.grounding_chunks if hasattr(c, "web")
                    ]
                return g
        except Exception:
            pass
        return None


# ── OpenAI 兼容适配器（硅基流动 / 智谱）──────────────────────────────────────

class OpenAICompatAdapter(BaseAdapter):
    def __init__(self, cfg: dict):
        from openai import OpenAI

        provider = cfg.get("provider", "siliconflow")
        if provider == "bigmodel":
            default_url = "https://open.bigmodel.cn/api/paas/v4"
            api_key = os.getenv("BIGMODEL_API_KEY", "")
        else:  # siliconflow
            default_url = "https://api.siliconflow.cn/v1"
            api_key = os.getenv("SILICONFLOW_API_KEY", "")

        base_url = cfg.get("base_url", default_url)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = cfg.get("model", "THUDM/GLM-4-32B-0414")

    def list_models(self) -> list[str]:
        try:
            page = self.client.models.list()
            return sorted(m.id for m in page.data)
        except Exception:
            return []

    def call(self, system_prompt, user_content, gen, schema):
        full_system = system_prompt + "\n\n" + _schema_to_prompt(schema)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=gen.get("temperature", 1.0),
            top_p=gen.get("top_p", 0.95),
            max_tokens=gen.get("max_output_tokens", 8192),
            presence_penalty=gen.get("presence_penalty", 0.0),
            frequency_penalty=gen.get("frequency_penalty", 0.0),
        )

        if not response.choices:
            logger.warning("[OpenAICompat] response.choices 为空（可能触发内容过滤或账户异常）")
            return None, None, False

        text = response.choices[0].message.content
        if not text:
            logger.warning("[OpenAICompat] response.content 为空")
            return None, None, False

        result, repaired = clean_and_parse(text, "[OpenAICompat]")
        return result, None, repaired  # OpenAI 兼容接口不支持 grounding


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


def create_adapter(cfg: dict) -> BaseAdapter:
    """根据 config.yaml 中的 provider 字段创建对应适配器。"""
    provider = cfg.get("provider", "gemini")
    if provider == "gemini":
        return GeminiAdapter(cfg)
    elif provider in ("siliconflow", "bigmodel"):
        return OpenAICompatAdapter(cfg)
    else:
        raise ValueError(
            f"未知的 provider: {provider!r}，"
            "可选值: gemini / siliconflow / bigmodel"
        )
