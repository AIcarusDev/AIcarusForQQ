"""OpenAI-compatible transport primitives.

This module owns provider resolution, SDK client construction, provider-specific
generation argument normalization, and the raw chat-completions call. It does
not know about consciousness flow, XML tool protocol, or local tool execution.
"""

from __future__ import annotations

import logging
import os

import httpx
from openai import OpenAI

from .profiles import resolve_model_provider

logger = logging.getLogger("AICQ.llm.transport")


def _gemini_reasoning_none_supported(model: str) -> bool:
    normalized = (model or "").lower()
    return (
        "gemini-2.5" in normalized
        and "gemini-2.5-pro" not in normalized
    )


def normalize_generation_for_provider(
    gen: dict,
    *,
    thinking_control: str = "enable_thinking",
    model: str = "",
) -> dict:
    """Return generation config with provider-specific thinking controls applied."""
    gen = dict(gen or {})
    extra_body = dict(gen.get("extra_body") or {})
    enable_thinking = gen.get("enable_thinking", extra_body.get("enable_thinking"))

    if thinking_control == "enable_thinking":
        if "enable_thinking" in gen:
            extra_body["enable_thinking"] = bool(gen["enable_thinking"])
        elif "enable_thinking" not in extra_body:
            extra_body["enable_thinking"] = True
    else:
        extra_body.pop("enable_thinking", None)

    if (
        thinking_control == "reasoning_effort"
        and "reasoning_effort" not in gen
        and enable_thinking is False
        and _gemini_reasoning_none_supported(model)
    ):
        gen["reasoning_effort"] = "none"

    if extra_body:
        gen["extra_body"] = extra_body
    else:
        gen.pop("extra_body", None)
    return gen


def add_extra_generation_kwargs(create_kwargs: dict, gen: dict) -> None:
    reasoning_effort = gen.get("reasoning_effort")
    if reasoning_effort:
        create_kwargs["reasoning_effort"] = reasoning_effort


class OpenAICompatClient:
    """Raw OpenAI SDK transport for OpenAI-compatible endpoints."""

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
        self.vision_enabled = bool(cfg.get("vision", True))
        self._thinking_control: str = provider_cfg.get("thinking_control", "enable_thinking")

    def normalize_generation(self, gen: dict) -> dict:
        return normalize_generation_for_provider(
            gen,
            thinking_control=getattr(self, "_thinking_control", "enable_thinking"),
            model=getattr(self, "model", ""),
        )

    def list_models(self) -> list[str]:
        """返回该 provider 可用的模型 ID 列表。"""
        try:
            page = self.client.models.list()
            return sorted(model.id for model in page.data)
        except Exception:
            return []

    def create_chat_completion(
        self,
        *,
        all_messages: list,
        create_kwargs: dict,
    ):
        """发起 chat completion。"""
        return self.client.chat.completions.create(
            messages=all_messages,  # type: ignore
            **create_kwargs,
        )

    @staticmethod
    def to_openai_tools(declarations: list[dict]) -> list[dict]:
        """将工具声明转为 OpenAI function calling 格式。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": declaration["name"],
                    "description": declaration.get("description", ""),
                    "parameters": OpenAICompatClient.strip_extensions(
                        declaration.get("parameters", {})
                    ),
                },
            }
            for declaration in declarations
        ]

    @staticmethod
    def strip_extensions(obj: object) -> object:
        """递归去除 JSON Schema 中以 x- 开头的自定义扩展键，避免传入 LLM prompt。"""
        if isinstance(obj, dict):
            return {
                k: OpenAICompatClient.strip_extensions(v)
                for k, v in obj.items()
                if not k.startswith("x-")
            }
        if isinstance(obj, list):
            return [OpenAICompatClient.strip_extensions(item) for item in obj]
        return obj
