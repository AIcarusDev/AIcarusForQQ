"""OpenAI-compatible transport primitives.

This module owns provider resolution, SDK client construction, provider-specific
generation argument normalization, and the raw chat-completions call. It does
not know about consciousness flow, AIC Action, or local tool execution.
"""

from __future__ import annotations

import logging
import os
import re
import time
from types import SimpleNamespace
from typing import Any

import httpx
from openai import OpenAI

from .error_policy import is_transient_provider_failure
from .profiles import resolve_model_provider

logger = logging.getLogger("AICQ.llm.transport")

_TOP_LEVEL_SAMPLING_KEYS = {
    "top_p",
    "presence_penalty",
    "frequency_penalty",
}

_EXTRA_BODY_SAMPLING_KEYS = {
    "top_k",
    "min_p",
    "repeat_penalty",
}

_TRANSIENT_CREATE_MAX_ATTEMPTS = 2
_TRANSIENT_RETRY_DELAY_SECONDS = 0.5
_EXPLICIT_STREAM_TOKEN = re.compile(
    r"(?<![a-z0-9_])stream(?:ing)?(?![a-z0-9_])",
    re.IGNORECASE,
)


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
    existing_thinking = extra_body.get("thinking")
    if enable_thinking is None and isinstance(existing_thinking, dict):
        thinking_type = str(existing_thinking.get("type") or "").strip().lower()
        if thinking_type in {"enabled", "disabled"}:
            enable_thinking = thinking_type == "enabled"

    if thinking_control == "enable_thinking":
        extra_body.pop("thinking", None)
        if "enable_thinking" in gen:
            extra_body["enable_thinking"] = bool(gen["enable_thinking"])
        elif "enable_thinking" not in extra_body:
            extra_body["enable_thinking"] = True
    elif thinking_control == "thinking":
        extra_body.pop("enable_thinking", None)
        extra_body["thinking"] = {
            "type": "enabled" if enable_thinking is not False else "disabled"
        }
    else:
        extra_body.pop("enable_thinking", None)
        extra_body.pop("thinking", None)

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


def add_enabled_sampling_kwargs(create_kwargs: dict, gen: dict) -> None:
    advanced = gen.get("advanced_sampling")
    if not isinstance(advanced, dict):
        return

    extra_body = dict(create_kwargs.get("extra_body") or {})
    for key, cfg in advanced.items():
        if key not in _TOP_LEVEL_SAMPLING_KEYS and key not in _EXTRA_BODY_SAMPLING_KEYS:
            continue
        if not isinstance(cfg, dict) or not cfg.get("enabled"):
            continue
        value = cfg.get("value")
        if value is None:
            continue
        if key in _TOP_LEVEL_SAMPLING_KEYS:
            create_kwargs[key] = value
        else:
            extra_body[key] = value

    if extra_body:
        create_kwargs["extra_body"] = extra_body
    else:
        create_kwargs.pop("extra_body", None)


def prepare_streaming_create_kwargs(create_kwargs: dict) -> dict:
    """Return create kwargs for the preferred streaming chat-completion path."""
    request_kwargs = dict(create_kwargs)
    request_kwargs["stream"] = True
    request_kwargs.setdefault("stream_options", {"include_usage": True})
    return request_kwargs


def aggregate_chat_completion_stream(stream: Any) -> Any:
    """Consume OpenAI-compatible chat completion chunks into a response-like object."""
    return aggregate_chat_completion_stream_with_callbacks(stream)


def _observe_stream_callback(callback, *args: Any) -> None:
    try:
        callback(*args)
    except Exception as exc:
        if getattr(exc, "stream_abort", False):
            raise
        logger.debug("stream observer callback failed", exc_info=True)


def aggregate_chat_completion_stream_with_callbacks(
    stream: Any,
    *,
    on_text_delta=None,
    on_chunk=None,
) -> Any:
    """Consume chat completion chunks, optionally observing text deltas."""
    chunks_seen = 0
    choices_seen = False
    content_parts: list[str] = []
    finish_reason = None
    usage = None
    response_id = ""
    created = None
    model = ""
    tool_call_parts: dict[int, dict[str, Any]] = {}

    for chunk in stream:
        chunks_seen += 1
        if on_chunk is not None:
            _observe_stream_callback(on_chunk, chunk)
        response_id = response_id or str(getattr(chunk, "id", "") or "")
        created = created if created is not None else getattr(chunk, "created", None)
        model = model or str(getattr(chunk, "model", "") or "")
        if getattr(chunk, "usage", None) is not None:
            usage = getattr(chunk, "usage", None)

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        choices_seen = True
        choice = choices[0]
        if getattr(choice, "finish_reason", None):
            finish_reason = getattr(choice, "finish_reason", None)
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue

        content = getattr(delta, "content", None)
        if isinstance(content, str):
            content_parts.append(content)
            if on_text_delta is not None:
                _observe_stream_callback(on_text_delta, content)

        _merge_delta_tool_calls(tool_call_parts, getattr(delta, "tool_calls", None))
        _merge_delta_function_call(tool_call_parts, getattr(delta, "function_call", None))

    if not chunks_seen:
        return SimpleNamespace(choices=[], usage=usage)

    tool_calls = _build_aggregated_tool_calls(tool_call_parts)
    if not choices_seen and not content_parts and not tool_calls:
        return SimpleNamespace(
            id=response_id,
            created=created,
            model=model,
            choices=[],
            usage=usage,
        )

    message = SimpleNamespace(
        content="".join(content_parts),
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
        index=0,
    )
    return SimpleNamespace(
        id=response_id,
        created=created,
        model=model,
        choices=[choice],
        usage=usage,
    )


def _merge_delta_tool_calls(
    slots: dict[int, dict[str, Any]],
    tool_calls_delta: Any,
) -> None:
    if not tool_calls_delta:
        return
    for fallback_index, tool_call in enumerate(tool_calls_delta):
        raw_index = getattr(tool_call, "index", None)
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            index = fallback_index
        slot = slots.setdefault(index, _new_tool_call_slot())
        if getattr(tool_call, "id", None):
            slot["id"] = str(getattr(tool_call, "id"))
        if getattr(tool_call, "type", None):
            slot["type"] = str(getattr(tool_call, "type"))

        function = getattr(tool_call, "function", None)
        if function is None:
            continue
        if getattr(function, "name", None):
            slot["function"]["name"] += str(getattr(function, "name"))
        if getattr(function, "arguments", None):
            slot["function"]["arguments"] += str(getattr(function, "arguments"))


def _merge_delta_function_call(
    slots: dict[int, dict[str, Any]],
    function_call_delta: Any,
) -> None:
    if function_call_delta is None:
        return
    slot = slots.setdefault(0, _new_tool_call_slot())
    if getattr(function_call_delta, "name", None):
        slot["function"]["name"] += str(getattr(function_call_delta, "name"))
    if getattr(function_call_delta, "arguments", None):
        slot["function"]["arguments"] += str(getattr(function_call_delta, "arguments"))


def _new_tool_call_slot() -> dict[str, Any]:
    return {
        "id": "",
        "type": "function",
        "function": {"name": "", "arguments": ""},
    }


def _build_aggregated_tool_calls(tool_call_parts: dict[int, dict[str, Any]]) -> list[Any]:
    tool_calls: list[Any] = []
    for index in sorted(tool_call_parts):
        slot = tool_call_parts[index]
        function = slot["function"]
        tool_calls.append(
            SimpleNamespace(
                id=slot["id"] or f"call_{index}",
                type=slot["type"] or "function",
                function=SimpleNamespace(
                    name=function["name"],
                    arguments=function["arguments"],
                ),
            )
        )
    return tool_calls


def _is_stream_options_unsupported(exc: Exception) -> bool:
    text = str(exc).lower()
    return "stream_options" in text or "include_usage" in text


def _is_streaming_unsupported(exc: Exception) -> bool:
    text = str(exc).lower()
    if _EXPLICIT_STREAM_TOKEN.search(text) is None:
        return False
    if re.search(
        r"\bstream(?:ing)?\b\s+(?:is\s+)?(?:unsupported|not\s+supported|unavailable|disabled)\b",
        text,
    ):
        return True
    if re.search(
        r"\b(?:does\s+not|doesn't|cannot|can't)\s+support\s+(?:the\s+)?\bstream(?:ing)?\b",
        text,
    ):
        return True
    has_field_context = any(
        marker in text
        for marker in (
            "parameter",
            "argument",
            "field",
            "option",
            "value",
            "mode",
            "capability",
            "feature",
            "extra input",
            "extra_forbidden",
        )
    )
    has_rejection = any(
        marker in text
        for marker in (
            "unsupported",
            "not support",
            "unrecognized",
            "unknown",
            "extra_forbidden",
            "invalid",
            "unexpected",
            "not permitted",
        )
    )
    return has_field_context and has_rejection


def _create_chat_completion_with_transient_retry(
    client: Any,
    *,
    provider: str,
    all_messages: list,
    request_kwargs: dict,
) -> Any:
    """Create once, retrying only an explicitly transient upstream failure."""

    for attempt in range(1, _TRANSIENT_CREATE_MAX_ATTEMPTS + 1):
        try:
            return client.chat.completions.create(
                messages=all_messages,  # type: ignore
                **request_kwargs,
            )
        except Exception as exc:
            if (
                attempt >= _TRANSIENT_CREATE_MAX_ATTEMPTS
                or not is_transient_provider_failure(exc)
            ):
                raise
            request_mode = "流式" if request_kwargs.get("stream") is True else "整块"
            logger.warning(
                "[%s] 供应商上游临时失败，%.1fs 后重试同一%s请求 (%d/%d): %s",
                provider,
                _TRANSIENT_RETRY_DELAY_SECONDS,
                request_mode,
                attempt + 1,
                _TRANSIENT_CREATE_MAX_ATTEMPTS,
                exc,
            )
            if _TRANSIENT_RETRY_DELAY_SECONDS > 0:
                time.sleep(_TRANSIENT_RETRY_DELAY_SECONDS)

    raise AssertionError("unreachable transient retry state")


def _create_non_streaming_chat_completion(
    client: Any,
    *,
    provider: str,
    all_messages: list,
    create_kwargs: dict,
) -> Any:
    fallback_kwargs = dict(create_kwargs)
    fallback_kwargs.pop("stream", None)
    fallback_kwargs.pop("stream_options", None)
    return _create_chat_completion_with_transient_retry(
        client,
        provider=provider,
        all_messages=all_messages,
        request_kwargs=fallback_kwargs,
    )


def _looks_like_chat_completion(response: Any) -> bool:
    return hasattr(response, "choices") and not hasattr(response, "__next__")


def create_streamed_chat_completion(
    client: Any,
    *,
    provider: str,
    all_messages: list,
    create_kwargs: dict,
    on_text_delta=None,
    on_chunk=None,
) -> Any:
    """Create a streaming chat completion and aggregate chunks into response shape."""
    stream_kwargs = prepare_streaming_create_kwargs(create_kwargs)
    try:
        stream = _create_chat_completion_with_transient_retry(
            client,
            provider=provider,
            all_messages=all_messages,
            request_kwargs=stream_kwargs,
        )
        if _looks_like_chat_completion(stream):
            return stream
    except Exception as exc:
        if _is_stream_options_unsupported(exc):
            logger.warning(
                "[%s] stream_options 不受支持，改用不含 usage 的流式请求: %s",
                provider,
                exc,
            )
            without_usage_kwargs = dict(stream_kwargs)
            without_usage_kwargs.pop("stream_options", None)
            try:
                stream = _create_chat_completion_with_transient_retry(
                    client,
                    provider=provider,
                    all_messages=all_messages,
                    request_kwargs=without_usage_kwargs,
                )
                if _looks_like_chat_completion(stream):
                    return stream
            except Exception as retry_exc:
                if not _is_streaming_unsupported(retry_exc):
                    raise
                logger.warning(
                    "[%s] provider 不支持流式，临时回退整块响应: %s",
                    provider,
                    retry_exc,
                )
                return _create_non_streaming_chat_completion(
                    client,
                    provider=provider,
                    all_messages=all_messages,
                    create_kwargs=create_kwargs,
                )
        elif _is_streaming_unsupported(exc):
            logger.warning(
                "[%s] provider 不支持流式，临时回退整块响应: %s",
                provider,
                exc,
            )
            return _create_non_streaming_chat_completion(
                client,
                provider=provider,
                all_messages=all_messages,
                create_kwargs=create_kwargs,
            )
        else:
            raise
    return aggregate_chat_completion_stream_with_callbacks(
        stream,
        on_text_delta=on_text_delta,
        on_chunk=on_chunk,
    )


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
        self.assistant_prefill_supported: bool = bool(
            provider_cfg.get("supports_assistant_prefill", True)
        )

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
        on_text_delta=None,
        on_chunk=None,
    ):
        """发起 streaming chat completion and return an aggregated response."""
        return create_streamed_chat_completion(
            self.client,
            provider=self.provider,
            all_messages=all_messages,
            create_kwargs=create_kwargs,
            on_text_delta=on_text_delta,
            on_chunk=on_chunk,
        )

    def create_chat_completion_stream(
        self,
        *,
        all_messages: list,
        create_kwargs: dict,
    ):
        """发起原始 streaming chat completion，供未来按 chunk 处理的调用点使用。"""
        return _create_chat_completion_with_transient_retry(
            self.client,
            provider=self.provider,
            all_messages=all_messages,
            request_kwargs=prepare_streaming_create_kwargs(create_kwargs),
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
