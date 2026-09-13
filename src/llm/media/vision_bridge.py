"""vision_bridge.py — 视觉桥：为不支持视觉的模型自动描述图片

当 config.vision=false（或主模型不支持视觉）时，此模块：
  1. 调用独立的 VLM（通过 OpenAI 兼容端点）生成图片的文字描述
  2. 描述结果写入 统一图片索引
  3. 相同图片再次出现时直接复用缓存描述，不重复消费 token

同时提供 examine() 接口供 examine_image 工具调用精查。

配置段（config.yaml）：
  vision_bridge:
    enabled: true
    provider: "siliconflow"
    model: "Qwen/Qwen2-VL-7B-Instruct"
    describe_prompt: "请用2-4句话描述这张图片..."
    similarity_threshold: 10
    temperature: 0.3
    max_output_tokens: 512
    enable_thinking: false
"""

import base64
import logging
import os
from typing import Any, Optional

from .image_store import append_examination, update_description, read_image, description_claim
from .outbound_image import make_data_url
from llm.core.profiles import resolve_model_provider, resolve_model_thinking_control
from llm.core.transport import (
    ProviderRequestIdentity,
    add_extra_generation_kwargs,
    create_streamed_chat_completion,
    normalize_generation_for_provider,
)
from llm_usage_recorder import record_llm_usage

logger = logging.getLogger("AICQ.llm.media.vision")

_DEFAULT_DESCRIBE_PROMPT = (
    "请用2-4句话描述这张图片的主要内容，"
    "重点识别其中的文字、数字、人物、场景和关键视觉元素。"
    "描述要简洁、客观。"
)

_DEFAULT_EXAMINE_PROMPT_TMPL = (
    "请仔细观察这张图片，重点关注：{focus}。"
    "详细描述你在该区域或该方面的观察结果，包含文字、数字等关键信息。"
)


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class VisionBridge:
    """图片视觉桥——图片缓存 + 按需 VLM 描述生成。

    线程安全：OpenAI 同步客户端，process_entry() 应在线程池中调用。
    """

    def __init__(self, cfg: dict[str, Any]):
        """
        cfg: 完整 config.yaml 字典，或兼容旧调用传入的 vision_bridge 子字典。
        """
        full_cfg = cfg if isinstance(cfg.get("vision_bridge"), dict) else {"vision_bridge": cfg}
        bridge_cfg = full_cfg.get("vision_bridge", {})
        self._cfg = bridge_cfg
        self._enabled: bool = bool(bridge_cfg.get("enabled", False))
        self._model: str = bridge_cfg.get("model", "")
        self._provider: str = bridge_cfg.get("provider", "")
        self._base_url: str = ""
        self._api_key_env: str = ""
        self._thinking_control: str = "enable_thinking"
        self._request_identity = ProviderRequestIdentity()
        if self._enabled:
            provider_cfg: dict[str, Any] = dict(full_cfg)
            provider_cfg["provider"] = self._provider
            _provider_name, resolved, _providers = resolve_model_provider(provider_cfg)
            self._base_url = resolved.get("base_url", "")
            self._api_key_env = resolved.get("api_key_env", "")
            self._thinking_control = resolve_model_thinking_control(
                resolved,
                self._model,
            )
            self._request_identity = ProviderRequestIdentity.from_provider(resolved)
        self._describe_prompt: str = bridge_cfg.get("describe_prompt", _DEFAULT_DESCRIBE_PROMPT)
        self._sim_threshold: int = _coerce_int(bridge_cfg.get("similarity_threshold"), 10)
        self._generation: dict = {
            "temperature": _coerce_float(bridge_cfg.get("temperature"), 0.3),
            "max_output_tokens": _coerce_int(bridge_cfg.get("max_output_tokens"), 512),
            "enable_thinking": bool(bridge_cfg.get("enable_thinking", False)),
        }
        self._client = None  # openai.OpenAI，懒初始化

        if self._enabled and self._provider and self._model:
            self._init_client()

    # ── 初始化 ─────────────────────────────────────────

    def _init_client(self) -> None:
        try:
            from openai import OpenAI
            import httpx

            api_key = (
                os.environ.get(self._api_key_env, "")
                if self._api_key_env
                else ""
            )
            if not api_key:
                logger.warning(
                    "[VisionBridge] API Key 未设置（env: %s），视觉桥将不可用",
                    self._api_key_env,
                )
                return

            kwargs: dict = {"api_key": api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if default_headers := self._request_identity.headers_for():
                kwargs["default_headers"] = default_headers
            
            # 代理配置：直接从环境变量读取（OPENAI_PROXY）
            if proxy_url := os.environ.get("OPENAI_PROXY", "").strip() or None:
                http_client = httpx.Client(proxy=proxy_url)
                kwargs["http_client"] = http_client

            self._client = OpenAI(**kwargs)
            logger.info(
                "[VisionBridge] 初始化完成，模型: %s temperature=%s max_output_tokens=%s enable_thinking=%s",
                self._model,
                self._generation["temperature"],
                self._generation["max_output_tokens"],
                self._generation["enable_thinking"],
            )
        except ImportError:
            logger.warning("[VisionBridge] openai 库未安装，视觉桥不可用")
        except Exception as exc:
            logger.warning("[VisionBridge] 初始化失败: %s", exc)

    # ── 属性 ───────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """VisionBridge 是否就绪（配置已启用 + 客户端初始化成功）。"""
        return self._enabled and self._client is not None

    # ── 内部 VLM 调用 ──────────────────────────────────

    def _call_vlm(self, b64: str, mime: str, prompt: str, subfeature: str) -> str:
        """向 VLM 发送图片 + 文本提示，返回纯文本回复（同步）。"""
        if not self._client:
            raise RuntimeError("VisionBridge 未初始化")
        data_url = make_data_url(b64, mime)
        if not data_url:
            raise ValueError(f"图片无法转换为兼容的视觉输入: {mime}")
        gen = normalize_generation_for_provider(
            self._generation,
            thinking_control=self._thinking_control,
            model=self._model,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        request_kwargs: dict = {
            "model": self._model,
            "temperature": gen["temperature"],
            "max_tokens": gen["max_output_tokens"],
        }
        add_extra_generation_kwargs(request_kwargs, gen)
        if gen.get("extra_body"):
            request_kwargs["extra_body"] = gen["extra_body"]
        try:
            response = create_streamed_chat_completion(
                self._client,
                provider=self._provider or "vision_bridge",
                all_messages=messages,
                create_kwargs=self._request_identity.apply(
                    request_kwargs,
                    session_scope="vision-bridge",
                ),
            )
        except Exception:
            record_llm_usage(
                provider=self._provider,
                model=self._model,
                feature="vision_bridge",
                subfeature=subfeature,
                usage=None,
                status="error",
            )
            raise

        record_llm_usage(
            provider=self._provider,
            model=self._model,
            feature="vision_bridge",
            subfeature=subfeature,
            usage=getattr(response, "usage", None),
            status="success" if response.choices else "empty_choices",
        )
        return (response.choices[0].message.content or "").strip()

    # ── 公共方法 ───────────────────────────────────────

    def describe(self, image_ref: str, b64: str, mime: str) -> Optional[str]:
        """为图片生成初步描述并写入统一图片索引，返回描述文本。

        失败时返回 None（不抛异常）。
        """
        if not self.enabled:
            return None
        try:
            result = self._call_vlm(b64, mime, self._describe_prompt, "describe")
            if result:
                update_description(image_ref, result)
                logger.debug(
                    "[VisionBridge] 描述已生成: image_ref=%.8s …%s",
                    image_ref, result[:30].replace("\n", " "),
                )
            return result or None
        except Exception as exc:
            logger.warning(
                "[VisionBridge] describe 失败 (image_ref=%.8s): %s", image_ref, exc
            )
            return None

    def examine(
        self, image_ref: Optional[str], b64: str, mime: str, focus: str
    ) -> Optional[str]:
        """对图片进行带焦点的精细观察，结果追加到统一图片索引。

        image_ref 为 None 时结果仍返回，但不持久化。
        失败时返回 None（不抛异常）。
        """
        if not self.enabled:
            return None
        try:
            prompt = _DEFAULT_EXAMINE_PROMPT_TMPL.format(focus=focus)
            result = self._call_vlm(b64, mime, prompt, "examine")
            if result and image_ref:
                append_examination(image_ref, focus, result)
                logger.debug(
                    "[VisionBridge] examine 完成: focus=%r image_ref=%.8s", focus, image_ref
                )
            return result or None
        except Exception as exc:
            logger.warning(
                "[VisionBridge] examine 失败 (image_ref=%.8s focus=%r): %s",
                image_ref or "?", focus, exc,
            )
            return None

    def process_entry(self, entry: dict) -> None:
        """Share descriptions by exact registered image identity."""
        from .image_store import register_entry
        register_entry(entry)
        for ref, payload in (entry.get("images") or {}).items():
            record = read_image(ref)
            if not record or record.get("unavailable_status"):
                continue
            if self.enabled and not record.get("description"):
                with description_claim(ref) as acquired:
                    if acquired:
                        self.describe(ref, base64.b64encode(record["data"]).decode("ascii"), record["mime"])
                record = read_image(ref)
            payload["description"] = record.get("description")
            payload["examinations"] = record.get("examinations", [])
