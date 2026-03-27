"""vision_bridge.py — 视觉桥：为不支持视觉的模型自动描述图片

当 config.vision=false（或主模型不支持视觉）时，此模块：
  1. 调用独立的 VLM（通过 OpenAI 兼容端点）生成图片的文字描述
  2. 描述结果写入 image_cache sidecar (.meta.json)
  3. 相同图片再次出现时直接复用缓存描述，不重复消费 token

同时提供 examine() 接口供 examine_image 工具调用精查。

配置段（config.yaml）：
  vision_bridge:
    enabled: true
    provider: "siliconflow"
    model: "Qwen/Qwen2-VL-7B-Instruct"
    base_url: "https://api.siliconflow.cn/v1"
    api_key_env: "SILICONFLOW_API_KEY"
    describe_prompt: "请用2-4句话描述这张图片..."
    similarity_threshold: 10
"""

import base64
import logging
import os
from typing import Optional

from .image_cache import (
    append_examination,
    cache_image,
    load_meta,
    update_description,
)

logger = logging.getLogger("AICQ.vision_bridge")

_DEFAULT_DESCRIBE_PROMPT = (
    "请用2-4句话描述这张图片的主要内容，"
    "重点识别其中的文字、数字、人物、场景和关键视觉元素。"
    "描述要简洁、客观。"
)

_DEFAULT_EXAMINE_PROMPT_TMPL = (
    "请仔细观察这张图片，重点关注：{focus}。"
    "详细描述你在该区域或该方面的观察结果，包含文字、数字等关键信息。"
)


class VisionBridge:
    """图片视觉桥——图片缓存 + 按需 VLM 描述生成。

    线程安全：OpenAI 同步客户端，process_entry() 应在线程池中调用。
    """

    def __init__(self, cfg: dict):
        """
        cfg: config.yaml 中的 vision_bridge 子字典（可为空 dict）。
        """
        self._cfg = cfg
        self._enabled: bool = bool(cfg.get("enabled", False))
        self._model: str = cfg.get("model", "")
        self._base_url: str = cfg.get("base_url", "")
        self._api_key_env: str = cfg.get("api_key_env", "")
        self._describe_prompt: str = cfg.get("describe_prompt", _DEFAULT_DESCRIBE_PROMPT)
        self._sim_threshold: int = int(cfg.get("similarity_threshold", 10))
        self._client = None  # openai.OpenAI，懒初始化

        if self._enabled and self._model:
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
            
            # 代理配置：直接从环境变量读取（OPENAI_PROXY）
            if proxy_url := os.environ.get("OPENAI_PROXY", "").strip() or None:
                http_client = httpx.Client(proxy=proxy_url)
                kwargs["http_client"] = http_client

            self._client = OpenAI(**kwargs)
            logger.info(
                "[VisionBridge] 初始化完成，模型: %s", self._model
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

    def _call_vlm(self, b64: str, mime: str, prompt: str) -> str:
        """向 VLM 发送图片 + 文本提示，返回纯文本回复（同步）。"""
        if not self._client:
            raise RuntimeError("VisionBridge 未初始化")
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=512,
        )
        return (response.choices[0].message.content or "").strip()

    # ── 公共方法 ───────────────────────────────────────

    def describe(self, phash: str, b64: str, mime: str) -> Optional[str]:
        """为图片生成初步描述并写入 sidecar，返回描述文本。

        失败时返回 None（不抛异常）。
        """
        if not self.enabled:
            return None
        try:
            result = self._call_vlm(b64, mime, self._describe_prompt)
            if result:
                update_description(phash, result)
                logger.debug(
                    "[VisionBridge] 描述已生成: phash=%.8s …%s",
                    phash, result[:30].replace("\n", " "),
                )
            return result or None
        except Exception as exc:
            logger.warning(
                "[VisionBridge] describe 失败 (phash=%.8s): %s", phash, exc
            )
            return None

    def examine(
        self, phash: Optional[str], b64: str, mime: str, focus: str
    ) -> Optional[str]:
        """对图片进行带焦点的精细观察，结果追加到 sidecar。

        phash 为 None 时结果仍返回，但不持久化。
        失败时返回 None（不抛异常）。
        """
        if not self.enabled:
            return None
        try:
            prompt = _DEFAULT_EXAMINE_PROMPT_TMPL.format(focus=focus)
            result = self._call_vlm(b64, mime, prompt)
            if result and phash:
                append_examination(phash, focus, result)
                logger.debug(
                    "[VisionBridge] examine 完成: focus=%r phash=%.8s", focus, phash
                )
            return result or None
        except Exception as exc:
            logger.warning(
                "[VisionBridge] examine 失败 (phash=%.8s focus=%r): %s",
                phash or "?", focus, exc,
            )
            return None

    def process_entry(self, entry: dict) -> None:
        """处理一条上下文消息中的所有图片（同步，适合在线程池运行）。

        对每张图片：
          1. 计算 pHash，落盘（去重）
          2. 加载 sidecar，检查是否有已有描述
          3. 缓存有描述 → 直接复用；无描述且桥已启用 → 调 VLM 生成
          4. 将 phash / description / examinations 写回 img_info（内存）
        """
        images: dict = entry.get("images") or {}
        for ref, img_info in images.items():
            b64: str = img_info.get("base64", "")
            mime: str = img_info.get("mime", "image/jpeg")
            if not b64:
                continue

            # ── 1. 落盘 + pHash ──────────────────────
            try:
                raw = base64.b64decode(b64)
                phash, _ = cache_image(raw, mime)
            except Exception as exc:
                logger.warning(
                    "[VisionBridge] 图片缓存失败 (ref=%s): %s", ref, exc
                )
                # 即便缓存失败，仍设置空描述占位
                img_info.setdefault("phash", None)
                img_info.setdefault("description", None)
                img_info.setdefault("examinations", [])
                continue

            img_info["phash"] = phash

            if phash is None:
                img_info.setdefault("description", None)
                img_info.setdefault("examinations", [])
                continue

            # ── 2. 复用或生成描述 ──────────────────────
            meta = load_meta(phash)
            description: Optional[str] = meta.get("description")
            examinations: list = meta.get("examinations") or []

            if description:
                # 缓存命中，直接复用
                img_info["description"] = description
                img_info["examinations"] = examinations
                logger.debug(
                    "[VisionBridge] 复用缓存描述: phash=%.8s", phash
                )
            elif self.enabled:
                # 首次见到，调 VLM 描述
                new_desc = self.describe(phash, b64, mime)
                img_info["description"] = new_desc
                img_info["examinations"] = []
            else:
                img_info["description"] = None
                img_info["examinations"] = []
