"""client.py — 视频理解模型请求客户端与配置适配器。

负责：
1. 解析 video_understanding 配置（支持空配置检查与 provider 继承）
2. 文件默认 20 MiB、硬上限 128 MiB，另校验编码后请求体
3. 将视频转为 Base64 并构建内联多模态载荷（不走 File API，不覆盖默认温度与 resolution）
4. 适配 OpenAI-compatible（含 sub2api video_url）与 Gemini 原生协议
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import yaml

from .config import DEFAULT_MAX_SIZE_MB, HARD_MAX_SIZE_MB, DEFAULT_TIMEOUT_SECONDS, validate_video_settings

logger = logging.getLogger("AICQ.video")

DEFAULT_VIDEO_SYSTEM_INSTRUCTION = """You are a multimodal video analysis assistant. Please analyze the video content objectively, comprehensively, and accurately.

1. Your default behavior is to provide a **comprehensive and coherent summary of the core content** of the entire video. However, if the user provides specific questions, instructions, or points of focus alongside the video, **prioritize the user's instructions**.
   - If the user asks specific questions or requests targeted observations, prioritize providing a precise analysis and answer addressing those points; if key events or moments are involved, please specify the exact timestamps (e.g., [00:01 - 00:05]) whenever possible.

2. Do not attempt to continue the conversation with the user in the final output (e.g., "If you'd like, I can..." or "Would you like me to...") or provide extraneous explanations. This is a one-time task; the user cannot interact with you further.
3. Maintain objectivity in your analysis; do not include personal opinions or moral judgments in the final output.
4. The output language should match the user's input query; if the user does not provide a text query, output in English."""


class VideoProcessingError(RuntimeError):
    """视频处理与识别过程中抛出的业务异常。"""


def _load_raw_config() -> dict[str, Any]:
    """尝试获取系统运行时配置，若无则读取 config_user.yaml 作为 fallback。"""
    try:
        import app_state

        if getattr(app_state, "config", None):
            return app_state.config
    except Exception:
        pass

    config_path = Path(__file__).resolve().parents[3] / "config_user.yaml"
    if config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("[video] 读取 config_user.yaml 失败: %s", exc)
    return {}


def get_video_config() -> dict[str, Any]:
    """提取并规范化 video_understanding 配置项。"""
    try:
        from dotenv import load_dotenv

        env_file = Path(__file__).resolve().parents[3] / ".env"
        if env_file.is_file():
            load_dotenv(dotenv_path=env_file, override=False)
    except Exception:
        pass

    cfg = _load_raw_config()
    try:
        video_cfg = validate_video_settings(cfg.get("video_understanding") or {})
    except ValueError as exc:
        raise VideoProcessingError(str(exc)) from exc

    provider_name = video_cfg.get("provider")
    providers = cfg.get("model_providers") or {}
    base_provider = providers.get(provider_name, {}) if provider_name else {}

    base_url = str(video_cfg.get("base_url") or base_provider.get("base_url") or "").strip()

    api_key = str(video_cfg.get("api_key") or "").strip()
    if not api_key:
        api_key_env = video_cfg.get("api_key_env") or base_provider.get("api_key_env")
        if api_key_env:
            api_key = os.environ.get(str(api_key_env).strip(), "").strip()

    model = str(video_cfg.get("model") or "").strip()
    max_size_mb = float(video_cfg.get("max_size_mb") or DEFAULT_MAX_SIZE_MB)
    timeout = float(video_cfg.get("timeout") or DEFAULT_TIMEOUT_SECONDS)

    return {
        "protocol": str(video_cfg.get("protocol") or "auto").strip().lower(),
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "max_size_mb": max_size_mb,
        "timeout": timeout,
        "is_configured": bool(base_url and model),
    }


def detect_video_mime(file_path: Path) -> str:
    """Use container bytes, including staged files with a generic suffix."""
    from llm.media.video_store import _container_extension, VideoStoreError
    try:
        with file_path.open("rb") as stream:
            ext = _container_extension(stream.read(4096))
    except (OSError, VideoStoreError) as exc:
        raise VideoProcessingError("文件不是支持的视频容器") from exc
    mapping = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".flv": "video/x-flv",
    }
    return mapping[ext]


class VideoModelClient:
    """视频多模态识别客户端。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or get_video_config()

    def validate_file_size(self, file_path: Path) -> None:
        """检查视频文件大小，超过限制则抛出异常。"""
        if not file_path.is_file():
            raise VideoProcessingError(f"目标视频文件不存在: {file_path}")

        file_size = file_path.stat().st_size
        try:
            limit = validate_video_settings(self.config)["max_size_mb"]
        except ValueError as exc:
            raise VideoProcessingError(str(exc)) from exc
        max_bytes = int(min(limit, HARD_MAX_SIZE_MB) * 1024 * 1024)
        if file_size > max_bytes:
            actual_mb = file_size / (1024 * 1024)
            raise VideoProcessingError(
                f"视频文件大小 ({actual_mb:.2f}MB) 超出允许的硬限制 ({self.config['max_size_mb']:.1f}MB)，请先压缩或截取片段"
            )

    def _resolve_native_base_url(self, base_url: str) -> str | None:
        """根据配置的 base_url 推导 Gemini 原生端点根路径。"""
        parsed = urlsplit(base_url)
        path = parsed.path.rstrip("/")
        for suffix in ("/openai/v1/chat/completions", "/openai/chat/completions", "/openai/v1", "/openai"):
            if path.endswith(suffix):
                path = path[:-len(suffix)]
                break
        if parsed.hostname == "generativelanguage.googleapis.com":
            path = "/v1beta"  # mediaProcessing is a v1beta feature.
        elif path.endswith(("/v1beta", "/v1")) and "/antigravity/" in path:
            pass
        elif path.endswith("/v1beta"):
            pass
        elif parsed.port == 8081 or "sub2api" in (parsed.hostname or "").lower():
            for suffix in ("/v1/chat/completions", "/v1"):
                if path.endswith(suffix):
                    path = path[:-len(suffix)]
                    break
            path += "/antigravity/v1beta"
        else:
            return None
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))

    def analyze(
        self,
        file_path: Path,
        prompt: str | None = None,
        mode: str = "static",
        system_instruction: str = DEFAULT_VIDEO_SYSTEM_INSTRUCTION,
    ) -> str:
        """读取视频文件内联发送到多模态模型，获取视频识别分析结果。"""
        if not self.config.get("is_configured"):
            raise VideoProcessingError(
                "video_understanding 未在配置中完整设定（缺少 base_url 或 model）。请在 config_user.yaml 中配置 video_understanding"
            )

        self.validate_file_size(file_path)
        if mode not in {"static", "agentic"}:
            raise VideoProcessingError("不支持的视频分析模式")
        from .common import run_ffprobe, extract_metadata
        extract_metadata(run_ffprobe(file_path), file_path)
        mime_type = detect_video_mime(file_path)

        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        b64_data = base64.b64encode(raw_bytes).decode("ascii")

        base_url = self.config["base_url"].rstrip("/")
        api_key = self.config["api_key"]
        model = self.config["model"]
        timeout = self.config["timeout"]
        protocol = str(self.config.get("protocol") or "auto").strip().lower()

        native_base = None
        if protocol in ("auto", "gemini_native"):
            native_base = self._resolve_native_base_url(base_url)
            if not native_base and protocol == "gemini_native":
                native_base = base_url.rstrip("/")

        if native_base:
            try:
                return self._call_google_native(
                    native_base,
                    api_key,
                    model,
                    b64_data,
                    mime_type,
                    prompt,
                    timeout,
                    mode=mode,
                    system_instruction=system_instruction,
                )
            except Exception as exc:
                if protocol == "gemini_native" or mode == "agentic":
                    raise
                logger.warning("[video] 尝试 Google 原生端点失败，尝试备用兼容端点: %s", exc)

        if protocol == "gemini_native":
            raise VideoProcessingError(f"无法为 base_url={base_url} 解析 Gemini 原生端点")

        if mode == "agentic":
            raise VideoProcessingError("agentic 需要支持该模式的 Gemini 原生端点和模型；兼容协议不支持，请选择原生协议或显式改用 static")

        return self._call_openai_compatible(
            base_url,
            api_key,
            model,
            b64_data,
            mime_type,
            prompt,
            timeout,
            system_instruction=system_instruction,
        )

    def _call_openai_compatible(
        self,
        base_url: str,
        api_key: str,
        model: str,
        b64_data: str,
        mime_type: str,
        prompt: str | None,
        timeout: float,
        system_instruction: str = DEFAULT_VIDEO_SYSTEM_INSTRUCTION,
    ) -> str:
        """调用兼容 OpenAI 的端点（通过 video_url 传输 Base64 内联视频）。"""
        url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"

        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        messages: list[dict[str, Any]] = []
        if system_instruction and system_instruction.strip():
            messages.append(
                {
                    "role": "system",
                    "content": system_instruction.strip(),
                }
            )

        content_parts: list[dict[str, Any]] = [
            {
                "type": "video_url",
                "video_url": {
                    "url": f"data:{mime_type};base64,{b64_data}",
                },
            }
        ]
        if prompt and str(prompt).strip():
            content_parts.append(
                {
                    "type": "text",
                    "text": str(prompt).strip(),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": content_parts,
            }
        )

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        self._validate_payload_size(base_url, payload)

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
        except Exception as exc:
            logger.exception("[video] 请求模型服务端连接失败: %s", exc)
            raise VideoProcessingError(f"视频分析服务端网络请求失败: {exc}") from exc

        if resp.status_code != 200:
            error_body = resp.text[:500]
            logger.error("[video] 模型服务返回 HTTP %s: %s", resp.status_code, error_body)
            raise VideoProcessingError(f"视频分析服务返回错误 (HTTP {resp.status_code}): {error_body}")

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise VideoProcessingError("视频分析服务响应中未包含有效 choices 内容")

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise VideoProcessingError("视频分析服务返回内容为空")

        return str(content).strip()

    def _call_google_native(
        self,
        base_url: str,
        api_key: str,
        model: str,
        b64_data: str,
        mime_type: str,
        prompt: str | None,
        timeout: float,
        mode: str = "static",
        system_instruction: str = DEFAULT_VIDEO_SYSTEM_INSTRUCTION,
    ) -> str:
        """调用 Google Gemini 原生 generateContent 端点。"""
        url = f"{base_url}/models/{model}:generateContent"
        params: dict[str, str] = {}
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if api_key:
            if api_key.startswith("AIza"):
                params["key"] = api_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"

        media_processing = "AGENTIC" if str(mode).lower() == "agentic" else "STATIC"
        video_part: dict[str, Any] = {
            "inlineData": {
                "mimeType": mime_type,
                "data": b64_data,
            },
            "mediaProcessing": media_processing,
        }

        parts: list[dict[str, Any]] = [video_part]
        if prompt and str(prompt).strip():
            parts.append({"text": str(prompt).strip()})

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ]
        }
        if system_instruction and system_instruction.strip():
            payload["systemInstruction"] = {
                "parts": [
                    {
                        "text": system_instruction.strip(),
                    }
                ]
            }
        self._validate_payload_size(base_url, payload)

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, headers=headers, params=params, json=payload)
        except Exception as exc:
            logger.exception("[video] Google 原生端点请求失败: %s", exc)
            raise VideoProcessingError(f"Google 原生端点请求网络失败: {exc}") from exc

        if resp.status_code != 200:
            error_body = resp.text[:500]
            logger.error("[video] Google 原生端点返回 HTTP %s: %s", resp.status_code, error_body)
            raise VideoProcessingError(f"Google 原生端点返回错误 (HTTP {resp.status_code}): {error_body}")

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise VideoProcessingError("Google 原生端点响应中未包含有效 candidates 内容")

        parts = candidates[0].get("content", {}).get("parts") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p and not p.get("thought")]
        result_text = "".join(texts).strip()
        if not result_text:
            raise VideoProcessingError("Google 原生端点返回文本为空")

        return result_text

    def _validate_payload_size(self, base_url: str, payload: dict) -> None:
        # httpx uses compact UTF-8 JSON. Include all text and Base64 expansion.
        size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        limit = 20_000_000 if urlsplit(base_url).hostname == "generativelanguage.googleapis.com" else int(HARD_MAX_SIZE_MB * 1024 * 1024)
        if size > limit:
            raise VideoProcessingError(f"内联视频请求体 {size} 字节超过服务限制 {limit} 字节，请缩小视频或缩短提问后重试")
