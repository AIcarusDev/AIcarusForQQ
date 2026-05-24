"""examine_url_image.py — 下载并"看"一张网络图片

给定一个图片的 HTTP/HTTPS 直链，下载后通过 _multimodal_parts 机制传给
多模态模型，让模型能真正"看到"这张图。

适用场景：
  - 搜索到候选图片 URL 后，先预览确认内容是否符合预期，再决定是否发送。
  - 群友/私聊发来图片链接，想看看图里有什么。

条件启用：仅在 config["vision"] = True 时可用（对纯文字模型无意义）。
潜伏工具（ALWAYS_AVAILABLE=False），需先 get_tools 激活。
"""

import io
import logging
import os
import re

import httpx
from PIL import Image

logger = logging.getLogger("AICQ.tools")

# 超过此像素数量时缩小（长/宽均不超过此值），节省 token
_MAX_SIDE = 1280
# 下载体积上限，避免意外下载超大文件
_MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
# JPEG 压缩质量（缩放后重新编码时使用）
_JPEG_QUALITY = 85

ALWAYS_AVAILABLE: bool = False

DECLARATION: dict = {
    "name": "examine_url_image",
    "description": (
        "下载并查看一张网络图片。给定图片的 HTTP/HTTPS 直链，"
        "将图片内容直接展示给你，让你能看到图片的实际内容。"
        "常用于：先搜索到候选图片链接，通过此工具确认图片内容后，"
        "再用 send_message 的 image segment 将满意的图片发送给对方。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "图片的 HTTP 或 HTTPS 直链地址。",
            },
        },
        "required": ["url"],
    },
}


def condition(config: dict) -> bool:
    """仅视觉模型可用。"""
    return config.get("vision", True)


def execute(**kwargs) -> dict:
    url: str = str(kwargs.get("url", "")).strip()

    # 基本 URL 安全校验
    if not re.match(r"^https?://", url, re.IGNORECASE):
        return {"error": f"无效的图片 URL（必须以 http:// 或 https:// 开头）：{url!r}"}

    # 代理：优先使用通用环境变量，回退到 TAVILY_PROXY
    proxy_url = (
        os.environ.get("HTTP_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("TAVILY_PROXY", "").strip()
        or None
    )

    logger.info("[tools] examine_url_image: 开始下载 url=%s", url[:120])

    try:
        with httpx.Client(proxy=proxy_url, timeout=20.0, follow_redirects=True) as client:
            response = client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; AIcarus-image-viewer/1.0)"},
            )
            response.raise_for_status()

            # 检查 Content-Type 是否为图片
            content_type = response.headers.get("content-type", "")
            if content_type and not content_type.split(";")[0].strip().startswith("image/"):
                logger.warning(
                    "[tools] examine_url_image: 响应不是图片 content_type=%r url=%s",
                    content_type, url[:80],
                )
                return {"error": f"该 URL 返回的不是图片（Content-Type: {content_type}）", "url": url}

            raw_bytes = response.content
    except httpx.HTTPStatusError as e:
        logger.warning("[tools] examine_url_image: HTTP 错误 url=%s — %s", url[:80], e)
        return {"error": f"下载失败 (HTTP {e.response.status_code})", "url": url}
    except Exception as e:
        logger.warning("[tools] examine_url_image: 下载异常 url=%s — %s", url[:80], e)
        return {"error": f"下载失败：{e}", "url": url}

    if len(raw_bytes) > _MAX_DOWNLOAD_BYTES:
        return {"error": f"图片文件过大（{len(raw_bytes) // 1024} KB，上限 10 MB），已跳过。", "url": url}

    if not raw_bytes:
        return {"error": "服务器返回数据为空。", "url": url}

    # 用 Pillow 解析 + 按需缩放
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        original_format = (img.format or "").upper()
        original_size = img.size  # (width, height)

        # 转 RGB/RGBA 以便统一编码（避免调色板模式出错）
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in img.mode or img.mode == "P" else "RGB")

        # 缩放
        w, h = img.size
        if w > _MAX_SIDE or h > _MAX_SIDE:
            ratio = _MAX_SIDE / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.debug(
                "[tools] examine_url_image: 缩放 %dx%d → %dx%d",
                w, h, new_w, new_h,
            )

        # 编码为 PNG（无损且兼容 RGBA）
        buf = io.BytesIO()
        save_format = "PNG"
        out_mime = "image/png"
        img.save(buf, format=save_format, optimize=True)
        final_bytes = buf.getvalue()

    except Exception as e:
        logger.warning("[tools] examine_url_image: Pillow 处理失败 — %s", e)
        # Pillow 无法处理时直接返回原始字节
        mime = content_type.split(";")[0].strip() if content_type else "image/jpeg"
        final_bytes = raw_bytes
        out_mime = mime
        original_size = (0, 0)
        original_format = ""

    logger.info(
        "[tools] examine_url_image: 成功 url=%s size=%dx%d → %d bytes",
        url[:80], original_size[0], original_size[1], len(final_bytes),
    )

    return {
        "url": url,
        "original_size": f"{original_size[0]}x{original_size[1]}" if original_size != (0, 0) else "unknown",
        "original_format": original_format or "unknown",
        "note": "图片已展示给你，请根据内容决定是否通过 send_message 的 image segment 发送。",
        "_multimodal_parts": [
            {
                "mime_type": out_mime,
                "display_name": "preview.png",
                "data": final_bytes,
            }
        ],
    }
