"""web_browse.py — 无头浏览器网页浏览

使用 Playwright Chromium 真实渲染网页（支持 JS），返回：
  - 页面标题与正文摘要（文字模型可读）
  - 页面内所有图片直链列表（可配合 examine_url_image + send_message 使用）
  - 页面截图（_multimodal_parts，多模态模型可直接"看到"）

适用场景：
  - 访问某个图片站/图库页面，提取里面的图片链接
  - 打开用户发来的某个 URL，看看页面里有什么
  - 用 web_search 找到候选页面后，进一步浏览其内容

需要在环境变量中设置 PLAYWRIGHT_BROWSERS_PATH 指向浏览器安装目录。
潜伏工具（ALWAYS_AVAILABLE=False），需先 get_tools 激活。
"""

import io
import logging
import os
import re

logger = logging.getLogger("AICQ.tools")

# 截图最大边长（节省 token）
_MAX_SCREENSHOT_SIDE = 1280
# 提取图片链接的最大数量
_MAX_IMAGE_LINKS = 30
# 正文摘要最大字符数
_MAX_TEXT_CHARS = 800
# 页面加载等待超时（毫秒）
_PAGE_TIMEOUT_MS = 20_000

ALWAYS_AVAILABLE: bool = False

DECLARATION: dict = {
    "name": "web_browse",
    "description": (
        "用无头浏览器打开一个网页（支持 JavaScript 渲染），"
        "返回页面截图（你可以直接看到页面内容）、页面标题、正文摘要，"
        "以及页面中找到的所有图片直链列表。"
        "常用于：访问图片站页面提取图片链接，或查看某个 URL 的实际内容。"
        "获取图片链接后，可用 examine_url_image 预览，再用 send_message 的 image segment 发送。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "要访问的网页 URL（http 或 https）。",
            },
            "wait_for": {
                "type": "string",
                "enum": ["load", "networkidle"],
                "description": (
                    "等待策略。'load'（默认）：DOM 加载完成即返回，速度快；"
                    "'networkidle'：等待网络请求停止，适合需要等 JS 异步加载图片的页面，但较慢。"
                ),
            },
        },
        "required": ["url"],
    },
}


def condition(config: dict) -> bool:
    """仅视觉模型可用（截图对文字模型无意义；文字模型用 web_extract 即可）。"""
    return config.get("vision", True)


def _resize_png(png_bytes: bytes, max_side: int) -> bytes:
    """如果截图超出尺寸限制则等比缩小，返回 PNG bytes。"""
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(png_bytes))
        w, h = img.size
        if w <= max_side and h <= max_side:
            return png_bytes
        ratio = max_side / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception:
        return png_bytes


def _extract_image_urls(page, base_url: str) -> list[str]:
    """从已渲染的页面提取图片 URL，去重、过滤图标和 data URI。"""
    try:
        # 收集 <img src/srcset>、<source srcset>、背景图 CSS（粗略）和 og:image
        raw: list[str] = page.evaluate("""() => {
            const urls = new Set();
            // <img>
            document.querySelectorAll('img[src]').forEach(el => {
                const src = el.getAttribute('src');
                if (src) urls.add(src);
                // srcset: 取最后一项（通常分辨率最高）
                const ss = el.getAttribute('srcset');
                if (ss) {
                    const last = ss.trim().split(',').map(s => s.trim().split(/\\s+/)[0]).filter(Boolean);
                    last.forEach(u => urls.add(u));
                }
            });
            // <source srcset>
            document.querySelectorAll('source[srcset]').forEach(el => {
                el.getAttribute('srcset').trim().split(',')
                  .map(s => s.trim().split(/\\s+/)[0]).filter(Boolean)
                  .forEach(u => urls.add(u));
            });
            // og:image / twitter:image meta
            ['og:image', 'twitter:image'].forEach(prop => {
                const el = document.querySelector(`meta[property="${prop}"],meta[name="${prop}"]`);
                if (el) urls.add(el.getAttribute('content') || '');
            });
            return [...urls];
        }""")
    except Exception as e:
        logger.warning("[tools] web_browse: JS 提取图片失败 — %s", e)
        raw = []

    # 拼接相对 URL，过滤 data URI 和小图标
    from urllib.parse import urljoin

    result: list[str] = []
    seen: set[str] = set()
    for u in raw:
        u = u.strip()
        if not u or u.startswith("data:"):
            continue
        if u.startswith("//"):
            u = "https:" + u
        elif not u.startswith("http"):
            u = urljoin(base_url, u)
        # 跳过明显的小图标（路径含 favicon / icon，或扩展名为 .ico）
        lower = u.lower()
        if ".ico" in lower or "favicon" in lower:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
        if len(result) >= _MAX_IMAGE_LINKS:
            break
    return result


def execute(**kwargs) -> dict:
    url: str = str(kwargs.get("url", "")).strip()
    wait_for: str = str(kwargs.get("wait_for", "load")).strip()
    if wait_for not in ("load", "networkidle"):
        wait_for = "load"

    if not re.match(r"^https?://", url, re.IGNORECASE):
        return {"error": f"无效 URL（必须以 http:// 或 https:// 开头）：{url!r}"}

    # 读取浏览器路径
    browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip() or None

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        return {"error": "Playwright 未安装，请先运行 pip install playwright 并安装浏览器。"}

    logger.info("[tools] web_browse: 开始浏览 url=%s wait_for=%s", url[:100], wait_for)

    try:
        env_patch: dict[str, str] = {}
        if browsers_path:
            env_patch["PLAYWRIGHT_BROWSERS_PATH"] = browsers_path

        # 临时注入环境变量（Playwright 在 launch 时读取）
        _old_env = {}
        for k, v in env_patch.items():
            _old_env[k] = os.environ.get(k)
            os.environ[k] = v

        try:
            proxy_url = (
                os.environ.get("HTTP_PROXY")
                or os.environ.get("HTTPS_PROXY")
                or os.environ.get("TAVILY_PROXY", "").strip()
                or None
            )
            launch_kwargs: dict = {"headless": True}
            if proxy_url:
                launch_kwargs["proxy"] = {"server": proxy_url}

            with sync_playwright() as pw:
                browser = pw.chromium.launch(**launch_kwargs)
                ctx = browser.new_context(
                    viewport={"width": 1280, "height": 800},
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    locale="zh-CN",
                )
                page = ctx.new_page()
                try:
                    page.goto(url, timeout=_PAGE_TIMEOUT_MS, wait_until=wait_for)
                except PWTimeout:
                    logger.warning("[tools] web_browse: 页面加载超时，尝试继续 url=%s", url[:80])

                title = page.title() or ""

                # 正文摘要
                try:
                    body_text: str = page.evaluate("""() => {
                        const el = document.body;
                        return el ? el.innerText : '';
                    }""") or ""
                    body_text = re.sub(r"[ \t]+", " ", body_text)
                    body_text = re.sub(r"\n{3,}", "\n\n", body_text).strip()
                    if len(body_text) > _MAX_TEXT_CHARS:
                        body_text = body_text[:_MAX_TEXT_CHARS] + f"…（已截断，共 {len(body_text)} 字符）"
                except Exception:
                    body_text = ""

                # 图片链接提取
                image_urls = _extract_image_urls(page, url)

                # 截图
                try:
                    png_raw = page.screenshot(full_page=False, type="png")
                    png_bytes = _resize_png(png_raw, _MAX_SCREENSHOT_SIDE)
                except Exception as e:
                    logger.warning("[tools] web_browse: 截图失败 — %s", e)
                    png_bytes = None

                browser.close()
        finally:
            for k, v in _old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    except Exception as e:
        logger.warning("[tools] web_browse: 浏览失败 url=%s — %s", url[:80], e)
        return {"error": f"浏览失败：{e}", "url": url}

    logger.info(
        "[tools] web_browse: 完成 url=%s title=%r images=%d",
        url[:80], title[:60], len(image_urls),
    )

    result: dict = {
        "url": url,
        "title": title,
        "text_preview": body_text,
        "image_urls": image_urls,
        "image_count": len(image_urls),
        "note": (
            "截图已展示给你。image_urls 是页面中找到的图片直链，"
            "可用 examine_url_image 预览后再用 send_message 发送。"
        ),
    }

    if png_bytes:
        result["_multimodal_parts"] = [
            {
                "mime_type": "image/png",
                "display_name": "screenshot.png",
                "data": png_bytes,
            }
        ]

    return result
