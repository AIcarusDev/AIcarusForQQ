"""web_extract.py — Tavily 网页正文抓取"""

import logging
import os

import httpx

logger = logging.getLogger("AICQ.tools")

RESULT_MAX_CHARS: int = 300  # 正文可能很长，只保留前 300 字符作参考

DECLARATION: dict = {
    "name": "web_extract",
    "description": (
        "网页正文抓取工具。提取指定 URL 网页的完整正文内容（纯文本）。"
        "当你需要深入阅读某个网页的详细内容时可以调用（通常配合 web_search 使用，"
        "先搜索获取 URL，再用此工具提取感兴趣的页面正文）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "要提取正文的网页 URL。",
            },
            "motivation": {
                "type": "string",
            },
        },
        "required": ["url", "motivation"],
    },
}


def execute(url: str, **kwargs) -> dict:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logger.warning("[tools] web_extract: TAVILY_API_KEY 未配置")
        return {"error": "TAVILY_API_KEY 未配置，无法使用网页抓取"}
    proxy_url = os.environ.get("TAVILY_PROXY", "").strip() or None
    try:
        logger.info("[tools] web_extract: 开始抓取 url=%s", url[:100])
        with httpx.Client(proxy=proxy_url, timeout=30.0) as client:
            response = client.post(
                "https://api.tavily.com/extract",
                json={"api_key": api_key, "urls": [url]},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
        extracted = data.get("results", [])
        if not extracted:
            logger.warning("[tools] web_extract: 未能提取内容 url=%s", url)
            return {"error": "未能提取到网页内容", "url": url}
        page = extracted[0]
        raw_content = page.get("raw_content", "")
        # 截断过长内容防止 token 爆炸（保留前 8000 字符）
        if len(raw_content) > 8000:
            raw_content = raw_content[:8000] + "\n\n... [内容已截断，共 {} 字符]".format(
                len(page.get("raw_content", ""))
            )
        logger.info("[tools] web_extract: 抓取成功 url=%s content_len=%d", url[:100], len(raw_content))
        return {
            "url": page.get("url", url),
            "content": raw_content,
        }
    except httpx.HTTPStatusError as e:
        logger.warning("[tools] web_extract: HTTP 错误 url=%s — %s", url[:100], e)
        return {"error": f"网页抓取失败 (HTTP {e.response.status_code}): {e}", "url": url}
    except Exception as e:
        logger.warning("[tools] web_extract: 抓取异常 url=%s — %s", url[:100], e)
        return {"error": f"网页抓取失败: {e}", "url": url}
