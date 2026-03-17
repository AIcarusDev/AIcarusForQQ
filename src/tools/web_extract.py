"""web_extract.py — Tavily 网页正文抓取"""

import logging
import os

from tavily import TavilyClient

logger = logging.getLogger("AICQ.tools")

DECLARATION: dict = {
    "max_calls_per_response": 3,
    "name": "web_extract",
    "description": (
        "网页正文抓取工具。提取指定 URL 网页的完整正文内容（纯文本）。"
        "当你需要深入阅读某个网页的详细内容时可以调用（通常配合 web_search 使用，"
        "先搜索获取 URL，再用此工具提取感兴趣的页面正文）。"
        "返回内容仅自己可见。"
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
                "description": "调用此工具的动机或原因。",
            },
        },
        "required": ["url"],
    },
}


def _get_client() -> TavilyClient | None:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return None
    return TavilyClient(api_key=api_key)


def execute(url: str, **kwargs) -> dict:
    client = _get_client()
    if client is None:
        return {"error": "TAVILY_API_KEY 未配置，无法使用网页抓取"}
    try:
        response = client.extract(urls=[url])
        extracted = response.get("results", [])
        if not extracted:
            return {"error": "未能提取到网页内容", "url": url}
        page = extracted[0]
        raw_content = page.get("raw_content", "")
        # 截断过长内容防止 token 爆炸（保留前 8000 字符）
        if len(raw_content) > 8000:
            raw_content = raw_content[:8000] + "\n\n... [内容已截断，共 {} 字符]".format(
                len(page.get("raw_content", ""))
            )
        return {
            "url": page.get("url", url),
            "content": raw_content,
        }
    except Exception as e:
        logger.warning("[tools] Tavily 网页抓取失败: %s", e)
        return {"error": f"网页抓取失败: {e}", "url": url}
