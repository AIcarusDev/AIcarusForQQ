"""web_search.py — Tavily 联网搜索"""

import logging
import os
import re

import httpx

logger = logging.getLogger("AICQ.tools")

MAX_CONTENT_CHARS = 180

DECLARATION: dict = {
    "name": "web_search",
    "description": (
        "联网搜索工具。根据关键词搜索互联网，返回相关网页列表和短内容预览。"
        "当你需要查找实时信息、新闻、技术资料或任何你不确定或好奇的事实时可以调用。"
        "搜索结果只适合快速判断候选网页；如果需要阅读网页正文，则需要调用 web_extract。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词或问题。",
            },
            "max_results": {
                "type": "integer",
                "description": "返回结果数量，默认 5，最大 10。",
            },
            "motivation": {
                "type": "string",
            },
        },
        "required": ["query", "motivation"],
    },
}


def _compact_content(raw: str, max_chars: int = MAX_CONTENT_CHARS) -> tuple[str, bool, int]:
    """Collapse noisy page text into a short search-result preview."""
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    original_chars = len(text)
    if original_chars <= max_chars:
        return text, False, original_chars

    suffix = "..."
    return text[: max(0, max_chars - len(suffix))].rstrip() + suffix, True, original_chars


def execute(query: str, max_results: int = 5, **kwargs) -> dict:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logger.warning("[tools] web_search: TAVILY_API_KEY 未配置")
        return {"error": "TAVILY_API_KEY 未配置，无法使用联网搜索"}
    proxy_url = os.environ.get("TAVILY_PROXY", "").strip() or None
    try:
        logger.info("[tools] web_search: 开始搜索 query=%r max_results=%d", query, max_results)
        with httpx.Client(proxy=proxy_url, timeout=30.0) as client:
            response = client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": min(max_results, 10),
                    "include_answer": True,
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
        results = []
        truncated_count = 0
        for item in data.get("results", []):
            content, truncated, original_chars = _compact_content(item.get("content", ""))
            if truncated:
                truncated_count += 1
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": content,
                "content_truncated": truncated,
                "content_original_chars": original_chars,
                "score": item.get("score", 0),
            })
        logger.info(
            "[tools] web_search: 搜索完成 query=%r 结果数=%d 截断=%d",
            query,
            len(results),
            truncated_count,
        )
        return {
            "query": query,
            "answer": data.get("answer", ""),
            "results_count": len(results),
            "results": results,
        }
    except httpx.HTTPStatusError as e:
        logger.warning("[tools] web_search: HTTP 错误 query=%r — %s", query, e)
        return {"error": f"搜索失败 (HTTP {e.response.status_code}): {e}"}
    except Exception as e:
        logger.warning("[tools] web_search: 搜索异常 query=%r — %s", query, e)
        return {"error": f"搜索失败: {e}"}
