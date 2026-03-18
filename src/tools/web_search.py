"""web_search.py — Tavily 联网搜索"""

import logging
import os

import httpx

logger = logging.getLogger("AICQ.tools")

DECLARATION: dict = {
    "max_calls_per_response": 3,
    "name": "web_search",
    "description": (
        "联网搜索工具。根据关键词搜索互联网，返回相关网页列表及内容摘要。"
        "当你需要查找实时信息、新闻、技术资料或任何你不确定或好奇的事实时可以调用。"
        "返回内容仅自己可见。"
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
                "description": "调用此工具的动机或原因。",
            },
        },
        "required": ["query"],
    },
}


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
        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0),
            }
            for item in data.get("results", [])
        ]
        logger.info("[tools] web_search: 搜索完成 query=%r 结果数=%d", query, len(results))
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
