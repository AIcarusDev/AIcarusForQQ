"""web_search.py — 联网搜索路由（Tavily / 可选 SearXNG）"""

import logging
import os
import re
from typing import Any, Callable
from urllib.parse import urljoin

import httpx

logger = logging.getLogger("AICQ.tools")

MAX_CONTENT_CHARS = 180
_SEARXNG_TIMEOUT = 20.0
_TAVILY_TIMEOUT = 30.0

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
        },
        "required": ["query"],
    },
}


REQUIRES_CONTEXT: list[str] = ["config"]


def _compact_content(raw: str, max_chars: int = MAX_CONTENT_CHARS) -> tuple[str, bool, int]:
    """Collapse noisy page text into a short search-result preview."""
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    original_chars = len(text)
    if original_chars <= max_chars:
        return text, False, original_chars

    suffix = "..."
    return text[: max(0, max_chars - len(suffix))].rstrip() + suffix, True, original_chars


def _normalize_max_results(max_results: Any) -> int:
    try:
        parsed = int(max_results)
    except (TypeError, ValueError):
        parsed = 5
    return max(1, min(parsed, 10))


def _search_tavily(query: str, max_results: int = 5) -> dict:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logger.warning("[tools] web_search: TAVILY_API_KEY 未配置")
        return {"error": "TAVILY_API_KEY 未配置，无法使用联网搜索"}
    proxy_url = os.environ.get("TAVILY_PROXY", "").strip() or None
    try:
        logger.info("[tools] web_search: 开始搜索 query=%r max_results=%d", query, max_results)
        with httpx.Client(proxy=proxy_url, timeout=_TAVILY_TIMEOUT) as client:
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
            "source": "tavily",
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


def _normalize_searxng_url(base_url: str) -> str:
    base = str(base_url or "").strip()
    if not base:
        return ""
    if not re.match(r"^https?://", base, re.IGNORECASE):
        base = "http://" + base
    return base.rstrip("/") + "/"


def _search_searxng(
    *,
    query: str,
    max_results: int,
    base_url: str,
    language: str,
    safesearch: int,
) -> dict:
    base_url = _normalize_searxng_url(base_url)
    if not base_url:
        return {"error": "SearXNG 地址未配置"}

    url = urljoin(base_url, "search")
    try:
        logger.info("[tools] web_search: SearXNG 开始搜索 query=%r url=%s", query, base_url)
        with httpx.Client(timeout=_SEARXNG_TIMEOUT) as client:
            response = client.get(
                url,
                params={
                    "q": query,
                    "format": "json",
                    "language": language or "zh-CN",
                    "safesearch": safesearch,
                },
            )
            response.raise_for_status()
            data = response.json()

        results = []
        truncated_count = 0
        for item in data.get("results", [])[:max_results]:
            content, truncated, original_chars = _compact_content(item.get("content", ""))
            if truncated:
                truncated_count += 1
            engines = item.get("engines")
            if not isinstance(engines, list):
                engines = [item.get("engine", "")] if item.get("engine") else []
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": content,
                "content_truncated": truncated,
                "content_original_chars": original_chars,
                "score": item.get("score", 0),
                "engine": item.get("engine", ""),
                "engines": engines,
            })

        logger.info(
            "[tools] web_search: SearXNG 搜索完成 query=%r 结果数=%d 截断=%d",
            query,
            len(results),
            truncated_count,
        )
        return {
            "query": query,
            "source": "searxng",
            "answer": "",
            "results_count": len(results),
            "results": results,
            "unresponsive_engines": data.get("unresponsive_engines", []),
        }
    except httpx.HTTPStatusError as e:
        logger.warning("[tools] web_search: SearXNG HTTP 错误 query=%r — %s", query, e)
        return {"error": f"SearXNG 搜索失败 (HTTP {e.response.status_code}): {e}"}
    except Exception as e:
        logger.warning("[tools] web_search: SearXNG 搜索异常 query=%r — %s", query, e)
        return {"error": f"SearXNG 搜索失败: {e}"}


def _searxng_cfg(config: dict) -> dict:
    web_search_cfg = config.get("web_search", {}) if isinstance(config, dict) else {}
    if not isinstance(web_search_cfg, dict):
        return {}
    searxng = web_search_cfg.get("searxng", {})
    return searxng if isinstance(searxng, dict) else {}


def make_handler(config: dict) -> Callable:
    def execute(query: str, max_results: int = 5, **kwargs) -> dict:
        max_results_norm = _normalize_max_results(max_results)
        searxng = _searxng_cfg(config)
        if not bool(searxng.get("enabled", False)):
            return _search_tavily(query, max_results_norm)

        searx_result = _search_searxng(
            query=query,
            max_results=max_results_norm,
            base_url=str(searxng.get("base_url", "") or ""),
            language=str(searxng.get("language", "zh-CN") or "zh-CN"),
            safesearch=int(searxng.get("safesearch", 0) or 0),
        )
        if not searx_result.get("error") and searx_result.get("results_count", 0) > 0:
            return searx_result

        logger.warning(
            "[tools] web_search: SearXNG 不可用或无结果，回退 Tavily query=%r reason=%s",
            query,
            searx_result.get("error") or "empty results",
        )
        tavily_result = _search_tavily(query, max_results_norm)
        if isinstance(tavily_result, dict):
            tavily_result["fallback_from"] = {
                "source": "searxng",
                "error": searx_result.get("error", ""),
                "results_count": searx_result.get("results_count", 0),
            }
        return tavily_result

    return execute
