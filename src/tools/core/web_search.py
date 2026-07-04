"""web_search.py — 联网搜索路由（Tavily / 可选 SearXNG）"""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse

import httpx
from pydantic import Field

from tools.contract import ToolArgsModel, ToolContract

logger = logging.getLogger("AICQ.tools")

MAX_CONTENT_CHARS = 180
_SEARXNG_TIMEOUT = 20.0
_TAVILY_TIMEOUT = 30.0
_TRACKING_QUERY_PARAMS = {
    "fbclid",
    "gclid",
    "gbraid",
    "mc_cid",
    "mc_eid",
    "msclkid",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
    "yclid",
}


class WebSearchArgs(ToolArgsModel):
    queries: list[str] = Field(
        min_length=1,
        max_length=4,
        description="搜索关键词或问题列表；每一项会独立查询。",
    )
    max_results: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="返回结果数量，默认 5。",
    )
    allowed_domains: list[str] | None = Field(
        default=None,
        max_length=20,
        description="可选；只返回这些域名及其子域名的结果，例如 ['github.com']。",
    )
    blocked_domains: list[str] | None = Field(
        default=None,
        max_length=20,
        description="可选；排除这些域名及其子域名的结果，例如 ['example.com']。",
    )


TOOL_CONTRACT = ToolContract(
    name="web_search",
    description=(
        "联网搜索工具。根据关键词搜索互联网，返回相关网页列表和短内容预览。"
        "当你需要查找实时信息、新闻、技术资料或任何你不确定或好奇的事实时可以调用。"
        "搜索结果只适合快速判断候选网页；如果需要阅读网页正文，则需要调用 web_extract。"
        "支持指定域名、排除域名；支持多个查询并发。"
        "注意：搜索不是问答，是检索，有可能需要多次迭代；避免把一次未命中直接理解为'没有/找不到'，结果不佳时可尝试调整搜索关键词、搜索语言、或换方向。"
        "当前月份为 {year_month}。需要搜索最新信息、文档或时事时，请使用目前的年/月份，避免基于过时信息判断。"
    ),
    args_model=WebSearchArgs,
)


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


def _normalize_queries(raw_queries: Any, legacy_query: Any = None) -> list[str]:
    if isinstance(raw_queries, list):
        candidates = raw_queries
    elif raw_queries is None and legacy_query is not None:
        candidates = [legacy_query]
    else:
        candidates = [raw_queries]

    queries: list[str] = []
    for raw in candidates:
        query = str(raw or "").strip()
        if query and query not in queries:
            queries.append(query)
        if len(queries) >= 4:
            break
    return queries


def _normalize_domain(raw: Any) -> str:
    domain = str(raw or "").strip().lower()
    if not domain:
        return ""
    domain = domain.removeprefix("*.").removeprefix(".")
    if "://" not in domain:
        domain = "http://" + domain
    parsed = urlparse(domain)
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    return host.removeprefix("www.") if host.startswith("www.") else host


def _normalize_domains(raw_domains: Any) -> list[str]:
    if not isinstance(raw_domains, list):
        return []
    domains: list[str] = []
    for raw in raw_domains:
        domain = _normalize_domain(raw)
        if domain and domain not in domains:
            domains.append(domain)
    return domains


def _url_host(url: Any) -> str:
    try:
        host = (urlparse(str(url or "")).hostname or "").strip().lower().rstrip(".")
    except Exception:
        return ""
    return host.removeprefix("www.") if host.startswith("www.") else host


def _host_matches_domain(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def _result_allowed_by_domains(
    result: dict,
    *,
    allowed_domains: list[str],
    blocked_domains: list[str],
) -> bool:
    host = _url_host(result.get("url"))
    if not host:
        return not allowed_domains
    if any(_host_matches_domain(host, domain) for domain in blocked_domains):
        return False
    if allowed_domains and not any(_host_matches_domain(host, domain) for domain in allowed_domains):
        return False
    return True


def _filter_results_by_domains(
    results: list[dict],
    *,
    allowed_domains: list[str],
    blocked_domains: list[str],
) -> tuple[list[dict], int]:
    if not allowed_domains and not blocked_domains:
        return results, 0
    filtered = [
        result
        for result in results
        if _result_allowed_by_domains(
            result,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )
    ]
    return filtered, len(results) - len(filtered)


def _result_dedupe_key(result: dict) -> str:
    raw_url = str(result.get("url") or "").strip()
    if not raw_url:
        return ""
    try:
        parsed = urlparse(raw_url)
    except Exception:
        return raw_url.split("#", 1)[0].rstrip("/").lower()

    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        return raw_url.split("#", 1)[0].rstrip("/").lower()
    if host.startswith("www."):
        host = host.removeprefix("www.")

    path = re.sub(r"/+", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")

    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_QUERY_PARAMS
    ]
    query = urlencode(sorted(query_items), doseq=True)
    return f"{host}{path}?{query}" if query else f"{host}{path}"


def _dedupe_search_results(searches: list[dict]) -> tuple[list[dict], int, int]:
    seen: dict[str, tuple[dict, str]] = {}
    deduped_searches: list[dict] = []
    raw_count = 0
    duplicate_count = 0

    for search in searches:
        if not isinstance(search, dict):
            deduped_searches.append(search)
            continue

        search_copy = dict(search)
        query = str(search_copy.get("query") or "").strip()
        raw_results = search_copy.get("results")
        if not isinstance(raw_results, list):
            deduped_searches.append(search_copy)
            continue

        raw_count += len(raw_results)
        deduped_results: list[dict] = []
        duplicates_in_search = 0
        for raw_item in raw_results:
            item = dict(raw_item) if isinstance(raw_item, dict) else {"value": raw_item}
            key = _result_dedupe_key(item) if isinstance(item, dict) else ""
            if not key:
                deduped_results.append(item)
                continue

            seen_entry = seen.get(key)
            if seen_entry is not None:
                first_item, first_query = seen_entry
                duplicate_count += 1
                duplicates_in_search += 1
                matched_queries = first_item.setdefault(
                    "matched_queries",
                    [first_query] if first_query else [],
                )
                if query and query not in matched_queries:
                    matched_queries.append(query)
                continue

            seen[key] = (item, query)
            deduped_results.append(item)

        search_copy["results"] = deduped_results
        search_copy["results_count"] = len(deduped_results)
        if duplicates_in_search:
            search_copy["raw_results_count"] = len(raw_results)
            search_copy["duplicates_omitted"] = duplicates_in_search
        deduped_searches.append(search_copy)

    return deduped_searches, raw_count, duplicate_count


def _searxng_domain_query(
    query: str,
    *,
    allowed_domains: list[str],
    blocked_domains: list[str],
) -> str:
    parts: list[str] = []
    if allowed_domains:
        site_terms = " OR ".join(f"site:{domain}" for domain in allowed_domains)
        parts.append(f"({site_terms})")
    parts.extend(f"-site:{domain}" for domain in blocked_domains)
    parts.append(query)
    return " ".join(part for part in parts if part).strip()


def _search_tavily(
    query: str,
    max_results: int = 5,
    *,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> dict:
    allowed = _normalize_domains(allowed_domains)
    blocked = _normalize_domains(blocked_domains)
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logger.warning("[tools] web_search: TAVILY_API_KEY 未配置")
        return {"error": "TAVILY_API_KEY 未配置，无法使用联网搜索"}
    proxy_url = os.environ.get("TAVILY_PROXY", "").strip() or None
    try:
        logger.info("[tools] web_search: 开始搜索 query=%r max_results=%d", query, max_results)
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": min(max_results, 10),
            "include_answer": False,
        }
        if allowed:
            payload["include_domains"] = allowed
        if blocked:
            payload["exclude_domains"] = blocked
        with httpx.Client(proxy=proxy_url, timeout=_TAVILY_TIMEOUT) as client:
            response = client.post(
                "https://api.tavily.com/search",
                json=payload,
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
        results, filtered_count = _filter_results_by_domains(
            results,
            allowed_domains=allowed,
            blocked_domains=blocked,
        )
        logger.info(
            "[tools] web_search: 搜索完成 query=%r 结果数=%d 截断=%d 域名过滤=%d",
            query,
            len(results),
            truncated_count,
            filtered_count,
        )
        return {
            "query": query,
            "source": "tavily",
            "results_count": len(results),
            "results": results,
        }
    except httpx.HTTPStatusError as e:
        logger.warning("[tools] web_search: HTTP 错误 query=%r — %s", query, e)
        if e.response.status_code == 432:
            return {
                "error": f"搜索失败 (HTTP 432): {e}",
                "status_code": 432,
                "retry_with_browser": True,
            }
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
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> dict:
    allowed = _normalize_domains(allowed_domains)
    blocked = _normalize_domains(blocked_domains)
    base_url = _normalize_searxng_url(base_url)
    if not base_url:
        return {"error": "SearXNG 地址未配置"}

    url = urljoin(base_url, "search")
    search_query = _searxng_domain_query(
        query,
        allowed_domains=allowed,
        blocked_domains=blocked,
    )
    try:
        logger.info("[tools] web_search: SearXNG 开始搜索 query=%r url=%s", query, base_url)
        with httpx.Client(timeout=_SEARXNG_TIMEOUT) as client:
            response = client.get(
                url,
                params={
                    "q": search_query,
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
        results, filtered_count = _filter_results_by_domains(
            results,
            allowed_domains=allowed,
            blocked_domains=blocked,
        )

        logger.info(
            "[tools] web_search: SearXNG 搜索完成 query=%r 结果数=%d 截断=%d 域名过滤=%d",
            query,
            len(results),
            truncated_count,
            filtered_count,
        )
        return {
            "query": query,
            "source": "searxng",
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


def _search_one(
    query: str,
    *,
    max_results: int,
    searxng: dict,
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
) -> dict:
    if not bool(searxng.get("enabled", False)):
        result = _search_tavily(
            query,
            max_results,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )
        if isinstance(result, dict) and "query" not in result:
            return {"query": query, **result}
        return result

    searx_result = _search_searxng(
        query=query,
        max_results=max_results,
        base_url=str(searxng.get("base_url", "") or ""),
        language=str(searxng.get("language", "zh-CN") or "zh-CN"),
        safesearch=int(searxng.get("safesearch", 0) or 0),
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
    )
    if not searx_result.get("error") and searx_result.get("results_count", 0) > 0:
        return searx_result

    logger.warning(
        "[tools] web_search: SearXNG 不可用或无结果，回退 Tavily query=%r reason=%s",
        query,
        searx_result.get("error") or "empty results",
    )
    tavily_result = _search_tavily(
        query,
        max_results,
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
    )
    if isinstance(tavily_result, dict):
        tavily_result["fallback_from"] = {
            "source": "searxng",
            "error": searx_result.get("error", ""),
            "results_count": searx_result.get("results_count", 0),
        }
        if "query" not in tavily_result:
            tavily_result["query"] = query
    return tavily_result


def _multi_search_result(queries: list[str], searches: list[dict]) -> dict:
    searches, raw_count, duplicate_count = _dedupe_search_results(searches)
    total_count = sum(
        int(result.get("results_count", 0) or 0)
        for result in searches
        if isinstance(result, dict)
    )
    errors_count = sum(
        1
        for result in searches
        if isinstance(result, dict) and result.get("error")
    )
    result: dict[str, Any] = {
        "queries": queries,
        "results_count": total_count,
        "searches": searches,
    }
    if duplicate_count:
        result["raw_results_count"] = raw_count
        result["duplicates_omitted"] = duplicate_count
    if errors_count:
        result["errors_count"] = errors_count
    return result


def make_handler(config: dict) -> Callable:
    def execute(
        queries: list[str] | None = None,
        query: str | None = None,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        **kwargs,
    ) -> dict:
        search_queries = _normalize_queries(queries, query)
        if not search_queries:
            return {"error": "搜索关键词不能为空"}
        max_results_norm = _normalize_max_results(max_results)
        searxng = _searxng_cfg(config)

        def run_search(search_query: str) -> dict:
            return _search_one(
                search_query,
                max_results=max_results_norm,
                searxng=searxng,
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
            )

        if len(search_queries) == 1:
            return run_search(search_queries[0])

        with ThreadPoolExecutor(max_workers=len(search_queries)) as executor:
            searches = list(executor.map(run_search, search_queries))
        return _multi_search_result(search_queries, searches)

    return execute


def repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    repaired = dict(args)
    changes: list[str] = []
    legacy_query = repaired.pop("query", None)
    if legacy_query is not None:
        changes.append("query -> queries[0] (legacy)")
        if "queries" not in repaired:
            repaired["queries"] = [legacy_query]
    return repaired, changes


def sanitize_semantic_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str], str | None]:
    repaired = dict(args)
    queries = _normalize_queries(repaired.get("queries"))
    changes: list[str] = []
    if queries != repaired.get("queries"):
        repaired["queries"] = queries
        changes.append("queries: trimmed empty or duplicate entries")
    if not queries:
        return repaired, changes, "queries is empty"
    return repaired, changes, None
