import threading

import httpx

from llm.core.tool_calling import attach_tool_result_warnings
from tools.core import web_extract, web_search


class _Tavily432Client:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, **kwargs):
        return httpx.Response(432, request=httpx.Request("POST", url))


class _TavilyDomainClient:
    payload = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, **kwargs):
        type(self).payload = kwargs.get("json")
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "results": [
                    {
                        "title": "GitHub",
                        "url": "https://github.com/openai/codex",
                        "content": "ok",
                        "score": 0.9,
                    },
                    {
                        "title": "Docs",
                        "url": "https://docs.github.com/en",
                        "content": "ok",
                        "score": 0.8,
                    },
                    {
                        "title": "Blocked subdomain",
                        "url": "https://gist.github.com/demo",
                        "content": "blocked",
                        "score": 0.7,
                    },
                    {
                        "title": "Other",
                        "url": "https://example.com/demo",
                        "content": "other",
                        "score": 0.6,
                    },
                ]
            },
        )


class _SearxngDomainClient:
    params = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, **kwargs):
        type(self).params = kwargs.get("params")
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={
                "results": [
                    {
                        "title": "Allowed",
                        "url": "https://github.com/openai/codex",
                        "content": "ok",
                        "score": 1,
                        "engine": "test",
                    },
                    {
                        "title": "Blocked",
                        "url": "https://example.com/demo",
                        "content": "blocked",
                        "score": 0.5,
                        "engine": "test",
                    },
                ],
                "unresponsive_engines": [],
            },
        )


def test_web_search_432_guides_model_to_browser(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(web_search.httpx, "Client", _Tavily432Client)

    result = web_search._search_tavily("鲲 Galgame 补丁", 5)

    assert result["status_code"] == 432
    assert result["retry_with_browser"] is True
    assert "browser_control" not in result["error"]

    attach_tool_result_warnings(tool_name="web_search", args={"queries": ["鲲 Galgame 补丁"]}, result=result, flow=None)
    warning = result["warning"]
    assert warning["code"] == "TAVILY_432_USE_BROWSER"
    assert warning["suggested_tool"] == "namespace_manage"
    assert warning["suggested_namespace"] == "browser_use"
    assert warning["target_tool"] == "browser_control"


def test_web_search_tavily_maps_and_enforces_domain_filters(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(web_search.httpx, "Client", _TavilyDomainClient)

    result = web_search._search_tavily(
        "codex",
        5,
        allowed_domains=["https://github.com/openai"],
        blocked_domains=["*.gist.github.com"],
    )

    assert _TavilyDomainClient.payload["include_domains"] == ["github.com"]
    assert _TavilyDomainClient.payload["exclude_domains"] == ["gist.github.com"]
    assert result["results_count"] == 2
    assert [item["url"] for item in result["results"]] == [
        "https://github.com/openai/codex",
        "https://docs.github.com/en",
    ]


def test_web_search_searxng_adds_site_terms_and_filters_results(monkeypatch):
    monkeypatch.setattr(web_search.httpx, "Client", _SearxngDomainClient)

    result = web_search._search_searxng(
        query="codex",
        max_results=5,
        base_url="http://searxng.local",
        language="zh-CN",
        safesearch=0,
        allowed_domains=["github.com"],
        blocked_domains=["example.com"],
    )

    assert _SearxngDomainClient.params["q"] == "(site:github.com) -site:example.com codex"
    assert result["results_count"] == 1
    assert result["results"][0]["url"] == "https://github.com/openai/codex"


def test_web_search_handler_runs_multiple_queries(monkeypatch):
    calls = []

    def fake_search_tavily(query, max_results, *, allowed_domains=None, blocked_domains=None):
        calls.append((query, max_results, allowed_domains, blocked_domains))
        return {
            "query": query,
            "source": "tavily",
            "results_count": 1,
            "results": [{"title": query, "url": f"https://example.com/{query}", "content": ""}],
        }

    monkeypatch.setattr(web_search, "_search_tavily", fake_search_tavily)
    handler = web_search.make_handler({"web_search": {"searxng": {"enabled": False}}})

    result = handler(
        queries=[" codex ", "tavily", "codex", ""],
        max_results=3,
        allowed_domains=["example.com"],
    )

    assert sorted(calls) == sorted([
        ("codex", 3, ["example.com"], None),
        ("tavily", 3, ["example.com"], None),
    ])
    assert result["queries"] == ["codex", "tavily"]
    assert result["results_count"] == 2
    assert [item["query"] for item in result["searches"]] == ["codex", "tavily"]


def test_web_search_handler_runs_multiple_queries_concurrently(monkeypatch):
    started: list[str] = []
    lock = threading.Lock()
    all_started = threading.Event()

    def fake_search_tavily(query, max_results, *, allowed_domains=None, blocked_domains=None):
        with lock:
            started.append(query)
            if len(started) == 2:
                all_started.set()
        assert all_started.wait(2.0)
        return {
            "query": query,
            "source": "tavily",
            "results_count": 1,
            "results": [{"title": query, "url": f"https://example.com/{query}", "content": ""}],
        }

    monkeypatch.setattr(web_search, "_search_tavily", fake_search_tavily)
    handler = web_search.make_handler({"web_search": {"searxng": {"enabled": False}}})

    result = handler(queries=["codex", "tavily"], max_results=3)

    assert set(started) == {"codex", "tavily"}
    assert [item["query"] for item in result["searches"]] == ["codex", "tavily"]


def test_web_search_multi_query_deduplicates_result_urls():
    result = web_search._multi_search_result(
        ["codex", "openai codex"],
        [
            {
                "query": "codex",
                "source": "tavily",
                "results_count": 2,
                "results": [
                    {
                        "title": "Codex",
                        "url": "https://www.example.com/codex/?utm_source=news#top",
                        "content": "first",
                    },
                    {
                        "title": "Docs",
                        "url": "https://docs.example.com/codex",
                        "content": "docs",
                    },
                ],
            },
            {
                "query": "openai codex",
                "source": "tavily",
                "results_count": 2,
                "results": [
                    {
                        "title": "Codex duplicate",
                        "url": "https://example.com/codex",
                        "content": "duplicate",
                    },
                    {
                        "title": "Blog",
                        "url": "https://blog.example.com/codex",
                        "content": "blog",
                    },
                ],
            },
        ],
    )

    assert result["results_count"] == 3
    assert result["raw_results_count"] == 4
    assert result["duplicates_omitted"] == 1
    assert result["searches"][0]["results"][0]["matched_queries"] == ["codex", "openai codex"]
    assert result["searches"][1]["results_count"] == 1
    assert result["searches"][1]["raw_results_count"] == 2
    assert result["searches"][1]["duplicates_omitted"] == 1
    assert [item["url"] for item in result["searches"][1]["results"]] == [
        "https://blog.example.com/codex"
    ]


def test_web_search_repairs_legacy_query_argument():
    repaired, changes = web_search.repair_schema_args({"query": "codex"})

    assert repaired == {"queries": ["codex"]}
    assert changes == ["query -> queries[0] (legacy)"]


def test_web_extract_432_guides_model_to_browser(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(web_extract.httpx, "Client", _Tavily432Client)

    result = web_extract.execute("https://www.moyu.moe/")

    assert result["status_code"] == 432
    assert result["retry_with_browser"] is True
    assert result["url"] == "https://www.moyu.moe/"

    attach_tool_result_warnings(
        tool_name="web_extract",
        args={"url": "https://www.moyu.moe/"},
        result=result,
        flow=None,
    )
    warning = result["warning"]
    assert warning["code"] == "TAVILY_432_USE_BROWSER"
    assert warning["suggested_tool"] == "namespace_manage"
    assert warning["suggested_namespace"] == "browser_use"
    assert warning["target_tool"] == "browser_control"


def test_tavily_432_warning_factory_ignores_unrelated_tools():
    result = {"error": "搜索失败 (HTTP 432)", "status_code": 432}

    attach_tool_result_warnings(tool_name="calculator", args={}, result=result, flow=None)

    assert "warning" not in result
    assert "warnings" not in result
