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


def test_web_search_432_guides_model_to_browser(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(web_search.httpx, "Client", _Tavily432Client)

    result = web_search._search_tavily("鲲 Galgame 补丁", 5)

    assert result["status_code"] == 432
    assert result["retry_with_browser"] is True
    assert "browser_control" not in result["error"]

    attach_tool_result_warnings(tool_name="web_search", args={"query": "鲲 Galgame 补丁"}, result=result, flow=None)
    warning = result["warning"]
    assert warning["code"] == "TAVILY_432_USE_BROWSER"
    assert warning["suggested_tool"] == "namespace_manage"
    assert warning["suggested_namespace"] == "browser_use"
    assert warning["target_tool"] == "browser_control"
    assert "browser_control" in warning["message"]


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
    assert "https://www.moyu.moe/" in warning["message"]

def test_tavily_432_warning_factory_ignores_unrelated_tools():
    result = {"error": "搜索失败 (HTTP 432)", "status_code": 432}

    attach_tool_result_warnings(tool_name="calculator", args={}, result=result, flow=None)

    assert "warning" not in result
    assert "warnings" not in result
