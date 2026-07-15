import asyncio
import sys
from types import SimpleNamespace

import pytest
from quart import Quart

import app_state
from web.routes_dashboard import (
    _resolve_model_discovery_target,
    list_models_route,
)


def _config(*, requires_api_key=True):
    return {
        "model_providers": {
            "grok": {
                "base_url": "http://localhost:6657/v1",
                "api_key_env": "GROK_API_KEY",
                "requires_api_key": requires_api_key,
            }
        }
    }


def test_model_discovery_uses_saved_provider_key():
    target = _resolve_model_discovery_target(
        {
            "provider": "grok",
            "api_key": "",
            "base_url": "https://untrusted.example/v1",
        },
        _config(),
        {"GROK_API_KEY": "saved-key"},
    )

    assert target == ("http://localhost:6657/v1", "saved-key")


def test_model_discovery_explicit_key_overrides_saved_key():
    target = _resolve_model_discovery_target(
        {"provider": "grok", "api_key": "temporary-key"},
        _config(),
        {"GROK_API_KEY": "saved-key"},
    )

    assert target == ("http://localhost:6657/v1", "temporary-key")


def test_model_discovery_allows_provider_without_api_key():
    target = _resolve_model_discovery_target(
        {"provider": "grok"},
        _config(requires_api_key=False),
        {},
    )

    assert target == ("http://localhost:6657/v1", "openai-compat")


@pytest.mark.parametrize(
    ("request_data", "config", "message"),
    [
        ({"provider": "unknown"}, _config(), "未知的模型供应商"),
        ({"provider": "grok"}, _config(), "尚未设置 API Key"),
    ],
)
def test_model_discovery_rejects_invalid_provider_configuration(
    request_data,
    config,
    message,
):
    with pytest.raises(ValueError, match=message):
        _resolve_model_discovery_target(request_data, config, {})


def test_models_route_passes_saved_key_to_openai_client(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.models = SimpleNamespace(
                list=lambda: SimpleNamespace(
                    data=[SimpleNamespace(id="grok-chat-fast")]
                )
            )

    monkeypatch.setattr(app_state, "config", _config())
    monkeypatch.setenv("GROK_API_KEY", "saved-key")
    monkeypatch.delenv("OPENAI_PROXY", raising=False)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    async def scenario():
        app = Quart(__name__)
        async with app.test_request_context(
            "/models",
            method="POST",
            json={"provider": "grok", "api_key": ""},
        ):
            response = await list_models_route()
            return await response.get_json()

    result = asyncio.run(scenario())

    assert result == {"success": True, "models": ["grok-chat-fast"]}
    assert captured["api_key"] == "saved-key"
    assert captured["base_url"] == "http://localhost:6657/v1"
