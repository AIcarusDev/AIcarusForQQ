from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import timezone
from functools import partial
from types import SimpleNamespace

import pytest
from quart import Quart
import yaml

import app_state
import config_loader
from llm.compression.config import normalize_generation_config
from llm.prompt.output_requirements import COGNITION_LANGUAGES, normalize_output_requirements_config
from web import routes_settings


@pytest.mark.parametrize("language", COGNITION_LANGUAGES)
def test_supported_cognition_languages_round_trip(language):
    assert normalize_output_requirements_config(
        {"cognition_language": language}, strict=True,
    ) == {"cognition_language": language}


@pytest.mark.parametrize("value", [None, {}, "en", {"cognition_language": "unknown"}])
def test_old_or_invalid_config_uses_automatic_language(value):
    assert normalize_output_requirements_config(value) == {"cognition_language": "auto"}


def test_native_cognition_forces_thinking_without_changing_normal_mode():
    assert normalize_generation_config({
        "native_reasoning_as_cognition": True, "enable_thinking": False,
    })["enable_thinking"] is True
    assert normalize_generation_config({
        "native_reasoning_as_cognition": False, "enable_thinking": False,
    })["enable_thinking"] is False


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("language", ["auto", "zh-CN", "en"])
def test_main_prompt_routes_language_and_request_generation_snapshot(monkeypatch, native, language):
    from llm.prompt import container, user_prompt_builder as builder

    monkeypatch.setattr(app_state, "config", {"output_requirements": {"cognition_language": language}})
    monkeypatch.setattr(app_state, "GEN", {"native_reasoning_as_cognition": not native})
    monkeypatch.setattr(builder, "_build_world_prompt", lambda *args, **kwargs: "<world/>")
    monkeypatch.setattr(builder, "_build_active_skill_prompt_block", lambda: "")
    monkeypatch.setattr(builder._memory, "build_memory_xml", lambda *args, **kwargs: "")
    monkeypatch.setattr(container, "build_container_xml", lambda: "<container/>")
    rendered = []

    def render(**kwargs):
        rendered.append(kwargs)
        return "fixture-output-requirements"

    monkeypatch.setattr(builder, "build_output_requirements_prompt", render)
    session = SimpleNamespace(
        _timezone=timezone.utc, recalled_events=[], last_sender_id="", _nick_cache={},
    )
    sections = builder.build_main_user_prompt_sections(
        session, generation={"native_reasoning_as_cognition": native},
    )
    assert rendered == [{"native_reasoning_as_cognition": native, "cognition_language": language}]
    assert sections.output_requirements == "fixture-output-requirements"


def _settings_client(tmp_path, monkeypatch):
    config = {
        "provider": "fixture", "model": "fixture-model", "model_name": "Fixture",
        "model_providers": {"fixture": {
            "base_url": "https://example.invalid/v1", "requires_api_key": False,
        }},
        "generation": {},
        "memory": {"auto_archive": {"enabled": False}, "processing": {"enabled": False}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(app_state, "config", deepcopy(config))
    monkeypatch.setattr(app_state, "GEN", {})
    monkeypatch.setattr(app_state, "webui_only", True)
    monkeypatch.setattr(routes_settings, "load_dotenv", lambda **kwargs: None)
    monkeypatch.setattr(routes_settings, "save_config", partial(config_loader.save_config, config_path=str(path)))
    monkeypatch.setattr(routes_settings, "read_env_keys", lambda *args: {})
    monkeypatch.setattr(routes_settings, "read_env_values", lambda *args: {})
    monkeypatch.setattr(routes_settings, "read_env_proxies", lambda: {})
    monkeypatch.setattr(routes_settings, "load_skill_user_body", lambda *args: "")
    app = Quart(__name__)
    app.register_blueprint(routes_settings.settings_bp)
    return app.test_client(), path


def test_settings_persist_language_apply_generation_and_preserve_on_partial_save(tmp_path, monkeypatch):
    client, path = _settings_client(tmp_path, monkeypatch)

    async def scenario():
        initial = await (await client.get("/settings/full")).get_json()
        assert initial["output_requirements"] == {"cognition_language": "auto"}
        response = await client.post("/settings/full", json={
            "provider": "fixture", "model": "fixture-model",
            "output_requirements": {"cognition_language": "ja"},
            "generation": {"native_reasoning_as_cognition": True, "enable_thinking": False},
        })
        assert response.status_code == 200, await response.get_json()
        saved = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert saved["output_requirements"] == {"cognition_language": "ja"}
        assert saved["generation"]["enable_thinking"] is True
        assert app_state.config["output_requirements"] == saved["output_requirements"]
        assert app_state.GEN["enable_thinking"] is True

        partial_response = await client.post("/settings/full", json={
            "provider": "fixture", "model": "fixture-model", "self_name": "Fixture",
        })
        assert partial_response.status_code == 200
        reloaded = await (await client.get("/settings/full")).get_json()
        assert reloaded["output_requirements"] == {"cognition_language": "ja"}
        assert reloaded["generation"]["native_reasoning_as_cognition"] is True

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid", [None, "en", {"cognition_language": "unknown"}, {"cognition_language": []}])
def test_settings_reject_invalid_language_before_writes(tmp_path, monkeypatch, invalid):
    client, path = _settings_client(tmp_path, monkeypatch)
    before = path.read_bytes()
    config_before = deepcopy(app_state.config)
    writes = []
    monkeypatch.setattr(routes_settings, "save_env_key", lambda *args: writes.append(args))

    async def scenario():
        response = await client.post("/settings/full", json={
            "output_requirements": invalid,
            "api_keys": {"FIXTURE_KEY": "fixture"},
        })
        assert response.status_code == 400

    asyncio.run(scenario())
    assert path.read_bytes() == before
    assert app_state.config == config_before
    assert writes == []
