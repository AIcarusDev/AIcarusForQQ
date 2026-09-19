"""Settings saves must match the endpoint that will actually receive videos."""
import asyncio
from copy import deepcopy

import pytest
from quart import Quart

from tools.video.client import get_video_config, VideoModelClient, VideoProcessingError


@pytest.fixture
def settings_client(monkeypatch):
    import app_state
    from web import routes_settings
    config = {
        "provider": "new", "model": "main", "generation": {},
        "model_providers": {"new": {"base_url": "https://new.example/v1", "requires_api_key": False}},
        "memory": {"auto_archive": {"enabled": False}, "processing": {"enabled": False}},
        "video_understanding": {"base_url": "https://old.example/v1", "api_key": "fake-old",
                                "api_key_env": "FAKE_OLD_KEY", "model": "old", "provider": "new"},
    }
    monkeypatch.setattr(app_state, "config", config)
    monkeypatch.setattr(app_state, "webui_only", True)
    monkeypatch.setattr(app_state, "webui_standalone", False, raising=False)
    monkeypatch.setattr(app_state, "MODEL", "main")
    monkeypatch.setattr(app_state, "MODEL_NAME", "main")
    saved = []
    monkeypatch.setattr(routes_settings, "save_config", lambda cfg, **kw: saved.append(deepcopy(cfg)))
    app = Quart(__name__)
    app.register_blueprint(routes_settings.settings_bp)
    return app.test_client(), saved


def save(client, video):
    return asyncio.run(client.post("/settings/full", json={"provider": "new", "model": "main", "video_understanding": video}))


def test_provider_switch_clears_hidden_overrides(settings_client):
    client, saved = settings_client
    assert save(client, {"provider": "new", "model": "video"}).status_code == 200
    result = saved[0]["video_understanding"]
    assert not any(result.get(key) for key in ("base_url", "api_key", "api_key_env"))
    assert get_video_config()["base_url"] == "https://new.example/v1"
    assert get_video_config()["api_key"] == ""


def test_explicit_connection_is_editable_and_preserved(settings_client, monkeypatch):
    client, saved = settings_client
    monkeypatch.setenv("FAKE_NEW_KEY", "new-secret")
    assert save(client, {"connection_mode": "explicit", "provider": "new", "model": "video",
                         "base_url": "https://direct.example/v1beta", "api_key": "", "api_key_env": "FAKE_NEW_KEY"}).status_code == 200
    assert "provider" not in saved[0]["video_understanding"]
    assert get_video_config()["base_url"] == "https://direct.example/v1beta"
    assert get_video_config()["api_key"] == "new-secret"


@pytest.mark.parametrize("invalid", [
    {"max_size_mb": 1024}, {"max_size_mb": -1}, {"max_size_mb": "nan"},
    {"timeout": -1}, {"timeout": "inf"}, {"protocol": "typo"}, {"provider": "missing"},
    {"connection_mode": "explicit", "base_url": ""},
])
def test_invalid_settings_do_not_save_or_hot_apply(settings_client, invalid):
    import app_state
    client, saved = settings_client
    before = deepcopy(app_state.config)
    assert save(client, invalid).status_code == 400
    assert saved == []
    assert app_state.config == before


def test_hard_limit_cannot_be_disabled_in_runtime_config(tmp_path):
    file = tmp_path / "video.mp4"
    file.write_bytes(b"test")
    with pytest.raises(VideoProcessingError):
        VideoModelClient({"max_size_mb": 1024}).validate_file_size(file)
