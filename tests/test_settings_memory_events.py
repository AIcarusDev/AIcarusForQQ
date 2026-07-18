import asyncio
import sys
from pathlib import Path

from quart import Quart


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_default_memory_cfg_includes_events_controls():
    from web.routes_settings import _default_memory_cfg

    cfg = _default_memory_cfg({"memory": {}})

    assert cfg["events"] == {
        "recall_limit": 6,
        "world_query_chunks": 6,
        "cognition_query_chunks": 3,
    }
    assert cfg["processing"] == {
        "enabled": False,
        "event_structuring_enabled": False,
        "algorithmic_storyline_enabled": False,
        "dry_run": True,
        "solidify": False,
        "max_candidate_storylines_per_maintenance": 100,
        "maintenance_timeout_seconds": 300,
        "storyline_synthesis_max_inputs_per_maintenance": 32,
        "storyline_synthesis_max_retries": 3,
        "provider": "",
        "model": "",
        "generation": {
            "temperature": 0.2,
            "max_output_tokens": 4000,
            "enable_thinking": False,
        },
    }


def test_default_memory_cfg_drops_retired_capacity_limits():
    from web.routes_settings import _default_memory_cfg

    cfg = _default_memory_cfg({"memory": {"max_active": 8, "max_passive": 15}})

    assert "max_active" not in cfg
    assert "max_passive" not in cfg


def test_default_memory_cfg_preserves_existing_events_controls():
    from web.routes_settings import _default_memory_cfg

    cfg = _default_memory_cfg({
        "memory": {
            "events": {
                "recall_limit": 9,
                "world_query_chunks": 12,
                "cognition_query_chunks": 4,
            }
        }
    })

    assert cfg["events"]["recall_limit"] == 9
    assert cfg["events"]["world_query_chunks"] == 12
    assert cfg["events"]["cognition_query_chunks"] == 4


def test_default_memory_cfg_preserves_existing_processing_controls():
    from web.routes_settings import _default_memory_cfg

    cfg = _default_memory_cfg(
        {
            "memory": {
                "processing": {
                    "enabled": True,
                    "event_structuring_enabled": True,
                    "algorithmic_storyline_enabled": True,
                    "dry_run": False,
                    "solidify": True,
                    "max_candidate_storylines_per_maintenance": 12,
                    "maintenance_timeout_seconds": 0,
                    "storyline_synthesis_max_inputs_per_maintenance": 7,
                    "storyline_synthesis_max_retries": 5,
                    "provider": "memory",
                    "model": "memory-model",
                    "generation": {
                        "temperature": 0.1,
                        "max_output_tokens": 2048,
                        "enable_thinking": True,
                    },
                }
            }
        }
    )

    assert cfg["processing"]["enabled"] is True
    assert cfg["processing"]["event_structuring_enabled"] is True
    assert cfg["processing"]["algorithmic_storyline_enabled"] is True
    assert cfg["processing"]["dry_run"] is False
    assert cfg["processing"]["solidify"] is True
    assert cfg["processing"]["max_candidate_storylines_per_maintenance"] == 12
    assert cfg["processing"]["maintenance_timeout_seconds"] == 0
    assert cfg["processing"]["storyline_synthesis_max_inputs_per_maintenance"] == 7
    assert cfg["processing"]["storyline_synthesis_max_retries"] == 5
    assert cfg["processing"]["provider"] == "memory"
    assert cfg["processing"]["model"] == "memory-model"
    assert cfg["processing"]["generation"] == {
        "temperature": 0.1,
        "max_output_tokens": 2048,
        "enable_thinking": True,
    }


def test_guardian_save_is_independent_and_updates_sessions(monkeypatch):
    import app_state
    from web import routes_settings

    saved_configs = []
    session_updates = []
    monkeypatch.setattr(app_state, "config", {"guardian": "旧介绍"})
    monkeypatch.setattr(app_state, "MAX_CONTEXT", 10)
    monkeypatch.setattr(app_state, "TIMEZONE", "Asia/Shanghai")
    monkeypatch.setattr(app_state, "persona", "persona")
    monkeypatch.setattr(app_state, "SELF_NAME", "AIcarus")
    monkeypatch.setattr(app_state, "MODEL_NAME", "test-model")
    monkeypatch.setattr(
        routes_settings,
        "save_config",
        lambda config, **_kwargs: saved_configs.append(dict(config)),
    )
    monkeypatch.setattr(
        routes_settings,
        "init_session_globals",
        lambda **kwargs: session_updates.append(kwargs),
    )

    app = Quart(__name__)
    app.register_blueprint(routes_settings.settings_bp)

    async def scenario():
        client = app.test_client()
        response = await client.post(
            "/settings/guardian",
            json={"guardian": "  第一行\n第二行  "},
        )
        assert response.status_code == 200
        assert await response.get_json() == {
            "success": True,
            "guardian": "第一行\n第二行",
        }

        invalid = await client.post(
            "/settings/guardian",
            json={"guardian": {"name": "旧格式"}},
        )
        assert invalid.status_code == 400

    asyncio.run(scenario())

    assert saved_configs == [{"guardian": "第一行\n第二行"}]
    assert app_state.config == {"guardian": "第一行\n第二行"}
    assert session_updates[0]["guardian_info"] == "第一行\n第二行"
