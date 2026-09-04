from __future__ import annotations

import asyncio
import sys
from copy import deepcopy
from pathlib import Path

from quart import Quart


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_full_settings_save_preserves_unsubmitted_qq_file_transfer(monkeypatch):
    import app_state
    from web import routes_settings

    original_config = {
        "provider": "test-provider",
        "model": "test-model",
        "model_name": "Test Model",
        "model_providers": {
            "test-provider": {
                "base_url": "https://example.invalid/v1",
                "requires_api_key": False,
            }
        },
        "generation": {},
        "memory": {
            "auto_archive": {"enabled": False},
            "processing": {"enabled": False},
        },
        "platforms": {
            "qq": {
                "enabled": True,
                "adapter": {
                    "type": "napcat",
                    "name": "NapCat",
                    "debug_only": False,
                    "reverse_ws": {"host": "127.0.0.1", "port": 8078},
                    "file_transfer": {
                        "host_directory": r"C:\shared\transfer",
                        "adapter_directory": "/app/napcat/transfer",
                    },
                },
            }
        },
    }
    submitted_qq = {
        "enabled": True,
        "adapter": {
            "type": "napcat",
            "name": "NapCat",
            "debug_only": True,
            "reverse_ws": {"host": "0.0.0.0", "port": 9000},
        },
    }
    saved_configs = []
    monkeypatch.setattr(app_state, "config", deepcopy(original_config))
    monkeypatch.setattr(app_state, "webui_only", True)
    monkeypatch.setattr(app_state, "webui_standalone", False, raising=False)
    monkeypatch.setattr(app_state, "MODEL", "test-model")
    monkeypatch.setattr(app_state, "MODEL_NAME", "Test Model")
    monkeypatch.setattr(
        routes_settings,
        "save_config",
        lambda config, **_kwargs: saved_configs.append(deepcopy(config)),
    )

    app = Quart(__name__)
    app.register_blueprint(routes_settings.settings_bp)

    async def scenario():
        client = app.test_client()
        response = await client.post(
            "/settings/full",
            json={
                "provider": "test-provider",
                "model": "test-model",
                "platforms": {"qq": submitted_qq},
            },
        )
        assert response.status_code == 200

    asyncio.run(scenario())

    saved_adapter = saved_configs[0]["platforms"]["qq"]["adapter"]
    assert saved_adapter["debug_only"] is True
    assert saved_adapter["reverse_ws"] == {"host": "0.0.0.0", "port": 9000}
    assert saved_adapter["file_transfer"] == original_config["platforms"]["qq"]["adapter"]["file_transfer"]
