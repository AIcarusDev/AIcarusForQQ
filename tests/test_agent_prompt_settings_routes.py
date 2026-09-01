from __future__ import annotations

import asyncio
from pathlib import Path

from quart import Quart
import yaml

import app_state
from web import routes_settings


def _write_prompt_fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    prompt_paths = {
        key: tmp_path / f"{key}.md"
        for key in ("drive", "cognition_content", "cognition_prompt")
    }
    for key, path in prompt_paths.items():
        path.write_text(f"{key}-initial", encoding="utf-8", newline="")
    config_path = tmp_path / "config_user.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "prompt_files": {
                    key: str(path)
                    for key, path in prompt_paths.items()
                },
                "unrelated": {"preserved": True},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path, prompt_paths


def test_agent_prompt_settings_route_preserves_files_and_revision_guard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        config_path, prompt_paths = _write_prompt_fixture(tmp_path)
        config_before = config_path.read_bytes()
        monkeypatch.setattr(routes_settings, "_AGENT_PROMPT_CONFIG_PATH", config_path)
        monkeypatch.setattr(
            app_state,
            "config",
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
        )

        app = Quart(__name__)
        app.register_blueprint(routes_settings.settings_bp)
        client = app.test_client()

        loaded = await client.get("/settings/agent-prompt")
        assert loaded.status_code == 200
        snapshot = (await loaded.get_json())["data"]
        assert loaded.headers["ETag"] == f'"{snapshot["revision"]}"'
        assert loaded.headers["Cache-Control"] == "no-store"

        missing_revision = await client.patch(
            "/settings/agent-prompt",
            json={"values": snapshot["values"]},
        )
        assert missing_revision.status_code == 428

        values = {
            "drive": "  drive\n",
            "cognition_content": "",
            "cognition_prompt": "prompt\n\n",
        }
        saved_response = await client.patch(
            "/settings/agent-prompt",
            headers={"If-Match": f'"{snapshot["revision"]}"'},
            json={"values": values},
        )
        assert saved_response.status_code == 200
        saved = (await saved_response.get_json())["data"]
        assert saved["values"] == values
        assert saved["revision"] != snapshot["revision"]
        assert saved["applied"] is True
        assert saved["restart_required"] is False
        assert config_path.read_bytes() == config_before
        assert {
            key: path.read_text(encoding="utf-8")
            for key, path in prompt_paths.items()
        } == values

        prompt_paths["drive"].write_text(
            "externally-edited-drive",
            encoding="utf-8",
            newline="",
        )
        conflict = await client.patch(
            "/settings/agent-prompt",
            headers={"If-Match": saved["revision"]},
            json={"values": values},
        )
        conflict_payload = await conflict.get_json()
        assert conflict.status_code == 409
        assert conflict_payload["error"]["code"] == "agent_prompt_revision_conflict"
        assert conflict_payload["latest"]["values"]["drive"] == "externally-edited-drive"

    asyncio.run(scenario())


def test_agent_prompt_settings_route_rejects_invalid_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        config_path, _prompt_paths = _write_prompt_fixture(tmp_path)
        monkeypatch.setattr(routes_settings, "_AGENT_PROMPT_CONFIG_PATH", config_path)
        monkeypatch.setattr(
            app_state,
            "config",
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
        )

        app = Quart(__name__)
        app.register_blueprint(routes_settings.settings_bp)
        client = app.test_client()
        loaded = (await (await client.get("/settings/agent-prompt")).get_json())["data"]

        for invalid in (None, "x" * 200_001):
            response = await client.patch(
                "/settings/agent-prompt",
                headers={"If-Match": loaded["revision"]},
                json={"values": {**loaded["values"], "drive": invalid}},
            )
            assert response.status_code == 422

        unknown = await client.patch(
            "/settings/agent-prompt",
            headers={"If-Match": loaded["revision"]},
            json={"values": {**loaded["values"], "unknown": "value"}},
        )
        assert unknown.status_code == 422

    asyncio.run(scenario())

