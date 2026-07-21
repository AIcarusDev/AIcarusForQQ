from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from quart import Quart
import pytest
import yaml

import app_state
from web import routes_ui_v1_settings, settings_domains
from web.settings_domains import (
    SCHEMA_VERSION,
    SUPPORTED_DOMAINS,
    SettingsConflict,
    SettingsDomainStore,
)


def _write_config(path: Path) -> None:
    prompt_files = {
        key: str(path.parent / f"{key}.md")
        for key in ("drive", "cognition_content", "cognition_prompt")
    }
    for key, prompt_path in prompt_files.items():
        Path(prompt_path).write_text(f"{key}-fixture", encoding="utf-8", newline="")
    path.write_text(
        yaml.safe_dump(
            {
                "model_providers": {
                    "primary": {
                        "name": "Primary",
                        "base_url": "https://example.test/v1",
                        "api_key_env": "MODEL_PROVIDER_PRIMARY_API_KEY",
                        "requires_api_key": True,
                    }
                },
                "provider": "primary",
                "model": "model-a",
                "model_name": "Model A",
                "generation": {
                    "temperature": 0.7,
                    "max_output_tokens": 4096,
                    "enable_thinking": True,
                },
                "max_calls_per_minute": 15,
                "prompt_files": prompt_files,
                "tts": {
                    "enabled": False,
                    "host": "127.0.0.1",
                    "port": 8765,
                    "secret_token": "tts-private",
                    "max_concurrent_tasks_per_plugin": 8,
                },
                "unrelated": {"preserved": True},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _store(tmp_path: Path) -> SettingsDomainStore:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ("." + "env")
    persona_path = tmp_path / "persona.md"
    _write_config(config_path)
    env_path.write_text("MODEL_PROVIDER_PRIMARY_API_KEY=old-secret\n", encoding="utf-8")
    persona_path.write_text("persona", encoding="utf-8")
    app_state.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return SettingsDomainStore(
        config_path=config_path,
        env_path=env_path,
        persona_path=persona_path,
    )


def _set_path(value: dict, path: str, replacement: object) -> None:
    current = value
    parts = path.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = replacement


def _get_path(value: dict, path: str) -> object:
    current: object = value
    for part in path.split("."):
        assert isinstance(current, dict)
        current = current[part]
    return current


def test_domain_patch_is_scoped_and_revision_guarded(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.read("main-model")
    values = first["values"]
    values["model"] = "model-b"
    values["model_name"] = "Model B"

    saved = store.update(
        "main-model",
        revision=first["revision"],
        values=values,
        secret_commands={},
    )

    persisted = yaml.safe_load(store.config_path.read_text(encoding="utf-8"))
    assert persisted["model"] == "model-b"
    assert persisted["tts"]["secret_token"] == "tts-private"
    assert persisted["unrelated"] == {"preserved": True}
    assert saved["revision"] != first["revision"]
    assert saved["saved"] is True
    assert saved["restart_required"] is True

    with pytest.raises(SettingsConflict) as conflict:
        store.update(
            "main-model",
            revision=first["revision"],
            values=values,
            secret_commands={},
        )
    assert conflict.value.latest["values"]["model"] == "model-b"


@pytest.mark.parametrize("domain", sorted(SUPPORTED_DOMAINS))
def test_every_supported_domain_round_trips_its_versioned_snapshot(
    domain: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings_domains, "load_skill_user_body", lambda _name: "social-style")
    monkeypatch.setattr(settings_domains, "save_skill_user_body", lambda _name, _body: True)
    store = _store(tmp_path)

    initial = store.read(domain)
    secret_commands = {
        secret_id: {"command": "keep"}
        for secret_id in initial["secrets"]
    }
    saved = store.update(
        domain,
        revision=initial["revision"],
        values=initial["values"],
        secret_commands=secret_commands,
    )

    assert saved["domain"] == domain
    assert saved["schema_version"] == SCHEMA_VERSION
    assert saved["values"] == initial["values"]
    assert saved["saved"] is True
    assert isinstance(saved["applied"], bool)
    assert isinstance(saved["restart_required"], bool)
    assert yaml.safe_load(store.config_path.read_text(encoding="utf-8"))["unrelated"] == {
        "preserved": True,
    }
    serialized = json.dumps(saved, ensure_ascii=False)
    assert "old-secret" not in serialized
    assert "tts-private" not in serialized


@pytest.mark.parametrize(
    ("domain", "path", "replacement"),
    [
        ("providers", "model_providers.primary.name", "Primary Renamed"),
        ("main-model", "model_name", "Model A Renamed"),
        (
            "specialized-models",
            "cognition_compression.generation.temperature",
            0.42,
        ),
        ("persona", "persona", "updated persona"),
        ("agent-prompt", "drive", "updated drive"),
        ("qq-adapter", "adapter.name", "QA Adapter"),
        ("tts", "port", 9876),
        (
            "services",
            "service_env.QWEATHER_API_HOST",
            "https://weather.example.test",
        ),
        ("alerts", "smtp.AICQ_SMTP_HOST", "smtp.example.test"),
        ("advanced", "tools.send_message.message_shape", "single"),
    ],
)
def test_every_supported_domain_persists_changed_public_values(
    domain: str,
    path: str,
    replacement: object,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings_domains, "load_skill_user_body", lambda _name: "social-style")
    monkeypatch.setattr(settings_domains, "save_skill_user_body", lambda _name, _body: True)
    domain_root = tmp_path / domain
    domain_root.mkdir()
    store = _store(domain_root)
    initial = store.read(domain)
    values = initial["values"]
    _set_path(values, path, replacement)

    saved = store.update(
        domain,
        revision=initial["revision"],
        values=values,
        secret_commands={
            secret_id: {"command": "keep"}
            for secret_id in initial["secrets"]
        },
    )
    reloaded = SettingsDomainStore(
        config_path=store.config_path,
        env_path=store.env_path,
        persona_path=store.persona_path,
    ).read(domain)

    assert _get_path(saved["values"], path) == replacement
    assert _get_path(reloaded["values"], path) == replacement


def test_qq_adapter_rejects_half_configured_file_transfer_mapping(tmp_path: Path) -> None:
    store = _store(tmp_path)
    initial = store.read("qq-adapter")
    values = initial["values"]
    values["adapter"]["file_transfer"] = {
        "host_directory": r"C:\AICQ\transfer",
        "adapter_directory": "",
    }

    with pytest.raises(settings_domains.SettingsValidationError):
        store.update(
            "qq-adapter",
            revision=initial["revision"],
            values=values,
            secret_commands={},
        )


def test_persona_revision_detects_external_file_edits(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings_domains, "load_skill_user_body", lambda _name: "social-style")
    store = _store(tmp_path)
    app_state.persona = "stale-runtime-persona"
    initial = store.read("persona")
    assert initial["values"]["persona"] == "persona"

    store.persona_path.write_text("externally-edited-persona", encoding="utf-8")

    with pytest.raises(SettingsConflict) as conflict:
        store.update(
            "persona",
            revision=initial["revision"],
            values=initial["values"],
            secret_commands={},
        )

    assert conflict.value.latest["values"]["persona"] == "externally-edited-persona"
    assert store.persona_path.read_text(encoding="utf-8") == "externally-edited-persona"


def test_agent_prompt_domain_preserves_exact_files_without_rewriting_config(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    initial = store.read("agent-prompt")
    config_before = store.config_path.read_bytes()
    values = {
        "drive": "  drive\n",
        "cognition_content": "",
        "cognition_prompt": "prompt\n\n",
    }

    saved = store.update(
        "agent-prompt",
        revision=initial["revision"],
        values=values,
        secret_commands={},
    )

    config = yaml.safe_load(store.config_path.read_text(encoding="utf-8"))
    assert store.config_path.read_bytes() == config_before
    assert {
        key: Path(config["prompt_files"][key]).read_text(encoding="utf-8")
        for key in values
    } == values
    assert saved["values"] == values
    assert saved["applied"] is True
    assert saved["restart_required"] is False


def test_agent_prompt_revision_detects_external_file_edits(tmp_path: Path) -> None:
    store = _store(tmp_path)
    initial = store.read("agent-prompt")
    config = yaml.safe_load(store.config_path.read_text(encoding="utf-8"))
    drive_path = Path(config["prompt_files"]["drive"])
    drive_path.write_text("external-drive", encoding="utf-8", newline="")

    with pytest.raises(SettingsConflict) as conflict:
        store.update(
            "agent-prompt",
            revision=initial["revision"],
            values=initial["values"],
            secret_commands={},
        )

    assert conflict.value.latest["values"]["drive"] == "external-drive"
    assert drive_path.read_text(encoding="utf-8") == "external-drive"


def test_agent_prompt_rejects_non_string_and_oversized_values(tmp_path: Path) -> None:
    store = _store(tmp_path)
    initial = store.read("agent-prompt")

    for invalid in (None, "x" * 200_001):
        values = {**initial["values"], "drive": invalid}
        with pytest.raises(settings_domains.SettingsValidationError):
            store.update(
                "agent-prompt",
                revision=initial["revision"],
                values=values,
                secret_commands={},
            )

    with pytest.raises(settings_domains.SettingsValidationError):
        store.update(
            "agent-prompt",
            revision=initial["revision"],
            values={**initial["values"], "unknown": "value"},
            secret_commands={},
        )


def test_secret_contract_never_returns_secret_and_requires_explicit_command(tmp_path: Path) -> None:
    store = _store(tmp_path)
    initial = store.read("providers")
    secret_id = "provider_api_key::primary"

    assert initial["secrets"][secret_id]["configured"] is True
    assert initial["secrets"][secret_id]["masked_hint"].endswith("cret")
    assert "old-secret" not in json.dumps(initial, ensure_ascii=False)

    kept = store.update(
        "providers",
        revision=initial["revision"],
        values=initial["values"],
        secret_commands={secret_id: {"command": "keep"}},
    )
    assert "old-secret" in store.env_path.read_text(encoding="utf-8")

    replaced = store.update(
        "providers",
        revision=kept["revision"],
        values=kept["values"],
        secret_commands={secret_id: {"command": "replace", "value": "new-secret"}},
    )
    assert "new-secret" in store.env_path.read_text(encoding="utf-8")
    assert "new-secret" not in json.dumps(replaced, ensure_ascii=False)

    cleared = store.update(
        "providers",
        revision=replaced["revision"],
        values=replaced["values"],
        secret_commands={secret_id: {"command": "clear"}},
    )
    assert "MODEL_PROVIDER_PRIMARY_API_KEY=" not in store.env_path.read_text(encoding="utf-8")
    assert cleared["secrets"][secret_id]["configured"] is False


def test_proxy_replace_and_clear_sync_persisted_and_process_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = _store(tmp_path)
    monkeypatch.setenv("OPENAI_PROXY", "http://stale-runtime.test")
    initial = store.read("advanced")

    replaced = store.update(
        "advanced",
        revision=initial["revision"],
        values=initial["values"],
        secret_commands={
            "openai_proxy": {
                "command": "replace",
                "value": "http://127.0.0.1:7897",
            }
        },
    )
    assert os.environ["OPENAI_PROXY"] == "http://127.0.0.1:7897"
    assert "OPENAI_PROXY=http://127.0.0.1:7897" in store.env_path.read_text(encoding="utf-8")

    cleared = store.update(
        "advanced",
        revision=replaced["revision"],
        values=replaced["values"],
        secret_commands={"openai_proxy": {"command": "clear"}},
    )

    assert "OPENAI_PROXY=" not in store.env_path.read_text(encoding="utf-8")
    assert "OPENAI_PROXY" not in os.environ
    assert cleared["secrets"]["openai_proxy"]["configured"] is False


def test_settings_routes_return_428_and_409_with_latest_snapshot(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> None:
        store = _store(tmp_path)
        monkeypatch.setattr(routes_ui_v1_settings, "settings_store", store)
        app = Quart(__name__)
        app.register_blueprint(routes_ui_v1_settings.ui_v1_settings_bp)
        client = app.test_client()

        loaded = await client.get("/api/ui/v1/settings/main-model")
        assert loaded.status_code == 200
        snapshot = (await loaded.get_json())["data"]
        assert loaded.headers["ETag"] == f'"{snapshot["revision"]}"'
        assert loaded.headers["Cache-Control"] == "no-store"

        missing = await client.patch(
            "/api/ui/v1/settings/main-model",
            json={"values": snapshot["values"], "secrets": {}},
        )
        assert missing.status_code == 428

        first_save = await client.patch(
            "/api/ui/v1/settings/main-model",
            headers={"If-Match": f'"{snapshot["revision"]}"'},
            json={"values": {**snapshot["values"], "model": "model-b"}, "secrets": {}},
        )
        assert first_save.status_code == 200

        conflict = await client.patch(
            "/api/ui/v1/settings/main-model",
            headers={"If-Match": snapshot["revision"]},
            json={"values": {**snapshot["values"], "model": "model-c"}, "secrets": {}},
        )
        conflict_payload = await conflict.get_json()
        assert conflict.status_code == 409
        assert conflict_payload["error"]["code"] == "settings_revision_conflict"
        assert conflict_payload["latest"]["values"]["model"] == "model-b"

    asyncio.run(scenario())
