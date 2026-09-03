from __future__ import annotations

from pathlib import Path

import pytest

import config_loader
from config_loader import (
    AGENT_PROMPT_KEYS,
    PromptDocumentError,
    load_agent_prompt_docs,
    save_agent_prompt_docs,
)
from llm import session as session_module


def _prepare_prompt_root(root: Path) -> None:
    templates = {
        "drive": "drive-template\n",
        "cognition_content": "content-template\n",
        "cognition_prompt": "prompt-template\n",
    }
    for key, text in templates.items():
        directory = root / "config" / key
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{key}.md.template").write_text(
            text,
            encoding="utf-8",
            newline="",
        )


def _use_prompt_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(config_loader, "_BASE_DIR", str(root))


def test_missing_agent_prompt_files_are_seeded_without_overwriting_existing_or_empty_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_prompt_root(tmp_path)
    _use_prompt_root(monkeypatch, tmp_path)
    drive_path = tmp_path / "config" / "drive" / "drive.md"
    content_path = tmp_path / "config" / "cognition_content" / "cognition_content.md"
    drive_path.write_text("custom-drive", encoding="utf-8", newline="")
    content_path.write_text("", encoding="utf-8", newline="")

    loaded = load_agent_prompt_docs({})

    assert loaded == {
        "drive": "custom-drive",
        "cognition_content": "",
        "cognition_prompt": "prompt-template\n",
    }
    assert drive_path.read_text(encoding="utf-8") == "custom-drive"
    assert content_path.read_text(encoding="utf-8") == ""


def test_agent_prompt_paths_support_relative_and_absolute_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_prompt_root(tmp_path)
    _use_prompt_root(monkeypatch, tmp_path)
    absolute_content = tmp_path / "outside" / "content.md"
    absolute_content.parent.mkdir()
    absolute_content.write_text("absolute-content", encoding="utf-8")
    config = {
        "prompt_files": {
            "drive": "custom/drive.md",
            "cognition_content": str(absolute_content),
            "cognition_prompt": "custom/cognition-prompt.md",
        }
    }

    loaded = load_agent_prompt_docs(config)

    assert loaded["drive"] == "drive-template\n"
    assert loaded["cognition_content"] == "absolute-content"
    assert loaded["cognition_prompt"] == "prompt-template\n"
    assert (tmp_path / "custom" / "drive.md").is_file()
    assert (tmp_path / "custom" / "cognition-prompt.md").is_file()


def test_agent_prompt_save_preserves_exact_text_and_replaces_files_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_prompt_root(tmp_path)
    _use_prompt_root(monkeypatch, tmp_path)
    load_agent_prompt_docs({})
    values = {
        "drive": "  drive\n",
        "cognition_content": "",
        "cognition_prompt": "prompt\n\n",
    }

    save_agent_prompt_docs({}, values)

    assert load_agent_prompt_docs({}) == values
    assert list((tmp_path / "config").rglob("*.tmp")) == []


def test_agent_prompt_requires_a_template_for_first_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_prompt_root(monkeypatch, tmp_path)

    with pytest.raises(PromptDocumentError, match="模板不存在"):
        load_agent_prompt_docs({})


def test_session_reloads_complete_agent_prompt_snapshot_and_uses_last_good_on_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_prompt_root(tmp_path)
    _use_prompt_root(monkeypatch, tmp_path)
    initial = load_agent_prompt_docs({})
    synthetic_template = "|".join(
        "{" + key + "}"
        for key in (*AGENT_PROMPT_KEYS, "persona", "self_name", "model_name", "guardian_info")
    )
    monkeypatch.setattr(session_module, "SYSTEM_PROMPT", synthetic_template)
    conversation = session_module.ConversationSession(
        _persona="persona-fixture",
        _self_name="self-fixture",
        _model_name="model-fixture",
        _agent_prompt_docs=dict(initial),
    )

    first = conversation.build_system_prompt()
    (tmp_path / "config" / "drive" / "drive.md").write_text(
        "drive-updated",
        encoding="utf-8",
        newline="",
    )
    second = conversation.build_system_prompt()

    assert first.split("|")[:3] == [
        initial["drive"],
        initial["cognition_content"],
        initial["cognition_prompt"],
    ]
    assert second.split("|")[:3] == [
        "drive-updated",
        initial["cognition_content"],
        initial["cognition_prompt"],
    ]

    (tmp_path / "config" / "drive" / "drive.md").write_bytes(b"\xff")
    assert conversation.build_system_prompt() == second


def test_system_prompt_selects_native_reasoning_contract_from_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_prompt_root(tmp_path)
    _use_prompt_root(monkeypatch, tmp_path)
    initial = load_agent_prompt_docs({})
    prompt_template = "|".join(
        (
            "{response_sequence}",
            "{cognition_intro}",
            "{motive_intro}",
            "{cognition_output_block}",
        )
    )
    explicit_parts = {
        "response_sequence": "explicit-sequence",
        "cognition_intro": "explicit-cognition",
        "motive_intro": "explicit-motive",
        "cognition_output_block": "explicit-output",
    }
    native_parts = {
        "response_sequence": "native-sequence",
        "cognition_intro": "native-cognition",
        "motive_intro": "native-motive",
        "cognition_output_block": "",
    }
    monkeypatch.setattr(session_module, "SYSTEM_PROMPT", prompt_template)
    monkeypatch.setattr(
        session_module,
        "EXPLICIT_COGNITION_PROMPT_PARTS",
        explicit_parts,
    )
    monkeypatch.setattr(
        session_module,
        "NATIVE_REASONING_PROMPT_PARTS",
        native_parts,
    )
    conversation = session_module.ConversationSession(
        _agent_prompt_docs=dict(initial),
    )

    assert conversation.build_system_prompt() == "|".join(explicit_parts.values())
    assert conversation.build_system_prompt(
        native_reasoning_as_cognition=True
    ) == "|".join(native_parts.values())
