from __future__ import annotations

from pathlib import Path

import pytest

import config_loader
from config_loader import (
    load_agent_prompt_docs,
    save_agent_prompt_docs,
)
from llm import session as session_module


def _prepare_prompt_root(root: Path) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)


def _use_prompt_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(config_loader, "_BASE_DIR", str(root))


def test_missing_agent_prompt_files_are_seeded_without_overwriting_existing_or_empty_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_prompt_root(tmp_path)
    _use_prompt_root(monkeypatch, tmp_path)
    instruction_path = tmp_path / "config" / "instruction.md"
    instruction_path.write_text("", encoding="utf-8", newline="")

    loaded = load_agent_prompt_docs({})

    assert loaded == {
        "instruction": "",
    }
    assert instruction_path.read_text(encoding="utf-8") == ""


def test_agent_prompt_paths_support_relative_and_absolute_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_prompt_root(tmp_path)
    _use_prompt_root(monkeypatch, tmp_path)
    absolute_instruction = tmp_path / "outside" / "instruction.md"
    absolute_instruction.parent.mkdir()
    absolute_instruction.write_text("absolute-instruction", encoding="utf-8")
    config = {
        "prompt_files": {
            "instruction": str(absolute_instruction),
        }
    }

    loaded = load_agent_prompt_docs(config)

    assert loaded["instruction"] == "absolute-instruction"


def test_agent_prompt_save_preserves_exact_text_and_replaces_files_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_prompt_root(tmp_path)
    _use_prompt_root(monkeypatch, tmp_path)
    load_agent_prompt_docs({})
    values = {
        "instruction": "custom instruction\r\n\r\n",
    }

    save_agent_prompt_docs({}, values)

    assert load_agent_prompt_docs({}) == values
    assert list((tmp_path / "config").rglob("*.tmp")) == []


def test_agent_prompt_initializes_missing_instruction_as_empty_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_prompt_root(monkeypatch, tmp_path)

    assert load_agent_prompt_docs({}) == {"instruction": ""}
    assert (tmp_path / "config" / "instruction.md").read_bytes() == b""


def test_legacy_prompt_file_keys_do_not_break_instruction_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_prompt_root(monkeypatch, tmp_path)
    config = {
        "prompt_files": {
            "drive": "missing/drive.md",
            "cognition_content": "missing/content.md",
            "cognition_prompt": "missing/prompt.md",
        }
    }

    assert load_agent_prompt_docs(config) == {"instruction": ""}


def test_session_reloads_complete_agent_prompt_snapshot_and_uses_last_good_on_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_prompt_root(tmp_path)
    _use_prompt_root(monkeypatch, tmp_path)
    initial = load_agent_prompt_docs({})
    conversation = session_module.ConversationSession(
        _persona="persona-fixture",
        _self_name="self-fixture",
        _model_name="model-fixture",
        _agent_prompt_docs=dict(initial),
    )

    first = conversation.build_prompt_prelude()
    (tmp_path / "config" / "instruction.md").write_text(
        "instruction <updated>",
        encoding="utf-8",
        newline="",
    )
    second = conversation.build_prompt_prelude()

    assert first.instruction == ""
    assert "`<instruction>`" not in first.system_prompt
    assert "instruction &lt;updated&gt;" in second.instruction
    assert "`<instruction>`" in second.system_prompt

    (tmp_path / "config" / "instruction.md").write_bytes(b"\xff")
    assert conversation.build_prompt_prelude() == second


def test_system_prompt_ignores_native_reasoning_when_temporarily_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_prompt_root(tmp_path)
    _use_prompt_root(monkeypatch, tmp_path)
    initial = load_agent_prompt_docs({})
    conversation = session_module.ConversationSession(
        _agent_prompt_docs=dict(initial),
    )

    default_prompt = conversation.build_system_prompt(native_reasoning_as_cognition=False)
    native_prompt = conversation.build_system_prompt(native_reasoning_as_cognition=True)

    assert default_prompt == native_prompt


def test_guardian_card_and_notice_are_gated_and_escaped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_prompt_root(tmp_path)
    _use_prompt_root(monkeypatch, tmp_path)

    without_guardian = session_module.ConversationSession().build_prompt_prelude()
    with_guardian = session_module.ConversationSession(
        _guardian_info="guardian <&>",
    ).build_prompt_prelude()

    assert without_guardian.guardian_card == ""
    assert "`<guardian_card>`" not in without_guardian.system_prompt
    assert "guardian &lt;&amp;&gt;" in with_guardian.guardian_card
    assert "`<guardian_card>`" in with_guardian.system_prompt
