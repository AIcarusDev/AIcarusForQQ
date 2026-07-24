from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_TEMPLATE = ROOT / "src" / "templates" / "settings.html"


def _source() -> str:
    return SETTINGS_TEMPLATE.read_text(encoding="utf-8")


def test_legacy_settings_exposes_agent_prompt_editor() -> None:
    source = _source()

    assert 'data-section-button="agent_prompt"' in source
    assert 'id="section-agent_prompt"' in source
    assert '["agent_prompt", "Agent Prompt"]' in source
    assert source.index('["persona", "角色"]') < source.index(
        '["agent_prompt", "Agent Prompt"]'
    )
    for field_id in (
        "agent-prompt-drive",
        "agent-prompt-cognition-content",
        "agent-prompt-cognition-prompt",
    ):
        assert f'id="{field_id}"' in source
    assert source.count('maxlength="200000"') >= 3
    assert 'id="agent-prompt-edit-btn"' in source
    assert 'id="agent-prompt-save-btn"' in source
    assert 'id="agent-prompt-cancel-btn"' in source
    assert "function openAgentPromptEdit()" in source
    assert "function cancelAgentPromptEdit()" in source
    assert "function normalizeAgentPromptLineEndings(" in source
    assert "function preserveAgentPromptLineEndings(" in source


def test_legacy_agent_prompt_uses_revision_guarded_domain_api() -> None:
    source = _source()

    assert 'fetch("/api/ui/v1/settings/agent-prompt")' in source
    assert 'fetch("/api/ui/v1/settings/agent-prompt", {' in source
    assert 'method: "PATCH"' in source
    assert '"If-Match": `"${state.agentPromptSnapshot.revision}"`' in source
    assert "res.status === 409" in source
    assert "state.agentPromptConflict = payload.latest || null" in source
    assert "hydrateAgentPromptSnapshot(payload.data)" in source


def test_legacy_agent_prompt_has_independent_edit_cancel_and_unload_guards() -> None:
    source = _source()

    assert "function agentPromptHasUnsavedChanges()" in source
    assert "function saveAgentPrompt()" in source
    assert "preserveAgentPromptDraft" in source
    assert "agentPromptDraft" in source
    assert "!agentPromptHasUnsavedChanges()" in source


def test_guardian_uses_preview_edit_confirm_and_cancel_flow() -> None:
    source = _source()

    assert 'id="guardian-preview"' in source
    assert 'id="guardian-edit-btn"' in source
    assert 'id="guardian-save-btn"' in source
    assert 'id="guardian-cancel-btn"' in source
    assert "function openGuardianEdit()" in source
    assert "function cancelGuardianEdit(" in source
    assert "state.guardianEditing" in source
