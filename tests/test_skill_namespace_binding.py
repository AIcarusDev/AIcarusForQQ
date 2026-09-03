from __future__ import annotations

import app_state
import llm.prompt.user_prompt_builder as prompt_builder
import skills.registry as skill_registry
from skills import build_skill_block_for_namespaces, load_skill_resource
from tools.core import recall_skill_resource
from tools.namespaces import (
    NamespaceRegistry,
    NamespaceRuntimeState,
    NamespaceSpec,
    load_namespace_registry,
)


def test_namespace_registry_records_bound_skill():
    registry = load_namespace_registry()

    assert registry.get("qq_social").skill == "qq-social-style"
    assert registry.get("core_chat").skill == "core-chat"
    assert registry.get("core").skill == ""


def test_skill_body_strips_file_metadata(monkeypatch, tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(
        "---\n"
        "name: qq-social-style\n"
        "description: Editable test skill.\n"
        "---\n"
        "# User heading\n\n"
        "Editable body.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        skill_registry,
        "ensure_skill_user_file",
        lambda skill_id: skill_file if skill_id == "qq-social-style" else None,
    )

    body = skill_registry.load_skill_body.__wrapped__("qq-social-style")

    assert body == "## User heading\n\nEditable body."
    assert "name: qq-social-style" not in body


def test_core_chat_skill_loads_user_editable_body_without_metadata(monkeypatch, tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(
        "---\n"
        "name: core-chat\n"
        "---\n"
        "Core chat test body.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        skill_registry,
        "ensure_skill_user_file",
        lambda skill_id: skill_file if skill_id == "core-chat" else None,
    )

    body = skill_registry.load_skill_body.__wrapped__("core-chat")

    assert body == "Core chat test body."
    assert "name: core-chat" not in body


def test_skill_resource_loader_reads_reference_file(monkeypatch, tmp_path):
    references = tmp_path / "sample-skill" / "references"
    references.mkdir(parents=True)
    (references / "sample.md").write_text("Synthetic resource body.\n", encoding="utf-8")
    monkeypatch.setattr(skill_registry, "_SKILLS_DIR", tmp_path)

    result = load_skill_resource("sample-skill", "sample")

    assert result["ok"] is True
    assert result["skill"] == "sample-skill"
    assert result["resource"] == "sample"
    assert result["path"] == "references/sample.md"
    assert result["truncated"] is False
    assert result["content"] == "Synthetic resource body."


def test_skill_resource_loader_rejects_unsafe_resource_id():
    result = load_skill_resource("qq-social-style", "../SKILL")

    assert result["skill"] == "qq-social-style"
    assert result["resource"] == "../SKILL"
    assert result["ok"] is False
    assert result["error"]


def test_recall_skill_resource_tool_requires_active_skill(monkeypatch):
    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    monkeypatch.setattr(app_state, "namespace_runtime_state", state)

    blocked = recall_skill_resource.execute("qq-social-style", "test")
    assert blocked["ok"] is False
    assert blocked["error"]
    assert blocked["skill"] == "qq-social-style"
    assert blocked["resource"] == "test"

    calls = []
    monkeypatch.setattr(
        recall_skill_resource,
        "load_skill_resource",
        lambda skill, resource: calls.append((skill, resource)) or {
            "ok": True,
            "skill": skill,
            "resource": resource,
        },
    )

    state.open("qq_social", registry, 1)
    result = recall_skill_resource.execute("qq-social-style", "test")
    assert result["ok"] is True
    assert calls == [("qq-social-style", "test")]


def test_skill_block_follows_active_namespace_lifecycle(monkeypatch):
    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    loaded = []
    monkeypatch.setattr(
        skill_registry,
        "load_skill_body",
        lambda skill_id: loaded.append(skill_id) or "synthetic body",
    )

    build_skill_block_for_namespaces(state.active_namespaces(registry), registry)
    assert loaded == []

    state.open("qq_social", registry, 1)
    build_skill_block_for_namespaces(state.active_namespaces(registry), registry)
    assert loaded == ["qq-social-style"]

    state.close("qq_social", registry)
    build_skill_block_for_namespaces(state.active_namespaces(registry), registry)
    assert loaded == ["qq-social-style"]


def test_core_chat_skill_block_follows_namespace_lifecycle(monkeypatch):
    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    loaded = []
    monkeypatch.setattr(
        skill_registry,
        "load_skill_body",
        lambda skill_id: loaded.append(skill_id) or "synthetic body",
    )

    state.open("core_chat", registry, 1)
    build_skill_block_for_namespaces(state.active_namespaces(registry), registry)

    assert loaded == ["core-chat"]


def test_skill_block_renders_multiple_unique_skills(monkeypatch):
    registry = NamespaceRegistry(
        namespaces={
            "alpha": NamespaceSpec(name="alpha", skill="skill-a"),
            "beta": NamespaceSpec(name="beta", skill="skill-b"),
            "gamma": NamespaceSpec(name="gamma", skill="skill-a"),
        },
        order=("alpha", "beta", "gamma"),
        tool_to_namespace={},
    )
    loaded = []
    monkeypatch.setattr(
        skill_registry,
        "load_skill_body",
        lambda skill_id: loaded.append(skill_id) or "synthetic body",
    )

    skill_block = build_skill_block_for_namespaces(("alpha", "beta", "gamma"), registry)

    assert loaded == ["skill-a", "skill-b"]
    assert '<skill name="skill-a" from="namespace.alpha">' in skill_block
    assert '<skill name="skill-b" from="namespace.beta">' in skill_block
    assert "namespace.gamma" not in skill_block


def test_active_skill_helper_delegates_using_namespace_state(monkeypatch):
    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    monkeypatch.setattr(app_state, "namespace_runtime_state", state)
    calls = []
    monkeypatch.setattr(
        prompt_builder,
        "build_skill_block_for_namespaces",
        lambda active, bound_registry: calls.append((tuple(active), bound_registry)),
    )

    prompt_builder._build_active_skill_prompt_block()
    assert calls == [(tuple(state.active_namespaces(registry)), registry)]

    state.open("qq_social", registry, 1)
    prompt_builder._build_active_skill_prompt_block()
    assert calls[-1] == (tuple(state.active_namespaces(registry)), registry)
