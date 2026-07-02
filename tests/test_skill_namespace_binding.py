from __future__ import annotations

import app_state
import skills.registry as skill_registry
from llm.core.tool_calling.aic_action import build_aic_action_message
from llm.prompt.user_prompt_builder import _build_active_skill_prompt_block
from skills import build_skill_block_for_namespaces, load_skill_body, load_skill_resource
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
    assert registry.get("core").skill == ""


def test_skill_body_strips_file_metadata():
    body = load_skill_body("qq-social-style")

    assert body.startswith("## 风格")
    assert "name: qq-social-style" not in body
    assert "<skill>" not in body


def test_skill_resource_loader_reads_reference_file():
    result = load_skill_resource("qq-social-style", "test")

    assert result["ok"] is True
    assert result["skill"] == "qq-social-style"
    assert result["resource"] == "test"
    assert result["path"] == "references/test.md"
    assert result["truncated"] is False
    assert "This is a test resource" in result["content"]


def test_skill_resource_loader_rejects_unsafe_resource_id():
    result = load_skill_resource("qq-social-style", "../SKILL")

    assert result == {
        "skill": "qq-social-style",
        "resource": "../SKILL",
        "ok": False,
        "error": "invalid resource id",
    }


def test_recall_skill_resource_tool_requires_active_skill(monkeypatch):
    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    monkeypatch.setattr(app_state, "namespace_runtime_state", state)

    blocked = recall_skill_resource.execute("qq-social-style", "test")
    assert blocked == {
        "ok": False,
        "error": "skill is not active; open the related namespace first",
        "skill": "qq-social-style",
        "resource": "test",
    }

    state.open("qq_social", registry, 1)
    result = recall_skill_resource.execute("qq-social-style", "test")
    assert result["ok"] is True
    assert result["path"] == "references/test.md"
    assert "这是一个测试用资源" in result["content"]


def test_skill_block_follows_active_namespace_lifecycle():
    registry = load_namespace_registry()
    state = NamespaceRuntimeState()

    assert build_skill_block_for_namespaces(state.active_namespaces(registry), registry) == ""

    state.open("qq_social", registry, 1)
    block = build_skill_block_for_namespaces(state.active_namespaces(registry), registry)
    assert block.startswith('<skills>\n<skill name="qq-social-style">\n## 风格')
    assert block.endswith("</skill>\n</skills>")
    assert "resource_catalog" not in block
    assert "按需回忆技巧" not in block

    state.close("qq_social", registry)
    assert build_skill_block_for_namespaces(state.active_namespaces(registry), registry) == ""


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
    monkeypatch.setattr(
        skill_registry,
        "load_skill_body",
        lambda skill_id: {
            "skill-a": "Alpha body",
            "skill-b": "Beta body",
        }.get(skill_id, ""),
    )

    block = build_skill_block_for_namespaces(("alpha", "beta", "gamma"), registry)

    assert block == (
        "<skills>\n"
        '<skill name="skill-a">\n'
        "Alpha body\n"
        "</skill>\n"
        '<skill name="skill-b">\n'
        "Beta body\n"
        "</skill>\n"
        "</skills>"
    )


def test_prompt_helper_renders_skill_only_when_namespace_active(monkeypatch):
    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    monkeypatch.setattr(app_state, "namespace_runtime_state", state)

    assert _build_active_skill_prompt_block() == ""

    state.open("qq_social", registry, 1)
    assert _build_active_skill_prompt_block().startswith(
        '<skills>\n<skill name="qq-social-style">\n## 风格'
    )


def test_namespace_model_view_does_not_expose_skill_metadata():
    action_message = build_aic_action_message(
        [],
        namespace_blocks=[
            {
                "name": "qq_social",
                "description": "QQ 社交消息发送。",
                "active": False,
            },
        ],
    )

    assert '<namespace name="qq_social" description="QQ 社交消息发送。" active="false"/>' in action_message
    assert "qq-social-style" not in action_message
    assert "skill" not in action_message
