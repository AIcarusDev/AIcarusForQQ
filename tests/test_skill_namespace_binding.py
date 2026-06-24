from __future__ import annotations

import app_state
from llm.core.tool_calling.xml_protocol import build_tools_xml_message
from llm.prompt.user_prompt_builder import _build_active_skill_prompt_block
from skills import build_skill_block_for_namespaces, load_skill_body
from tools.namespaces import NamespaceRuntimeState, load_namespace_registry


def test_namespace_registry_records_bound_skill():
    registry = load_namespace_registry()

    assert registry.get("qq_social").skill == "qq-social-style"
    assert registry.get("core").skill == ""


def test_skill_body_strips_file_metadata():
    body = load_skill_body("qq-social-style")

    assert body.startswith("## 风格")
    assert "name: qq-social-style" not in body
    assert "<skill>" not in body


def test_skill_block_follows_active_namespace_lifecycle():
    registry = load_namespace_registry()
    state = NamespaceRuntimeState()

    assert build_skill_block_for_namespaces(state.active_namespaces(registry), registry) == ""

    state.open("qq_social", registry, 1)
    block = build_skill_block_for_namespaces(state.active_namespaces(registry), registry)
    assert block.startswith("<skill>\n## 风格")
    assert block.endswith("</skill>")
    assert "resource_catalog" not in block
    assert "按需回忆技巧" not in block

    state.close("qq_social", registry)
    assert build_skill_block_for_namespaces(state.active_namespaces(registry), registry) == ""


def test_prompt_helper_renders_skill_only_when_namespace_active(monkeypatch):
    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    monkeypatch.setattr(app_state, "namespace_runtime_state", state)

    assert _build_active_skill_prompt_block() == ""

    state.open("qq_social", registry, 1)
    assert _build_active_skill_prompt_block().startswith("<skill>\n## 风格")


def test_namespace_model_view_does_not_expose_skill_metadata():
    xml = build_tools_xml_message(
        [],
        namespace_blocks=[
            {
                "name": "qq_social",
                "description": "QQ 社交消息发送。",
                "active": False,
            },
        ],
    )

    assert '<namespace name="qq_social" description="QQ 社交消息发送。" active="false"/>' in xml
    assert "qq-social-style" not in xml
    assert "skill" not in xml
