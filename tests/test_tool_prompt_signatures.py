from __future__ import annotations

from pathlib import Path

from tools import build_tools
from tools.namespaces import NamespaceRuntimeState, load_namespace_registry
from tools.prompt_signatures import build_prompt_signature, strip_schema_descriptions
from llm.core.tool_calling.xml_protocol import build_tools_xml_message
import tools as tools_package


def test_build_prompt_signature_preserves_descriptions_as_line_comments():
    declaration = {
        "name": "wait",
        "description": "核心等待工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 15,
                    "description": "等待秒数，范围 1~15。",
                },
                "mode": {
                    "type": "string",
                    "enum": ["quiet", "watch"],
                    "description": "等待模式。",
                },
            },
            "required": ["seconds"],
        },
    }

    signature = build_prompt_signature(declaration)

    assert "// 核心等待工具。" in signature
    assert "wait(args: {" in signature
    assert "seconds: number; // 等待秒数，范围 1~15。" in signature
    assert 'mode?: "quiet" | "watch"; // 等待模式。' in signature


def test_strip_schema_descriptions_keeps_validation_keywords():
    declaration = {
        "name": "wait",
        "description": "核心等待工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 15,
                    "description": "等待秒数。",
                    "x-coerce-integer": True,
                },
                "description": {
                    "type": "string",
                    "description": "业务参数名本身叫 description 时必须保留。",
                },
            },
            "required": ["seconds", "description"],
        },
    }

    stripped = strip_schema_descriptions(declaration)

    assert "核心等待工具" not in repr(stripped)
    assert "等待秒数" not in repr(stripped)
    assert "业务参数名本身叫 description" not in repr(stripped)
    assert "description" in stripped["parameters"]["properties"]
    assert stripped["parameters"]["properties"]["seconds"]["minimum"] == 1
    assert stripped["parameters"]["properties"]["seconds"]["maximum"] == 15
    assert stripped["parameters"]["properties"]["seconds"]["x-coerce-integer"] is True


def test_build_prompt_signature_adds_constraints_when_description_omits_them():
    declaration = {
        "name": "search_history",
        "description": "搜索历史。",
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "搜索关键词列表。",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["keywords"],
        },
    }

    signature = build_prompt_signature(declaration)

    assert "keywords: string[]; // 搜索关键词列表。 至少 1 项" in signature
    assert "limit?: number; // 范围 1~5" in signature


def test_send_message_package_exports_curated_prompt_signature():
    class Session:
        conv_type = "private"
        conv_id = "demo"
        conv_name = "demo"
        context_messages = []

    class QQAdapter:
        connected = True
        _loop = None

    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    state.open("qq_social", registry, 0)

    collection = build_tools(
        {},
        namespace_state=state,
        current_round=0,
        session=Session(),
        qq_adapter_client=QQAdapter(),
    )

    signature = collection.active_specs["send_message"].prompt_signature

    assert '| { command: "text"; content: string }' in signature
    assert '| { command: "image"; image_ref: string }' in signature
    assert 'command: "text" | "at" | "image" | "sticker"' not in signature


def test_send_voice_prompt_signature_tracks_dynamic_tts_workers(monkeypatch):
    import app_state
    from tools.qq.qq_social import send_voice

    class TTS:
        def list_plugins(self):
            return [
                {
                    "plugin_id": "main",
                    "llm_schema": {
                        "description": "普通语音",
                        "properties": {
                            "voice": {"type": "string", "description": "音色名称。"},
                            "speed": {
                                "type": "number",
                                "minimum": 0.5,
                                "maximum": 2,
                                "description": "语速。",
                            },
                        },
                    },
                },
                {
                    "plugin_id": "song",
                    "llm_schema": {
                        "description": "歌声合成",
                        "properties": {
                            "style": {
                                "type": "string",
                                "enum": ["pop", "rock"],
                                "description": "演唱风格。",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "description": "标签。",
                            },
                        },
                    },
                },
            ]

    monkeypatch.setattr(app_state, "tts_server", TTS(), raising=False)

    signature = send_voice.get_prompt_signature()

    assert 'plugin_id: "main" | "song"' in signature
    assert "text?: string;" in signature
    assert "voice?: string; // 音色名称。" in signature
    assert "speed?: number; // 语速。；范围 0.5~2" in signature
    assert 'style?: "pop" | "rock"; // 演唱风格。' in signature
    assert "tags?: string[]; // 标签。；至少 1 项" in signature


def test_all_discovered_first_party_tools_export_handwritten_prompt_signatures():
    missing = []
    for mod in tools_package._tool_modules:
        signature = getattr(mod, "PROMPT_SIGNATURE", None)
        get_signature = getattr(mod, "get_prompt_signature", None)
        if not isinstance(signature, str) and not callable(get_signature):
            missing.append(getattr(mod, "__name__", repr(mod)))

    assert missing == []


def test_all_tool_declaration_files_include_prompt_signature_source():
    missing = []
    for path in Path("src/tools").rglob("*.py"):
        if path.name.startswith("_"):
            continue
        text = path.read_text(encoding="utf-8-sig")
        if "DECLARATION" not in text:
            continue
        if "PROMPT_SIGNATURE" in text or "get_prompt_signature" in text:
            continue
        missing.append(str(path))

    assert missing == []


def test_all_visible_tools_render_prompt_signatures_without_schema_descriptions():
    class Session:
        conv_type = "group"
        conv_id = "demo"
        conv_name = "demo"
        context_messages = []

    class QQAdapter:
        connected = True
        _loop = None

    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    for namespace in registry.order:
        spec = registry.get(namespace)
        if spec and spec.visible and spec.openable and not spec.permanent:
            state.open(namespace, registry, 0)

    collection = build_tools(
        {},
        namespace_state=state,
        current_round=0,
        session=Session(),
        qq_adapter_client=QQAdapter(),
    )
    xml = build_tools_xml_message([], namespace_blocks=collection.namespace_prompt_blocks())

    assert not [spec.name for spec in collection.active_specs.values() if not spec.prompt_signature.strip()]
    assert '"parameters"' not in xml
    assert '"type":"object"' not in xml
    assert not [
        spec.name
        for spec in collection.active_specs.values()
        if _has_schema_description_keyword(spec.declaration)
    ]


def _has_schema_description_keyword(value, *, in_properties: bool = False) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "description" and not in_properties:
                return True
            if _has_schema_description_keyword(child, in_properties=(key == "properties")):
                return True
    if isinstance(value, list):
        return any(_has_schema_description_keyword(item) for item in value)
    return False
