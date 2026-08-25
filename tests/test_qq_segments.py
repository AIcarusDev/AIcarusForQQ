from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from platforms.qq.adapter import segments as segments_mod
from platforms.qq.adapter.segments import (
    ImageLoadError,
    build_content_segments,
    get_reply_message_id,
    llm_segments_to_qq_adapter,
    qq_adapter_segments_to_text,
)


def test_qq_adapter_segments_to_text_uses_safe_human_labels():
    message = [
        {"type": "text", "data": {"text": "hello "}},
        {"type": "at", "data": {"qq": "bot"}},
        {"type": "face", "data": {"id": "14"}},
        {"type": "image", "data": {}},
        {"type": "file", "data": {"name": "notes.txt"}},
        {"type": "unknown", "data": {}},
    ]

    text = qq_adapter_segments_to_text(message, bot_id="bot", bot_display_name="AICQ")

    assert "hello" in text
    assert "@AICQ" in text
    assert "notes.txt" in text
    assert "[unknown]" in text


def test_build_content_segments_keeps_structured_cards_and_media_refs():
    message = [
        {"type": "text", "data": {"content": "payload"}},
        {"type": "at", "data": {"qq": "u_alice", "name": "Alice"}},
        {"type": "image", "data": {"subType": 1}},
        {"type": "record", "data": {"duration": "2.5"}},
        {
            "type": "json",
            "data": {"data": '{"app":"music","title":"Track","desc":"Demo"}'},
        },
    ]

    parts = build_content_segments(message)

    assert parts[0] == {"type": "text", "text": "payload"}
    assert parts[1] == {"type": "mention", "uid": "u_alice", "display": "@Alice"}
    assert parts[2]["type"] == "sticker"
    assert "image_ref" in parts[2]
    assert "ref" not in parts[2]
    assert parts[3] == {"type": "voice", "label": "voice", "duration": 2.5} or parts[3] == {
        "type": "voice",
        "label": "\u8bed\u97f3",
        "duration": 2.5,
    }
    assert parts[4]["type"] == "card"
    assert parts[4]["kind"] == "music"
    assert parts[4]["title"] == "Track"


def test_get_reply_message_id_reads_reply_segment():
    assert get_reply_message_id([{"type": "reply", "data": {"id": "msg-1"}}]) == "msg-1"
    assert get_reply_message_id([{"type": "text", "data": {"text": "x"}}]) is None


def test_llm_segments_to_qq_adapter_inserts_reply_and_at_spacing():
    result = llm_segments_to_qq_adapter(
        [
            {"command": "at", "user_id": "u_alice"},
            {"command": "text", "content": "hello"},
        ],
        reply_message_id="msg-1",
    )

    assert result[0] == {"type": "reply", "data": {"id": "msg-1"}}
    assert result[1] == {"type": "at", "data": {"qq": "u_alice"}}
    assert result[2] == {"type": "text", "data": {"text": " "}}
    assert result[3] == {"type": "text", "data": {"text": "hello"}}


def test_llm_segments_to_qq_adapter_requires_image_ref():
    with pytest.raises(ImageLoadError):
        llm_segments_to_qq_adapter([{"command": "image"}])


def test_llm_segments_to_qq_adapter_loads_browser_image_by_ref(monkeypatch):
    monkeypatch.setattr(segments_mod, "_load_browser_image_as_base64", lambda ref: f"base64://{ref}")

    result = llm_segments_to_qq_adapter([{"command": "image", "image_ref": "img_ref"}])

    assert result == [{"type": "image", "data": {"file": "base64://img_ref"}}]


def test_send_adapter_rejects_non_artifact_browser_refs(monkeypatch):
    import browser

    monkeypatch.setattr(browser, "read_sendable_browser_image_file", lambda _ref: None)

    with pytest.raises(ImageLoadError, match="browser image_ref not found"):
        llm_segments_to_qq_adapter([
            {"command": "image", "image_ref": "viewport-screenshot-ref"},
        ])


def test_send_adapter_accepts_validated_immutable_artifact(tmp_path, monkeypatch):
    import browser.session as browser_session
    from browser.image_resources import (
        BrowserImageArtifactStore,
        BrowserImageResourceRegistry,
    )

    output = io.BytesIO()
    Image.new("RGB", (32, 24), (1, 2, 3)).save(output, format="PNG")
    original = output.getvalue()
    registry = BrowserImageResourceRegistry()
    resource = registry.register(
        source_url="https://cdn.example/original.png",
        page_url="https://example/",
        identity="main:0",
        alt="original",
        rect={"x": 0, "y": 0, "width": 32, "height": 24},
        natural_size=(32, 24),
    )
    assert resource is not None
    store = BrowserImageArtifactStore(tmp_path)
    artifact = store.persist(
        original,
        resource=resource,
        strategy="response_body",
        declared_mime="image/png",
    )
    monkeypatch.setattr(browser_session, "_BROWSER_IMAGE_ARTIFACT_STORE", store)

    result = llm_segments_to_qq_adapter([
        {"command": "image", "image_ref": artifact.image_ref},
    ])

    encoded = result[0]["data"]["file"].removeprefix("base64://")
    assert base64.b64decode(encoded) == original

