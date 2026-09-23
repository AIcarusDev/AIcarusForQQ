from __future__ import annotations

import base64
import asyncio
import io

import pytest
from PIL import Image

from platforms.qq.adapter import segments as segments_mod
from platforms.qq.adapter import events as qq_events
from platforms.qq.adapter.segments import (
    ImageLoadError,
    build_content_segments,
    get_reply_message_id,
    llm_segments_to_qq_adapter,
    qq_adapter_segments_to_text,
)
from platforms.chat.xml_builder import _render_content_xml
from platforms.qq.adapter.segments import _determine_content_type
from platforms.qq.unread import _render_preview_text


def test_qq_adapter_segments_to_text_preserves_fixture_content():
    message = [
        {"type": "text", "data": {"text": "hello "}},
        {"type": "at", "data": {"qq": "bot"}},
        {"type": "file", "data": {"name": "notes.txt"}},
    ]

    text = qq_adapter_segments_to_text(message, bot_id="bot", bot_display_name="AICQ")

    assert "hello" in text
    assert "@AICQ" in text
    assert "notes.txt" in text


def test_native_face_keeps_own_type_in_chat_history_and_legacy_emoji_renders_as_face():
    message = [
        {"type": "text", "data": {"text": "你好"}},
        {"type": "face", "data": {"id": "364", "raw": {"faceText": "/超级赞"}}},
        {"type": "text", "data": {"text": "！"}},
    ]
    parts = build_content_segments(message)
    assert parts == [
        {"type": "text", "text": "你好"},
        {"type": "face", "id": "364", "des": "/超级赞"},
        {"type": "text", "text": "！"},
    ]
    assert qq_adapter_segments_to_text(message) == "你好[超级赞]！"
    assert _render_content_xml({"content_segments": parts}) == (
        '    <content type="text">你好</content>'
        '<content type="face" id="364">/超级赞</content>'
        '<content type="text">！</content>'
    )
    assert _render_preview_text({"content_segments": parts}) == "你好[超级赞]！"
    assert _determine_content_type([message[1]]) == "face"

    legacy = {"content_segments": [{"type": "emoji", "id": "5", "name": "流泪"}]}
    assert _render_content_xml(legacy) == '    <content type="face" id="5">/流泪</content>'
    assert _render_preview_text(legacy) == "[流泪]"


def test_native_face_without_description_still_keeps_id():
    parts = build_content_segments([{"type": "face", "data": {"id": 469}}])
    assert parts == [{"type": "face", "id": "469"}]
    assert _render_content_xml({"content_segments": parts}) == (
        '    <content type="face" id="469">[表情:469]</content>'
    )

    entry = asyncio.run(qq_events.qq_adapter_event_to_context({
        "post_type": "message",
        "message_type": "private",
        "message_id": "face-469",
        "time": 1760000,
        "sender": {"user_id": "sender", "nickname": "Sender"},
        "message": [{"type": "face", "data": {"id": 469}}],
    }))
    assert entry is not None
    assert entry["content_type"] == "face"
    assert entry["content_segments"] == parts
    assert _render_content_xml(entry) == (
        '    <content type="face" id="469">[表情:469]</content>'
    )


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


def test_file_segment_keeps_pre_download_metadata_without_marking_it_downloaded():
    parts = build_content_segments([
        {
            "type": "file",
            "data": {
                "file": "notes.txt",
                "file_id": "remote-file-1",
                "file_size": "2048",
                "busid": 102,
                "url": "https://example.invalid/should-not-be-persisted",
            },
        }
    ])

    assert parts == [{
        "type": "file",
        "filename": "notes.txt",
        "size_bytes": 2048,
        "is_downloaded": False,
        "file_id": "remote-file-1",
        "busid": 102,
    }]
    assert "url" not in parts[0]


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
    monkeypatch.setattr(segments_mod, "_load_image_as_base64", lambda ref: f"base64://{ref}")

    result = llm_segments_to_qq_adapter([{"command": "image", "image_ref": "img_ref"}])

    assert result == [{"type": "image", "data": {"file": "base64://img_ref"}}]


def test_llm_segments_to_qq_adapter_sends_materialized_local_image():
    result = llm_segments_to_qq_adapter([{
        "command": "image",
        "_local_image_base64": "cG5nLWJ5dGVz",
        "_local_image_ref": "img_0123456789abcdef0123456789abcdef",
    }])

    assert result == [{
        "type": "image",
        "data": {"file": "base64://cG5nLWJ5dGVz"},
    }]


def test_send_adapter_rejects_non_artifact_browser_refs(monkeypatch):
    import browser

    monkeypatch.setattr(browser, "read_sendable_browser_image_file", lambda _ref: None)

    with pytest.raises(ImageLoadError):
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


def test_video_segment_keeps_metadata_and_video_ref():
    from platforms.chat.xml_builder import _render_content_chunks, _render_content_xml

    message = [
        {
            "type": "video",
            "data": {
                "url": "https://example.com/demo.mp4",
                "file_size": 1048576,
                "file_id": "vid-12345",
                "file_name": "demo.mp4",
                "duration": "15.0",
            },
        },
    ]

    parts = build_content_segments(message)
    assert len(parts) == 1
    seg = parts[0]
    assert seg["type"] == "video"
    assert "video_ref" in seg
    assert seg["video_ref"].count("_") == 1
    assert seg["url"] == "https://example.com/demo.mp4"
    assert seg["file_size"] == 1048576
    assert seg["file_id"] == "vid-12345"
    assert seg["file_name"] == "demo.mp4"
    assert seg["duration"] == 15.0

    chunks = _render_content_chunks(parts)
    assert len(chunks) == 1
    ct, text_repr, attrs = chunks[0]
    assert ct == "video"
    assert f'video_ref="{seg["video_ref"]}"' in text_repr
    assert "15" in text_repr
    assert 'size="1MB"' in attrs

    xml = _render_content_xml({"content_segments": parts})
    assert '<content type="video"' in xml
    assert f'video_ref="{seg["video_ref"]}"' in xml


