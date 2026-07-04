from __future__ import annotations

from llm.prompt.unread_builder import _render_preview_text, build_unread_info_xml
from llm.session import ChatSession


def _session(
    *,
    conv_type: str,
    conv_id: str,
    conv_name: str = "",
    unread_count: int = 1,
    context_messages: list[dict] | None = None,
) -> ChatSession:
    session = ChatSession()
    session.set_conversation_meta(conv_type, conv_id, conv_name)
    session.unread_count = unread_count
    session.context_messages = context_messages or []
    return session


def test_unread_preview_renders_segments_as_plain_text_without_type_or_sentinels():
    group = _session(
        conv_type="group",
        conv_id="group_test_001",
        conv_name="Sandbox Group",
        unread_count=13,
        context_messages=[
            {
                "role": "user",
                "sender_name": "Test Sender",
                "timestamp": "2026-06-25T08:55:00+08:00",
                "content_type": "text",
                "content_segments": [
                    {"type": "text", "text": "这能对吗！"},
                    {"type": "image", "image_ref": "abcdef123456"},
                ],
            }
        ],
    )

    xml = build_unread_info_xml({"group_group_test_001": group}, "temp_test_001")

    assert '<session type="group" id="group_test_001" name="Sandbox Group" unread="13">' in xml
    assert 'sender="Test Sender">这能对吗！[图片]</preview>' in xml
    assert "type=" not in xml.split("<preview", 1)[1].split(">", 1)[0]
    assert "\x00" not in xml
    assert "abcdef123456" not in xml


def test_unread_preview_downgrades_non_text_segments_to_text_labels():
    text = _render_preview_text(
        {
            "content_type": "text",
            "content_segments": [
                {"type": "voice", "duration": 2.5},
                {"type": "video"},
                {"type": "sticker", "image_ref": "abc123abc123"},
                {"type": "file", "filename": "notes.txt"},
                {"type": "forward"},
                {"type": "card", "label": "小程序卡片"},
                {"type": "poke", "label": "戳一戳"},
            ],
        },
        max_len=200,
    )

    assert text == "[语音][视频][动画表情][文件:notes.txt][合并转发][小程序卡片][戳一戳]"
    assert "\x00" not in text


def test_unread_session_does_not_use_id_as_group_name():
    group = _session(
        conv_type="group",
        conv_id="group_test_001",
        unread_count=1,
        context_messages=[
            {
                "role": "user",
                "sender_name": "Test Sender",
                "timestamp": "2026-06-25T08:55:00+08:00",
                "content": "hello",
                "content_type": "text",
            }
        ],
    )

    xml = build_unread_info_xml({"group_group_test_001": group}, "temp_test_001")

    assert '<session type="group" id="group_test_001" unread="1">' in xml
    assert 'name="group_test_001"' not in xml
