from __future__ import annotations

from datetime import datetime, timedelta, timezone

from platforms.qq.unread import (
    _render_preview_text,
    build_recent_active_sessions_xml,
    build_unread_info_xml,
)
from llm.session import ConversationSession


def _session(
    *,
    conv_type: str,
    conv_id: str,
    conv_name: str = "",
    unread_count: int = 1,
    context_messages: list[dict] | None = None,
) -> ConversationSession:
    session = ConversationSession()
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

    xml = build_unread_info_xml({"qq:group:group_test_001": group}, "qq:temp:test_001")

    assert '<session type="group" id="group_test_001" name="Sandbox Group" unread="13">' in xml
    assert 'sender="Test Sender">这能对吗！[图片]</preview>' in xml
    assert "type=" not in xml.split("<preview", 1)[1].split(">", 1)[0]
    assert "\x00" not in xml
    assert "abcdef123456" not in xml


def test_unread_preview_uses_safe_metadata_without_media_refs():
    text = _render_preview_text(
        {
            "content_type": "text",
            "content_segments": [
                {"type": "sticker", "image_ref": "abc123abc123"},
                {"type": "file", "filename": "notes.txt"},
                {"type": "card", "label": "小程序卡片"},
                {"type": "poke", "label": "戳一戳"},
            ],
        },
        max_len=200,
    )

    assert text
    assert "notes.txt" in text
    assert "小程序卡片" in text
    assert "戳一戳" in text
    assert "abc123abc123" not in text
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

    xml = build_unread_info_xml({"qq:group:group_test_001": group}, "qq:temp:test_001")

    assert '<session type="group" id="group_test_001" unread="1">' in xml
    assert 'name="group_test_001"' not in xml


def test_recent_active_sessions_render_closed_session_tags_without_preview():
    newer = _session(
        conv_type="group",
        conv_id="456",
        conv_name="另一个群",
        unread_count=0,
        context_messages=[
            {
                "role": "bot",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content": "ok",
                "content_type": "text",
            }
        ],
    )
    older = _session(
        conv_type="private",
        conv_id="10001",
        conv_name="Bob",
        unread_count=0,
        context_messages=[
            {
                "role": "user",
                "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=23)).isoformat(),
                "content": "seen but no reply",
                "content_type": "text",
            }
        ],
    )

    xml = build_recent_active_sessions_xml(
        {
            "qq:private:10001": older,
            "qq:group:456": newer,
        },
        "",
    )

    assert "<recent_active_sessions>" in xml
    assert '<session type="group" id="456" name="另一个群" last_active=' in xml
    assert '<session type="private" id="10001" nickname="Bob" last_active=' in xml
    assert "<preview" not in xml
    assert xml.index('id="456"') < xml.index('id="10001"')


def test_recent_active_sessions_limit_to_five_newest():
    base_time = datetime(2026, 7, 5, 10, tzinfo=timezone.utc)
    session_map = {
        f"qq:group:{i}": _session(
            conv_type="group",
            conv_id=str(i),
            conv_name=f"群{i}",
            unread_count=0,
            context_messages=[
                {
                    "role": "user",
                    "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
                    "content": f"seen {i}",
                    "content_type": "text",
                }
            ],
        )
        for i in range(7)
    }

    xml = build_recent_active_sessions_xml(session_map, "")

    assert xml.count("<session ") == 5
    assert xml.index('id="6"') < xml.index('id="5"') < xml.index('id="4"')
    assert 'id="1"' not in xml
    assert 'id="0"' not in xml


def test_recent_active_sessions_skip_unread_sessions():
    unread = _session(
        conv_type="group",
        conv_id="42",
        conv_name="测试群",
        unread_count=1,
        context_messages=[
            {
                "role": "user",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content": "hello",
                "content_type": "text",
            }
        ],
    )
    read = _session(
        conv_type="group",
        conv_id="456",
        conv_name="另一个群",
        unread_count=0,
        context_messages=[
            {
                "role": "user",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content": "already seen",
                "content_type": "text",
            }
        ],
    )

    xml = build_recent_active_sessions_xml(
        {
            "qq:group:42": unread,
            "qq:group:456": read,
        },
        "",
    )

    assert 'id="456"' in xml
    assert 'id="42"' not in xml


def test_unread_info_limits_to_five_newest_and_backfills_after_processed():
    base_time = datetime(2026, 7, 5, 10, tzinfo=timezone.utc)
    session_map = {
        f"qq:group:{i}": _session(
            conv_type="group",
            conv_id=str(i),
            conv_name=f"群{i}",
            unread_count=1,
            context_messages=[
                {
                    "role": "user",
                    "sender_name": f"用户{i}",
                    "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
                    "content": f"unread {i}",
                    "content_type": "text",
                }
            ],
        )
        for i in range(7)
    }

    xml = build_unread_info_xml(session_map, "")

    assert xml.count("<session ") == 5
    assert xml.index('id="6"') < xml.index('id="5"') < xml.index('id="4"')
    assert 'id="1"' not in xml
    assert 'id="0"' not in xml

    session_map["qq:group:6"].clear_unread_messages()
    xml = build_unread_info_xml(session_map, "")

    assert xml.count("<session ") == 5
    assert 'id="6"' not in xml
    assert 'id="1"' in xml
    assert 'id="0"' not in xml
    assert xml.index('id="5"') < xml.index('id="4"') < xml.index('id="3"')
