from __future__ import annotations

import asyncio
import json

from llm.prompt.xml_builder import build_chat_log_xml
from platforms.qq.adapter import events as qq_events
from platforms.qq.adapter.segments import build_content_segments


def _group_meta() -> dict:
    return {
        "type": "group",
        "id": "prompt_xml_test_group",
        "name": "Prompt XML Test Group",
    }


def test_miniapp_card_omits_raw_payload_from_segments_and_prompt():
    payload = {
        "ver": "1.0.0.19",
        "prompt": "[QQ小程序]〖中文字幕〗【初音ミク】疯帽匠与永不完结的茶会【ぱぺっつ】",
        "config": {
            "token": "synthetic-card-token",
            "forward": 1,
        },
        "app": "com.tencent.miniapp_01",
        "meta": {
            "detail_1": {
                "title": "哔哩哔哩",
                "desc": "〖中文字幕〗【初音ミク】疯帽匠与永不完结的茶会【ぱぺっつ】",
                "preview": "https://qq.ugcimg.cn/v1/base64-trash",
                "url": "m.q.qq.com/a/s/synthetic-card",
                "shareTemplateId": "SYNTHETIC_TEMPLATE",
            }
        },
    }
    segments = build_content_segments([
        {"type": "json", "data": {"data": json.dumps(payload, ensure_ascii=False)}}
    ])

    assert segments[0]["type"] == "card"
    assert segments[0]["kind"] == "miniapp"
    assert "raw" not in segments[0]

    xml = build_chat_log_xml(
        [
            {
                "role": "user",
                "message_id": "msg-card",
                "sender_id": "user_card_sender",
                "sender_name": "萱流表bot",
                "sender_card": "萱流表bot",
                "sender_nickname": "暗世界の电流表",
                "timestamp": "2026-06-26T12:00:00+08:00",
                "content": "[卡片消息]",
                "content_type": "text",
                "content_segments": segments,
            }
        ],
        _group_meta(),
    )

    assert '<content type="card" kind="miniapp">' in xml
    assert "<title>哔哩哔哩</title>" in xml
    assert "<summary>〖中文字幕〗【初音ミク】疯帽匠与永不完结的茶会【ぱぺっつ】</summary>" in xml
    assert "<raw" not in xml
    assert "synthetic-card-token" not in xml
    assert "shareTemplateId" not in xml
    assert "base64-trash" not in xml


def test_group_sender_card_and_nickname_are_separate_in_prompt_xml():
    entry = asyncio.run(
        qq_events.qq_adapter_event_to_context(
            {
                "post_type": "message",
                "message_type": "group",
                "message_id": "msg-card",
                "time": 1760000,
                "sender": {
                    "user_id": "user_card_sender",
                    "card": "萱流表bot",
                    "nickname": "暗世界の电流表",
                    "role": "member",
                    "title": "🎶萱萱萱萱",
                    "level": "100",
                },
                "message": [{"type": "text", "data": {"text": "hello"}}],
            }
        )
    )

    assert entry is not None
    assert entry["sender_name"] == "萱流表bot"
    assert entry["sender_card"] == "萱流表bot"
    assert entry["sender_nickname"] == "暗世界の电流表"

    xml = build_chat_log_xml([entry], _group_meta())

    assert 'card="萱流表bot"' in xml
    assert 'nickname="暗世界の电流表"' in xml
    assert 'nickname="萱流表bot"' not in xml


def test_legacy_group_sender_name_renders_as_display_not_fake_nickname():
    xml = build_chat_log_xml(
        [
            {
                "role": "user",
                "message_id": "legacy-1",
                "sender_id": "legacy_user_without_identity_snapshot",
                "sender_name": "旧群名片",
                "timestamp": "2026-06-26T12:00:00+08:00",
                "content": "hello",
                "content_type": "text",
            }
        ],
        _group_meta(),
    )

    assert 'display="旧群名片"' in xml
    assert 'nickname="旧群名片"' not in xml


def test_pending_self_message_omits_internal_id_and_renders_state():
    xml = build_chat_log_xml(
        [
            {
                "role": "bot",
                "message_id": "pending_abc123",
                "sender_id": "bot",
                "sender_name": "Bot",
                "timestamp": "2026-06-26T12:00:00+08:00",
                "content": "重启好了",
                "content_type": "text",
                "content_segments": [{"type": "text", "text": "重启好了"}],
                "delivery_state": "pending",
            }
        ],
        _group_meta(),
    )

    assert "pending_abc123" not in xml
    assert '<message timestamp=' in xml
    assert 'state="pending"' in xml
    assert '<sender id="self"/>' in xml
    assert "<content type=\"text\">重启好了</content>" in xml


def test_failed_self_message_omits_legacy_failed_id_and_renders_state():
    xml = build_chat_log_xml(
        [
            {
                "role": "bot",
                "message_id": "failed_e1f3d2a4",
                "sender_id": "bot",
                "sender_name": "Bot",
                "timestamp": "2026-06-26T12:00:00+08:00",
                "content": "重启好了",
                "content_type": "send_failed",
                "content_segments": [{"type": "text", "text": "重启好了"}],
            }
        ],
        _group_meta(),
    )

    assert "failed_e1f3d2a4" not in xml
    assert 'state="failed"' in xml
    assert "<content type=\"text\">重启好了</content>" in xml


def test_confirmed_self_message_keeps_real_actionable_id():
    xml = build_chat_log_xml(
        [
            {
                "role": "bot",
                "message_id": "-1174946",
                "sender_id": "bot",
                "sender_name": "Bot",
                "timestamp": "2026-06-26T12:00:00+08:00",
                "content": "重启好了",
                "content_type": "text",
                "content_segments": [{"type": "text", "text": "重启好了"}],
            }
        ],
        _group_meta(),
    )

    assert 'id="-1174946"' in xml
    assert "state=" not in xml

