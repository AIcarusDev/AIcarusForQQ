from __future__ import annotations

import asyncio
import sqlite3
from string import Formatter

import pytest

from llm.prompt.user_prompt_builder import _wrap_platform_block_with_world
from llm.prompt.prompt import SYSTEM_PROMPT
from llm.session import create_session
from platforms import AttentionEvent, PlatformRegistry, PlatformWorldBlock
from platforms.core.prompt import render_dialogue
from platforms.qq import QQRuntime


def test_system_prompt_guardian_placeholder_contract():
    fields = {
        field_name
        for _, field_name, _, _ in Formatter().parse(SYSTEM_PROMPT)
        if field_name is not None
    }

    assert "guardian_info" in fields
    assert "guardian" not in fields


@pytest.fixture(autouse=True)
def qq_platform_runtime():
    import app_state

    previous = getattr(app_state, "platform_registry", None)
    app_state.platform_registry = PlatformRegistry()
    app_state.platform_registry.register(QQRuntime({}))
    yield
    app_state.platform_registry = previous


def test_world_wraps_platform_block_with_account_attrs():
    world = _wrap_platform_block_with_world(
        PlatformWorldBlock(
            name="qq",
            attrs={"account_id": '123"45', "account_name": "A&B"},
            content="<des>QQ platform home view</des>\n<unread_info/>\n<current_session/>",
        ),
        "2026年 夏天，7月1日，上午10点0分",
    )

    assert world.startswith("<world>\n<current_time>")
    assert "<attention_events/>" in world
    assert world.index("<attention_events/>") < world.index('<platform name="qq"')
    assert (
        '<platform name="qq" account_id="123&quot;45" account_name="A&amp;B">' in world
    )
    assert "<des>" in world
    assert "QQ platform home view" in world
    assert world.index("<des>") < world.index("<unread_info/>")
    assert "<qq>" not in world
    assert "</qq>" not in world
    assert "</platform>" in world


def test_world_attention_events_filter_current_platform_and_deduplicate():
    import app_state

    class FakeAttentionRuntime:
        platform = "fake"

        def attention_events(self, *, now=None):
            return [
                AttentionEvent(name="qq", age="1s", level="mention"),
                AttentionEvent(name="core", age="14m", level="normal"),
                AttentionEvent(name="core", age="2s", level="mention"),
            ]

    app_state.platform_registry.register(FakeAttentionRuntime())

    world = _wrap_platform_block_with_world(
        PlatformWorldBlock(
            name="qq",
            attrs={},
            content="<des>QQ platform home view</des>\n<current_session/>",
        ),
        "2026年 夏天，7月1日，上午10点0分",
    )

    assert "<attention_events>" in world
    assert '<event type="platform" name="qq"' not in world
    assert world.count('name="core"') == 1
    assert '<event type="platform" name="core" age="2s" level="mention"/>' in world
    assert world.index("<attention_events>") < world.index('<platform name="qq"')


def test_world_allows_self_closing_platform_when_no_page_is_open():
    world = _wrap_platform_block_with_world(
        PlatformWorldBlock(name="", attrs={"page": "none"}, content=None),
        "2026年 夏天，7月1日，上午10点0分",
    )

    assert "<attention_events/>" in world
    assert '<platform page="none"/>' in world
    assert "</platform>" not in world


def test_qq_runtime_world_block_reports_only_account_attrs():
    class FakeClient:
        connected = True
        bot_id = '123"45'
        bot_name = "LocalSelfName"

    session = create_session()
    runtime = QQRuntime({}, client=FakeClient())
    runtime.update_account('123"45', "A&B")

    block = runtime.world_block(
        session,
        current_time="ignored by qq content",
        chat_log="<current_session/>",
    )

    assert block.name == "qq"
    assert block.attrs == {"account_id": '123"45', "account_name": "A&B"}
    assert "<platform" not in block.content
    assert "<current_time>" not in block.content


def test_qq_runtime_account_name_does_not_fallback_to_local_self_name():
    class FakeClient:
        connected = True
        bot_id = "123"
        bot_name = "Icarus"

    runtime = QQRuntime({}, client=FakeClient())

    assert runtime.account.account_id == "123"
    assert runtime.account.account_name == ""


def test_core_platform_history_dialogue_loads_db_window(monkeypatch, tmp_path):
    import database

    db_path = tmp_path / "AICQ.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_path))
    asyncio.run(database.init_db())

    async def seed_messages():
        for index in range(1, 5):
            await database.save_chat_message(
                "core:private:guardian",
                {
                    "role": "user",
                    "message_id": f"coremsg_{index}",
                    "sender_id": "guardian",
                    "sender_name": "监护人",
                    "timestamp": f"2026-07-06T22:1{index}:00+08:00",
                    "content": f"history message {index}",
                    "content_type": "text",
                    "content_segments": [
                        {"type": "text", "text": f"history message {index}"}
                    ],
                },
            )

    asyncio.run(seed_messages())
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM chat_messages WHERE session_key=? AND message_id=?",
            ("core:private:guardian", "coremsg_2"),
        ).fetchone()
    assert row is not None

    session = create_session("core:private:guardian")
    session.context_messages = [
        {
            "role": "user",
            "message_id": "live_only",
            "timestamp": "2026-07-06T23:00:00+08:00",
            "content": "latest live message",
            "content_type": "text",
        }
    ]
    session.chat_window_view = {
        "mode": "history",
        "top_db_id": int(row[0]),
        "page_size": 2,
    }

    dialogue = render_dialogue(session)

    assert '<dialogue mode="history" has_previous="true">' in dialogue
    assert (
        '<guardian id="coremsg_2" time="2026-07-06T22:12:00+08:00">history message 2</guardian>'
        in dialogue
    )
    assert (
        '<guardian id="coremsg_3" time="2026-07-06T22:13:00+08:00">history message 3</guardian>'
        in dialogue
    )
    assert "latest live message" not in dialogue
