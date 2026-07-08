from __future__ import annotations

from datetime import datetime, timezone

import pytest

from llm.prompt.user_prompt_builder import _wrap_platform_block_with_world, build_main_user_prompt
from llm.session import create_session, init_session_globals, sessions
from platforms import AttentionEvent, PlatformRegistry, PlatformWorldBlock
from platforms.core import CLOSED_PLATFORM_FOCUS, CoreRuntime
from platforms.qq import QQRuntime
from platforms.qq.session_context import HOME_FOCUS


@pytest.fixture(autouse=True)
def qq_platform_runtime():
    import app_state

    previous = getattr(app_state, "platform_registry", None)
    app_state.platform_registry = PlatformRegistry()
    app_state.platform_registry.register(QQRuntime({}))
    yield
    app_state.platform_registry = previous


def _prompt_text(prompt: str | list) -> str:
    if isinstance(prompt, str):
        return prompt
    return "".join(
        str(part.get("text", ""))
        for part in prompt
        if isinstance(part, dict) and part.get("type") == "text"
    )


def test_system_prompt_formats_self_name():
    init_session_globals(
        max_context=10,
        timezone=None,
        persona="persona text",
        self_name="Aicarus",
        model_name="model-x",
    )
    session = create_session()

    prompt = session.build_system_prompt()

    assert "你是 Aicarus" in prompt
    assert "{self_name}" not in prompt


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
    assert '<platform name="qq" account_id="123&quot;45" account_name="A&amp;B">' in world
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


def test_main_user_prompt_allows_qq_platform_without_current_session():
    sessions.clear()
    session = create_session()

    prompt = _prompt_text(build_main_user_prompt(session))

    assert '<platform name="qq" account_id="" account_name="">' in prompt
    assert "<des>" in prompt
    assert "QQ platform home view" in prompt
    assert "`recent_active_sessions` lists recently active sessions" in prompt
    assert "<unread_info/>" in prompt
    assert "<recent_active_sessions/>" in prompt
    assert "<current_session/>" in prompt
    assert "<chat_logs" not in prompt
    assert prompt.index("<des>") < prompt.index("<unread_info/>")
    assert prompt.index("<unread_info/>") < prompt.index("<recent_active_sessions/>")
    assert prompt.index("<recent_active_sessions/>") < prompt.index("<current_session/>")


def test_main_user_prompt_treats_qq_home_focus_as_without_current_session():
    sessions.clear()
    session = create_session(HOME_FOCUS)

    prompt = _prompt_text(build_main_user_prompt(session))

    assert "<recent_active_sessions/>" in prompt
    assert "<current_session/>" in prompt
    assert "<chat_logs" not in prompt


def test_main_user_prompt_keeps_unread_info_on_qq_platform_without_current_session():
    sessions.clear()
    current = create_session()
    unread = create_session()
    unread.set_conversation_meta("group", "42", "测试群")
    unread.unread_count = 1
    unread.context_messages = [
        {
            "role": "user",
            "sender_name": "Alice",
            "timestamp": "2026-07-05T10:00:00+08:00",
            "content": "hello",
            "content_type": "text",
        }
    ]
    sessions[unread.key] = unread

    prompt = _prompt_text(build_main_user_prompt(current))

    assert '<session type="group" id="42" name="测试群" unread="1">' in prompt
    assert "<recent_active_sessions/>" in prompt
    assert "<current_session/>" in prompt
    assert "<chat_logs" not in prompt
    assert prompt.index("<unread_info>") < prompt.index("<current_session/>")


def test_main_user_prompt_includes_recent_active_sessions_only_on_qq_home():
    sessions.clear()
    current = create_session()
    recent = create_session("qq:group:456")
    recent.set_conversation_meta("group", "456", "另一个群")
    recent.context_messages = [
        {
            "role": "bot",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": "刚才聊过",
            "content_type": "text",
        }
    ]
    sessions[recent.key] = recent

    prompt = _prompt_text(build_main_user_prompt(current))

    assert "<recent_active_sessions>" in prompt
    assert '<session type="group" id="456" name="另一个群" last_active=' in prompt
    assert "<preview" not in prompt
    assert prompt.index("<unread_info/>") < prompt.index("<recent_active_sessions>")
    assert prompt.index("</recent_active_sessions>") < prompt.index("<current_session/>")


def test_main_user_prompt_omits_recent_active_sessions_when_current_session_expands():
    sessions.clear()
    current = create_session("qq:group:42")
    current.set_conversation_meta("group", "42", "当前群")
    other = create_session("qq:group:456")
    other.set_conversation_meta("group", "456", "另一个群")
    other.context_messages = [
        {
            "role": "bot",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": "刚才聊过",
            "content_type": "text",
        }
    ]
    sessions[other.key] = other

    prompt = _prompt_text(build_main_user_prompt(current))

    assert "<recent_active_sessions" not in prompt
    assert "currently opened QQ chat window" in prompt
    assert "QQ platform home view" not in prompt
    assert "<current_session" in prompt
    assert "<chat_logs" in prompt


def test_core_platform_prompt_uses_minimal_dialogue_with_empty_des():
    import app_state

    sessions.clear()
    app_state.platform_registry.register(CoreRuntime({}))
    session = create_session("core:private:guardian")
    session.context_messages = [
        {
            "role": "user",
            "message_id": "coremsg_1",
            "timestamp": "2026-07-06T22:18:03+08:00",
            "content": "我们先不管 QQ，聊一下 core 平台。",
            "content_type": "text",
        },
        {
            "role": "bot",
            "message_id": "coremsg_2",
            "timestamp": "2026-07-06T22:18:05+08:00",
            "reply_to": "coremsg_1",
            "content": "我理解，这是一个本地 1v1 私聊入口。",
            "content_type": "text",
        },
    ]

    prompt = _prompt_text(build_main_user_prompt(session))

    assert "<attention_events/>" in prompt
    assert '<platform name="core" transport="webui">' in prompt
    assert "<des></des>" in prompt
    assert '<dialogue mode="current" has_previous="false">' in prompt
    assert (
        '<guardian id="coremsg_1" time="2026-07-06T22:18:03+08:00">'
        "我们先不管 QQ，聊一下 core 平台。"
        "</guardian>"
    ) in prompt
    assert (
        '<self id="coremsg_2" time="2026-07-06T22:18:05+08:00" reply_to="coremsg_1">'
        "我理解，这是一个本地 1v1 私聊入口。"
        "</self>"
    ) in prompt
    assert "<current_session" not in prompt
    assert "<chat_logs" not in prompt


def test_closed_platform_focus_renders_no_platform_page():
    import app_state

    sessions.clear()
    app_state.platform_registry.register(CoreRuntime({}))
    session = create_session(CLOSED_PLATFORM_FOCUS)

    prompt = _prompt_text(build_main_user_prompt(session))

    assert '<platform page="none"/>' in prompt
    assert "<dialogue" not in prompt
    assert "<guardian" not in prompt
