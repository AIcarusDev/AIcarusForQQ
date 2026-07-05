from __future__ import annotations

from datetime import datetime, timezone

from llm.prompt.user_prompt_builder import _wrap_chat_log_with_world, build_main_user_prompt
from llm.session import create_session, init_session_globals, sessions
from platforms.qq.session_context import HOME_FOCUS


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


def test_world_wraps_chat_log_in_platform_with_account_attrs():
    world = _wrap_chat_log_with_world(
        "<current_session/>",
        "<unread_info/>",
        "2026年 夏天，7月1日，上午10点0分",
        platform_name="qq",
        account_id='123"45',
        account_name="A&B",
    )

    assert '<platform name="qq" account_id="123&quot;45" account_name="A&amp;B">' in world
    assert "<des>" in world
    assert "QQ platform home view" in world
    assert "current chat window" not in world
    assert world.index("<des>") < world.index("<unread_info/>")
    assert "<qq>" not in world
    assert "</qq>" not in world
    assert "</platform>" in world


def test_main_user_prompt_allows_qq_platform_without_current_session():
    sessions.clear()
    session = create_session()

    prompt = _prompt_text(build_main_user_prompt(session))

    assert '<platform name="qq"' in prompt
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
