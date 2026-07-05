from __future__ import annotations

from llm.prompt.user_prompt_builder import _wrap_chat_log_with_world
from llm.session import create_session, init_session_globals


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
    assert world.index("<des>") < world.index("<unread_info/>")
    assert "<qq>" not in world
    assert "</qq>" not in world
    assert "</platform>" in world
