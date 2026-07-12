from __future__ import annotations

import asyncio
from types import SimpleNamespace

from llm.session import ConversationSession, sessions
from platforms.focus import FocusRef
from platforms.qq.adapter import recovery


def test_build_targets_only_includes_numeric_qq_sessions(monkeypatch):
    async def fake_load_chat_sessions():
        return [
            {
                "session_key": "core:private:guardian",
                "focus_platform": "core",
                "conv_type": "private",
                "conv_id": "guardian",
                "conv_name": "Core 聊天页面",
            },
            {
                "session_key": "qq:group:123456",
                "focus_platform": "qq",
                "conv_type": "group",
                "conv_id": "123456",
                "conv_name": "测试群",
            },
            {
                "session_key": "qq:private:not-a-qq-id",
                "focus_platform": "qq",
                "conv_type": "private",
                "conv_id": "not-a-qq-id",
                "conv_name": "脏数据",
            },
        ]

    original_sessions = dict(sessions)
    sessions.clear()
    sessions.update(
        {
            "core:private:guardian": ConversationSession(
                focus=FocusRef("core", "private", "guardian", "Core 聊天页面")
            ),
            "qq:private:654321": ConversationSession(
                focus=FocusRef("qq", "private", "654321", "QQ 好友")
            ),
        }
    )
    monkeypatch.setattr(recovery, "load_chat_sessions", fake_load_chat_sessions)

    try:
        targets = asyncio.run(
            recovery._build_targets(recovery.RecoveryConfig(seed_from_whitelist=False))
        )
    finally:
        sessions.clear()
        sessions.update(original_sessions)

    assert [(target.session_key, target.conv_id) for target in targets] == [
        ("qq:group:123456", "123456"),
        ("qq:private:654321", "654321"),
    ]


def test_run_recovery_skips_failed_target_and_continues(monkeypatch):
    targets = [
        recovery.RecoveryTarget("qq:private:111", "private", "111"),
        recovery.RecoveryTarget("qq:private:222", "private", "222"),
    ]
    attempted: list[str] = []

    async def fake_build_targets(_cfg):
        return targets

    async def fake_recover_target(_client, target, _cfg):
        attempted.append(target.session_key)
        if target.conv_id == "111":
            raise ValueError("bad target")
        return 2, 3

    monkeypatch.setattr(recovery, "_build_targets", fake_build_targets)
    monkeypatch.setattr(recovery, "_recover_target", fake_recover_target)
    monkeypatch.setattr(recovery, "_recovery_generation", 7)

    asyncio.run(
        recovery._run_recovery(
            SimpleNamespace(connected=True),
            recovery.RecoveryConfig(),
            generation=7,
        )
    )

    assert attempted == ["qq:private:111", "qq:private:222"]


def test_fetch_history_ignores_non_numeric_session_id():
    class Client:
        async def send_api_raw(self, *_args, **_kwargs):
            raise AssertionError("invalid target must not reach the QQ adapter")

    messages = asyncio.run(
        recovery._fetch_history_messages(
            Client(),
            recovery.RecoveryTarget(
                "qq:private:guardian",
                "private",
                "guardian",
            ),
            anchor_message_id=None,
            page_size=50,
        )
    )

    assert messages == []
