from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from platforms.qq.tools.qq_runtime import enter_qq_session as enter


@pytest.fixture(autouse=True)
def allow_sessions(monkeypatch):
    monkeypatch.setattr(enter, "_qq_cfg", lambda: {"whitelist": {"enabled": False}})


@pytest.mark.parametrize("source,error", [(('300', '群'), None), (None, None), (('300', '群'), '非群成员')])
def test_incomplete_temp_source_is_repaired_only_from_valid_group(monkeypatch, source, error):
    monkeypatch.setattr(enter, "_resolve_existing_temp", AsyncMock(return_value={"name": "梦华", "temp_source_group_id": ""}))
    monkeypatch.setattr(enter, "_current_group_source", AsyncMock(return_value=source))
    monkeypatch.setattr(enter, "_validate_group_member", AsyncMock(return_value=error))
    monkeypatch.setattr(enter, "_qq_client", lambda: object())
    result = asyncio.run(enter._resolve_temp_target("200"))
    if source and not error:
        assert result["temp_source_group_id"] == "300"
        assert result["name"] == "梦华"
        assert enter._validate_group_member.await_args.args[1:] == ("300", "200")
    else:
        assert "error" in result


def test_complete_temp_source_is_preserved(monkeypatch):
    existing = {"temp_source_group_id": "300"}
    monkeypatch.setattr(enter, "_resolve_existing_temp", AsyncMock(return_value=existing))
    source = AsyncMock()
    monkeypatch.setattr(enter, "_current_group_source", source)
    assert asyncio.run(enter._resolve_temp_target("200")) == existing
    source.assert_not_awaited()


def test_unknown_friend_status_is_not_treated_as_nonfriend(monkeypatch):
    from types import SimpleNamespace
    adapter = SimpleNamespace(connected=True, send_api=AsyncMock(return_value=None))
    monkeypatch.setattr(enter, "_qq_client", lambda: adapter)
    assert asyncio.run(enter._is_friend(adapter, "200")) is None
    assert "error" in asyncio.run(enter._resolve_enter_target("private", "200"))


def test_repair_updates_live_and_persisted_session(monkeypatch):
    import app_state
    import database
    import llm.session as session_module
    from types import SimpleNamespace
    from platforms.focus import FocusRef
    from platforms.qq.adapter.conversation import make_temp_session_key

    asyncio.run(database.init_db())
    monkeypatch.setattr(session_module, "sessions", {})
    target = session_module.get_or_create_session(make_temp_session_key("200"))
    target.set_conversation_meta("temp", "200", "梦华")
    monkeypatch.setattr(app_state, "main_loop", SimpleNamespace(is_running=lambda: True))
    monkeypatch.setattr(app_state, "current_focus", FocusRef("qq", "group", "300", "群"))
    monkeypatch.setattr(enter, "run_coroutine_sync", lambda coro, loop, timeout: asyncio.run(coro))
    monkeypatch.setattr(enter, "_qq_client", lambda: SimpleNamespace(connected=True, send_api=AsyncMock(return_value=[])))
    monkeypatch.setattr(enter, "_validate_group_member", AsyncMock(return_value=None))
    result = enter.execute(type="private", id="200")
    assert result["ok"] is True
    assert result["now_focusing"]["source_group_id"] == "300"
    assert target.temp_source_group_id == "300"
    persisted = asyncio.run(enter._load_persisted_temp("200"))
    assert persisted["temp_source_group_id"] == "300"
