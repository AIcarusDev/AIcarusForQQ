from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import database
from platforms.qq.friends import FriendService


def client():
    return SimpleNamespace(bot_id="100", send_api_raw=AsyncMock(return_value={"status": "ok", "retcode": 0, "data": None}))


def request(**changes):
    return {"request_type": "friend", "self_id": 100, "user_id": 200,
            "flag": "request-1", "comment": '<hello & "friend">', "time": 123, **changes}


def test_requests_persist_deduplicate_and_are_account_scoped():
    async def scenario():
        await database.init_db()
        adapter = client()
        service = FriendService(adapter)
        assert await service.receive(request())
        assert not await service.receive(request())
        restored = FriendService(adapter)
        assert len((await restored.list_requests())["requests"]) == 1
        assert "&lt;hello &amp;" in restored.render_pending()
        adapter.bot_id = "101"
        assert restored.pending == []
        assert (await restored.list_requests())["requests"] == []
        assert not await restored.receive(request())
    asyncio.run(scenario())


@pytest.mark.parametrize("approve,state", [(True, "approved"), (False, "rejected")])
def test_decision_checks_raw_response_and_cannot_reprocess(approve, state):
    async def scenario():
        await database.init_db()
        adapter = client()
        service = FriendService(adapter)
        await service.receive(request())
        results = await asyncio.gather(service.decide("request-1", approve, "备注"), service.decide("request-1", approve))
        assert results[0] == {"ok": True, "user_id": "200", "state": state}
        assert results[1]["ok"] is False
        adapter.send_api_raw.assert_awaited_once_with("set_friend_add_request", {"flag": "request-1", "approve": approve, "remark": "备注"})
        assert not await service.receive(request())
        assert not (await FriendService(adapter).list_requests())["requests"]
    asyncio.run(scenario())


@pytest.mark.parametrize("response", [None, {"status": "failed", "retcode": 1}, {"status": "ok", "retcode": 1}])
def test_failed_or_unknown_decision_does_not_claim_success(response):
    async def scenario():
        await database.init_db()
        adapter = client()
        adapter.send_api_raw.return_value = response
        service = FriendService(adapter)
        await service.receive(request())
        assert not (await service.decide("request-1", True))["ok"]
        assert len(service.pending) == 1
        assert not (await service.delete("200"))["ok"]
    asyncio.run(scenario())


def test_unknown_and_other_request_types_are_not_actionable():
    async def scenario():
        await database.init_db()
        adapter = client()
        service = FriendService(adapter)
        for event in (request(request_type="group"), request(flag=""), request(user_id=0)):
            assert not await service.receive(event)
        assert not (await service.decide("missing", True))["ok"]
        adapter.send_api_raw.assert_not_awaited()
        assert await service.delete("200") == {"ok": True, "user_id": "200"}
        adapter.send_api_raw.assert_awaited_once_with("delete_friend", {"user_id": 200})
    asyncio.run(scenario())


def test_request_notification_and_world_visibility(monkeypatch):
    import app_state
    import platforms.registry
    from platforms.qq.handler import _handle_qq_friend_request
    from platforms.qq.runtime import QQRuntime
    from unittest.mock import Mock

    async def scenario():
        await database.init_db()
        runtime = QQRuntime({"enabled": True})
        adapter = client()
        runtime.client = adapter
        runtime.friends = FriendService(adapter)
        monkeypatch.setattr(platforms.registry, "get_platform", lambda name: runtime)
        hub = SimpleNamespace(publish_threadsafe=Mock())
        monkeypatch.setattr(app_state, "runtime_event_hub", hub)
        monkeypatch.setattr(app_state, "main_loop", asyncio.get_running_loop())
        await _handle_qq_friend_request(request())
        await _handle_qq_friend_request(request())
        hub.publish_threadsafe.assert_called_once()
        assert runtime.attention_events()[0].level == "mention"
        assert "request-1" in runtime.friends.render_pending()
    asyncio.run(scenario())


def test_friend_tools_are_discoverable_and_mutations_are_serial():
    from tools import build_tools

    collection = build_tools({"platforms": {"qq": {"enabled": True}}}, current_platform="qq")
    for name in ("list_friend_requests", "handle_friend_request", "delete_friend"):
        spec = collection.all_specs[f"qq_contacts.{name}"]
        assert spec.externally_perceptible == (name != "list_friend_requests")
    from platforms.qq.tools.qq_contacts.delete_friend import execute
    assert "error" in execute(user_id="not-a-qq-id")


def test_websocket_request_dispatch():
    import json
    from platforms.qq.adapter.client import QQAdapterClient

    async def scenario():
        handled = asyncio.Event()
        received = []

        async def handler(event):
            received.append(event)
            handled.set()

        class Socket:
            remote_address = "test"
            request = SimpleNamespace(headers={"X-Self-ID": "100"})

            async def __aiter__(self):
                yield json.dumps({"post_type": "request", **request()})
                await asyncio.wait_for(handled.wait(), 1)

        adapter = QQAdapterClient()
        adapter.set_request_handler(handler)
        await adapter._connection_handler(Socket())
        assert received == [{"post_type": "request", **request()}]
    asyncio.run(scenario())
