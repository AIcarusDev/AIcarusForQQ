from __future__ import annotations

import asyncio

import pytest
from quart import Quart
from quart.testing import WebsocketResponseError

import app_state
from agent_events import (
    agent_event_stats,
    clear_agent_events_for_test,
    emit_agent_event,
)
from web import auth, debug_server, routes_agent


def test_http_and_websocket_share_session_auth(monkeypatch) -> None:
    async def scenario() -> None:
        clear_agent_events_for_test()
        monkeypatch.setattr(
            app_state,
            "config",
            {
                "webui_auth": {
                    "enabled": True,
                    "password_hash": auth._hash_password("secret1"),
                    "skipped_setup": False,
                    "session_days": 7,
                }
            },
        )
        monkeypatch.setattr(app_state, "shutdown_event", asyncio.Event())
        app = Quart(__name__)
        auth.install_auth(app)
        app.register_blueprint(auth.auth_bp)
        app.register_blueprint(routes_agent.agent_bp)
        client = app.test_client()

        denied_http = await client.get("/api/agent/state")
        assert denied_http.status_code == 401

        async with client.websocket("/agent/ws/events") as websocket:
            with pytest.raises(WebsocketResponseError) as denied_ws:
                await websocket.receive()
        assert denied_ws.value.response.status_code == 401

        login = await client.post("/api/auth/login", json={"password": "secret1"})
        assert login.status_code == 200
        async with client.websocket("/agent/ws/events") as websocket:
            snapshot = await websocket.receive_json()
            assert snapshot["type"] == "snapshot"
            assert snapshot["stream_id"] == agent_event_stats()["stream_id"]

    asyncio.run(scenario())


def test_agent_stream_resets_stale_cursor_after_process_generation_change(monkeypatch) -> None:
    async def scenario() -> None:
        clear_agent_events_for_test()
        monkeypatch.setattr(app_state, "shutdown_event", asyncio.Event())
        emitted = emit_agent_event("round_started", round_id="round-restart")
        app = Quart(__name__)
        app.register_blueprint(routes_agent.agent_bp)
        client = app.test_client()

        async with client.websocket(
            "/agent/ws/events",
            query_string={
                "since": emitted["seq"] + 100,
                "stream_id": "previous-process",
            },
        ) as websocket:
            snapshot = await websocket.receive_json()

        assert snapshot["cursor_reset"] is True
        assert snapshot["stream_id"] == agent_event_stats()["stream_id"]
        assert [event["seq"] for event in snapshot["events"]] == [emitted["seq"]]

    asyncio.run(scenario())


def test_log_stream_resets_stale_cursor_after_process_generation_change(monkeypatch) -> None:
    async def scenario() -> None:
        debug_server._log_buffer.clear()
        monkeypatch.setattr(app_state, "shutdown_event", asyncio.Event())
        record = {
            "level": "INFO",
            "name": "test.realtime",
            "message": "after restart",
        }
        debug_server.add_log_record(record)
        app = Quart(__name__)
        app.register_blueprint(debug_server.debug_bp)
        client = app.test_client()

        async with client.websocket(
            "/log/ws/log",
            query_string={
                "since": int(record["seq"]) + 100,
                "stream_id": "previous-process",
            },
        ) as websocket:
            snapshot = await websocket.receive_json()

        assert snapshot["cursor_reset"] is True
        assert snapshot["stream_id"] == debug_server._log_stream_id
        assert [item["seq"] for item in snapshot["records"]] == [record["seq"]]
        debug_server._log_buffer.clear()

    asyncio.run(scenario())
