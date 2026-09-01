from __future__ import annotations

import asyncio

from platforms.qq.adapter.client import QQAdapterClient
from platforms.qq.adapter.client import _safe_adapter_message, _safe_api_params_for_log


def test_adapter_diagnostics_redact_file_resources_and_local_paths() -> None:
    params = {
        "file": "base64://c2VjcmV0",
        "message": [
            {"type": "image", "data": {"file": r"C:\Users\private\secret.png"}}
        ],
    }

    safe = _safe_api_params_for_log(params)

    assert safe["file"] == "<base64 payload: 8 chars>"
    assert safe["message"][0]["data"]["file"] == "<local file resource>"
    assert _safe_adapter_message(r"failed, uri=C:\Users\private\secret.bin") == (
        "<adapter message redacted: contained a local path>"
    )
    assert _safe_adapter_message("ordinary adapter failure") == "ordinary adapter failure"


def test_active_disconnect_resolves_pending_api_waits() -> None:
    async def scenario() -> None:
        client = QQAdapterClient()
        active_socket = object()
        client._ws = active_socket  # type: ignore[assignment]
        pending = asyncio.get_running_loop().create_future()
        client._api_futures["echo"] = pending

        assert client._clear_connection_if_current(active_socket) is True  # type: ignore[arg-type]
        assert await pending == {
            "status": "failed",
            "retcode": None,
            "message": "QQ adapter WebSocket 连接已断开",
            "wording": "QQ adapter WebSocket 连接已断开",
        }
        assert client._api_futures == {}

    asyncio.run(scenario())


def test_stale_disconnect_does_not_resolve_current_connection_waits() -> None:
    async def scenario() -> None:
        client = QQAdapterClient()
        active_socket = object()
        stale_socket = object()
        client._ws = active_socket  # type: ignore[assignment]
        pending = asyncio.get_running_loop().create_future()
        client._api_futures["echo"] = pending

        assert client._clear_connection_if_current(stale_socket) is False  # type: ignore[arg-type]
        assert pending.done() is False
        assert client._api_futures == {"echo": pending}

        pending.cancel()

    asyncio.run(scenario())


def test_sent_event_waiter_is_matched_once_and_removed() -> None:
    async def scenario() -> None:
        client = QQAdapterClient()
        token, pending = client.register_sent_event_waiter(
            lambda event: event.get("message_id") == 42
        )

        client._dispatch_sent_event_waiters({"message_id": 41})
        assert pending.done() is False
        assert token in client._sent_event_waiters

        matched = {"message_id": 42, "message": [{"type": "file", "data": {}}]}
        client._dispatch_sent_event_waiters(matched)
        assert await pending == matched
        assert token not in client._sent_event_waiters

        client._dispatch_sent_event_waiters({"message_id": 42, "extra": True})
        assert await pending == matched

    asyncio.run(scenario())


def test_active_disconnect_resolves_sent_event_waiters() -> None:
    async def scenario() -> None:
        client = QQAdapterClient()
        active_socket = object()
        client._ws = active_socket  # type: ignore[assignment]
        token, pending = client.register_sent_event_waiter(lambda _event: True)

        assert client._clear_connection_if_current(active_socket) is True  # type: ignore[arg-type]
        assert await pending is None
        assert token not in client._sent_event_waiters

    asyncio.run(scenario())
