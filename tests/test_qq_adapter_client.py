from __future__ import annotations

import asyncio

from platforms.qq.adapter.client import QQAdapterClient


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
