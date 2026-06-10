from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection

logger = logging.getLogger("AICQ.qqrtc.server")

PROTOCOL_VERSION = "1.0"


@dataclass
class QQRTCPluginSession:
    plugin_id: str
    ws: ServerConnection
    capabilities: dict[str, Any] = field(default_factory=dict)
    last_pong_at: float = field(default_factory=time.monotonic)
    _send_queue: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)
    _sender_task: asyncio.Task | None = None

    def start_sender(self) -> None:
        if self._sender_task is None:
            self._sender_task = asyncio.create_task(
                self._sender_loop(),
                name=f"qqrtc_sender:{self.plugin_id}",
            )

    async def _sender_loop(self) -> None:
        while True:
            item = await self._send_queue.get()
            if item is None:
                break
            try:
                await self.ws.send(item)
            except Exception:
                logger.debug("QQRTC send loop stopped for %s", self.plugin_id, exc_info=True)
                break

    async def send_json(self, data: dict[str, Any]) -> None:
        await self._send_queue.put(json.dumps(data, ensure_ascii=False))

    async def close(self, *, close_ws: bool = True) -> None:
        await self._send_queue.put(None)
        if close_ws:
            try:
                await self.ws.close()
            except Exception:
                logger.debug("QQRTC websocket close failed for %s", self.plugin_id, exc_info=True)
        if self._sender_task and self._sender_task is not asyncio.current_task():
            try:
                await asyncio.wait_for(self._sender_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._sender_task.cancel()
            except Exception:
                logger.debug("QQRTC sender task ended with error for %s", self.plugin_id, exc_info=True)


class QQRTCServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8776,
        secret_token: str = "",
        *,
        event_buffer_size: int = 200,
        ping_interval: float = 30.0,
        pong_timeout: float = 75.0,
        register_timeout: float = 10.0,
    ) -> None:
        self._host = host
        self._port = int(port)
        self._bound_port = int(port)
        self._secret_token = secret_token
        self._event_buffer: deque[dict[str, Any]] = deque(maxlen=max(1, int(event_buffer_size)))
        self._active_calls: dict[str, dict[str, Any]] = {}
        self._plugins: dict[str, QQRTCPluginSession] = {}
        self._pending_commands: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._ping_interval = float(ping_interval)
        self._pong_timeout = float(pong_timeout)
        self._register_timeout = float(register_timeout)
        self._server: Any = None

    @property
    def bound_port(self) -> int:
        return self._bound_port

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await websockets.serve(
            self._handle_connection,
            self._host,
            self._port,
            max_size=8 * 1024 * 1024,
        )
        sockets = getattr(self._server, "sockets", None)
        if sockets:
            self._bound_port = int(sockets[0].getsockname()[1])
        logger.info("QQRTC Server listening on ws://%s:%d", self._host, self._bound_port)

    async def stop(self) -> None:
        for plugin_id in list(self._plugins):
            await self._cleanup_plugin(plugin_id, close_ws=True)
        for command_id, future in list(self._pending_commands.items()):
            if not future.done():
                future.set_exception(RuntimeError("QQRTC server stopped"))
            self._pending_commands.pop(command_id, None)
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_connection(self, ws: ServerConnection) -> None:
        session: QQRTCPluginSession | None = None
        ping_task: asyncio.Task | None = None
        try:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=self._register_timeout)
            except asyncio.TimeoutError:
                logger.warning("QQRTC plugin failed to register within %.1fs", self._register_timeout)
                return

            data = self._safe_parse(raw)
            if not data or data.get("type") != "register":
                await ws.send(json.dumps({
                    "type": "register_ack",
                    "accepted": False,
                    "reason": "register_required",
                }))
                return

            session = await self._on_register(ws, data)
            if session is None:
                return

            ping_task = asyncio.create_task(
                self._ping_loop(session),
                name=f"qqrtc_ping:{session.plugin_id}",
            )

            async for message in ws:
                if isinstance(message, str):
                    await self._on_message(session, self._safe_parse(message))
        except websockets.ConnectionClosed:
            pass
        except Exception:
            logger.exception("Unexpected error in QQRTC connection")
        finally:
            if ping_task:
                ping_task.cancel()
                try:
                    await ping_task
                except (asyncio.CancelledError, Exception):
                    pass
            if session:
                await self._cleanup_plugin(session.plugin_id, session=session, close_ws=False)

    async def _on_register(self, ws: ServerConnection, data: dict[str, Any]) -> QQRTCPluginSession | None:
        if self._secret_token and data.get("secret_token") != self._secret_token:
            await ws.send(json.dumps({"type": "register_ack", "accepted": False, "reason": "invalid_token"}))
            return None
        if data.get("protocol_version") != PROTOCOL_VERSION:
            await ws.send(json.dumps({
                "type": "register_ack",
                "accepted": False,
                "reason": "unsupported_protocol_version",
            }))
            return None

        plugin_id = str(data.get("plugin_id") or "").strip()
        if not plugin_id:
            await ws.send(json.dumps({"type": "register_ack", "accepted": False, "reason": "plugin_id_required"}))
            return None

        if plugin_id in self._plugins:
            logger.info("QQRTC plugin %s reconnected, closing old session", plugin_id)
            await self._cleanup_plugin(plugin_id, close_ws=True)

        session = QQRTCPluginSession(
            plugin_id=plugin_id,
            ws=ws,
            capabilities=data.get("capabilities") if isinstance(data.get("capabilities"), dict) else {},
        )
        session.start_sender()
        self._plugins[plugin_id] = session
        await session.send_json({
            "type": "register_ack",
            "plugin_id": plugin_id,
            "accepted": True,
            "reason": None,
        })
        logger.info("QQRTC plugin registered: %s", plugin_id)
        return session

    async def _on_message(self, session: QQRTCPluginSession, data: dict[str, Any] | None) -> None:
        if not data:
            return
        msg_type = data.get("type")
        if msg_type == "pong":
            session.last_pong_at = time.monotonic()
        elif msg_type == "event":
            event = data.get("event") if isinstance(data.get("event"), dict) else {}
            self._record_event(session.plugin_id, event)
        elif msg_type == "command_result":
            self._complete_command(data)

    def _record_event(self, plugin_id: str, event: dict[str, Any]) -> None:
        if not event:
            return
        event = dict(event)
        event.setdefault("plugin_id", plugin_id)
        event.setdefault("received_at", time.time())
        self._event_buffer.append(event)

        session_id = str(event.get("session_id") or "").strip()
        if not session_id:
            return
        prev = self._active_calls.get(session_id, {})
        merged = {
            **prev,
            "session_id": session_id,
            "caller_id": event.get("caller_id") or prev.get("caller_id", ""),
            "callee_id": event.get("callee_id") or prev.get("callee_id", ""),
            "peer_id": event.get("peer_id") or prev.get("peer_id", ""),
            "direction": event.get("direction") or prev.get("direction", "unknown"),
            "last_sub_type": event.get("sub_type") or prev.get("last_sub_type", ""),
            "last_event_at": event.get("received_at"),
            "plugin_id": plugin_id,
        }
        if "first_event_at" not in merged:
            merged["first_event_at"] = event.get("received_at")
        self._active_calls[session_id] = merged

    def _complete_command(self, data: dict[str, Any]) -> None:
        command_id = str(data.get("command_id") or "")
        future = self._pending_commands.pop(command_id, None)
        if future and not future.done():
            future.set_result(data)

    async def _ping_loop(self, session: QQRTCPluginSession) -> None:
        while True:
            await asyncio.sleep(self._ping_interval)
            if time.monotonic() - session.last_pong_at > self._pong_timeout:
                logger.warning("QQRTC plugin %s missed pong timeout, closing", session.plugin_id)
                await session.close()
                return
            await session.send_json({"type": "ping", "timestamp": time.time()})

    def list_plugins(self) -> list[dict[str, Any]]:
        return [
            {
                "plugin_id": session.plugin_id,
                "capabilities": session.capabilities,
                "connected": True,
            }
            for session in self._plugins.values()
        ]

    def list_events(self, limit: int = 20, session_id: str | None = None) -> list[dict[str, Any]]:
        events = list(self._event_buffer)
        if session_id:
            events = [event for event in events if str(event.get("session_id") or "") == session_id]
        limit = max(1, min(int(limit or 20), len(events) or 1))
        return events[-limit:]

    def list_active_calls(self) -> list[dict[str, Any]]:
        return sorted(
            self._active_calls.values(),
            key=lambda item: float(item.get("last_event_at") or 0),
            reverse=True,
        )

    async def dispatch_command(
        self,
        action: str,
        parameters: dict[str, Any] | None = None,
        *,
        plugin_id: str | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        target_id = plugin_id.strip() if plugin_id else ""
        session = self._plugins.get(target_id) if target_id else next(iter(self._plugins.values()), None)
        if session is None:
            raise RuntimeError("没有在线 QQRTC 插件")

        command_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_commands[command_id] = future
        await session.send_json({
            "type": "command",
            "command_id": command_id,
            "action": action,
            "parameters": parameters or {},
        })
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending_commands.pop(command_id, None)

    async def _cleanup_plugin(
        self,
        plugin_id: str,
        *,
        session: QQRTCPluginSession | None = None,
        close_ws: bool,
    ) -> None:
        current = self._plugins.get(plugin_id)
        if session is not None and current is not session:
            return
        session = self._plugins.pop(plugin_id, None)
        if not session:
            return
        await session.close(close_ws=close_ws)
        logger.info("QQRTC plugin unregistered: %s", plugin_id)

    @staticmethod
    def _safe_parse(raw: str | bytes) -> dict[str, Any] | None:
        if isinstance(raw, bytes):
            return None
        try:
            data = json.loads(raw)
        except Exception:
            return None
        return data if isinstance(data, dict) else None
