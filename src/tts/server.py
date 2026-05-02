from __future__ import annotations

import asyncio
import inspect
import json
import logging
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import websockets
from websockets.asyncio.server import ServerConnection

logger = logging.getLogger("AICQ.tts.server")

PROTOCOL_VERSION = "1.0"
MAX_TASK_ID_BYTES = 256
DEFAULT_MAX_CONCURRENT_TASKS_PER_PLUGIN = 8
AudioChunkCallback = Callable[[str, bytes], Any | Awaitable[Any]]


@dataclass
class PluginSession:
    plugin_id: str
    ws: ServerConnection
    audio_format: dict[str, Any]
    llm_schema: dict[str, Any]
    max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT_TASKS_PER_PLUGIN
    last_pong_at: float = field(default_factory=time.monotonic)
    active_tasks: set[str] = field(default_factory=set)
    _send_queue: asyncio.Queue[str | bytes | None] = field(default_factory=asyncio.Queue)
    _sender_task: asyncio.Task | None = None

    def start_sender(self) -> None:
        if self._sender_task is None:
            self._sender_task = asyncio.create_task(
                self._sender_loop(),
                name=f"tts_sender:{self.plugin_id}",
            )

    async def _sender_loop(self) -> None:
        while True:
            item = await self._send_queue.get()
            if item is None:
                break
            try:
                await self.ws.send(item)
            except Exception:
                logger.debug("TTS send loop stopped for %s", self.plugin_id, exc_info=True)
                break

    async def send(self, data: str | bytes) -> None:
        await self._send_queue.put(data)

    async def close(self, *, close_ws: bool = True) -> None:
        await self._send_queue.put(None)
        if close_ws:
            try:
                await self.ws.close()
            except Exception:
                logger.debug("TTS websocket close failed for %s", self.plugin_id, exc_info=True)
        if self._sender_task and self._sender_task is not asyncio.current_task():
            try:
                await asyncio.wait_for(self._sender_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._sender_task.cancel()
            except Exception:
                logger.debug("TTS sender task ended with error for %s", self.plugin_id, exc_info=True)


class TTSServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        secret_token: str = "",
        on_audio_chunk: AudioChunkCallback | None = None,
        *,
        ping_interval: float = 30.0,
        pong_timeout: float = 75.0,
        register_timeout: float = 10.0,
        max_concurrent_tasks_per_plugin: int = DEFAULT_MAX_CONCURRENT_TASKS_PER_PLUGIN,
    ) -> None:
        self._host = host
        self._port = int(port)
        self._bound_port = int(port)
        self._secret_token = secret_token
        self._on_audio_chunk = on_audio_chunk
        self._ping_interval = float(ping_interval)
        self._pong_timeout = float(pong_timeout)
        self._register_timeout = float(register_timeout)
        self._max_concurrent_tasks_per_plugin = max(1, int(max_concurrent_tasks_per_plugin))
        self._plugins: dict[str, PluginSession] = {}
        self._pending_tasks: dict[str, asyncio.Future[None]] = {}
        self._task_plugins: dict[str, str] = {}
        self._server: Any = None

    @property
    def bound_port(self) -> int:
        return self._bound_port

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await websockets.serve(self._handle_connection, self._host, self._port)
        sockets = getattr(self._server, "sockets", None)
        if sockets:
            self._bound_port = int(sockets[0].getsockname()[1])
        logger.info("TTS Server listening on ws://%s:%d", self._host, self._bound_port)

    async def stop(self) -> None:
        for plugin_id in list(self._plugins):
            await self._cleanup_plugin(plugin_id, RuntimeError("TTS server stopped"))
        for task_id, future in list(self._pending_tasks.items()):
            if not future.done():
                future.set_exception(RuntimeError("TTS server stopped"))
            self._pending_tasks.pop(task_id, None)
            self._task_plugins.pop(task_id, None)
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_connection(self, ws: ServerConnection) -> None:
        session: PluginSession | None = None
        ping_task: asyncio.Task | None = None
        try:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=self._register_timeout)
            except asyncio.TimeoutError:
                logger.warning("TTS plugin failed to register within %.1fs", self._register_timeout)
                return

            data = self._safe_parse(raw)
            if not data or data.get("type") != "register":
                await self._send_register_ack(ws, False, reason="register_required")
                return

            session = await self._on_register(ws, data)
            if session is None:
                return

            ping_task = asyncio.create_task(
                self._ping_loop(session),
                name=f"tts_ping:{session.plugin_id}",
            )

            async for message in ws:
                if isinstance(message, bytes):
                    self._parse_binary_frame(message)
                elif isinstance(message, str):
                    await self._on_control_message(session, self._safe_parse(message))
        except websockets.ConnectionClosed:
            pass
        except Exception:
            logger.exception("Unexpected error in TTS connection")
        finally:
            if ping_task:
                ping_task.cancel()
                try:
                    await ping_task
                except (asyncio.CancelledError, Exception):
                    pass
            if session:
                await self._cleanup_plugin(
                    session.plugin_id,
                    RuntimeError(f"TTS plugin {session.plugin_id!r} disconnected"),
                    session=session,
                    close_ws=False,
                )

    async def _on_register(self, ws: ServerConnection, data: dict[str, Any]) -> PluginSession | None:
        if self._secret_token and data.get("secret_token") != self._secret_token:
            await self._send_register_ack(ws, False, reason="invalid_token")
            return None

        if data.get("protocol_version") != PROTOCOL_VERSION:
            await self._send_register_ack(ws, False, reason="unsupported_protocol_version")
            return None

        plugin_id = str(data.get("plugin_id") or "").strip()
        if not plugin_id:
            await self._send_register_ack(ws, False, reason="plugin_id_required")
            return None

        if plugin_id in self._plugins:
            logger.info("TTS plugin %s reconnected, closing old session", plugin_id)
            await self._cleanup_plugin(plugin_id, RuntimeError("TTS plugin reconnected"))

        session = PluginSession(
            plugin_id=plugin_id,
            ws=ws,
            audio_format=data.get("audio_format") if isinstance(data.get("audio_format"), dict) else {},
            llm_schema=data.get("llm_schema") if isinstance(data.get("llm_schema"), dict) else {},
            max_concurrent_tasks=self._max_concurrent_tasks_per_plugin,
        )
        session.start_sender()
        self._plugins[plugin_id] = session
        await session.send(json.dumps({
            "type": "register_ack",
            "plugin_id": plugin_id,
            "accepted": True,
            "reason": None,
        }))
        logger.info("TTS plugin registered: %s", plugin_id)
        return session

    async def _send_register_ack(self, ws: ServerConnection, accepted: bool, *, reason: str | None) -> None:
        await ws.send(json.dumps({
            "type": "register_ack",
            "accepted": accepted,
            "reason": reason,
        }))

    async def _on_control_message(self, session: PluginSession, data: dict[str, Any] | None) -> None:
        if not data:
            return
        msg_type = data.get("type")
        if msg_type == "status":
            self._handle_status(data)
        elif msg_type == "pong":
            session.last_pong_at = time.monotonic()

    def _handle_status(self, data: dict[str, Any]) -> None:
        task_id = str(data.get("task_id") or "")
        status = data.get("status")
        future = self._pending_tasks.get(task_id)
        if not future or future.done():
            return
        if status == "completed":
            future.set_result(None)
            self._release_task(task_id)
        elif status == "error":
            future.set_exception(
                RuntimeError(f"TTS error {data.get('error_code')}: {data.get('message')}")
            )
            self._release_task(task_id)

    def _parse_binary_frame(self, buf: bytes) -> None:
        if len(buf) < 4:
            logger.debug("TTS binary frame too short, discarded")
            return
        id_length = struct.unpack(">I", buf[:4])[0]
        if id_length > MAX_TASK_ID_BYTES or len(buf) < 4 + id_length:
            logger.warning("Invalid TTS binary frame: id_length=%d", id_length)
            return
        task_id = buf[4:4 + id_length].decode("utf-8", errors="replace")
        if task_id not in self._pending_tasks:
            logger.debug("TTS audio chunk for unknown task %s discarded", task_id)
            return
        pcm_data = buf[4 + id_length:]
        if self._on_audio_chunk is None:
            return
        try:
            result = self._on_audio_chunk(task_id, pcm_data)
            if inspect.isawaitable(result):
                asyncio.create_task(result, name=f"tts_audio_chunk:{task_id}")
        except Exception:
            logger.warning("TTS audio chunk callback failed for task %s", task_id, exc_info=True)

    async def _ping_loop(self, session: PluginSession) -> None:
        while True:
            await asyncio.sleep(self._ping_interval)
            if time.monotonic() - session.last_pong_at > self._pong_timeout:
                logger.warning("TTS plugin %s missed pong timeout, closing", session.plugin_id)
                await session.close()
                return
            await session.send(json.dumps({"type": "ping", "timestamp": time.time()}))

    async def dispatch_task(self, plugin_id: str, text: str, parameters: dict[str, Any] | None = None) -> str:
        session = self._plugins.get(plugin_id)
        if not session:
            raise RuntimeError(f"TTS plugin {plugin_id!r} not connected")
        if len(session.active_tasks) >= session.max_concurrent_tasks:
            raise RuntimeError(f"TTS plugin {plugin_id!r} reached concurrent task limit")

        task_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        future.add_done_callback(self._consume_unobserved_exception)
        self._pending_tasks[task_id] = future
        self._task_plugins[task_id] = plugin_id
        session.active_tasks.add(task_id)
        await session.send(json.dumps({
            "type": "task",
            "task_id": task_id,
            "text": text,
            "parameters": parameters or {},
        }))
        return task_id

    async def wait_task(self, task_id: str, timeout: float = 60.0) -> None:
        future = self._pending_tasks.get(task_id)
        if not future:
            raise KeyError(f"Unknown task_id: {task_id}")
        try:
            await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        finally:
            self._pending_tasks.pop(task_id, None)
            self._release_task(task_id)

    def list_plugins(self) -> list[dict[str, Any]]:
        return [
            {
                "plugin_id": session.plugin_id,
                "audio_format": session.audio_format,
                "llm_schema": session.llm_schema,
                "active_tasks": len(session.active_tasks),
                "max_concurrent_tasks": session.max_concurrent_tasks,
            }
            for session in self._plugins.values()
        ]

    def get_plugin_info(self, plugin_id: str) -> dict[str, Any] | None:
        session = self._plugins.get(plugin_id)
        if session is None:
            return None
        return {
            "plugin_id": session.plugin_id,
            "audio_format": session.audio_format,
            "llm_schema": session.llm_schema,
            "active_tasks": len(session.active_tasks),
            "max_concurrent_tasks": session.max_concurrent_tasks,
        }

    def select_plugin_id(self, preferred_plugin_id: str | None = None) -> str | None:
        if preferred_plugin_id:
            preferred_plugin_id = preferred_plugin_id.strip()
            if preferred_plugin_id in self._plugins:
                return preferred_plugin_id
        if not self._plugins:
            return None
        return next(iter(self._plugins))

    async def _cleanup_plugin(
        self,
        plugin_id: str,
        error: Exception | None = None,
        *,
        session: PluginSession | None = None,
        close_ws: bool = True,
    ) -> None:
        current = self._plugins.get(plugin_id)
        if session is not None and current is not session:
            return
        session = self._plugins.pop(plugin_id, None)
        if not session:
            return
        for task_id in list(session.active_tasks):
            future = self._pending_tasks.get(task_id)
            if future and not future.done():
                future.set_exception(error or RuntimeError(f"TTS plugin {plugin_id!r} disconnected"))
            self._release_task(task_id)
        await session.close(close_ws=close_ws)
        logger.info("TTS plugin unregistered: %s", plugin_id)

    def _release_task(self, task_id: str) -> None:
        plugin_id = self._task_plugins.pop(task_id, None)
        if plugin_id:
            session = self._plugins.get(plugin_id)
            if session:
                session.active_tasks.discard(task_id)

    @staticmethod
    def _consume_unobserved_exception(future: asyncio.Future[None]) -> None:
        if not future.done() or future.cancelled():
            return
        try:
            future.exception()
        except (asyncio.CancelledError, Exception):
            pass

    @staticmethod
    def _safe_parse(raw: str | bytes) -> dict[str, Any] | None:
        if isinstance(raw, bytes):
            return None
        try:
            data = json.loads(raw)
        except Exception:
            return None
        return data if isinstance(data, dict) else None
