"""debug_server.py — 日志 WebUI（Quart Blueprint）

提供：
  - /log              日志页面（聊天记录 + 调试日志两个 Tab）
  - /debug            重定向 → /log（向后兼容）
  - /log/ws/chat      WebSocket 实时推送 QQ 聊天事件
  - /log/ws/log       WebSocket 实时推送后端日志记录
  - /debug/ws         原 NapCat XML WebSocket（保留供内部使用）
  - /debug/api/status NapCat 连接状态轮询

作为 Quart Blueprint 注册到主 app。
"""

import asyncio
import json
from collections import deque
from datetime import datetime

from quart import Blueprint, jsonify, redirect, render_template, websocket as quart_ws

debug_bp = Blueprint("debug", __name__)

# ── 广播队列集合 ─────────────────────────────────────────
_debug_queues: set[asyncio.Queue] = set()   # 原 XML Inspector WS
_chat_queues: set[asyncio.Queue] = set()    # 聊天记录 WS
_log_queues: set[asyncio.Queue] = set()     # 后端日志 WS

# 历史缓冲（新连接接入时先补发）
_chat_buffer: deque = deque(maxlen=200)
_log_buffer: deque = deque(maxlen=2000)

# 由 app.py 注入
_timezone = None
_napcat_client = None


def init_debug(timezone, napcat_client=None) -> None:
    """设置调试模块使用的时区，并可选注入 NapCat 客户端引用。"""
    global _timezone, _napcat_client
    _timezone = timezone
    _napcat_client = napcat_client


# ── 聊天事件广播 ─────────────────────────────────────────

async def broadcast_chat_event(payload: dict) -> None:
    """向所有连接的聊天 WS 客户端广播事件（在 event loop 中调用）。"""
    _chat_buffer.append(payload)
    if not _chat_queues:
        return
    data = json.dumps(payload, ensure_ascii=False)
    for q in list(_chat_queues):
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass


# ── 日志记录广播 ─────────────────────────────────────────

def add_log_record(record_dict: dict) -> None:
    """从任意线程添加日志记录，线程安全地调度到 event loop 分发。"""
    _log_buffer.append(record_dict)
    try:
        import app_state
        loop = getattr(app_state, "main_loop", None)
        if loop and loop.is_running():
            loop.call_soon_threadsafe(_put_log_to_queues, record_dict)
    except Exception:
        pass


def _put_log_to_queues(record_dict: dict) -> None:
    """在 event loop 线程中向所有日志 WS 队列推送记录。"""
    if not _log_queues:
        return
    data = json.dumps(record_dict, ensure_ascii=False)
    for q in list(_log_queues):
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass


# ── 原 debug XML 广播（保留供 napcat_handler 使用）────────

async def broadcast_debug_xml(xml_str: str, raw_event: dict) -> None:
    """向所有已连接的调试 WebSocket 客户端广播消息。"""
    if not _debug_queues:
        return

    from zoneinfo import ZoneInfo
    tz = _timezone or ZoneInfo("Asia/Shanghai")

    payload = json.dumps(
        {
            "xml": xml_str,
            "raw": raw_event,
            "timestamp": datetime.now(tz).isoformat(),
        },
        ensure_ascii=False,
    )
    for q in list(_debug_queues):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


# ── 路由 ─────────────────────────────────────────────────

@debug_bp.route("/debug/api/status")
async def debug_status():
    """轮询接口：返回 NapCat 连接状态，供前端替代推送。"""
    connected = bool(_napcat_client and _napcat_client.connected)
    bot_id = _napcat_client.bot_id if (_napcat_client and _napcat_client.bot_id) else ""
    return jsonify({"napcat_connected": connected, "bot_id": bot_id})


@debug_bp.route("/debug")
async def debug_redirect():
    """向后兼容重定向。"""
    return redirect("/log")


@debug_bp.route("/log")
async def log_page():
    return await render_template("log.html")


@debug_bp.websocket("/debug/ws")
async def debug_ws():
    """原 XML Inspector WebSocket（保留）。"""
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    _debug_queues.add(queue)
    try:
        async def _sender():
            while True:
                data = await queue.get()
                await quart_ws.send(data)

        sender_task = asyncio.ensure_future(_sender())
        try:
            while True:
                await quart_ws.receive()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            sender_task.cancel()
    finally:
        _debug_queues.discard(queue)


@debug_bp.websocket("/log/ws/chat")
async def log_ws_chat():
    """聊天记录 WebSocket：先补发历史缓冲，再实时推送新事件。"""
    queue: asyncio.Queue = asyncio.Queue(maxsize=512)
    _chat_queues.add(queue)
    try:
        for item in list(_chat_buffer):
            await quart_ws.send(json.dumps(item, ensure_ascii=False))

        async def _sender():
            while True:
                data = await queue.get()
                await quart_ws.send(data)

        sender_task = asyncio.ensure_future(_sender())
        try:
            while True:
                await quart_ws.receive()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            sender_task.cancel()
    finally:
        _chat_queues.discard(queue)


@debug_bp.websocket("/log/ws/log")
async def log_ws_log():
    """日志 WebSocket：先补发历史缓冲，再实时推送新记录。"""
    queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
    _log_queues.add(queue)
    try:
        for item in list(_log_buffer):
            await quart_ws.send(json.dumps(item, ensure_ascii=False))

        async def _sender():
            while True:
                data = await queue.get()
                await quart_ws.send(data)

        sender_task = asyncio.ensure_future(_sender())
        try:
            while True:
                await quart_ws.receive()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            sender_task.cancel()
    finally:
        _log_queues.discard(queue)
