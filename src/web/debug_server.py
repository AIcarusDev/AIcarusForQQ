"""debug_server.py — 日志 WebUI（Quart Blueprint）

提供：
  - /log              日志页面（聊天记录 + 调试日志两个 Tab）
  - /debug            重定向 → /log（向后兼容）
  - /log/ws/chat      WebSocket 实时推送 QQ 聊天事件
  - /log/ws/log       WebSocket 实时推送后端日志记录
  - /debug/ws         原 QQ adapter XML WebSocket（保留供内部使用）
  - /debug/ws/status  QQ adapter 连接状态 WebSocket
  - /debug/api/status QQ adapter 连接状态轮询

作为 Quart Blueprint 注册到主 app。
"""

import asyncio
import itertools
import json
import re
from collections import deque
from datetime import datetime

from quart import Blueprint, jsonify, redirect, render_template, request, websocket as quart_ws

debug_bp = Blueprint("debug", __name__)

# ── 广播队列集合 ───────────────────────────────
_debug_queues: set[asyncio.Queue] = set()   # 原 XML Inspector WS
_chat_queues: set[asyncio.Queue] = set()    # 聊天记录 WS
_log_queues: set[asyncio.Queue] = set()     # 后端日志 WS
_status_queues: set[asyncio.Queue] = set()  # QQ adapter 连接状态 WS

# 历史缓冲（新连接接入时先补发）
_chat_buffer: deque = deque(maxlen=200)
_log_buffer: deque = deque(maxlen=2000)

# 日志全局递增序号（用于重连增量同步、去重、计数）
_log_seq_counter = itertools.count(1)

# 由 app.py 注入
_timezone = None
_qq_adapter_client = None

_FILE_LOG_HEADER_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) "
    r"\[(?P<level>[A-Z]+)\] (?P<name>\S+) (?P<file>.+):(?P<lineno>\d+)$"
)


def _parse_log_text(text: str) -> list[dict]:
    """Parse records written by log_config.FileFormatter."""
    records: list[dict] = []
    current: dict | None = None
    message_lines: list[str] = []

    def flush() -> None:
        nonlocal current, message_lines
        if current is None:
            message_lines = []
            return
        while message_lines and message_lines[-1] == "":
            message_lines.pop()
        current["message"] = "\n".join(message_lines)
        records.append(current)
        current = None
        message_lines = []

    for line in text.splitlines():
        match = _FILE_LOG_HEADER_RE.match(line)
        if match:
            flush()
            data = match.groupdict()
            timestamp = data["timestamp"]
            current = {
                "level": data["level"],
                "name": data["name"],
                "message": "",
                "time": timestamp.split(" ", 1)[1],
                "timestamp": timestamp,
                "file": data["file"],
                "lineno": int(data["lineno"]),
                "source": "file",
            }
            continue
        if current is not None:
            message_lines.append(line)

    flush()
    return records


def init_debug(timezone, qq_adapter_client=None) -> None:
    """设置调试模块使用的时区，并可选注入 QQ adapter 客户端引用。"""
    global _timezone, _qq_adapter_client
    _timezone = timezone
    _qq_adapter_client = qq_adapter_client


def _qq_adapter_status_payload() -> dict:
    connected = bool(_qq_adapter_client and _qq_adapter_client.connected)
    bot_id = _qq_adapter_client.bot_id if (_qq_adapter_client and _qq_adapter_client.bot_id) else ""
    adapter = getattr(_qq_adapter_client, "adapter", "") if _qq_adapter_client else ""
    adapter_name = getattr(_qq_adapter_client, "adapter_name", "") if _qq_adapter_client else ""
    return {
        "qq_adapter_connected": connected,
        "bot_id": bot_id,
        "adapter": adapter,
        "adapter_name": adapter_name,
        "status_ws_supported": True,
    }


async def broadcast_qq_adapter_status() -> None:
    """Push the current QQ adapter status to sidebar listeners."""
    if not _status_queues:
        return
    payload = _qq_adapter_status_payload()
    payload["type"] = "qq_adapter_status"
    data = json.dumps(payload, ensure_ascii=False)
    for q in list(_status_queues):
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass


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
    """从任意线程添加日志记录，线程安全地调度到 event loop 分发。

    会为记录注入全局单调递增的 seq，供前端增量同步、去重、计数使用。
    """
    record_dict["seq"] = next(_log_seq_counter)
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


# ── 原 debug XML 广播（保留供 qq_adapter_handler 使用）────────

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
    """轮询接口：返回 QQ adapter 连接状态，供前端替代推送。"""
    return jsonify(_qq_adapter_status_payload())


@debug_bp.route("/debug/api/qq_adapter/get_forward_msg", methods=["POST"])
async def debug_get_forward_msg():
    """Local-only debug endpoint for inspecting adapter forward-message payloads."""
    remote_addr = request.headers.get("X-Forwarded-For", request.remote_addr or "")
    if remote_addr not in {"127.0.0.1", "::1", "localhost"}:
        return jsonify({"ok": False, "error": "local requests only"}), 403
    if not _qq_adapter_client or not _qq_adapter_client.connected:
        return jsonify({"ok": False, "error": "QQ adapter is not connected"}), 503

    payload = await request.get_json(silent=True) or {}
    forward_id = str(payload.get("id") or payload.get("forward_id") or "").strip()
    if not forward_id:
        return jsonify({"ok": False, "error": "missing id"}), 400

    data = await _qq_adapter_client.send_api("get_forward_msg", {"id": forward_id}, timeout=15.0)
    if data is None:
        api_error = getattr(_qq_adapter_client, "last_api_error", None) or {}
        return jsonify({"ok": False, "error": api_error.get("message") or "empty adapter response"}), 502
    return jsonify({"ok": True, "data": data})


@debug_bp.route("/debug/api/qq_adapter/get_group_msg_history", methods=["POST"])
async def debug_get_group_msg_history():
    """Local-only debug endpoint for inspecting group-history adapter payloads."""
    remote_addr = request.headers.get("X-Forwarded-For", request.remote_addr or "")
    if remote_addr not in {"127.0.0.1", "::1", "localhost"}:
        return jsonify({"ok": False, "error": "local requests only"}), 403
    if not _qq_adapter_client or not _qq_adapter_client.connected:
        return jsonify({"ok": False, "error": "QQ adapter is not connected"}), 503

    payload = await request.get_json(silent=True) or {}
    group_id = str(payload.get("group_id") or "").strip()
    if not group_id:
        return jsonify({"ok": False, "error": "missing group_id"}), 400
    params = {
        "group_id": int(group_id),
        "count": int(payload.get("count") or 50),
        "parse_mult_msg": bool(payload.get("parse_mult_msg", True)),
    }
    if payload.get("message_seq") not in (None, ""):
        params["message_seq"] = int(payload["message_seq"])
    if payload.get("reverse_order") not in (None, ""):
        params["reverse_order"] = bool(payload["reverse_order"])

    data = await _qq_adapter_client.send_api("get_group_msg_history", params, timeout=15.0)
    if data is None:
        api_error = getattr(_qq_adapter_client, "last_api_error", None) or {}
        return jsonify({"ok": False, "error": api_error.get("message") or "empty adapter response"}), 502
    return jsonify({"ok": True, "data": data})


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


@debug_bp.websocket("/debug/ws/status")
async def status_ws():
    """QQ adapter 连接状态 WebSocket：连接时先下发快照，后续推送变化。"""
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    _status_queues.add(queue)
    try:
        await quart_ws.send(json.dumps(_qq_adapter_status_payload(), ensure_ascii=False))

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
        _status_queues.discard(queue)


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
    """日志 WebSocket：可通过 ?since=<seq> 只拉增量，默认全量。

    存在 since 参数时只补发 seq > since 的记录，用于重连去重。
    历史会以一条 snapshot 消息批量下发，避免逐条 send 塑造延迟。
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
    _log_queues.add(queue)
    try:
        # 解析 since
        try:
            since = int(quart_ws.args.get("since", "0") or 0)
        except Exception:
            since = 0
        # 带上 snapshot 一次发完
        history = [it for it in list(_log_buffer) if it.get("seq", 0) > since]
        await quart_ws.send(json.dumps({"type": "snapshot", "records": history}, ensure_ascii=False))

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
