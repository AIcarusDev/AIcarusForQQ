"""debug_server.py — 调试 WebUI（Quart Blueprint）

提供：
  - /debug       调试页面
  - /debug/ws    WebSocket 实时推送 NapCat 消息 XML

作为 Quart Blueprint 注册到主 app。
"""

import asyncio
import json
from datetime import datetime

from quart import Blueprint, render_template, websocket as quart_ws

debug_bp = Blueprint("debug", __name__)

# ── 广播队列集合 ─────────────────────────────────────────
_debug_queues: set[asyncio.Queue] = set()

# 由 app.py 注入
_timezone = None


def init_debug(timezone) -> None:
    """设置调试模块使用的时区。"""
    global _timezone
    _timezone = timezone


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

@debug_bp.route("/debug")
async def debug_page():
    return await render_template("debug.html")


@debug_bp.websocket("/debug/ws")
async def debug_ws():
    """调试用 WebSocket —— 实时推送 NapCat 消息 XML。"""
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
