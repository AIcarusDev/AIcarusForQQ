"""Agent View routes."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress

from quart import Blueprint, jsonify, render_template, request, websocket as quart_ws

import app_state
from agent_events import (
    agent_event_stats,
    snapshot_events,
    subscribe_agent_events,
    unsubscribe_agent_events,
)
from database import load_chat_sessions, load_recent_bot_turns

agent_bp = Blueprint("agent", __name__)


async def _receive_ws_or_shutdown() -> bool:
    shutdown_event = getattr(app_state, "shutdown_event", None)
    if shutdown_event is None:
        await quart_ws.receive()
        return False
    if shutdown_event.is_set():
        return True

    receive_task = asyncio.create_task(quart_ws.receive())
    shutdown_task = asyncio.create_task(shutdown_event.wait())
    try:
        done, _pending = await asyncio.wait(
            {receive_task, shutdown_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if shutdown_task in done and shutdown_event.is_set():
            return True
        await receive_task
        return False
    finally:
        for task in (receive_task, shutdown_task):
            if not task.done():
                task.cancel()
        with suppress(Exception):
            await asyncio.gather(receive_task, shutdown_task, return_exceptions=True)


async def _cancel_task(task: asyncio.Task) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task


@agent_bp.route("/agent")
async def agent_page():
    return await render_template(
        "agent.html",
        active_page="agent",
        bot_name=getattr(app_state, "BOT_NAME", ""),
    )


@agent_bp.route("/api/agent/state")
async def agent_state():
    turns = await load_recent_bot_turns(limit=24)
    sessions = await load_chat_sessions()
    return jsonify({
        "current_focus": app_state.current_focus,
        "bot_name": getattr(app_state, "BOT_NAME", ""),
        "provider": getattr(getattr(app_state, "adapter", None), "provider", ""),
        "model": getattr(getattr(app_state, "adapter", None), "model", ""),
        "sessions": sessions,
        "recent_turns": turns,
        "events": snapshot_events(limit=300),
        "stats": agent_event_stats(),
    })


@agent_bp.route("/api/agent/turns")
async def agent_turns():
    try:
        limit = int(request.args.get("limit", "24") or 24)
    except Exception:
        limit = 24
    try:
        before = int(request.args.get("before", "0") or 0)
    except Exception:
        before = 0
    turns = await load_recent_bot_turns(limit=limit, before=before or None)
    return jsonify({"turns": turns})


@agent_bp.websocket("/agent/ws/events")
async def agent_ws_events():
    queue = subscribe_agent_events()
    try:
        try:
            since = int(quart_ws.args.get("since", "0") or 0)
        except Exception:
            since = 0
        await quart_ws.send(json.dumps({
            "type": "snapshot",
            "events": snapshot_events(since=since),
            "stats": agent_event_stats(),
        }, ensure_ascii=False))

        async def _sender():
            while True:
                event = await queue.get()
                await quart_ws.send(json.dumps(event, ensure_ascii=False))

        sender_task = asyncio.ensure_future(_sender())
        try:
            while True:
                if await _receive_ws_or_shutdown():
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            await _cancel_task(sender_task)
    finally:
        unsubscribe_agent_events(queue)


__all__ = ["agent_bp"]
