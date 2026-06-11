# Copyright (C) 2026  AIcarusDev
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Dashboard and focus routes for the WebUI."""

import asyncio
import logging
import mimetypes
import os
import time
from datetime import datetime

import aiosqlite
from quart import Blueprint, Response, jsonify, render_template, request

import app_state
from database import (
    DB_PATH,
    load_chat_messages,
    load_chat_sessions,
    load_recent_bot_turns,
)

logger = logging.getLogger("AICQ.web.dashboard")

dashboard_bp = Blueprint("dashboard", __name__)
_start_time = time.time()


@dashboard_bp.route("/")
async def home():
    return await render_template("home.html", active_page="home")


@dashboard_bp.route("/api/status")
async def api_status():
    """Dashboard status API used by home.html polling."""
    uptime_sec = int(time.time() - _start_time)

    memory_counts = {"entity_profiles": 0, "entities": 0, "groups": 0, "sessions": 0}
    today_messages = 0

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            for tbl, key in (
                ("entity_profiles", "entity_profiles"),
                ("entities", "entities"),
                ("groups", "groups"),
                ("chat_sessions", "sessions"),
            ):
                try:
                    async with db.execute(f"SELECT COUNT(*) AS n FROM {tbl}") as cur:
                        row = await cur.fetchone()
                        memory_counts[key] = row["n"] if row else 0
                except Exception:
                    pass

            today_start = int(
                datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000
            )
            try:
                async with db.execute(
                    "SELECT COUNT(*) AS n FROM bot_turns WHERE created_at >= ?", (today_start,)
                ) as cur:
                    row = await cur.fetchone()
                    today_messages = row["n"] if row else 0
            except Exception:
                pass

    except Exception as e:
        logger.warning("api_status DB query failed: %s", e)

    return jsonify({
        "current_focus": app_state.current_focus,
        "today_messages": today_messages,
        "memory_counts": memory_counts,
        "uptime_seconds": uptime_sec,
        "bot_name": app_state.BOT_NAME,
        "model": app_state.MODEL,
    })


@dashboard_bp.route("/models", methods=["POST"])
async def list_models_route():
    data = await request.get_json() or {}

    base_url = (data.get("base_url") or "").strip()
    api_key = (data.get("api_key") or "").strip() or "openai-compat"

    if not base_url:
        return jsonify({"success": False, "error": "未提供 base_url", "models": []}), 400

    try:
        import httpx
        from openai import OpenAI

        proxy_url = os.getenv("OPENAI_PROXY", "").strip() or None
        client_kwargs: dict = {"api_key": api_key, "base_url": base_url}
        if proxy_url:
            client_kwargs["http_client"] = httpx.Client(proxy=proxy_url)

        client = OpenAI(**client_kwargs)
        models = await asyncio.to_thread(
            lambda: sorted(m.id for m in client.models.list().data)
        )
        return jsonify({"success": True, "models": models})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "models": []}), 500


@dashboard_bp.route("/focus")
async def focus_page():
    from runtime.emergency_reset import expected_confirmation
    return await render_template(
        "focus.html",
        active_page="focus",
        emergency_reset_confirmation=expected_confirmation(),
    )


@dashboard_bp.route("/api/focus/state")
async def focus_state():
    """Focus state API: current focus, known sessions, and recent bot turns."""
    sessions_list = await load_chat_sessions()
    turns = await load_recent_bot_turns(limit=15)
    return jsonify({
        "current_focus": app_state.current_focus,
        "sessions": sessions_list,
        "recent_turns": turns,
    })


@dashboard_bp.route("/api/focus/context")
async def focus_context():
    """Return recent messages for the selected session, including image data."""
    key = (request.args.get("key") or "").strip() or app_state.current_focus
    if not key:
        return jsonify({"session_key": None, "messages": []})
    messages = await load_chat_messages(key, limit=40)
    return jsonify({"session_key": key, "messages": messages})


@dashboard_bp.route("/api/browser/state")
async def browser_state():
    """Return the latest browser surface rendered into <world> for focus.html."""
    try:
        from browser.session import browser_debug_state, browser_world_view_state

        activity = browser_debug_state()
        world_view = browser_world_view_state()
        world_view["history"] = activity.get("history", [])
        world_view["activity_latest"] = activity.get("latest")
        return jsonify(world_view)
    except Exception:
        logger.warning("browser_state failed", exc_info=True)
        return jsonify({
            "active": False,
            "runtime_active": False,
            "state": "unavailable",
            "source": "world",
            "latest": None,
            "history": [],
            "error": "load failed",
        }), 500


@dashboard_bp.route("/api/browser/image/<image_ref>")
async def browser_image(image_ref: str):
    """Serve browser_control cached image bytes for inline rendering."""
    try:
        from browser.session import browser_image_path

        path = browser_image_path(image_ref)
    except Exception:
        logger.warning("browser image lookup failed ref=%s", image_ref, exc_info=True)
        return jsonify({"error": "load failed"}), 500

    if path is None or not path.is_file():
        return jsonify({"error": "not found"}), 404
    data = await asyncio.to_thread(path.read_bytes)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return Response(data, content_type=mime, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@dashboard_bp.route("/api/sticker/<sticker_id>")
async def sticker_serve(sticker_id: str):
    """Serve sticker image bytes for inline rendering in the focus view."""
    if not sticker_id.isalnum():
        return jsonify({"error": "invalid id"}), 400
    try:
        from llm.media.sticker_collection import load_sticker_bytes
        result = await asyncio.to_thread(load_sticker_bytes, sticker_id)
    except Exception:
        return jsonify({"error": "load failed"}), 500
    if result is None:
        return jsonify({"error": "not found"}), 404
    data, mime = result
    return Response(data, content_type=mime, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
