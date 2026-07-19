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
import hashlib
import logging
import mimetypes
import os
import time
import uuid
from datetime import datetime

import aiosqlite
from quart import Blueprint, Response, jsonify, render_template, request

import app_state
from database import (
    DB_PATH,
    get_chat_message_by_id,
    get_existing_chat_message_ids,
    load_chat_messages,
    load_chat_sessions,
    load_recent_bot_turns,
    save_chat_message,
    upsert_chat_session,
)
from llm.core.profiles import get_model_providers
from llm.session import get_or_create_session, sessions
from platforms.core import CORE_MAIN_FOCUS
from platforms.focus import current_focus_key

logger = logging.getLogger("AICQ.web.dashboard")

dashboard_bp = Blueprint("dashboard", __name__)
_start_time = time.time()
_core_chat_send_lock = asyncio.Lock()


@dashboard_bp.route("/")
async def home():
    return await render_template("home.html", active_page="home")


@dashboard_bp.route("/api/status")
async def api_status():
    """Dashboard status API used by home.html polling."""
    uptime_sec = int(time.time() - _start_time)

    memory_counts = {"events": 0, "predicates": 0, "participants": 0, "relations": 0, "sources": 0, "cognition_sources": 0}
    today_messages = 0

    try:
        from memory.repo.events import ensure_schema as _ensure_memory_schema

        await _ensure_memory_schema()
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            for tbl, key in (
                ("MemoryEvents", "events"),
                ("MemoryPredicates", "predicates"),
                ("MemoryParticipants", "participants"),
                ("MemoryRelations", "relations"),
                ("MemoryEventSources", "sources"),
                ("CognitionSources", "cognition_sources"),
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
        "current_focus": current_focus_key(app_state.current_focus),
        "today_messages": today_messages,
        "memory_counts": memory_counts,
        "uptime_seconds": uptime_sec,
        "self_name": app_state.SELF_NAME,
        "model": app_state.MODEL,
    })


@dashboard_bp.route("/models", methods=["POST"])
async def list_models_route():
    data = await request.get_json() or {}

    try:
        base_url, api_key = _resolve_model_discovery_target(
            data,
            getattr(app_state, "config", {}) or {},
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc), "models": []}), 400

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


def _resolve_model_discovery_target(
    request_data: dict,
    config: dict,
    environ=None,
) -> tuple[str, str]:
    """Resolve model discovery from the saved provider, with an optional key override."""
    provider_id = _payload_text(request_data.get("provider"))
    provider = get_model_providers(config).get(provider_id)
    if provider is None:
        raise ValueError("未知的模型供应商，请先保存供应商配置")

    base_url = provider.get("base_url", "")
    if not base_url:
        raise ValueError(f"模型供应商 {provider_id!r} 未配置 base_url")

    explicit_api_key = _payload_text(request_data.get("api_key"))
    env_name = provider.get("api_key_env", "")
    environment = os.environ if environ is None else environ
    saved_api_key = _payload_text(environment.get(env_name)) if env_name else ""
    api_key = explicit_api_key or saved_api_key

    if not api_key:
        if provider.get("requires_api_key", True):
            raise ValueError(f"模型供应商 {provider_id!r} 尚未设置 API Key")
        api_key = "openai-compat"

    return base_url, api_key


def _payload_text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


@dashboard_bp.route("/focus")
async def focus_page():
    return await render_template(
        "focus.html",
        active_page="focus",
    )


@dashboard_bp.route("/chat")
async def chat_page():
    cfg = getattr(app_state, "config", {}) or {}
    platforms = cfg.get("platforms", {}) if isinstance(cfg.get("platforms", {}), dict) else {}
    core_cfg = platforms.get("core", {}) if isinstance(platforms.get("core", {}), dict) else {}
    return await render_template(
        "chat.html",
        active_page="chat",
        self_name=getattr(app_state, "SELF_NAME", ""),
        core_account_name=core_cfg.get("account_name", ""),
        core_account_id=core_cfg.get("account_id", ""),
    )


def _guardian_meta() -> tuple[str, str]:
    return "guardian", "监护人"


def _wake_for_core_message() -> None:
    focus_key = current_focus_key(app_state.current_focus)
    if not focus_key:
        return
    focus_session = sessions.get(focus_key)
    if focus_session is None:
        return
    wake_remark = "收到 core 平台消息"
    hub = getattr(app_state, "runtime_event_hub", None)
    loop = getattr(app_state, "main_loop", None)
    if hub is not None and loop is not None and loop.is_running():
        hub.publish_threadsafe(
            loop,
            {"type": "attention", "reason": wake_remark, "from": CORE_MAIN_FOCUS.key()},
            target=focus_key,
        )
        return
    focus_session.last_wake_reason = wake_remark
    focus_session.sleep_wake_from = CORE_MAIN_FOCUS.key()
    if focus_session.sleep_wake_event is not None:
        focus_session.sleep_wake_event.set()
        return
    focus_session.sleep_pending_wake = True
    focus_session.sleep_pending_wake_at = time.time()


@dashboard_bp.route("/api/core/chat", methods=["GET"])
async def core_chat_messages():
    """Return recent DB-backed messages for the Core WebUI chat page."""
    try:
        limit = max(1, min(200, int(request.args.get("limit", "80"))))
    except (TypeError, ValueError):
        limit = 80
    messages = await load_chat_messages(CORE_MAIN_FOCUS.key(), limit=limit)
    return jsonify({
        "session_key": CORE_MAIN_FOCUS.key(),
        "messages": messages,
    })


@dashboard_bp.route("/api/core/chat", methods=["POST"])
async def core_chat_send():
    """Persist a guardian message into the Core platform conversation."""
    data = await request.get_json() or {}
    content = str(data.get("content") or data.get("text") or "").strip()
    if not content:
        return jsonify({"ok": False, "error": "消息内容不能为空"}), 400

    raw_client_id = data.get("client_id")
    client_id = str(raw_client_id or "").strip()
    if raw_client_id is not None and not client_id:
        return jsonify({"ok": False, "error": "client_id 不能为空"}), 400
    if len(client_id) > 128:
        return jsonify({"ok": False, "error": "client_id 不能超过 128 个字符"}), 400

    if client_id:
        digest = hashlib.sha256(client_id.encode("utf-8")).hexdigest()[:32]
        message_id = f"core_ui_{digest}"
    else:
        # 旧版客户端未携带 client_id 时继续保持原有行为。
        message_id = f"core_{uuid.uuid4().hex}"

    guardian_id, guardian_name = _guardian_meta()
    timestamp = datetime.now(getattr(app_state, "TIMEZONE", None)).isoformat()
    entry = {
        "role": "user",
        "message_id": message_id,
        "sender_id": guardian_id,
        "sender_name": guardian_name,
        "timestamp": timestamp,
        "content": content,
        "content_type": "text",
        "content_segments": [{"type": "text", "text": content}],
    }

    async with _core_chat_send_lock:
        if client_id:
            existing_ids = await get_existing_chat_message_ids(
                CORE_MAIN_FOCUS.key(),
                [message_id],
            )
            if message_id in existing_ids:
                existing = await get_chat_message_by_id(message_id)
                return jsonify({
                    "ok": True,
                    "duplicate": True,
                    "client_id": client_id,
                    "session_key": CORE_MAIN_FOCUS.key(),
                    "message": existing or entry,
                })

        session = get_or_create_session(CORE_MAIN_FOCUS)
        if not session.conv_type:
            session.set_conversation_meta(
                CORE_MAIN_FOCUS.target_type,
                CORE_MAIN_FOCUS.target_id,
                CORE_MAIN_FOCUS.target_name,
                platform=CORE_MAIN_FOCUS.platform,
            )
        session.add_to_context(entry)
        session.mark_unread_message(entry["message_id"])
        await save_chat_message(CORE_MAIN_FOCUS.key(), entry)
        await upsert_chat_session(
            CORE_MAIN_FOCUS.key(),
            CORE_MAIN_FOCUS.target_type,
            CORE_MAIN_FOCUS.target_id,
            CORE_MAIN_FOCUS.target_name,
        )

        if app_state.current_focus is None:
            from consciousness import trigger_first_activation

            trigger_first_activation(initial_focus=CORE_MAIN_FOCUS)
        else:
            _wake_for_core_message()
            first_input_event = getattr(app_state, "first_input_event", None)
            if first_input_event is not None:
                first_input_event.set()

    return jsonify({
        "ok": True,
        "duplicate": False,
        "client_id": client_id or None,
        "session_key": CORE_MAIN_FOCUS.key(),
        "message": entry,
    })


@dashboard_bp.route("/api/focus/state")
async def focus_state():
    """Focus state API: current focus, known sessions, and recent bot turns."""
    sessions_list = await load_chat_sessions()
    turns = await load_recent_bot_turns(limit=15)
    return jsonify({
        "current_focus": current_focus_key(app_state.current_focus),
        "sessions": sessions_list,
        "recent_turns": turns,
    })


@dashboard_bp.route("/api/focus/context")
async def focus_context():
    """Return recent messages for the selected session, including image data."""
    key = (request.args.get("key") or "").strip() or current_focus_key(app_state.current_focus)
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
        logger.warning("browser image lookup failed image_ref=%s", image_ref, exc_info=True)
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
