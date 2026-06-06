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

"""routes_chat.py — Web 聊天相关路由

Blueprint：首页、聊天、主动循环、清空上下文、模型切换等。
"""

import asyncio
from copy import deepcopy
import logging
import os
import time
import traceback
import uuid
from datetime import datetime

import aiosqlite
from quart import Blueprint, render_template, request, jsonify, Response, send_file

import app_state
from config_loader import save_model_override
from database import (
    DB_PATH, upsert_chat_session, save_chat_message, save_bot_turn,
    load_chat_sessions, load_recent_bot_turns, load_chat_messages,
)
from llm.core.llm_core import call_model_and_process
from llm.core.provider import create_adapter
from llm.core.profiles import get_model_providers, get_selected_provider_name
from llm.session import create_session, update_session_model_name, sessions

logger = logging.getLogger("AICQ.web.chat")

chat_bp = Blueprint("chat", __name__)
_start_time = time.time()


async def _run_web_model(session, ctx_before, log_tag, extra_fields=None, error_context=""):
    """调用模型并保存结果，供 chat/cycle 端点复用。

    NOTE: web 路径与让 bot 永动思考的 ``consciousness.main_loop`` 是两条独立路径，
    两者共享 ``app_state.llm_lock`` 以串行化 LLM 调用，但 web 不会修改
    ``current_focus``（那是主循环独享的状态）。
    """
    async with app_state.llm_lock:
        try:
            await app_state.rate_limiter.acquire()
            # 模型调用前预先做 FTS5 记忆召回 + 事件召回，结果存入 session
            try:
                await session.prepare_memory_recall()
            except Exception:
                logger.warning("[%s] prepare_memory_recall 失败，跳过本轮召回", log_tag, exc_info=True)
            result, tool_calls_log, system_prompt = (
                await asyncio.to_thread(call_model_and_process, session)
            )
            if isinstance(result, dict) and result.get("aborted_by_runtime_reset"):
                return jsonify({
                    "success": False,
                    "error": "运行时已被紧急恢复，本轮结果已丢弃",
                    "aborted_by_runtime_reset": True,
                }), 409
            if result is None:
                logger.warning("[%s] 模型返回为空", log_tag)
                return jsonify({"success": False, "error": "模型返回为空（可能被安全过滤拦截）"}), 502

            for _entry in session.context_messages[ctx_before:]:
                await save_chat_message("web", _entry)
            await upsert_chat_session("web", session.conv_type, session.conv_id, session.conv_name)
            await save_bot_turn(
                turn_id=uuid.uuid4().hex,
                conv_type=session.conv_type,
                conv_id=session.conv_id,
                result=result,
                tool_calls_log=tool_calls_log,
            )

            resp = {
                "success": True,
                "data": result,
                "system_prompt": system_prompt,
                "tool_calls_log": tool_calls_log,
            }
            if extra_fields:
                resp.update(extra_fields)
            return jsonify(resp)
        except Exception as e:
            logger.error("[%s] 异常\n%s%s", log_tag, error_context, traceback.format_exc())
            return jsonify({"success": False, "error": str(e)}), 500


@chat_bp.route("/")
async def home():
    return await render_template("home.html", active_page="home")


@chat_bp.route("/test")
async def test_page():
    return await render_template("index.html", active_page="test")


@chat_bp.route("/api/status")
async def api_status():
    """仪表盘状态 API — 供 home.html 轮询。"""
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

            # Today's outgoing messages (bot_turns created today)
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
        "current_focus":  app_state.current_focus,
        "today_messages": today_messages,
        "memory_counts":  memory_counts,
        "uptime_seconds": uptime_sec,
        "bot_name":       app_state.BOT_NAME,
        "model":          app_state.MODEL,
    })


@chat_bp.route("/chat", methods=["POST"])
async def chat():
    session = sessions["web"]

    data = await request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"success": False, "error": "消息不能为空"}), 400

    user_id = data.get("user_id", "user_001")
    user_name = data.get("user_name", "测试用户")
    message_id = f"msg_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now(app_state.TIMEZONE).isoformat()

    # 同步 conv_name / conv_id，确保 prompt 里 <other> 标签与前端输入一致
    session.set_conversation_meta("private", user_id, user_name)

    ctx_before = len(session.context_messages)
    session.add_to_context({
        "role": "user",
        "message_id": message_id,
        "sender_id": user_id,
        "sender_name": user_name,
        "sender_role": "",
        "timestamp": timestamp,
        "content": user_message,
        "content_type": "text",
    })

    if app_state.llm_lock.locked():
        return jsonify({"success": False, "error": "机器人正忙，请稍后再试"}), 429

    return await _run_web_model(
        session, ctx_before,
        log_tag="/chat",
        extra_fields={"message_id": message_id},
        error_context=f"user_message: {user_message}\nuser_id: {user_id}\n",
    )


@chat_bp.route("/cycle", methods=["POST"])
async def cycle():
    """主动循环：前端确认上一轮消息渲染后调用。"""
    session = sessions["web"]

    ctx_before = len(session.context_messages)

    if app_state.llm_lock.locked():
        return jsonify({"success": False, "error": "机器人正忙，请稍后再试"}), 429

    return await _run_web_model(session, ctx_before, log_tag="/cycle")


@chat_bp.route("/clear", methods=["POST"])
async def clear_context():
    _s = create_session()
    _s.set_conversation_meta("private", "web_user", "网页用户")
    sessions["web"] = _s
    return jsonify({"success": True})


@chat_bp.route("/config", methods=["GET"])
async def get_config_route():
    provider = get_selected_provider_name(app_state.config)
    return jsonify({
        "provider": provider,
        "model_providers": get_model_providers(app_state.config),
        "model": app_state.MODEL,
    })


@chat_bp.route("/models", methods=["POST"])
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


@chat_bp.route("/switch_provider", methods=["POST"])
async def switch_provider():
    data = await request.get_json() or {}
    provider = (data.get("provider") or "").strip()
    model = (data.get("model") or "").strip()

    if not provider or not model:
        return jsonify({"success": False, "error": "provider 和 model 不能为空"}), 400

    new_cfg = deepcopy(app_state.config)
    new_cfg["provider"] = provider
    new_cfg["model"] = model
    new_cfg["model_name"] = model

    try:
        new_adapter = create_adapter(new_cfg)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        return jsonify({"success": False, "error": f"服务端缺少依赖: {e}"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    # 意识流独立于 adapter，热重载时无需迁移
    app_state.config = new_cfg
    app_state.adapter = new_adapter
    app_state.MODEL = model
    app_state.MODEL_NAME = model
    update_session_model_name(model)

    save_model_override(provider, model, model)
    return jsonify({"success": True, "provider": provider, "model": model})


# ── 焦点视图 ──────────────────────────────────────────────────────────────────

@chat_bp.route("/focus")
async def focus_page():
    from runtime.emergency_reset import expected_confirmation
    return await render_template(
        "focus.html",
        active_page="focus",
        emergency_reset_confirmation=expected_confirmation(),
    )


@chat_bp.route("/api/focus/state")
async def focus_state():
    """焦点状态 API：返回 current_focus + 已知会话列表 + 最近 15 轮 bot_turns。"""
    sessions_list = await load_chat_sessions()
    turns = await load_recent_bot_turns(limit=15)
    return jsonify({
        "current_focus": app_state.current_focus,
        "sessions": sessions_list,
        "recent_turns": turns,
    })


@chat_bp.route("/api/focus/context")
async def focus_context():
    """返回指定会话（默认为焦点会话）的最近 40 条消息，含图片数据。"""
    key = (request.args.get("key") or "").strip() or app_state.current_focus
    if not key:
        return jsonify({"session_key": None, "messages": []})
    messages = await load_chat_messages(key, limit=40)
    return jsonify({"session_key": key, "messages": messages})


@chat_bp.route("/api/browser/state")
async def browser_state():
    """返回浏览器工具最近截图和操作历史，用于 WebUI 观察面板。"""
    try:
        from tools.browser_session import browser_debug_state

        return jsonify(browser_debug_state())
    except Exception as exc:
        return jsonify({"active": False, "latest": None, "history": [], "error": str(exc)}), 500


@chat_bp.route("/api/browser/image/<image_ref>")
async def browser_image(image_ref: str):
    """按 image_ref 返回浏览器视口截图或缓存图片。"""
    from tools.browser_session import browser_image_path

    path = browser_image_path(image_ref)
    if path is None or not path.exists():
        return jsonify({"error": "image not found"}), 404
    return await send_file(path)


@chat_bp.route("/api/sticker/<sticker_id>")
async def sticker_serve(sticker_id: str):
    """直接提供表情包图片（供焦点视图 inline 渲染）。"""
    # 防止路径穿越：sticker_id 只允许字母数字
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
