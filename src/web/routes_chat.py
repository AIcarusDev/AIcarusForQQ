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
import logging
import time
import traceback
import uuid
from datetime import datetime

import aiosqlite
from quart import Blueprint, render_template, request, jsonify

import app_state
from config_loader import save_model_override
from database import DB_PATH, upsert_chat_session, save_chat_message, save_bot_turn
from llm.core.llm_core import call_model_and_process, commit_bot_messages_web
from llm.core.provider import create_adapter
from llm.session import create_session, update_session_model_name, sessions

logger = logging.getLogger("AICQ.app")

chat_bp = Blueprint("chat", __name__)
_start_time = time.time()


async def _run_web_model(session, ctx_before, log_tag, extra_fields=None, error_context=""):
    """调用模型并保存结果，供 chat/cycle 端点复用。"""
    async with app_state.consciousness_lock:
        app_state.current_focus = "web"
        try:
            await app_state.rate_limiter.acquire()
            result, grounding, system_prompt, user_prompt, repaired, tool_calls_log = (
                await asyncio.to_thread(call_model_and_process, session)
            )
            if result is None:
                logger.warning("[%s] 模型返回为空", log_tag)
                return jsonify({"success": False, "error": "模型返回为空（可能被安全过滤拦截）"}), 502

            commit_bot_messages_web(session, result)

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
                "grounding": grounding,
                "json_repaired": repaired,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "tool_calls_log": tool_calls_log,
            }
            if extra_fields:
                resp.update(extra_fields)
            return jsonify(resp)
        except Exception as e:
            logger.error("[%s] 异常\n%s%s", log_tag, error_context, traceback.format_exc())
            return jsonify({"success": False, "error": str(e)}), 500
        finally:
            app_state.current_focus = None


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

    memory_counts = {"persons": 0, "accounts": 0, "groups": 0, "sessions": 0}
    today_messages = 0
    recent_activity: list = []

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            for tbl, key in (("persons", "persons"), ("accounts", "accounts"),
                             ("groups", "groups"), ("chat_sessions", "sessions")):
                async with db.execute(f"SELECT COUNT(*) AS n FROM {tbl}") as cur:
                    row = await cur.fetchone()
                    memory_counts[key] = row["n"] if row else 0

            # Today's outgoing messages (bot_turns created today)
            today_start = int(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
            async with db.execute(
                "SELECT COUNT(*) AS n FROM bot_turns WHERE created_at >= ?", (today_start,)
            ) as cur:
                row = await cur.fetchone()
                today_messages = row["n"] if row else 0

            # Recent activity log
            try:
                async with db.execute(
                    "SELECT entry_type, enter_remark, created_at FROM activity_log ORDER BY created_at DESC LIMIT 10"
                ) as cur:
                    async for row in cur:
                        recent_activity.append({
                            "action":     row["entry_type"],
                            "detail":     row["enter_remark"],
                            "created_at": row["created_at"],
                        })
            except Exception:
                pass  # activity_log table may not exist yet

    except Exception as e:
        logger.warning("api_status DB query failed: %s", e)

    return jsonify({
        "current_focus":  app_state.current_focus,
        "today_messages": today_messages,
        "memory_counts":  memory_counts,
        "uptime_seconds": uptime_sec,
        "bot_name":       app_state.BOT_NAME,
        "model":          app_state.MODEL,
        "recent_activity": recent_activity,
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

    if app_state.consciousness_lock.locked():
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

    if app_state.consciousness_lock.locked():
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
    return jsonify({
        "provider": app_state.config.get("provider", "gemini"),
        "model": app_state.MODEL,
        "base_url": app_state.config.get("base_url", ""),
    })


@chat_bp.route("/models", methods=["POST"])
async def list_models_route():
    data = await request.get_json() or {}
    provider = (data.get("provider") or app_state.config.get("provider", "gemini")).strip()
    base_url = (data.get("base_url") or "").strip()

    tmp_cfg = dict(app_state.config)
    tmp_cfg["provider"] = provider
    if base_url:
        tmp_cfg["base_url"] = base_url
    elif "base_url" in tmp_cfg:
        del tmp_cfg["base_url"]

    try:
        tmp_adapter = create_adapter(tmp_cfg)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e), "models": []}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "models": []}), 500

    try:
        models = tmp_adapter.list_models()
        return jsonify({"success": True, "models": models})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "models": []}), 500


@chat_bp.route("/switch_provider", methods=["POST"])
async def switch_provider():
    data = await request.get_json() or {}
    provider = (data.get("provider") or "").strip()
    model = (data.get("model") or "").strip()
    base_url = (data.get("base_url") or "").strip()

    if not provider or not model:
        return jsonify({"success": False, "error": "provider 和 model 不能为空"}), 400

    new_cfg = dict(app_state.config)
    new_cfg["provider"] = provider
    new_cfg["model"] = model
    new_cfg["model_name"] = model
    if base_url:
        new_cfg["base_url"] = base_url
    elif "base_url" in new_cfg:
        del new_cfg["base_url"]

    try:
        new_adapter = create_adapter(new_cfg)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        return jsonify({"success": False, "error": f"服务端缺少依赖: {e}"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    app_state.config = new_cfg
    app_state.adapter = new_adapter
    app_state.MODEL = model
    app_state.MODEL_NAME = model
    update_session_model_name(model)

    save_model_override(provider, model, model, base_url or None)
    return jsonify({"success": True, "provider": provider, "model": model})
