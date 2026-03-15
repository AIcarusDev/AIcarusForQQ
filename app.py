"""app.py — 主入口

职责：
  - 初始化日志、加载配置
  - 注册蓝图 & 路由
  - NapCat 集成启停
  - 启动 Quart 服务
"""

import asyncio
import logging
import signal
import sys
import traceback
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from quart import Quart, render_template, request, jsonify

from config_loader import load_config, save_model_override
from provider import create_adapter
from schema import RESPONSE_SCHEMA
from tools import TOOL_DECLARATIONS, TOOL_REGISTRY
from session import (
    init_session_globals,
    update_session_model_name,
    create_session,
    get_or_create_session,
    sessions,
    extract_bot_messages,
)
from debug_server import debug_bp, init_debug, broadcast_debug_xml
from napcat_handler import (
    NapcatClient,
    napcat_event_to_context,
    napcat_event_to_debug_xml,
    llm_segments_to_napcat,
    should_respond,
)

load_dotenv()

# ── 日志配置 ──────────────────────────────────────────────
_log_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d\n%(message)s\n",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log_handler = RotatingFileHandler(
    "mita.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_log_handler.setFormatter(_log_formatter)
logging.root.setLevel(logging.DEBUG)
logging.root.addHandler(_log_handler)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_formatter)
logging.root.addHandler(_console_handler)

logger = logging.getLogger("mita.app")

# ── 加载配置 ──────────────────────────────────────────────
config, persona = load_config()

MODEL = config.get("model", "gemini-2.0-flash")
MODEL_NAME = config.get("model_name", MODEL)
GEN = config.get("generation", {})
TIMEZONE = ZoneInfo(config.get("timezone", "Asia/Shanghai"))
MAX_CYCLES = config.get("max_cycles", 3)
MAX_CONTEXT = 20
BOT_NAME = config.get("bot_name", "小懒猫")

adapter = create_adapter(config)

# ── 初始化各子模块 ────────────────────────────────────────
init_session_globals(
    max_context=MAX_CONTEXT,
    timezone=TIMEZONE,
    persona=persona,
    model_name=MODEL_NAME,
)
init_debug(TIMEZONE)

# 创建 Web 默认会话
sessions["web"] = create_session()

# ── Quart App ─────────────────────────────────────────────
app = Quart(__name__)
app.json.sort_keys = False  # type: ignore[attr-defined]
app.register_blueprint(debug_bp)


# ── 核心：调用模型并处理结果 ──────────────────────────────

def call_model_and_process(session):
    """调用模型、更新上下文。

    返回 (result, grounding, system_prompt, user_prompt, repaired, tool_calls_log)。
    """
    # 构建 system_prompt_builder：接受 tool_budget 字典，返回完整 system prompt
    # provider 在工具调用循环中会多次调用它以获取最新配额信息
    system_prompt_builder = lambda tool_budget: session.build_system_prompt(tool_budget=tool_budget)
    chat_log = session.build_chat_log_xml()

    result, grounding, repaired, tool_calls_log = adapter.call(
        system_prompt_builder,
        chat_log,
        GEN,
        RESPONSE_SCHEMA,
        tool_declarations=TOOL_DECLARATIONS,
        tool_registry=TOOL_REGISTRY,
    )

    # 用最终的 system prompt（初始配额）作为返回值，方便调试
    system_prompt = system_prompt_builder({})

    if result is None:
        return None, None, system_prompt, chat_log, False, tool_calls_log

    if session.remaining_cycles <= 0:
        result["cycle_action"] = "stop"

    now_ts = datetime.now(TIMEZONE).isoformat()
    for text in extract_bot_messages(result):
        session.add_to_context({
            "role": "bot",
            "message_id": f"msg_{uuid.uuid4().hex[:8]}",
            "sender_id": "bot",
            "sender_name": BOT_NAME,
            "timestamp": now_ts,
            "content": text,
        })

    session.previous_cycle_json = result
    return result, grounding, system_prompt, chat_log, repaired, tool_calls_log


# ══════════════════════════════════════════════════════════
#  路由
# ══════════════════════════════════════════════════════════

@app.route("/")
async def index():
    return await render_template("index.html")


@app.route("/chat", methods=["POST"])
async def chat():
    session = sessions["web"]

    data = await request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"success": False, "error": "消息不能为空"}), 400

    user_id = data.get("user_id", "user_001")
    user_name = data.get("user_name", "测试用户")
    message_id = f"msg_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now(TIMEZONE).isoformat()

    session.add_to_context({
        "role": "user",
        "message_id": message_id,
        "sender_id": user_id,
        "sender_name": user_name,
        "timestamp": timestamp,
        "content": user_message,
    })

    session.remaining_cycles = MAX_CYCLES

    try:
        result, grounding, system_prompt, user_prompt, repaired, tool_calls_log = (
            await asyncio.to_thread(call_model_and_process, session)
        )
        if result is None:
            logger.warning("[/chat] 模型返回为空（可能被安全过滤拦截）")
            return jsonify({"success": False, "error": "模型返回为空（可能被安全过滤拦截）"}), 502

        return jsonify({
            "success": True,
            "data": result,
            "message_id": message_id,
            "grounding": grounding,
            "remaining_cycles": session.remaining_cycles,
            "json_repaired": repaired,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "tool_calls_log": tool_calls_log,
        })
    except Exception as e:
        logger.error(
            "[/chat] 异常\nuser_message: %s\nuser_id: %s\n%s",
            user_message, user_id, traceback.format_exc(),
        )
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/cycle", methods=["POST"])
async def cycle():
    """主动循环：前端确认上一轮消息渲染后调用。"""
    session = sessions["web"]

    if session.remaining_cycles <= 0:
        return jsonify({"success": False, "error": "没有剩余循环次数"}), 400

    session.remaining_cycles -= 1

    try:
        result, grounding, system_prompt, user_prompt, repaired, tool_calls_log = (
            await asyncio.to_thread(call_model_and_process, session)
        )
        if result is None:
            session.remaining_cycles += 1
            logger.warning(
                "[/cycle] 模型返回为空，remaining_cycles 已回滚至 %d",
                session.remaining_cycles,
            )
            return jsonify({"success": False, "error": "模型返回为空（可能被安全过滤拦截）"}), 502

        return jsonify({
            "success": True,
            "data": result,
            "grounding": grounding,
            "remaining_cycles": session.remaining_cycles,
            "json_repaired": repaired,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "tool_calls_log": tool_calls_log,
        })
    except Exception as e:
        session.remaining_cycles += 1
        logger.error(
            "[/cycle] 异常，remaining_cycles 已回滚至 %d\n%s",
            session.remaining_cycles, traceback.format_exc(),
        )
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/clear", methods=["POST"])
async def clear_context():
    sessions["web"] = create_session()
    return jsonify({"success": True})


@app.route("/config", methods=["GET"])
async def get_config_route():
    return jsonify({
        "provider": config.get("provider", "gemini"),
        "model": MODEL,
        "base_url": config.get("base_url", ""),
    })


@app.route("/models", methods=["POST"])
async def list_models_route():
    data = await request.get_json() or {}
    provider = (data.get("provider") or config.get("provider", "gemini")).strip()
    base_url = (data.get("base_url") or "").strip()

    tmp_cfg = dict(config)
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


@app.route("/switch_provider", methods=["POST"])
async def switch_provider():
    global adapter, MODEL, MODEL_NAME, config

    data = await request.get_json() or {}
    provider = (data.get("provider") or "").strip()
    model = (data.get("model") or "").strip()
    base_url = (data.get("base_url") or "").strip()

    if not provider or not model:
        return jsonify({"success": False, "error": "provider 和 model 不能为空"}), 400

    new_cfg = dict(config)
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

    config = new_cfg
    adapter = new_adapter
    MODEL = model
    MODEL_NAME = model
    update_session_model_name(model)

    save_model_override(provider, model, model, base_url or None)
    return jsonify({"success": True, "provider": provider, "model": model})


# ══════════════════════════════════════════════════════════
#  NapCat 集成
# ══════════════════════════════════════════════════════════

napcat_cfg = config.get("napcat", {})
napcat_enabled = napcat_cfg.get("enabled", False)
napcat_client = NapcatClient(bot_name=BOT_NAME) if napcat_enabled else None


async def _handle_napcat_message(event: dict, conversation_id: str) -> None:
    """NapCat 消息到达时的处理回调。"""
    assert napcat_client is not None
    bot_id = napcat_client.bot_id
    debug_xml = napcat_event_to_debug_xml(event, bot_id=bot_id, timezone=TIMEZONE)
    await broadcast_debug_xml(debug_xml, event)

    if napcat_cfg.get("debug_only", False):
        return

    if not should_respond(event, napcat_client.bot_id, BOT_NAME):
        logger.debug("NapCat 消息不需要回复 (conv=%s)", conversation_id)
        return

    session = get_or_create_session(conversation_id)

    ctx_entry = napcat_event_to_context(event, bot_id=napcat_client.bot_id, timezone=TIMEZONE)
    if not ctx_entry:
        return
    session.add_to_context(ctx_entry)

    session.remaining_cycles = MAX_CYCLES

    try:
        result, _, _, _, _, _ = await asyncio.to_thread(call_model_and_process, session)
    except Exception:
        logger.exception("NapCat LLM 调用失败 (conv=%s)", conversation_id)
        return

    if result is None:
        logger.warning("NapCat LLM 返回为空 (conv=%s)", conversation_id)
        return

    msg_type = event.get("message_type", "")
    group_id = event.get("group_id") if msg_type == "group" else None
    user_id = event.get("sender", {}).get("user_id") if msg_type == "private" else None

    for msg in result.get("messages", []):
        segments = msg.get("segments", [])
        reply_id = msg.get("reply_message_id") or None
        napcat_segs = llm_segments_to_napcat(segments, reply_message_id=reply_id)
        if not napcat_segs:
            continue
        await napcat_client.send_message(
            group_id=group_id, user_id=user_id, message=napcat_segs
        )

    # 主动循环
    while (
        result
        and result.get("cycle_action") == "continue"
        and session.remaining_cycles > 0
    ):
        session.remaining_cycles -= 1
        try:
            result, _, _, _, _, _ = await asyncio.to_thread(call_model_and_process, session)
        except Exception:
            logger.exception("NapCat 主动循环 LLM 调用失败 (conv=%s)", conversation_id)
            break

        if result is None:
            break

        for msg in result.get("messages", []):
            segments = msg.get("segments", [])
            reply_id = msg.get("reply_message_id") or None
            napcat_segs = llm_segments_to_napcat(segments, reply_message_id=reply_id)
            if not napcat_segs:
                continue
            await napcat_client.send_message(
                group_id=group_id, user_id=user_id, message=napcat_segs
            )


if napcat_client:
    napcat_client.set_message_handler(_handle_napcat_message)


# ── 生命周期 ─────────────────────────────────────────────

@app.before_serving
async def startup():
    if napcat_client:
        host = napcat_cfg.get("host", "127.0.0.1")
        port = napcat_cfg.get("port", 8078)
        await napcat_client.start(host=host, port=port)
        logger.info("NapCat 集成已启用，等待连接: ws://%s:%d", host, port)
    else:
        logger.info("NapCat 集成未启用（napcat.enabled = false）")


@app.after_serving
async def shutdown():
    if napcat_client:
        await napcat_client.stop()


# ══════════════════════════════════════════════════════════
#  启动入口
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Windows 下修复 Ctrl+C 无法终止的问题：
    # Quart 内部使用 hypercorn，在 Windows 上 asyncio 事件循环
    # 不会传递 SIGINT，导致 Ctrl+C 被吞掉。
    # 将 SIGINT 恢复为默认的 C 级处理器即可立即终止。
    if sys.platform == "win32":
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    srv = config.get("server", {})
    app.run(debug=srv.get("debug", True), port=srv.get("port", 5000))
