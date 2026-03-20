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
import time
import traceback
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from quart import Quart, render_template, request, jsonify

from config_loader import load_config, save_config, save_persona, save_model_override, read_env_keys, save_env_key, read_env_proxies, save_env_proxy
from provider import create_adapter
from schema import RESPONSE_SCHEMA
from tools import build_tools
from vision_bridge import VisionBridge
from image_cache import evict_cache
from session import (
    init_session_globals,
    update_session_model_name,
    update_bot_info,
    create_session,
    get_or_create_session,
    sessions,
    extract_bot_messages,
)
from debug_server import debug_bp, init_debug, broadcast_debug_xml
from napcat import (
    NapcatClient,
    napcat_event_to_context,
    napcat_event_to_debug_xml,
    llm_segments_to_napcat,
    should_respond,
)
from database import (
    init_db,
    get_bot_self,
    upsert_bot_self,
    get_group_info,
    upsert_group,
    upsert_account,
    upsert_membership,
    get_display_name,
    get_person_map,
)
from memory import init_memory_db, run_memory_pipeline
from log_config import setup_logging

load_dotenv()

# ── 日志配置 ──────────────────────────────────────────────
setup_logging()

logger = logging.getLogger("AICQ.app")

# ── 加载配置 ──────────────────────────────────────────────
config, persona = load_config()

MODEL = config.get("model", "gemini-2.0-flash")
MODEL_NAME = config.get("model_name", MODEL)
GEN = config.get("generation", {})
TIMEZONE = ZoneInfo((config.get("timezone") or "").strip() or "Asia/Shanghai")
MAX_CYCLES = config.get("max_cycles", 3)
MAX_CONTEXT = 20
BOT_NAME = config.get("bot_name", "小懒猫")

adapter = create_adapter(config)

# ── 视觉桥（图片描述缓存）────────────────────────────────
vision_bridge = VisionBridge(config.get("vision_bridge", {}))

# ── 初始化各子模块 ────────────────────────────────────────
init_session_globals(
    max_context=MAX_CONTEXT,
    timezone=TIMEZONE,
    persona=persona,
    model_name=MODEL_NAME,
)
# 创建 Web 默认会话（按私聊格式）
_web_session = create_session()
_web_session.set_conversation_meta("private", "web_user", "网页用户")
sessions["web"] = _web_session

# ── Quart App ─────────────────────────────────────────────
app = Quart(__name__)
app.json.sort_keys = False  # type: ignore[attr-defined]
app.register_blueprint(debug_bp)


# ── 核心：调用模型并处理结果 ──────────────────────────────

def call_model_and_process(session):
    """调用模型、更新上下文。

    返回 (result, grounding, system_prompt, user_prompt_display, repaired, tool_calls_log)。
    """
    # 构建 system_prompt_builder：接受 tool_budget 字典，返回完整 system prompt
    # provider 在工具调用循环中会多次调用它以获取最新配额信息
    def system_prompt_builder(tool_budget, rounds_used=0, max_rounds=None, tool_budget_suffix=""):
        return session.build_system_prompt(tool_budget=tool_budget, rounds_used=rounds_used, max_rounds=max_rounds, tool_budget_suffix=tool_budget_suffix)
    chat_log = session.build_chat_log_xml()
    chat_log_display = session.get_chat_log_display()

    # 按会话上下文动态扩展工具集
    # get_group_members 需要 napcat_client + group_id，非群聊时传 None 自动跳过
    # get_self_image 的 vision 条件判断已内置在工具模块的 condition() 中
    # examine_image 需要 session + vision_bridge（未启用时传 None 自动跳过）
    logger.info("[app] 构建工具集开始 conv_type=%s", session.conv_type)
    tool_declarations, tool_registry = build_tools(
        config,
        napcat_client=napcat_client if session.conv_type == "group" else None,
        group_id=session.conv_id if session.conv_type == "group" else None,
        session=session,
        vision_bridge=vision_bridge if vision_bridge.enabled else None,
    )
    logger.info("[app] 构建工具集完成 tools_count=%d", len(tool_declarations))

    logger.info("[app] LLM 调用开始 model=%s provider=%s", MODEL, adapter.provider)
    result, grounding, repaired, tool_calls_log, system_prompt = adapter.call(
        system_prompt_builder,
        chat_log,
        GEN,
        RESPONSE_SCHEMA,
        tool_declarations=tool_declarations,
        tool_registry=tool_registry,
    )

    if result is None:
        logger.warning("[app] LLM 调用失败，返回 None")
        return None, None, system_prompt, chat_log_display, False, tool_calls_log
    
    logger.info("[app] LLM 调用完成 repaired=%s tool_calls=%d", repaired, len(tool_calls_log))

    if session.remaining_cycles <= 0:
        result["loop_control"] = "break"
        logger.info("[app] 对话循环已达上限，标记为中断")

    now_ts = datetime.now(TIMEZONE).isoformat()
    bot_sender_id = session._qq_id or "bot"
    bot_sender_name = session._qq_name or BOT_NAME
    bot_msg_count = 0
    for bot_msg in extract_bot_messages(result):
        bot_msg_count += 1
        session.add_to_context({
            "role": "bot",
            "message_id": f"msg_{uuid.uuid4().hex[:8]}",
            "sender_id": bot_sender_id,
            "sender_name": bot_sender_name,
            "sender_role": "",
            "timestamp": now_ts,
            "content": bot_msg["text"],
            "content_type": "text",
            "content_segments": bot_msg["content_segments"],
        })
    
    logger.info("[app] 提取机器人消息完成 count=%d", bot_msg_count)

    session.previous_cycle_json = result
    return result, grounding, system_prompt, chat_log_display, repaired, tool_calls_log


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
        "sender_role": "",
        "timestamp": timestamp,
        "content": user_message,
        "content_type": "text",
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
    _s = create_session()
    _s.set_conversation_meta("private", "web_user", "网页用户")
    sessions["web"] = _s
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
#  Settings WebUI
# ══════════════════════════════════════════════════════════

@app.route("/settings")
async def settings_page():
    return await render_template("settings.html")


@app.route("/settings/full", methods=["GET"])
async def settings_get():
    """返回完整配置供前端填充表单。"""
    cfg = dict(config)
    # 不把 base_url 留空 key
    return jsonify({
        "provider": cfg.get("provider", "gemini"),
        "model": cfg.get("model", ""),
        "model_name": cfg.get("model_name", ""),
        "base_url": cfg.get("base_url", ""),
        "vision": cfg.get("vision", True),
        "vision_bridge": cfg.get("vision_bridge", {}),
        "generation": cfg.get("generation", {}),
        "thinking": cfg.get("thinking", {}),
        "max_cycles": cfg.get("max_cycles", 5),
        "bot_name": cfg.get("bot_name", ""),
        "timezone": cfg.get("timezone", "Asia/Shanghai"),
        "napcat": cfg.get("napcat", {}),
        "persona": persona,
        "api_keys": read_env_keys(),
        "proxies": read_env_proxies(),
    })


@app.route("/settings/full", methods=["POST"])
async def settings_save():
    """保存完整配置：写 config.yaml、persona.md、.env API Key，热重载 adapter。"""
    global adapter, MODEL, MODEL_NAME, config, persona, vision_bridge

    data = await request.get_json() or {}

    # ── 写 API Key（只写非掩码值）──────────────────────────
    for key_name in ("GEMINI_API_KEY", "SILICONFLOW_API_KEY", "BIGMODEL_API_KEY", "VISION_BRIDGE_API_KEY"):
        val = (data.get("api_keys") or {}).get(key_name, "")
        if val:
            try:
                save_env_key(key_name, val)
            except ValueError:
                pass
    
    # ── 写代理配置到 .env（总是处理，包括空值用于删除）────────────────────
    for proxy_name in ("GEMINI_PROXY", "OPENAI_PROXY", "TAVILY_PROXY"):
        val = (data.get("proxies") or {}).get(proxy_name, "")
        try:
            save_env_proxy(proxy_name, val)
        except ValueError:
            pass
    
    load_dotenv(override=True)  # 重新载入 .env 到 os.environ

    # ── 构建新 config ──────────────────────────────────────
    new_cfg = dict(config)
    if "provider" in data:
        new_cfg["provider"] = data["provider"]
    if "model" in data:
        new_cfg["model"] = data["model"]
    if "model_name" in data:
        new_cfg["model_name"] = data["model_name"] or data.get("model", new_cfg.get("model", ""))
    if "base_url" in data:
        if data["base_url"]:
            new_cfg["base_url"] = data["base_url"]
        elif "base_url" in new_cfg:
            del new_cfg["base_url"]
    if "generation" in data and isinstance(data["generation"], dict):
        new_cfg["generation"] = data["generation"]
    if "thinking" in data and isinstance(data["thinking"], dict):
        new_cfg["thinking"] = data["thinking"]
    if "max_cycles" in data:
        new_cfg["max_cycles"] = int(data["max_cycles"])
    if "bot_name" in data:
        new_cfg["bot_name"] = data["bot_name"]
    if "timezone" in data:
        tz_val = (data.get("timezone") or "").strip() or "Asia/Shanghai"
        new_cfg["timezone"] = tz_val
    if "napcat" in data and isinstance(data["napcat"], dict):
        new_cfg["napcat"] = data["napcat"]
    if "vision" in data:
        new_cfg["vision"] = bool(data["vision"])
    if "vision_bridge" in data and isinstance(data["vision_bridge"], dict):
        vb_data = data["vision_bridge"]
        new_vb = dict(new_cfg.get("vision_bridge", {}))
        if "enabled" in vb_data:
            new_vb["enabled"] = bool(vb_data["enabled"])
        if "base_url" in vb_data:
            new_vb["base_url"] = vb_data["base_url"]
        if "model" in vb_data:
            new_vb["model"] = vb_data["model"]
        new_vb["api_key_env"] = "VISION_BRIDGE_API_KEY"
        new_cfg["vision_bridge"] = new_vb

    # ── 热重载 adapter ────────────────────────────────────
    try:
        new_adapter = create_adapter(new_cfg)
    except Exception as e:
        return jsonify({"success": False, "error": f"adapter 初始化失败: {e}"}), 400

    # ── 写 persona.md ─────────────────────────────────────
    new_persona = data.get("persona", persona)
    save_persona(new_persona)

    # ── 写 config.yaml ────────────────────────────────────
    save_config(new_cfg)

    # ── 应用到运行时 ──────────────────────────────────────
    config = new_cfg
    adapter = new_adapter
    persona = new_persona
    MODEL = new_cfg.get("model", MODEL)
    MODEL_NAME = new_cfg.get("model_name", MODEL_NAME)
    vision_bridge = VisionBridge(new_cfg.get("vision_bridge", {}))
    update_session_model_name(MODEL_NAME)
    init_session_globals(
        max_context=MAX_CONTEXT,
        timezone=ZoneInfo(new_cfg["timezone"]),
        persona=new_persona,
        model_name=MODEL_NAME,
    )

    return jsonify({"success": True})


# ══════════════════════════════════════════════════════════
#  NapCat 集成
# ══════════════════════════════════════════════════════════

napcat_cfg = config.get("napcat", {})
napcat_enabled = napcat_cfg.get("enabled", False)
napcat_client = NapcatClient(bot_name=BOT_NAME) if napcat_enabled else None
init_debug(TIMEZONE, napcat_client)


async def _send_decision_messages(
    decision: dict,
    group_id,
    user_id,
    pending_bot_entries: list,
    llm_elapsed: float,
) -> None:
    """将 decision 中的 send_messages 逐条发送到 NapCat，并回填真实消息 ID。"""
    assert napcat_client is not None
    _first_msg = True
    for msg in decision.get("send_messages") or []:
        segments = msg.get("segments", [])
        reply_id = msg.get("quote") or None
        napcat_segs = llm_segments_to_napcat(segments, reply_message_id=reply_id)
        if not napcat_segs:
            continue
        # 判断此消息是否有文本内容（与 extract_bot_messages 逻辑一致，有则对应一条上下文条目）
        msg_has_text = bool("".join(
            seg.get("params", {}).get("content", "") if seg.get("command") == "text"
            else f"@{seg.get('params', {}).get('user_id', '')}" if seg.get("command") == "at"
            else ""
            for seg in segments
        ))
        send_result = await napcat_client.send_message(
            group_id=group_id, user_id=user_id, message=napcat_segs,
            llm_elapsed=llm_elapsed if _first_msg else 0.0,
        )
        _first_msg = False
        if msg_has_text and pending_bot_entries:
            entry = pending_bot_entries.pop(0)
            if send_result and send_result.get("message_id") is not None:
                entry["message_id"] = str(send_result["message_id"])


async def _handle_napcat_message(event: dict, conversation_id: str) -> None:
    """NapCat 消息到达时的处理回调。"""
    assert napcat_client is not None
    bot_id = napcat_client.bot_id
    debug_xml = await napcat_event_to_debug_xml(event, bot_id=bot_id, timezone=TIMEZONE)
    await broadcast_debug_xml(debug_xml, event)

    if napcat_cfg.get("debug_only", False):
        return

    # 白名单过滤：私聊用户 + 群组
    whitelist_cfg = napcat_cfg.get("whitelist", {})
    private_whitelist = [str(u) for u in whitelist_cfg.get("private_users", [])]
    group_whitelist = [str(g) for g in whitelist_cfg.get("group_ids", [])]
    msg_type = event.get("message_type", "")
    sender_id = str(event.get("sender", {}).get("user_id", ""))
    group_id_str = str(event.get("group_id", ""))
    if msg_type == "private":
        if private_whitelist and sender_id not in private_whitelist:
            logger.debug("私聊来自非白名单用户 %s，忽略", sender_id)
            return
    elif msg_type == "group":
        if group_whitelist and group_id_str not in group_whitelist:
            logger.debug("群聊来自非白名单群组 %s，忽略", group_id_str)
            return
    else:
        logger.debug("未知消息类型 %s，忽略 (conv=%s)", msg_type, conversation_id)
        return

    # 纯多模态消息（无文字）暂不处理
    message_segs = event.get("message", [])
    has_real_text = any(
        seg.get("type") == "text" and seg.get("data", {}).get("text", "").strip()
        for seg in message_segs
    )
    has_image = any(seg.get("type") == "image" for seg in message_segs)
    has_unhandled_media_only = (
        not has_real_text
        and not has_image
        and any(seg.get("type") in ("record", "video") for seg in message_segs)
    )
    if has_unhandled_media_only:
        logger.debug("纯语音/视频消息，暂不处理 (conv=%s)", conversation_id)
        return

    need_respond = should_respond(event, napcat_client.bot_id, BOT_NAME)
    if not need_respond:
        logger.debug("NapCat 消息不触发回复，静默记入上下文 (conv=%s)", conversation_id)

    session = get_or_create_session(conversation_id)

    # 设置/更新会话元信息（群名、私聊昵称等）；同步发送者信息到 DB
    msg_type = event.get("message_type", "")
    sender = event.get("sender", {})
    sender_id = str(sender.get("user_id", ""))
    sender_nickname = sender.get("nickname", "")
    if msg_type == "group":
        group_id = str(event.get("group_id", ""))
        sender_card = sender.get("card", "") or sender_nickname
        sender_role = sender.get("role", "member")
        sender_title = sender.get("title", "")
        if not session.conv_type:
            group_name, member_count, bot_card = await get_group_info(group_id)
            session.set_conversation_meta("group", group_id, group_name, member_count)
            session._qq_card = bot_card
        # 懒同步：每次收到消息时更新发送者的账号和群成员关系
        await upsert_membership(
            "qq", sender_id, group_id,
            cardname=sender_card,
            title=sender_title,
            permission_level=sender_role,
        )
    elif msg_type == "private":
        peer_id = str(sender.get("user_id", ""))
        peer_name = sender.get("nickname", "")
        if not session.conv_type:
            session.set_conversation_meta("private", peer_id, peer_name)
        # 懒同步：更新私聊对方的账号信息
        await upsert_account("qq", peer_id, nickname=peer_name)

    bot_display = session._qq_card or session._qq_name or ""
    ctx_entry = await napcat_event_to_context(event, bot_id=napcat_client.bot_id, bot_display_name=bot_display, timezone=TIMEZONE)
    if not ctx_entry:
        return

    # 图片落盘 + pHash 去重 + 视觉描述（有图时在后台线程执行，不阻塞事件循环）
    if ctx_entry.get("images"):
        await asyncio.to_thread(vision_bridge.process_entry, ctx_entry)

    session.add_to_context(ctx_entry)

    if not need_respond:
        return

    # 严格唯一性：同一会话 LLM 处理期间（思考→工具调用→输出→发送完毕）绝不允许第二次激活
    if session.processing_lock.locked():
        logger.info("会话 %s LLM 正在处理中，消息已记入上下文，本次激活跳过", conversation_id)
        return

    async with session.processing_lock:
        session.remaining_cycles = MAX_CYCLES

        # 记录调用前的上下文长度，用于找到 bot 新增的条目并回填真实消息 ID
        ctx_before = len(session.context_messages)
        _t0 = time.monotonic()
        try:
            result, _, _, _, _, _ = await asyncio.to_thread(call_model_and_process, session)
        except Exception:
            logger.exception("NapCat LLM 调用失败 (conv=%s)", conversation_id)
            return
        _llm_elapsed = time.monotonic() - _t0
        logger.info("LLM 响应耗时 %.2fs (conv=%s)", _llm_elapsed, conversation_id)

        if result is None:
            logger.warning("NapCat LLM 返回为空 (conv=%s)", conversation_id)
            return

        msg_type = event.get("message_type", "")
        group_id = event.get("group_id") if msg_type == "group" else None
        user_id = event.get("sender", {}).get("user_id") if msg_type == "private" else None

        decision = result.get("decision") or {}
        # 收集新增的 bot 上下文条目，发送后将真实 QQ message_id 回填进去
        pending_bot_entries = list(session.context_messages[ctx_before:])
        await _send_decision_messages(decision, group_id, user_id, pending_bot_entries, _llm_elapsed)

        # 主动循环
        while (
            result
            and result.get("loop_control") == "continue"
            and session.remaining_cycles > 0
        ):
            session.remaining_cycles -= 1
            ctx_before = len(session.context_messages)
            _t0 = time.monotonic()
            try:
                result, _, _, _, _, _ = await asyncio.to_thread(call_model_and_process, session)
            except Exception:
                logger.exception("NapCat 主动循环 LLM 调用失败 (conv=%s)", conversation_id)
                break
            _llm_elapsed = time.monotonic() - _t0
            logger.info("LLM 响应耗时（主动循环）%.2fs (conv=%s)", _llm_elapsed, conversation_id)

            if result is None:
                break

            decision = result.get("decision") or {}
            pending_bot_entries = list(session.context_messages[ctx_before:])
            await _send_decision_messages(decision, group_id, user_id, pending_bot_entries, _llm_elapsed)

        # 意识流全部结束后，在锁内快照，锁外异步触发记忆流水线
        _mem_ctx_snapshot = list(session.context_messages)
        _mem_xml_snapshot = session.get_chat_log_display()
        _mem_source = "group" if event.get("message_type") == "group" else "private"

    # 处理锁已释放 —— 批量解析 person_id 后 fire-and-forget
    _mem_sender_ids = list(dict.fromkeys(
        str(m["sender_id"])
        for m in _mem_ctx_snapshot
        if m.get("sender_id") and m.get("role") != "bot"
    ))
    _mem_person_map = await get_person_map("qq", _mem_sender_ids)
    asyncio.create_task(
        run_memory_pipeline(
            _mem_ctx_snapshot,
            _mem_xml_snapshot,
            adapter,
            _mem_person_map,
            _mem_source,
        ),
        name=f"memory_pipeline_{conversation_id}",
    )


if napcat_client:
    napcat_client.set_message_handler(_handle_napcat_message)

    async def _handle_napcat_recall(event: dict) -> None:
        """处理群/私聊撤回通知，将对应上下文条目替换为撤回提示。"""
        notice_type = event.get("notice_type", "")
        message_id = str(event.get("message_id", ""))
        if not message_id:
            return

        if notice_type == "group_recall":
            group_id = str(event.get("group_id", ""))
            operator_id = str(event.get("user_id", ""))
            conv_id = f"group_{group_id}"
            operator_name = await get_display_name("qq", operator_id, group_id or None)
        else:  # friend_recall
            peer_id = str(event.get("user_id", ""))
            conv_id = f"private_{peer_id}"
            operator_name = await get_display_name("qq", peer_id, None)

        session = sessions.get(conv_id)
        if not session:
            return

        timestamp = datetime.now(TIMEZONE).isoformat()
        found = session.mark_message_recalled(message_id, operator_name, timestamp)
        if found:
            logger.debug("撤回通知已处理: conv=%s msg_id=%s operator=%s", conv_id, message_id, operator_name)

    napcat_client.set_recall_handler(_handle_napcat_recall)


# ── 生命周期 ─────────────────────────────────────────────

@app.before_serving
async def startup():
    await init_db()
    # 启动时清理过期 / 超量的图片缓存
    _evict_cfg = config.get("vision_bridge", {}).get("cache_eviction", {})
    try:
        _max_age = int(_evict_cfg.get("max_age_days", 30))
    except (ValueError, TypeError):
        logger.warning("[startup] cache_eviction.max_age_days 配置无效，已回退到默认值 30")
        _max_age = 30
    try:
        _max_size = int(_evict_cfg.get("max_size_mb", 0))
    except (ValueError, TypeError):
        logger.warning("[startup] cache_eviction.max_size_mb 配置无效，已回退到默认值 0")
        _max_size = 0
    if _max_age or _max_size:
        await asyncio.to_thread(evict_cache, max_age_days=_max_age, max_size_mb=_max_size)
    await init_memory_db()
    # 启动时从数据库恢复上次同步的 bot 账号信息（NapCat 尚未连接时也能展示）
    saved_qq_id, saved_qq_name = await get_bot_self()
    if saved_qq_id:
        update_bot_info(saved_qq_id, saved_qq_name)
    if napcat_client:
        host = napcat_cfg.get("host", "127.0.0.1")
        port = napcat_cfg.get("port", 8078)

        async def _sync_bot_profile() -> None:
            """NapCat 连接后同步机器人自身信息。"""
            assert napcat_client is not None
            bot_id = napcat_client.bot_id
            if not bot_id:
                logger.warning("同步跳过：bot_id 未知")
                return

            login_info = await napcat_client.send_api("get_login_info", {})
            if login_info:
                qq_id = str(login_info.get("user_id", bot_id))
                nickname = login_info.get("nickname", "")
                await upsert_bot_self(qq_id, nickname)
                update_bot_info(qq_id, nickname)

            group_list = await napcat_client.send_api("get_group_list", {})
            if not group_list:
                logger.warning("获取群列表失败，跳过群名片同步")
                return

            for group in group_list:
                group_id = str(group.get("group_id", ""))
                group_name = group.get("group_name", "")
                member_count = int(group.get("member_count", 0))
                if not group_id:
                    continue
                member_info = await napcat_client.send_api(
                    "get_group_member_info",
                    {"group_id": int(group_id), "user_id": int(bot_id)},
                )
                bot_card = ""
                if member_info:
                    bot_card = member_info.get("card") or member_info.get("nickname", "")
                await upsert_group(group_id, group_name, bot_card, member_count)

            logger.info("机器人自身信息同步完成")

        napcat_client.set_connect_handler(_sync_bot_profile)
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
