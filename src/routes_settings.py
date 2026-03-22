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

"""routes_settings.py — 设置页面路由

Blueprint：设置页面展示、完整配置读写、热重载 adapter。
"""

import contextlib
import logging
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from quart import Blueprint, render_template, request, jsonify

import app_state
from config_loader import (
    save_config,
    save_persona,
    save_chat_example,
    read_env_keys,
    save_env_key,
    read_env_proxies,
    save_env_proxy,
)
from provider import create_adapter, build_watcher_adapter_cfg
from rate_limiter import MinuteRateLimiter
from session import init_session_globals, update_session_model_name
from vision_bridge import VisionBridge

logger = logging.getLogger("AICQ.app")

settings_bp = Blueprint("settings", __name__)


@settings_bp.route("/settings")
async def settings_page():
    return await render_template("settings.html")


@settings_bp.route("/settings/full", methods=["GET"])
async def settings_get():
    """返回完整配置供前端填充表单。"""
    cfg = dict(app_state.config)
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
        "max_calls_per_minute": cfg.get("max_calls_per_minute", 15),
        "bot_name": cfg.get("bot_name", ""),
        "timezone": cfg.get("timezone", "Asia/Shanghai"),
        "napcat": cfg.get("napcat", {}),
        "watcher": cfg.get("watcher", {}),
        "persona": app_state.persona,
        "chat_example": app_state.chat_example,
        "api_keys": read_env_keys(),
        "proxies": read_env_proxies(),
    })


@settings_bp.route("/settings/full", methods=["POST"])
async def settings_save():
    """保存完整配置：写 config.yaml、persona.md、.env API Key，热重载 adapter。"""
    data = await request.get_json() or {}

    # ── 写 API Key（只写非掩码值）──────────────────────────
    for key_name in ("GEMINI_API_KEY", "SILICONFLOW_API_KEY", "BIGMODEL_API_KEY", "VISION_BRIDGE_API_KEY"):
        if val := (data.get("api_keys") or {}).get(key_name, ""):
            with contextlib.suppress(ValueError):
                save_env_key(key_name, val)
    
    # ── 写代理配置到 .env（总是处理，包括空值用于删除）────────────────────
    for proxy_name in ("GEMINI_PROXY", "OPENAI_PROXY", "TAVILY_PROXY"):
        val = (data.get("proxies") or {}).get(proxy_name, "")
        with contextlib.suppress(ValueError):
            save_env_proxy(proxy_name, val)
    
    load_dotenv(override=True)  # 重新载入 .env 到 os.environ

    # ── 构建新 config ──────────────────────────────────────
    new_cfg = dict(app_state.config)
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
    if "max_calls_per_minute" in data:
        new_cfg["max_calls_per_minute"] = int(data["max_calls_per_minute"])
    if "bot_name" in data:
        new_cfg["bot_name"] = data["bot_name"]
    if "timezone" in data:
        tz_val = (data.get("timezone") or "").strip() or "Asia/Shanghai"
        new_cfg["timezone"] = tz_val
    if "napcat" in data and isinstance(data["napcat"], dict):
        new_cfg["napcat"] = data["napcat"]
    if "watcher" in data and isinstance(data["watcher"], dict):
        wd = data["watcher"]
        new_watcher = dict(new_cfg.get("watcher", {}))
        for key in ("enabled", "model", "model_name", "interval", "interval_jitter"):
            if key in wd:
                new_watcher[key] = wd[key]
        for key in ("provider", "base_url"):
            if key in wd:
                if wd[key]:
                    new_watcher[key] = wd[key]
                elif key in new_watcher:
                    del new_watcher[key]
        if "generation" in wd and isinstance(wd["generation"], dict):
            new_watcher["generation"] = wd["generation"]
        new_cfg["watcher"] = new_watcher
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
    new_persona = data.get("persona", app_state.persona)
    save_persona(new_persona)

    # ── 写 chat_example.md ────────────────────────────────
    new_chat_example = data.get("chat_example", app_state.chat_example)
    save_chat_example(new_chat_example)

    # ── 写 config.yaml ────────────────────────────────────
    save_config(new_cfg)

    # ── 应用到运行时 ──────────────────────────────────────
    app_state.config = new_cfg
    app_state.adapter = new_adapter
    # ── 热重载 watcher adapter ────────────────────────────────
    new_watcher_cfg = new_cfg.get("watcher", {})
    app_state.watcher_cfg = new_watcher_cfg
    if new_watcher_cfg.get("enabled", False):
        try:
            app_state.watcher_adapter = create_adapter(build_watcher_adapter_cfg(new_cfg, new_watcher_cfg))
        except Exception as e:
            logger.warning("热重载 watcher adapter 失败: %s", e)
    else:
        app_state.watcher_adapter = None
    app_state.persona = new_persona
    app_state.chat_example = new_chat_example
    app_state.MODEL = new_cfg.get("model", app_state.MODEL)
    app_state.MODEL_NAME = new_cfg.get("model_name", app_state.MODEL_NAME)
    app_state.MAX_CALLS_PER_MINUTE = new_cfg.get("max_calls_per_minute", 15)
    app_state.rate_limiter = MinuteRateLimiter(app_state.MAX_CALLS_PER_MINUTE)
    app_state.vision_bridge = VisionBridge(new_cfg.get("vision_bridge", {}))
    update_session_model_name(app_state.MODEL_NAME)
    init_session_globals(
        max_context=app_state.MAX_CONTEXT,
        timezone=ZoneInfo(new_cfg["timezone"]),
        persona=new_persona,
        chat_example=new_chat_example,
        model_name=app_state.MODEL_NAME,
    )

    return jsonify({"success": True})
