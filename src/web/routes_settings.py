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
import llm.prompt.activity_log as _activity_log
import llm.prompt.memory as _memory
from config_loader import (
    save_config,
    save_instructions,
    save_persona,
    read_env_keys,
    save_env_key,
    read_env_proxies,
    save_env_proxy,
)
from llm.core.provider import create_adapter, build_watcher_adapter_cfg, build_is_adapter_cfg
from llm.core.rate_limiter import MinuteRateLimiter
from llm.session import init_session_globals, update_session_model_name
from llm.media.vision_bridge import VisionBridge

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
        "generation": {
            **cfg.get("generation", {}),
            "retry_on_new_message": cfg.get("generation", {}).get("retry_on_new_message", True),
            "final_reminder": cfg.get("generation", {}).get("final_reminder", True),
        },
        "thinking": cfg.get("thinking", {}),
        "max_calls_per_minute": cfg.get("max_calls_per_minute", 15),
        "bot_name": cfg.get("bot_name", ""),
        "guardian": cfg.get("guardian", {"name": "", "id": ""}),
        "timezone": cfg.get("timezone", "Asia/Shanghai"),
        "napcat": cfg.get("napcat", {}),
        "watcher": cfg.get("watcher", {}),
        "is": cfg.get("is", {}),
        "activity_log": cfg.get("activity_log", {}),
        "memory": cfg.get("memory", {}),
        "typing_speed": cfg.get("typing_speed", 1.0),
        "persona": app_state.persona,
        "instructions": app_state.instructions,
        "api_keys": read_env_keys(),
        "proxies": read_env_proxies(),
    })


@settings_bp.route("/settings/full", methods=["POST"])
async def settings_save():
    """保存完整配置：写 config.yaml、persona.md、.env API Key，热重载 adapter。"""
    data = await request.get_json() or {}

    # ── 写 API Key（只写非掩码值）──────────────────────────
    for key_name in (
        "GEMINI_API_KEY",
        "DASHSCOPE_API_KEY",
        "SILICONFLOW_API_KEY",
        "BIGMODEL_API_KEY",
        "LMSTUDIO_API_KEY",
        "VISION_BRIDGE_API_KEY",
    ):
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
        new_gen = dict(new_cfg.get("generation", {}))
        new_gen.update(data["generation"])
        if "retry_on_new_message" in data["generation"]:
            new_gen["retry_on_new_message"] = bool(data["generation"]["retry_on_new_message"])
        if "final_reminder" in data["generation"]:
            new_gen["final_reminder"] = bool(data["generation"]["final_reminder"])
        new_cfg["generation"] = new_gen
    if "thinking" in data and isinstance(data["thinking"], dict):
        new_cfg["thinking"] = data["thinking"]
    if "max_calls_per_minute" in data:
        new_cfg["max_calls_per_minute"] = int(data["max_calls_per_minute"])
    if "typing_speed" in data:
        speed_val = float(data["typing_speed"])
        new_cfg["typing_speed"] = speed_val if speed_val > 0 else 1.0
    if "bot_name" in data:
        new_cfg["bot_name"] = data["bot_name"]
    if "guardian" in data and isinstance(data["guardian"], dict):
        gd = data["guardian"]
        new_guardian = dict(new_cfg.get("guardian", {}))
        if "name" in gd:
            new_guardian["name"] = gd["name"]
        if "id" in gd:
            new_guardian["id"] = gd["id"]
        new_cfg["guardian"] = new_guardian
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
    if "is" in data and isinstance(data["is"], dict):
        is_data = data["is"]
        new_is = dict(new_cfg.get("is", {}))
        if "enabled" in is_data:
            new_is["enabled"] = bool(is_data["enabled"])
        for key in ("model", "model_name"):
            if key in is_data:
                new_is[key] = is_data[key]
        for key in ("provider", "base_url"):
            if key in is_data:
                if is_data[key]:
                    new_is[key] = is_data[key]
                elif key in new_is:
                    del new_is[key]
        if "generation" in is_data and isinstance(is_data["generation"], dict):
            cleaned = {k: v for k, v in is_data["generation"].items() if v is not None}
            new_is["generation"] = cleaned
        if "thinking" in is_data and isinstance(is_data["thinking"], dict):
            if is_data["thinking"].get("level"):
                new_is["thinking"] = is_data["thinking"]
            elif "thinking" in new_is:
                del new_is["thinking"]
        if "vision" in is_data:
            new_is["vision"] = bool(is_data["vision"])
        new_cfg["is"] = new_is
    if "activity_log" in data and isinstance(data["activity_log"], dict):
        al_data = data["activity_log"]
        new_al = dict(new_cfg.get("activity_log", {}))
        if "max_entries" in al_data:
            new_al["max_entries"] = max(3, int(al_data["max_entries"]))
        new_cfg["activity_log"] = new_al
    if "memory" in data and isinstance(data["memory"], dict):
        mem_data = data["memory"]
        new_mem = dict(new_cfg.get("memory", {}))
        if "max_active" in mem_data:
            new_mem["max_active"] = max(1, int(mem_data["max_active"]))
        if "max_passive" in mem_data:
            new_mem["max_passive"] = max(1, int(mem_data["max_passive"]))
        new_cfg["memory"] = new_mem
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
        if "describe_prompt" in vb_data:
            new_vb["describe_prompt"] = vb_data["describe_prompt"]
        if "similarity_threshold" in vb_data:
            new_vb["similarity_threshold"] = int(vb_data["similarity_threshold"])
        if "whitelist" in vb_data and isinstance(vb_data["whitelist"], dict):
            new_vb_wl = dict(new_vb.get("whitelist", {}))
            if "private_users" in vb_data["whitelist"]:
                new_vb_wl["private_users"] = [str(u) for u in vb_data["whitelist"]["private_users"]]
            new_vb["whitelist"] = new_vb_wl
        if "cache_eviction" in vb_data and isinstance(vb_data["cache_eviction"], dict):
            new_vb_ce = dict(new_vb.get("cache_eviction", {}))
            ce = vb_data["cache_eviction"]
            if "max_age_days" in ce:
                new_vb_ce["max_age_days"] = int(ce["max_age_days"])
            if "max_size_mb" in ce:
                new_vb_ce["max_size_mb"] = int(ce["max_size_mb"])
            new_vb["cache_eviction"] = new_vb_ce
        new_vb["api_key_env"] = "VISION_BRIDGE_API_KEY"
        new_cfg["vision_bridge"] = new_vb

    # ── 热重载 adapter ────────────────────────────────────
    try:
        new_adapter = create_adapter(new_cfg)
    except Exception as e:
        return jsonify({"success": False, "error": f"adapter 初始化失败: {e}"}), 400

    # ── 写 instructions.md ───────────────────────────────────────
    new_instructions = data.get("instructions", app_state.instructions)
    save_instructions(new_instructions)
    # ── 写 config.yaml ────────────────────────────────────
    save_config(new_cfg)

    # ── 应用到运行时 ──────────────────────────────────────
    app_state.config = new_cfg
    app_state.adapter = new_adapter
    _activity_log.configure(int(new_cfg.get("activity_log", {}).get("max_entries", 10)))
    _mem_cfg = new_cfg.get("memory", {})
    _memory.configure(
        max_active=int(_mem_cfg.get("max_active", 8)),
        max_passive=int(_mem_cfg.get("max_passive", 15)),
    )
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
    # ── 热重载 IS adapter ────────────────────────────────
    new_is_cfg = new_cfg.get("is", {})
    app_state.is_cfg = new_is_cfg
    if new_is_cfg.get("model") or new_is_cfg.get("provider"):
        try:
            app_state.is_adapter = create_adapter(build_is_adapter_cfg(new_cfg, new_is_cfg))
        except Exception as e:
            logger.warning("热重载 IS adapter 失败: %s", e)
    else:
        app_state.is_adapter = None
    app_state.instructions = new_instructions
    app_state.MODEL = new_cfg.get("model", app_state.MODEL)
    app_state.MODEL_NAME = new_cfg.get("model_name", app_state.MODEL_NAME)
    app_state.MAX_CALLS_PER_MINUTE = new_cfg.get("max_calls_per_minute", 15)
    app_state.rate_limiter = MinuteRateLimiter(app_state.MAX_CALLS_PER_MINUTE)
    app_state.vision_bridge = VisionBridge(new_cfg.get("vision_bridge", {}))
    update_session_model_name(app_state.MODEL_NAME)
    init_session_globals(
        max_context=app_state.MAX_CONTEXT,
        timezone=ZoneInfo(new_cfg["timezone"]),
        persona=app_state.persona,
        instructions=new_instructions,
        model_name=app_state.MODEL_NAME,
        guardian_name=new_cfg.get("guardian", {}).get("name", ""),
        guardian_id=new_cfg.get("guardian", {}).get("id", ""),
    )
    return jsonify({"success": True})


@settings_bp.route("/settings/persona", methods=["POST"])
async def persona_save():
    """独立保存 persona.md，并热更新运行时 persona。"""
    data = await request.get_json() or {}
    new_persona = data.get("persona", "")
    save_persona(new_persona)
    app_state.persona = new_persona
    cfg = app_state.config
    init_session_globals(
        max_context=app_state.MAX_CONTEXT,
        timezone=ZoneInfo(cfg.get("timezone", "Asia/Shanghai")),
        persona=new_persona,
        instructions=app_state.instructions,
        model_name=app_state.MODEL_NAME,
        guardian_name=cfg.get("guardian", {}).get("name", ""),
        guardian_id=cfg.get("guardian", {}).get("id", ""),
    )
    return jsonify({"success": True})
