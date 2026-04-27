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

import asyncio
import contextlib
from copy import deepcopy
import logging
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from quart import Blueprint, render_template, request, jsonify

import app_state
from config_loader import (
    save_config,
    save_persona,
    read_env_keys,
    save_env_key,
    read_env_proxies,
    save_env_proxy,
    read_env_smtp,
    save_env_smtp,
)
from llm.core.provider import create_adapter, build_is_adapter_cfg
from llm.core.profiles import (
    get_configured_api_key_names,
    get_openai_profiles,
    get_selected_profile_name,
    normalize_profile_config_inplace,
)
from llm.core.rate_limiter import MinuteRateLimiter
from llm.session import init_session_globals, update_session_model_name
from llm.media.vision_bridge import VisionBridge

logger = logging.getLogger("AICQ.web.settings")

settings_bp = Blueprint("settings", __name__)


@settings_bp.route("/settings")
async def settings_page():
    return await render_template("settings.html")


@settings_bp.route("/settings/full", methods=["GET"])
async def settings_get():
    """返回完整配置供前端填充表单。"""
    cfg = deepcopy(app_state.config)
    normalize_profile_config_inplace(cfg)
    # 不把 base_url 留空 key
    return jsonify({
        "profile": get_selected_profile_name(cfg),
        "provider": get_selected_profile_name(cfg),
        "openai_profiles": get_openai_profiles(cfg),
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
        "alerting": cfg.get("alerting", {
            "enabled": False,
            "heartbeat_timeout": 120,
            "cooldown": 600,
            "subject_prefix": "[AIcarus 告警]",
        }),
        "smtp": await asyncio.to_thread(read_env_smtp),
        "is": cfg.get("is", {}),
        "memory": cfg.get("memory", {}),
        "typing_speed": cfg.get("typing_speed", 1.0),
        "persona": app_state.persona,
        "api_keys": await asyncio.to_thread(read_env_keys, get_configured_api_key_names(cfg)),
        "proxies": await asyncio.to_thread(read_env_proxies),
    })


@settings_bp.route("/settings/full", methods=["POST"])
async def settings_save():
    """保存完整配置：写 config.yaml、persona.md、.env API Key，热重载 adapter。"""
    data = await request.get_json() or {}

    # ── 写 API Key 和代理（线程池，避免阻塞事件循环）──────
    api_keys_data = dict(data.get("api_keys") or {})
    proxies_data = dict(data.get("proxies") or {})
    smtp_data = dict(data.get("smtp") or {})

    def _write_env():
        for key_name, val in api_keys_data.items():
            if val:
                with contextlib.suppress(ValueError):
                    save_env_key(key_name, val)
        for proxy_name in ("OPENAI_PROXY", "TAVILY_PROXY"):
            val = proxies_data.get(proxy_name, "")
            with contextlib.suppress(ValueError):
                save_env_proxy(proxy_name, val)
        if smtp_data:
            with contextlib.suppress(ValueError):
                save_env_smtp(smtp_data)
        load_dotenv(override=True)

    await asyncio.to_thread(_write_env)

    # ── 构建新 config ──────────────────────────────────────
    new_cfg = deepcopy(app_state.config)
    if "openai_profiles" in data:
        if not isinstance(data["openai_profiles"], dict):
            return jsonify({"success": False, "error": "openai_profiles 必须是对象"}), 400
        new_cfg["openai_profiles"] = data["openai_profiles"]
    if "profile" in data or "provider" in data:
        new_cfg["profile"] = data.get("profile") or data.get("provider")
    new_cfg.pop("provider", None)
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
    if "alerting" in data and isinstance(data["alerting"], dict):
        ad = data["alerting"]
        new_alerting = dict(new_cfg.get("alerting", {}))
        if "enabled" in ad:
            new_alerting["enabled"] = bool(ad["enabled"])
        if "heartbeat_timeout" in ad:
            new_alerting["heartbeat_timeout"] = max(30, int(ad["heartbeat_timeout"]))
        if "cooldown" in ad:
            new_alerting["cooldown"] = max(0, int(ad["cooldown"]))
        if "subject_prefix" in ad:
            new_alerting["subject_prefix"] = str(ad["subject_prefix"]).strip() or "[AIcarus 告警]"
        new_cfg["alerting"] = new_alerting
    if "is" in data and isinstance(data["is"], dict):
        is_data = data["is"]
        new_is = dict(new_cfg.get("is", {}))
        if "enabled" in is_data:
            new_is["enabled"] = bool(is_data["enabled"])
        for key in ("model", "model_name"):
            if key in is_data:
                new_is[key] = is_data[key]
        for key in ("profile", "provider", "base_url"):
            if key not in is_data:
                continue
            target_key = "profile" if key in {"profile", "provider"} else key
            if is_data[key]:
                new_is[target_key] = is_data[key]
            elif target_key in new_is:
                del new_is[target_key]
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
    if "memory" in data and isinstance(data["memory"], dict):
        mem_data = data["memory"]
        new_mem = dict(new_cfg.get("memory", {}))
        if "max_active" in mem_data:
            new_mem["max_active"] = max(1, int(mem_data["max_active"]))
        if "max_passive" in mem_data:
            new_mem["max_passive"] = max(1, int(mem_data["max_passive"]))
        # 兼容旧前端传 max_entries：映射到 max_passive
        if "max_entries" in mem_data and "max_passive" not in mem_data:
            new_mem["max_passive"] = max(1, int(mem_data["max_entries"]))
        if "auto_archive" in mem_data and isinstance(mem_data["auto_archive"], dict):
            aa_data = mem_data["auto_archive"]
            new_aa = dict(new_mem.get("auto_archive", {}))
            if "generation" in aa_data and isinstance(aa_data["generation"], dict):
                gen_data = aa_data["generation"]
                new_gen = dict(new_aa.get("generation", {}))
                if "temperature" in gen_data:
                    new_gen["temperature"] = max(0.0, min(2.0, float(gen_data["temperature"])))
                if "max_output_tokens" in gen_data:
                    new_gen["max_output_tokens"] = max(256, int(gen_data["max_output_tokens"]))
                new_aa["generation"] = new_gen
            new_mem["auto_archive"] = new_aa
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

    normalize_profile_config_inplace(new_cfg)

    # ── 热重载 adapter + 写 config（全部在线程池，避免阻塞事件循环）──────────
    # create_adapter / VisionBridge 会初始化 httpx.Client，属于慢同步操作
    def _create_and_save():
        adapter = create_adapter(new_cfg)
        is_cfg_ = new_cfg.get("is", {})
        is_adapter_ = None
        if is_cfg_.get("model") or is_cfg_.get("profile"):
            is_adapter_ = create_adapter(build_is_adapter_cfg(new_cfg, is_cfg_))
        save_config(new_cfg)
        vb = VisionBridge(new_cfg.get("vision_bridge", {}))
        return adapter, is_cfg_, is_adapter_, vb

    try:
        new_adapter, new_is_cfg, new_is_adapter, new_vision_bridge = await asyncio.to_thread(_create_and_save)
    except Exception as e:
        return jsonify({"success": False, "error": f"adapter 初始化失败: {e}"}), 400

    # ── 应用到运行时 ──────────────────────────────────────
    app_state.config = new_cfg
    app_state.adapter = new_adapter
    # ── 热重载 IS adapter ────────────────────────────────
    app_state.is_cfg = new_is_cfg
    app_state.is_adapter = new_is_adapter
    app_state.MODEL = new_cfg.get("model", app_state.MODEL)
    app_state.MODEL_NAME = new_cfg.get("model_name", app_state.MODEL_NAME)
    app_state.MAX_CALLS_PER_MINUTE = new_cfg.get("max_calls_per_minute", 15)
    app_state.MAX_CONTEXT = int(new_cfg.get("max_context", 10))
    app_state.rate_limiter = MinuteRateLimiter(app_state.MAX_CALLS_PER_MINUTE)
    app_state.vision_bridge = new_vision_bridge
    update_session_model_name(app_state.MODEL_NAME)
    init_session_globals(
        max_context=app_state.MAX_CONTEXT,
        timezone=ZoneInfo(new_cfg["timezone"]),
        persona=app_state.persona,
        model_name=app_state.MODEL_NAME,
        guardian_name=new_cfg.get("guardian", {}).get("name", ""),
        guardian_id=new_cfg.get("guardian", {}).get("id", ""),
    )

    # ── 热重载 AlertManager 与 NapcatClient 心跳监视 ──────
    try:
        from alerting import AlertManager
        new_alerting_cfg = new_cfg.get("alerting", {}) or {}
        new_alert = AlertManager(new_alerting_cfg)
        app_state.alert_manager = new_alert
        if app_state.napcat_client is not None:
            if new_alert.enabled:
                app_state.napcat_client.set_alert_manager(
                    new_alert,
                    heartbeat_timeout=float(new_alerting_cfg.get("heartbeat_timeout", 120)),
                )
            else:
                # 关闭告警：解绑 alert，watchdog 仍在跑但不会发邮件
                app_state.napcat_client.set_alert_manager(None, heartbeat_timeout=120.0)
    except Exception:
        logger.exception("热重载 AlertManager 失败")

    return jsonify({"success": True})


@settings_bp.route("/settings/alerting/test", methods=["POST"])
async def alerting_test():
    """触发一次测试告警邮件，验证 SMTP 配置可用。

    使用当前 .env 中已写入的 SMTP 凭据（前端必须先点"保存并应用"再点测试）。
    """
    from alerting import AlertManager
    cfg = (app_state.config.get("alerting", {}) or {}).copy()
    # 测试时强制启用，并使用独立主题前缀以便区分
    cfg["enabled"] = True
    cfg["subject_prefix"] = cfg.get("subject_prefix", "[AIcarus 告警]") + "[WebUI 测试]"
    mgr = AlertManager(cfg)
    try:
        await mgr.notify_disconnect("WebUI 测试: 这是一封测试邮件，可忽略")
        return jsonify({"success": True, "message": "已尝试发送测试邮件，请到收件箱确认"})
    except Exception as e:
        logger.exception("发送测试告警邮件失败")
        return jsonify({"success": False, "error": str(e)}), 500


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
        model_name=app_state.MODEL_NAME,
        guardian_name=cfg.get("guardian", {}).get("name", ""),
        guardian_id=cfg.get("guardian", {}).get("id", ""),
    )
    return jsonify({"success": True})
