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
    read_env_values,
    save_env_key,
    save_env_value,
    read_env_proxies,
    save_env_proxy,
    read_env_smtp,
    save_env_smtp,
    read_env_imap,
    save_env_imap,
)
from llm.core.provider import create_adapter, build_is_adapter_cfg, build_slow_thinking_adapter_cfg
from llm.core.profiles import (
    get_configured_api_key_names,
    get_model_providers,
    get_selected_provider_name,
    normalize_profile_config_inplace,
    sanitize_model_providers,
)
from llm.core.rate_limiter import MinuteRateLimiter
from llm.session import init_session_globals, update_session_model_name
from llm.media.vision_bridge import VisionBridge

logger = logging.getLogger("AICQ.web.settings")

settings_bp = Blueprint("settings", __name__)

SETTINGS_AUXILIARY_API_KEY_NAMES = (
    "TAVILY_API_KEY",
    "QWEATHER_API_KEY",
)
SETTINGS_AUXILIARY_ENV_NAMES = (
    "QWEATHER_API_HOST",
)


def _get_settings_api_key_names(cfg: dict) -> tuple[str, ...]:
    names = set(get_configured_api_key_names(cfg))
    names.update(SETTINGS_AUXILIARY_API_KEY_NAMES)
    return tuple(sorted(name for name in names if name))


@settings_bp.route("/settings")
async def settings_page():
    return await render_template("settings.html")


@settings_bp.route("/settings/full", methods=["GET"])
async def settings_get():
    """返回完整配置供前端填充表单。"""
    cfg = deepcopy(app_state.config)
    normalize_profile_config_inplace(cfg)
    return jsonify({
        "provider": get_selected_provider_name(cfg),
        "model_providers": get_model_providers(cfg),
        "model": cfg.get("model", ""),
        "model_name": cfg.get("model_name", ""),
        "vision": cfg.get("vision", True),
        "vision_bridge": cfg.get("vision_bridge", {}),
        "generation": {
            **cfg.get("generation", {}),
            "retry_on_new_message": cfg.get("generation", {}).get("retry_on_new_message", True),
            "final_reminder": cfg.get("generation", {}).get("final_reminder", True),
        },
        "max_calls_per_minute": cfg.get("max_calls_per_minute", 15),
        "bot_name": cfg.get("bot_name", ""),
        "guardian": cfg.get("guardian", {"name": "", "id": ""}),
        "timezone": cfg.get("timezone", "Asia/Shanghai"),
        "napcat": cfg.get("napcat", {}),
        "tts": cfg.get("tts", {
            "enabled": False,
            "host": "127.0.0.1",
            "port": 8765,
            "secret_token": "",
            "max_concurrent_tasks_per_plugin": 8,
        }),
        "alerting": cfg.get("alerting", {
            "enabled": False,
            "heartbeat_timeout": 120,
            "cooldown": 600,
            "subject_prefix": "[AIcarus 告警]",
            "napcat_restart": {
                "enabled": False,
                "command": "",
                "args": [],
                "cwd": "",
                "stop_command": "",
                "stop_image_names": ["NapCatWinBootMain.exe"],
                "stop_path_filter": "",
                "force_kill_by_image_name": False,
                "stop_grace_seconds": 3,
                "cooldown_seconds": 300,
                "max_attempts_per_hour": 4,
                "recovery_grace_seconds": 45,
                "qrcode_globs": ["**/qrcode*.png", "cache/qrcode*.png"],
            },
            "email_control": {
                "enabled": False,
                "allowed_commands": ["REQUEST", "RESTART", "STATUS"],
                "token_ttl_seconds": 600,
                "poll_interval": 30,
                "reuse_smtp_credentials": True,
            },
        }),
        "smtp": await asyncio.to_thread(read_env_smtp),
        "imap": await asyncio.to_thread(read_env_imap),
        "is": cfg.get("is", {}),
        "memory": cfg.get("memory", {}),
        "slow_thinking": cfg.get("slow_thinking", {}),
        "typing_speed": cfg.get("typing_speed", 1.0),
        "persona": app_state.persona,
        "api_keys": await asyncio.to_thread(read_env_keys, _get_settings_api_key_names(cfg)),
        "service_env": await asyncio.to_thread(read_env_values, SETTINGS_AUXILIARY_ENV_NAMES),
        "proxies": await asyncio.to_thread(read_env_proxies),
    })


@settings_bp.route("/settings/providers", methods=["POST"])
async def settings_save_providers():
    """独立保存模型供应商，不阻塞于整页模型绑定校验。"""
    data = await request.get_json() or {}
    raw_model_providers = data.get("model_providers", {})
    if not isinstance(raw_model_providers, dict):
        return jsonify({"success": False, "error": "model_providers 必须是对象"}), 400

    api_keys_data = dict(data.get("api_keys") or {})

    def _write_env():
        for key_name, val in api_keys_data.items():
            if val:
                with contextlib.suppress(ValueError):
                    save_env_key(key_name, val)
        load_dotenv(override=True)

    await asyncio.to_thread(_write_env)

    new_cfg = deepcopy(app_state.config)
    new_cfg.pop("profiles", None)
    new_cfg.pop("openai_profiles", None)
    new_cfg["model_providers"] = sanitize_model_providers(
        raw_model_providers,
        dedupe_display_names=True,
    )
    normalize_profile_config_inplace(new_cfg)

    await asyncio.to_thread(save_config, new_cfg)
    app_state.config = new_cfg

    return jsonify({
        "success": True,
        "model_providers": get_model_providers(new_cfg),
        "api_keys": await asyncio.to_thread(
            read_env_keys,
            _get_settings_api_key_names(new_cfg),
        ),
    })


@settings_bp.route("/settings/full", methods=["POST"])
async def settings_save():
    """保存完整配置：写 config.yaml、persona.md、.env API Key，热重载 adapter。"""
    data = await request.get_json() or {}

    # ── 写 API Key 和代理（线程池，避免阻塞事件循环）──────
    api_keys_data = dict(data.get("api_keys") or {})
    service_env_data = dict(data.get("service_env") or {})
    proxies_data = dict(data.get("proxies") or {})
    smtp_data = dict(data.get("smtp") or {})
    imap_data = dict(data.get("imap") or {})

    def _write_env():
        for key_name, val in api_keys_data.items():
            if val:
                with contextlib.suppress(ValueError):
                    save_env_key(key_name, val)
        for key_name, val in service_env_data.items():
            with contextlib.suppress(ValueError):
                save_env_value(key_name, val)
        for proxy_name in ("OPENAI_PROXY", "TAVILY_PROXY"):
            if proxy_name in proxies_data:
                with contextlib.suppress(ValueError):
                    save_env_proxy(proxy_name, proxies_data.get(proxy_name, ""))
        if smtp_data:
            with contextlib.suppress(ValueError):
                save_env_smtp(smtp_data)
        if imap_data:
            with contextlib.suppress(ValueError):
                save_env_imap(imap_data)
        load_dotenv(override=True)

    await asyncio.to_thread(_write_env)

    # ── 构建新 config ──────────────────────────────────────
    new_cfg = deepcopy(app_state.config)
    new_cfg.pop("profiles", None)
    new_cfg.pop("openai_profiles", None)
    if "model_providers" in data:
        if not isinstance(data["model_providers"], dict):
            return jsonify({"success": False, "error": "model_providers 必须是对象"}), 400
        new_cfg["model_providers"] = sanitize_model_providers(
            data["model_providers"],
            dedupe_display_names=True,
        )
    if "provider" in data:
        new_cfg["provider"] = data.get("provider")
    new_cfg.pop("profile", None)
    new_cfg.pop("base_url", None)
    new_cfg.pop("api_key_env", None)
    if "model" in data:
        new_cfg["model"] = data["model"]
    if "model_name" in data:
        new_cfg["model_name"] = data["model_name"] or data.get("model", new_cfg.get("model", ""))
    if "generation" in data and isinstance(data["generation"], dict):
        new_gen = dict(new_cfg.get("generation", {}))
        new_gen.update(data["generation"])
        if "retry_on_new_message" in data["generation"]:
            new_gen["retry_on_new_message"] = bool(data["generation"]["retry_on_new_message"])
        if "final_reminder" in data["generation"]:
            new_gen["final_reminder"] = bool(data["generation"]["final_reminder"])
        new_cfg["generation"] = new_gen
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
    if "tts" in data and isinstance(data["tts"], dict):
        td = data["tts"]
        new_tts = dict(new_cfg.get("tts", {}))
        if "enabled" in td:
            new_tts["enabled"] = bool(td["enabled"])
        if "host" in td:
            new_tts["host"] = str(td.get("host") or "127.0.0.1").strip() or "127.0.0.1"
        if "port" in td:
            new_tts["port"] = max(1, min(65535, int(td["port"])))
        if "secret_token" in td:
            new_tts["secret_token"] = str(td.get("secret_token") or "")
        if "max_concurrent_tasks_per_plugin" in td:
            new_tts["max_concurrent_tasks_per_plugin"] = max(
                1,
                min(128, int(td["max_concurrent_tasks_per_plugin"])),
            )
        new_cfg["tts"] = new_tts
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
        # NapCat 自动重启子节点
        if "napcat_restart" in ad and isinstance(ad["napcat_restart"], dict):
            nr_in = ad["napcat_restart"]
            nr_out = dict(new_alerting.get("napcat_restart", {}))
            if "enabled" in nr_in:
                nr_out["enabled"] = bool(nr_in["enabled"])
            if "command" in nr_in:
                nr_out["command"] = str(nr_in["command"] or "").strip()
            if "args" in nr_in and isinstance(nr_in["args"], list):
                nr_out["args"] = [str(a) for a in nr_in["args"]]
            if "cwd" in nr_in:
                nr_out["cwd"] = str(nr_in["cwd"] or "").strip()
            if "stop_command" in nr_in:
                nr_out["stop_command"] = str(nr_in["stop_command"] or "").strip()
            if "stop_image_names" in nr_in and isinstance(nr_in["stop_image_names"], list):
                nr_out["stop_image_names"] = [
                    str(n).strip() for n in nr_in["stop_image_names"] if str(n).strip()
                ]
            if "stop_path_filter" in nr_in:
                nr_out["stop_path_filter"] = str(nr_in["stop_path_filter"] or "").strip()
            if "force_kill_by_image_name" in nr_in:
                nr_out["force_kill_by_image_name"] = bool(nr_in["force_kill_by_image_name"])
            if "stop_grace_seconds" in nr_in:
                nr_out["stop_grace_seconds"] = max(0, int(nr_in["stop_grace_seconds"]))
            if "cooldown_seconds" in nr_in:
                nr_out["cooldown_seconds"] = max(30, int(nr_in["cooldown_seconds"]))
            if "max_attempts_per_hour" in nr_in:
                nr_out["max_attempts_per_hour"] = max(1, int(nr_in["max_attempts_per_hour"]))
            if "recovery_grace_seconds" in nr_in:
                nr_out["recovery_grace_seconds"] = max(5, int(nr_in["recovery_grace_seconds"]))
            if "qrcode_globs" in nr_in and isinstance(nr_in["qrcode_globs"], list):
                nr_out["qrcode_globs"] = [str(g) for g in nr_in["qrcode_globs"] if str(g).strip()]
            new_alerting["napcat_restart"] = nr_out
        # 邮件远程指令子节点（Phase 3）
        if "email_control" in ad and isinstance(ad["email_control"], dict):
            ec_in = ad["email_control"]
            ec_out = dict(new_alerting.get("email_control", {}))
            if "enabled" in ec_in:
                ec_out["enabled"] = bool(ec_in["enabled"])
            if "allowed_commands" in ec_in and isinstance(ec_in["allowed_commands"], list):
                allowed_pool = {"REQUEST", "RESTART", "STOP", "STATUS", "KILL_AICQ"}
                cleaned = []
                for c in ec_in["allowed_commands"]:
                    cu = str(c).strip().upper()
                    if cu in allowed_pool and cu not in cleaned:
                        cleaned.append(cu)
                # REQUEST 为握手入口，必须保留，否则用户无法主动要 token
                if "REQUEST" not in cleaned:
                    cleaned.insert(0, "REQUEST")
                ec_out["allowed_commands"] = cleaned
            if "token_ttl_seconds" in ec_in:
                ec_out["token_ttl_seconds"] = max(60, min(7 * 24 * 3600, int(ec_in["token_ttl_seconds"])))
            if "poll_interval" in ec_in:
                ec_out["poll_interval"] = max(10, min(600, int(ec_in["poll_interval"])))
            if "reuse_smtp_credentials" in ec_in:
                ec_out["reuse_smtp_credentials"] = bool(ec_in["reuse_smtp_credentials"])
            new_alerting["email_control"] = ec_out
        new_cfg["alerting"] = new_alerting
    if "is" in data and isinstance(data["is"], dict):
        is_data = data["is"]
        new_is = dict(new_cfg.get("is", {}))
        if "enabled" in is_data:
            new_is["enabled"] = bool(is_data["enabled"])
        for key in ("model",):
            if key in is_data:
                if is_data[key]:
                    new_is[key] = is_data[key]
                else:
                    new_is.pop(key, None)
        if "provider" in is_data:
            provider = is_data.get("provider")
            if provider:
                new_is["provider"] = provider
            else:
                new_is.pop("provider", None)
        new_is.pop("profile", None)
        new_is.pop("base_url", None)
        new_is.pop("api_key_env", None)
        if "generation" in is_data and isinstance(is_data["generation"], dict):
            cleaned = {k: v for k, v in is_data["generation"].items() if v is not None}
            new_is["generation"] = cleaned
        if "vision" in is_data:
            new_is["vision"] = bool(is_data["vision"])
        new_cfg["is"] = new_is
    if "slow_thinking" in data and isinstance(data["slow_thinking"], dict):
        st_data = data["slow_thinking"]
        new_st = dict(new_cfg.get("slow_thinking", {}))
        if "enabled" in st_data:
            new_st["enabled"] = bool(st_data["enabled"])
        for key in ("model",):
            if key in st_data:
                if st_data[key]:
                    new_st[key] = st_data[key]
                else:
                    new_st.pop(key, None)
        if "provider" in st_data:
            provider = st_data.get("provider")
            if provider:
                new_st["provider"] = provider
            else:
                new_st.pop("provider", None)
        new_st.pop("profile", None)
        new_st.pop("base_url", None)
        new_st.pop("api_key_env", None)
        if "generation" in st_data and isinstance(st_data["generation"], dict):
            gen_data = st_data["generation"]
            new_gen = dict(new_st.get("generation", {}))
            if "temperature" in gen_data:
                new_gen["temperature"] = max(0.0, min(2.0, float(gen_data["temperature"])))
            if "max_output_tokens" in gen_data:
                new_gen["max_output_tokens"] = max(64, int(gen_data["max_output_tokens"]))
            new_st["generation"] = new_gen
        new_cfg["slow_thinking"] = new_st
    if "vision" in data:
        new_cfg["vision"] = bool(data["vision"])
    if "vision_bridge" in data and isinstance(data["vision_bridge"], dict):
        vb_data = data["vision_bridge"]
        new_vb = dict(new_cfg.get("vision_bridge", {}))
        if "enabled" in vb_data:
            new_vb["enabled"] = bool(vb_data["enabled"])
        if "provider" in vb_data:
            provider = vb_data.get("provider")
            if provider:
                new_vb["provider"] = provider
            else:
                new_vb.pop("provider", None)
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
        new_vb.pop("profile", None)
        new_vb.pop("base_url", None)
        new_vb.pop("api_key_env", None)
        new_cfg["vision_bridge"] = new_vb

    def _payload_binding_error(label: str, payload_part: dict, required: bool = True) -> str | None:
        provider = (payload_part.get("provider") or "").strip()
        model = (payload_part.get("model") or "").strip()
        if required and (not provider or not model):
            return f"{label} 必须同时选择供应商并填写模型 ID"
        return None

    for error in (
        _payload_binding_error("主模型", data),
        _payload_binding_error("IS 中断哨兵", data.get("is", {}), bool(data.get("is", {}).get("enabled", True))) if isinstance(data.get("is"), dict) else None,
        _payload_binding_error("Vision Bridge", data.get("vision_bridge", {}), bool(data.get("vision_bridge", {}).get("enabled", False))) if isinstance(data.get("vision_bridge"), dict) else None,
        _payload_binding_error("慢思考模型", data.get("slow_thinking", {}), bool(data.get("slow_thinking", {}).get("enabled", False))) if isinstance(data.get("slow_thinking"), dict) else None,
    ):
        if error:
            return jsonify({"success": False, "error": error}), 400

    normalize_profile_config_inplace(new_cfg)

    providers = get_model_providers(new_cfg)

    def _validate_model_binding(label: str, cfg_part: dict, required: bool = True) -> str | None:
        provider = (cfg_part.get("provider") or "").strip()
        model = (cfg_part.get("model") or "").strip()
        if not required and not provider and not model:
            return None
        if not provider or not model:
            return f"{label} 必须同时选择供应商并填写模型 ID"
        if provider not in providers:
            return f"{label} 选择了未定义的供应商: {provider}"
        return None

    for error in (
        _validate_model_binding("主模型", new_cfg),
        _validate_model_binding("IS 中断哨兵", new_cfg.get("is", {}), bool(new_cfg.get("is", {}).get("enabled", True))),
        _validate_model_binding("Vision Bridge", new_cfg.get("vision_bridge", {}), bool(new_cfg.get("vision_bridge", {}).get("enabled", False))),
        _validate_model_binding("慢思考模型", new_cfg.get("slow_thinking", {}), bool(new_cfg.get("slow_thinking", {}).get("enabled", False))),
    ):
        if error:
            return jsonify({"success": False, "error": error}), 400

    # ── 热重载 adapter + 写 config（全部在线程池，避免阻塞事件循环）──────────
    # create_adapter / VisionBridge 会初始化 httpx.Client，属于慢同步操作
    def _create_and_save():
        adapter = create_adapter(new_cfg)
        is_cfg_ = new_cfg.get("is", {})
        is_adapter_ = None
        if is_cfg_.get("enabled", True):
            is_adapter_ = create_adapter(build_is_adapter_cfg(new_cfg, is_cfg_))
        st_cfg_ = new_cfg.get("slow_thinking", {})
        st_adapter_ = None
        if st_cfg_.get("enabled", True) and st_cfg_.get("provider") and st_cfg_.get("model"):
            st_adapter_ = create_adapter(build_slow_thinking_adapter_cfg(new_cfg, st_cfg_))
        save_config(new_cfg)
        vb = VisionBridge(new_cfg)
        return adapter, is_cfg_, is_adapter_, st_cfg_, st_adapter_, vb

    try:
        new_adapter, new_is_cfg, new_is_adapter, new_st_cfg, new_st_adapter, new_vision_bridge = await asyncio.to_thread(_create_and_save)
    except Exception as e:
        return jsonify({"success": False, "error": f"adapter 初始化失败: {e}"}), 400

    # ── 应用到运行时 ──────────────────────────────────────
    app_state.config = new_cfg
    app_state.adapter = new_adapter
    # ── 热重载 IS adapter ────────────────────────────────
    app_state.is_cfg = new_is_cfg
    app_state.is_adapter = new_is_adapter
    # ── 热重载 slow_thinking adapter ─────────────────────
    app_state.slow_thinking_cfg = new_st_cfg
    app_state.slow_thinking_adapter = new_st_adapter
    app_state.MODEL = new_cfg.get("model", app_state.MODEL)
    app_state.MODEL_NAME = new_cfg.get("model_name", app_state.MODEL_NAME)
    app_state.MAX_CALLS_PER_MINUTE = new_cfg.get("max_calls_per_minute", 15)
    app_state.MAX_CONTEXT = int(new_cfg.get("max_context", 10))
    app_state.napcat_cfg = new_cfg.get("napcat", {}) or {}
    app_state.tts_cfg = new_cfg.get("tts", {}) or {}
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
        from napcat_supervisor import NapcatSupervisor
        new_alerting_cfg = new_cfg.get("alerting", {}) or {}
        new_alert = AlertManager(new_alerting_cfg)
        # 迁移远程指令 token 注册表：避免“保存设置”时把已发出的 token 全部作废，
        # 导致用户回信被判 token missing。
        old_alert = app_state.alert_manager
        if old_alert is not None:
            try:
                new_alert._pending_tokens.update(getattr(old_alert, "_pending_tokens", {}))
                new_alert._recent_msgids.update(getattr(old_alert, "_recent_msgids", {}))
            except (AttributeError, TypeError):
                pass
        app_state.alert_manager = new_alert
        # NapCat 监管器热重载
        new_supervisor = NapcatSupervisor(
            new_alerting_cfg.get("napcat_restart", {}) or {},
            client=app_state.napcat_client,
            alert=new_alert,
        )
        app_state.napcat_supervisor = new_supervisor
        if app_state.napcat_client is not None:
            if new_alert.enabled:
                app_state.napcat_client.set_alert_manager(
                    new_alert,
                    heartbeat_timeout=float(new_alerting_cfg.get("heartbeat_timeout", 120)),
                )
            else:
                # 关闭告警：解绑 alert，watchdog 仍在跑但不会发邮件
                app_state.napcat_client.set_alert_manager(None, heartbeat_timeout=120.0)
            # 同步重启能力
            app_state.napcat_client.set_supervisor(
                new_supervisor if new_supervisor.is_configured() else None
            )
        # ── 邮件远程指令控制器热重载（Phase 3）────────────
        from email_controller import EmailController
        old_ec = app_state.email_controller
        if old_ec is not None:
            try:
                await old_ec.stop()
            except Exception:
                logger.warning("热重载：停旧 EmailController 异常", exc_info=True)
        new_ec = EmailController(
            new_alerting_cfg,
            supervisor=new_supervisor,
            alert=new_alert,
        )
        app_state.email_controller = new_ec
        try:
            await new_ec.start()
        except Exception:
            logger.warning("热重载：启新 EmailController 异常", exc_info=True)
    except Exception:
        logger.exception("热重载 AlertManager 失败")

    # ── 热重载 TTS 插件服务端 ───────────────────────
    try:
        from tts import TTSServer

        old_tts_server = app_state.tts_server
        if old_tts_server is not None:
            await old_tts_server.stop()

        def _buffer_tts_audio(task_id: str, pcm: bytes) -> None:
            app_state.tts_audio_buffers.setdefault(task_id, bytearray()).extend(pcm)

        app_state.tts_audio_buffers.clear()
        if app_state.tts_cfg.get("enabled", False):
            new_tts_server = TTSServer(
                host=app_state.tts_cfg.get("host", "127.0.0.1"),
                port=int(app_state.tts_cfg.get("port", 8765)),
                secret_token=app_state.tts_cfg.get("secret_token", ""),
                on_audio_chunk=_buffer_tts_audio,
                max_concurrent_tasks_per_plugin=int(
                    app_state.tts_cfg.get("max_concurrent_tasks_per_plugin", 8)
                ),
            )
            await new_tts_server.start()
            app_state.tts_server = new_tts_server
        else:
            app_state.tts_server = None
    except Exception:
        app_state.tts_server = None
        logger.exception("热重载 TTS 插件服务端失败")

    return jsonify({"success": True})


@settings_bp.route("/settings/alerting/test", methods=["POST"])
async def alerting_test():
    """触发一次测试告警邮件，验证 SMTP 配置可用。

    使用当前 .env 中已写入的 SMTP 凭据（前端必须先点"保存并应用"再点测试）。
    ⚠️ 必须复用全局 app_state.alert_manager，否则签发的远程指令 token
       只会进临时实例的注册表，等用户回复邮件时全局实例查不到 token。
    """
    mgr = app_state.alert_manager
    if mgr is None:
        return jsonify({"success": False, "error": "AlertManager 尚未初始化"}), 500

    # 临时启用 + 改前缀，发完恢复
    saved_enabled = mgr.cfg.get("enabled", False)
    saved_prefix = mgr.cfg.get("subject_prefix", "[AIcarus 告警]")
    mgr.cfg["enabled"] = True
    mgr.cfg["subject_prefix"] = saved_prefix + "[WebUI 测试]"
    try:
        await mgr.notify_disconnect("WebUI 测试: 这是一封测试邮件，可忽略")
        return jsonify({"success": True, "message": "已尝试发送测试邮件，请到收件箱确认"})
    except Exception as e:
        logger.exception("发送测试告警邮件失败")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        mgr.cfg["enabled"] = saved_enabled
        mgr.cfg["subject_prefix"] = saved_prefix


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
