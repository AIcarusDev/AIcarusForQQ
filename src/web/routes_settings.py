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
import mimetypes
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from quart import Blueprint, render_template, request, jsonify, send_file

import app_state
from config_loader import (
    save_config,
    save_persona,
    save_prompt_doc,
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
from llm.core.provider import (
    create_adapter,
    build_is_adapter_cfg,
    build_archiver_adapter_cfg,
    build_slow_thinking_adapter_cfg,
    build_compression_adapter_cfg,
)
from llm.compression.config import normalize_generation_config
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
from qq_adapter.config import normalize_qq_adapter_config
from browser.config import normalize_browser_control_config

logger = logging.getLogger("AICQ.web.settings")

settings_bp = Blueprint("settings", __name__)

SETTINGS_AUXILIARY_API_KEY_NAMES = (
    "TAVILY_API_KEY",
    "QWEATHER_API_KEY",
)
SETTINGS_AUXILIARY_ENV_NAMES = (
    "QWEATHER_API_HOST",
)


def _default_compression_cfg(cfg: dict, gen_cfg: dict) -> dict:
    """Return explicit UI defaults without changing saved config."""
    compression_cfg = deepcopy(cfg.get("cognition_compression", {}) or {})
    if not compression_cfg.get("provider"):
        compression_cfg["provider"] = get_selected_provider_name(cfg)
    if not compression_cfg.get("model"):
        compression_cfg["model"] = cfg.get("model", "")
    compression_gen = dict(compression_cfg.get("generation", {}) or {})
    compression_gen.setdefault(
        "temperature",
        gen_cfg.get("cognition_compression_temperature", 0.3),
    )
    compression_gen.setdefault(
        "max_output_tokens",
        gen_cfg.get("cognition_compression_max_output_tokens", 2000),
    )
    compression_cfg["generation"] = compression_gen
    return compression_cfg


def _default_memory_cfg(cfg: dict) -> dict:
    """Return memory config with UI-visible defaults without changing saved config."""
    memory_cfg = deepcopy(cfg.get("memory", {}) or {})
    auto_archive = memory_cfg.get("auto_archive")
    if isinstance(auto_archive, dict):
        auto_archive = dict(auto_archive)
    else:
        auto_archive = {}
    auto_archive.setdefault("enabled", True)
    memory_cfg["auto_archive"] = auto_archive
    return memory_cfg


def _section_enabled(cfg_part: object, default: bool = True) -> bool:
    if not isinstance(cfg_part, dict):
        return default
    return bool(cfg_part.get("enabled", default))


def _get_settings_api_key_names(cfg: dict) -> tuple[str, ...]:
    names = set(get_configured_api_key_names(cfg))
    names.update(SETTINGS_AUXILIARY_API_KEY_NAMES)
    return tuple(sorted(name for name in names if name))


def _default_web_search_cfg(cfg: dict) -> dict:
    """Return explicit UI defaults without mutating saved config."""
    raw = cfg.get("web_search", {}) if isinstance(cfg, dict) else {}
    web_search = deepcopy(raw) if isinstance(raw, dict) else {}
    searxng_raw = web_search.get("searxng", {})
    searxng = deepcopy(searxng_raw) if isinstance(searxng_raw, dict) else {}
    searxng.setdefault("enabled", False)
    searxng.setdefault("base_url", "http://127.0.0.1:8888")
    searxng.setdefault("language", "zh-CN")
    searxng.setdefault("safesearch", 0)
    web_search["searxng"] = searxng
    return web_search


def _qq_adapter_runtime_signature(cfg: dict) -> tuple[bool, str, str, int]:
    try:
        port = int(cfg.get("port", 8078))
    except (TypeError, ValueError):
        port = 8078
    return (
        bool(cfg.get("enabled", False)),
        str(cfg.get("adapter", "napcat") or "napcat"),
        str(cfg.get("host", "127.0.0.1") or "127.0.0.1"),
        port,
    )


async def _reload_qq_adapter_client(
    new_cfg: dict,
    old_cfg: dict | None,
) -> None:
    """Apply QQ adapter enable/adapter/host/port changes without requiring a restart."""
    from qq_adapter import QQAdapterClient
    from qq_adapter_handler import register_qq_adapter_handlers
    from web.debug_server import broadcast_qq_adapter_status, init_debug

    old_client = app_state.qq_adapter_client
    old_sig = _qq_adapter_runtime_signature(old_cfg or {})
    new_sig = _qq_adapter_runtime_signature(new_cfg)

    if not new_sig[0]:
        if old_client is not None:
            await old_client.stop()
        app_state.qq_adapter_client = None
        init_debug(app_state.TIMEZONE, None)
        await broadcast_qq_adapter_status()
        return

    if old_client is not None and old_sig == new_sig:
        old_client.adapter = new_sig[1]
        old_client.adapter_name = str(new_cfg.get("name", "") or new_sig[1])
        old_client.set_status_change_handler(broadcast_qq_adapter_status)
        init_debug(app_state.TIMEZONE, old_client)
        await broadcast_qq_adapter_status()
        return

    if old_client is not None:
        await old_client.stop()

    client = QQAdapterClient(
        bot_name=app_state.BOT_NAME,
        adapter=new_sig[1],
        adapter_name=str(new_cfg.get("name", "") or new_sig[1]),
    )
    app_state.qq_adapter_client = client
    register_qq_adapter_handlers()
    client.set_status_change_handler(broadcast_qq_adapter_status)
    init_debug(app_state.TIMEZONE, client)
    await broadcast_qq_adapter_status()
    await client.start(new_sig[2], new_sig[3])


@settings_bp.route("/settings")
async def settings_page():
    return await render_template("settings.html")


@settings_bp.route("/settings/full", methods=["GET"])
async def settings_get():
    """返回完整配置供前端填充表单。"""
    cfg = deepcopy(app_state.config)
    normalize_profile_config_inplace(cfg)
    gen_cfg = normalize_generation_config(cfg.get("generation"))
    return jsonify({
        "provider": get_selected_provider_name(cfg),
        "model_providers": get_model_providers(cfg),
        "model": cfg.get("model", ""),
        "model_name": cfg.get("model_name", ""),
        "vision": cfg.get("vision", True),
        "vision_bridge": cfg.get("vision_bridge", {}),
        "generation": {
            **gen_cfg,
            "retry_on_new_message": gen_cfg.get("retry_on_new_message", True),
            "final_reminder": gen_cfg.get("final_reminder", True),
            "enable_thinking": gen_cfg.get("enable_thinking", True),
        },
        "max_calls_per_minute": cfg.get("max_calls_per_minute", 15),
        "bot_name": cfg.get("bot_name", ""),
        "guardian": cfg.get("guardian", {"name": "", "id": ""}),
        "timezone": cfg.get("timezone", "Asia/Shanghai"),
        "qq_adapter": cfg.get("qq_adapter", {}),
        "tts": cfg.get("tts", {
            "enabled": False,
            "host": "127.0.0.1",
            "port": 8765,
            "secret_token": "",
            "max_concurrent_tasks_per_plugin": 8,
        }),
        "web_search": _default_web_search_cfg(cfg),
        "browser_control": normalize_browser_control_config(cfg.get("browser_control")),
        "alerting": cfg.get("alerting", {
            "enabled": False,
            "heartbeat_timeout": 120,
            "cooldown": 600,
            "subject_prefix": "[AIcarus 告警]",
            "qq_adapter_restart": {
                "enabled": False,
                "command": "",
                "args": [],
                "cwd": "",
                "stop_command": "",
                "stop_image_names": ["QQ.exe"],
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
        "cognition_compression": _default_compression_cfg(cfg, gen_cfg),
        "memory": _default_memory_cfg(cfg),
        "slow_thinking": cfg.get("slow_thinking", {}),
        "typing_speed": cfg.get("typing_speed", 1.0),
        "persona": app_state.persona,
        "style": app_state.style_prompt,
        "social_tips_private": app_state.social_tips_private,
        "social_tips_group": app_state.social_tips_group,
        "social_tips_temp": app_state.social_tips_temp,
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
        if "enable_thinking" in data["generation"]:
            new_gen["enable_thinking"] = bool(data["generation"]["enable_thinking"])
        if "llm_contents_max_rounds" in data["generation"]:
            new_gen["llm_contents_max_rounds"] = int(data["generation"]["llm_contents_max_rounds"])
        if "cognition_compression_trigger_rounds" in data["generation"]:
            new_gen["cognition_compression_trigger_rounds"] = int(
                data["generation"]["cognition_compression_trigger_rounds"]
            )
        if "world_multimodal_image_limit" in data["generation"]:
            new_gen["world_multimodal_image_limit"] = int(
                data["generation"]["world_multimodal_image_limit"]
            )
        new_gen = normalize_generation_config(new_gen)
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
    if "qq_adapter" in data and isinstance(data["qq_adapter"], dict):
        new_cfg["qq_adapter"] = data["qq_adapter"]
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
    if "web_search" in data and isinstance(data["web_search"], dict):
        ws_data = data["web_search"]
        new_ws = dict(new_cfg.get("web_search", {}))
        if "searxng" in ws_data and isinstance(ws_data["searxng"], dict):
            sx_data = ws_data["searxng"]
            new_sx = dict(new_ws.get("searxng", {}))
            if "enabled" in sx_data:
                new_sx["enabled"] = bool(sx_data["enabled"])
            if "base_url" in sx_data:
                base_url = str(sx_data.get("base_url") or "").strip()
                new_sx["base_url"] = base_url or "http://127.0.0.1:8888"
            if "language" in sx_data:
                language = str(sx_data.get("language") or "").strip()
                new_sx["language"] = language or "zh-CN"
            if "safesearch" in sx_data:
                new_sx["safesearch"] = max(0, min(2, int(sx_data["safesearch"])))
            new_ws["searxng"] = new_sx
        new_cfg["web_search"] = new_ws
    if "browser_control" in data and isinstance(data["browser_control"], dict):
        new_cfg["browser_control"] = normalize_browser_control_config(data["browser_control"])
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
        # QQ adapter 自动重启子节点
        if "qq_adapter_restart" in ad and isinstance(ad["qq_adapter_restart"], dict):
            nr_in = ad["qq_adapter_restart"]
            nr_out = dict(new_alerting.get("qq_adapter_restart", {}))
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
            new_alerting["qq_adapter_restart"] = nr_out
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
    if "cognition_compression" in data and isinstance(data["cognition_compression"], dict):
        cc_data = data["cognition_compression"]
        new_cc = dict(new_cfg.get("cognition_compression", {}))
        for key in ("model",):
            if key in cc_data:
                if cc_data[key]:
                    new_cc[key] = cc_data[key]
                else:
                    new_cc.pop(key, None)
        if "provider" in cc_data:
            provider = cc_data.get("provider")
            if provider:
                new_cc["provider"] = provider
            else:
                new_cc.pop("provider", None)
        new_cc.pop("profile", None)
        new_cc.pop("base_url", None)
        new_cc.pop("api_key_env", None)
        if "generation" in cc_data and isinstance(cc_data["generation"], dict):
            gen_data = cc_data["generation"]
            new_gen = dict(new_cc.get("generation", {}))
            if "temperature" in gen_data:
                new_gen["temperature"] = max(0.0, min(2.0, float(gen_data["temperature"])))
            if "max_output_tokens" in gen_data:
                new_gen["max_output_tokens"] = max(256, int(gen_data["max_output_tokens"]))
            new_cc["generation"] = new_gen
        new_cfg["cognition_compression"] = new_cc
    if "memory" in data and isinstance(data["memory"], dict):
        mem_data = data["memory"]
        new_mem = dict(new_cfg.get("memory", {}))
        new_mem.pop("max_entries", None)
        if "max_active" in mem_data:
            new_mem["max_active"] = max(1, int(mem_data["max_active"]))
        if "max_passive" in mem_data:
            new_mem["max_passive"] = max(1, int(mem_data["max_passive"]))
        if "auto_archive" in mem_data and isinstance(mem_data["auto_archive"], dict):
            aa_data = mem_data["auto_archive"]
            new_aa = dict(new_mem.get("auto_archive", {}))
            if "enabled" in aa_data:
                new_aa["enabled"] = bool(aa_data["enabled"])
            for key in ("model",):
                if key in aa_data:
                    if aa_data[key]:
                        new_aa[key] = aa_data[key]
                    else:
                        new_aa.pop(key, None)
            if "provider" in aa_data:
                provider = aa_data.get("provider")
                if provider:
                    new_aa["provider"] = provider
                else:
                    new_aa.pop("provider", None)
            new_aa.pop("profile", None)
            new_aa.pop("base_url", None)
            new_aa.pop("api_key_env", None)
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

    normalize_qq_adapter_config(new_cfg)

    raw_auto_archive = (
        data.get("memory", {}).get("auto_archive", {})
        if isinstance(data.get("memory"), dict)
        else {}
    )
    auto_archive_required = _section_enabled(raw_auto_archive, True)

    for error in (
        _payload_binding_error("主模型", data),
        _payload_binding_error("IS 中断哨兵", data.get("is", {}), bool(data.get("is", {}).get("enabled", True))) if isinstance(data.get("is"), dict) else None,
        _payload_binding_error("上下文压缩模型", data.get("cognition_compression", {})) if isinstance(data.get("cognition_compression"), dict) else None,
        _payload_binding_error(
            "记忆归档模型",
            raw_auto_archive if isinstance(raw_auto_archive, dict) else {},
            auto_archive_required,
        ) if isinstance(data.get("memory"), dict) else None,
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

    new_auto_archive = new_cfg.get("memory", {}).get("auto_archive", {})

    for error in (
        _validate_model_binding("主模型", new_cfg),
        _validate_model_binding("IS 中断哨兵", new_cfg.get("is", {}), bool(new_cfg.get("is", {}).get("enabled", True))),
        _validate_model_binding("上下文压缩模型", new_cfg.get("cognition_compression", {}), bool(new_cfg.get("cognition_compression", {}))),
        _validate_model_binding(
            "记忆归档模型",
            new_auto_archive if isinstance(new_auto_archive, dict) else {},
            _section_enabled(new_auto_archive, True),
        ),
        _validate_model_binding("Vision Bridge", new_cfg.get("vision_bridge", {}), bool(new_cfg.get("vision_bridge", {}).get("enabled", False))),
        _validate_model_binding("慢思考模型", new_cfg.get("slow_thinking", {}), bool(new_cfg.get("slow_thinking", {}).get("enabled", False))),
    ):
        if error:
            return jsonify({"success": False, "error": error}), 400

    if getattr(app_state, "webui_only", False) or getattr(app_state, "webui_standalone", False):
        await asyncio.to_thread(save_config, new_cfg)
        app_state.config = new_cfg
        app_state.GEN = new_cfg.get("generation", {})
        app_state.MODEL = new_cfg.get("model", app_state.MODEL)
        app_state.MODEL_NAME = new_cfg.get("model_name", app_state.MODEL_NAME)
        return jsonify({"success": True, "applied": False})

    # ── 热重载 adapter + 写 config（全部在线程池，避免阻塞事件循环）──────────
    # create_adapter / VisionBridge 会初始化 httpx.Client，属于慢同步操作
    def _create_and_save():
        adapter = create_adapter(new_cfg)
        is_cfg_ = new_cfg.get("is", {})
        is_adapter_ = None
        if is_cfg_.get("enabled", True):
            is_adapter_ = create_adapter(build_is_adapter_cfg(new_cfg, is_cfg_))
        archiver_cfg_ = new_cfg.get("memory", {}).get("auto_archive", {})
        archiver_adapter_ = None
        if (
            archiver_cfg_.get("enabled", True)
            and archiver_cfg_.get("provider")
            and archiver_cfg_.get("model")
        ):
            archiver_adapter_ = create_adapter(
                build_archiver_adapter_cfg(new_cfg, archiver_cfg_)
            )
        compression_cfg_ = new_cfg.get("cognition_compression", {})
        compression_adapter_ = None
        if compression_cfg_.get("provider") and compression_cfg_.get("model"):
            compression_adapter_ = create_adapter(
                build_compression_adapter_cfg(new_cfg, compression_cfg_)
            )
        st_cfg_ = new_cfg.get("slow_thinking", {})
        st_adapter_ = None
        if st_cfg_.get("enabled", True) and st_cfg_.get("provider") and st_cfg_.get("model"):
            st_adapter_ = create_adapter(build_slow_thinking_adapter_cfg(new_cfg, st_cfg_))
        save_config(new_cfg)
        vb = VisionBridge(new_cfg)
        return (
            adapter,
            is_cfg_,
            is_adapter_,
            archiver_cfg_,
            archiver_adapter_,
            compression_cfg_,
            compression_adapter_,
            st_cfg_,
            st_adapter_,
            vb,
        )

    try:
        (
            new_adapter,
            new_is_cfg,
            new_is_adapter,
            new_archiver_cfg,
            new_archiver_adapter,
            new_compression_cfg,
            new_compression_adapter,
            new_st_cfg,
            new_st_adapter,
            new_vision_bridge,
        ) = await asyncio.to_thread(_create_and_save)
    except Exception as e:
        return jsonify({"success": False, "error": f"adapter 初始化失败: {e}"}), 400

    # ── 应用到运行时 ──────────────────────────────────────
    app_state.config = new_cfg
    app_state.adapter = new_adapter
    # ── 热重载 IS adapter ────────────────────────────────
    app_state.is_cfg = new_is_cfg
    app_state.is_adapter = new_is_adapter
    # ── 热重载 archiver adapter ──────────────────────────
    app_state.archiver_cfg = new_archiver_cfg
    app_state.archiver_adapter = new_archiver_adapter
    # ── 热重载上下文压缩 adapter ──────────────────────────
    app_state.cognition_compression_cfg = new_compression_cfg
    app_state.cognition_compression_adapter = new_compression_adapter
    # ── 热重载 slow_thinking adapter ─────────────────────
    app_state.slow_thinking_cfg = new_st_cfg
    app_state.slow_thinking_adapter = new_st_adapter
    app_state.MODEL = new_cfg.get("model", app_state.MODEL)
    app_state.MODEL_NAME = new_cfg.get("model_name", app_state.MODEL_NAME)
    app_state.GEN = new_cfg.get("generation", {})
    app_state.MAX_CALLS_PER_MINUTE = new_cfg.get("max_calls_per_minute", 15)
    app_state.MAX_CONTEXT = int(new_cfg.get("max_context", 10))
    app_state.TIMEZONE = ZoneInfo(new_cfg["timezone"])
    app_state.BOT_NAME = new_cfg.get("bot_name", app_state.BOT_NAME)
    old_qq_adapter_cfg = app_state.qq_adapter_cfg
    app_state.qq_adapter_cfg = new_cfg.get("qq_adapter", {}) or {}
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

    try:
        await _reload_qq_adapter_client(app_state.qq_adapter_cfg, old_qq_adapter_cfg)
    except Exception as exc:
        logger.exception("热重载 QQ adapter 失败")
        return jsonify({"success": False, "error": f"QQ adapter 热重载失败: {exc}"}), 400

    # ── 热重载 AlertManager 与 QQAdapterClient 心跳监视 ──────
    try:
        from alerting import AlertManager
        from qq_adapter_supervisor import QQAdapterSupervisor
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
        # QQ adapter 监管器热重载
        new_supervisor = QQAdapterSupervisor(
            new_alerting_cfg.get("qq_adapter_restart", {}) or {},
            client=app_state.qq_adapter_client,
            alert=new_alert,
        )
        app_state.qq_adapter_supervisor = new_supervisor
        if app_state.qq_adapter_client is not None:
            if new_alert.enabled:
                app_state.qq_adapter_client.set_alert_manager(
                    new_alert,
                    heartbeat_timeout=float(new_alerting_cfg.get("heartbeat_timeout", 120)),
                )
            else:
                # 关闭告警：解绑 alert，watchdog 仍在跑但不会发邮件
                app_state.qq_adapter_client.set_alert_manager(None, heartbeat_timeout=120.0)
            # 同步重启能力
            app_state.qq_adapter_client.set_supervisor(
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


# ── Prompt 文档独立保存（style / social_tips）──────────────────────────────────

_PROMPT_DOC_ATTR: dict[str, str] = {
    "style": "style_prompt",
    "social_tips_private": "social_tips_private",
    "social_tips_group": "social_tips_group",
    "social_tips_temp": "social_tips_temp",
}


@settings_bp.route("/settings/prompt_doc", methods=["POST"])
async def prompt_doc_save():
    """独立保存 style.md 或 social_tips/*.md，并热更新运行时。"""
    data = await request.get_json() or {}
    key = data.get("key", "")
    text = data.get("text", "")
    if key not in _PROMPT_DOC_ATTR:
        return jsonify({"success": False, "error": f"不支持的 key: {key}"}), 400
    await asyncio.to_thread(save_prompt_doc, key, text)
    setattr(app_state, _PROMPT_DOC_ATTR[key], text)
    init_session_globals(
        max_context=app_state.MAX_CONTEXT,
        timezone=app_state.TIMEZONE,
        persona=app_state.persona,
        model_name=app_state.MODEL_NAME,
        guardian_name=app_state.config.get("guardian", {}).get("name", ""),
        guardian_id=app_state.config.get("guardian", {}).get("id", ""),
        **{_PROMPT_DOC_ATTR[key]: text},
    )
    return jsonify({"success": True})


# ── Self Image 上传 / 列出 / 删除 / 查看 ──────────────────────────────────────

_SELF_IMAGE_DIR = Path(__file__).resolve().parents[2] / "config" / "self_image"
_ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


@settings_bp.route("/settings/self_image", methods=["GET"])
async def self_image_list():
    """列出 config/self_image/ 下的所有图片文件。"""
    _SELF_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for f in sorted(_SELF_IMAGE_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in _ALLOWED_IMAGE_EXTS:
            files.append({"name": f.name, "size": f.stat().st_size})
    return jsonify({"files": files})


@settings_bp.route("/settings/self_image/<path:filename>", methods=["GET"])
async def self_image_serve(filename: str):
    """提供 self_image 图片内容（防路径穿越）。"""
    target = (_SELF_IMAGE_DIR / filename).resolve()
    if not str(target).startswith(str(_SELF_IMAGE_DIR.resolve())):
        return jsonify({"error": "forbidden"}), 403
    if not target.is_file():
        return jsonify({"error": "not found"}), 404
    mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return await send_file(str(target), mimetype=mime)


@settings_bp.route("/settings/self_image", methods=["POST"])
async def self_image_upload():
    """上传图片到 config/self_image/。"""
    _SELF_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    files = await request.files
    uploaded = []
    for _field, f in files.items(multi=True):
        ext = Path(f.filename).suffix.lower() if f.filename else ""
        if ext not in _ALLOWED_IMAGE_EXTS:
            return jsonify({"success": False, "error": f"不支持的文件类型: {ext}"}), 400
        # 只保留安全文件名
        safe_name = Path(f.filename).name if f.filename else "image"
        dest = _SELF_IMAGE_DIR / safe_name
        data = f.read()
        await asyncio.to_thread(dest.write_bytes, data)
        uploaded.append(safe_name)
    return jsonify({"success": True, "uploaded": uploaded})


@settings_bp.route("/settings/self_image/<path:filename>", methods=["DELETE"])
async def self_image_delete(filename: str):
    """删除 config/self_image/ 下的指定文件（防路径穿越）。"""
    target = (_SELF_IMAGE_DIR / filename).resolve()
    if not str(target).startswith(str(_SELF_IMAGE_DIR.resolve())):
        return jsonify({"error": "forbidden"}), 403
    if not target.is_file():
        return jsonify({"error": "not found"}), 404
    await asyncio.to_thread(target.unlink)
    return jsonify({"success": True})


# ── 缓存管理 ────────────────────────────────────────────────────────────────

_BASE_DIR = Path(__file__).resolve().parents[2]
_CACHE_DIRS: dict[str, Path] = {
    "image": _BASE_DIR / "cache" / "image",
    "tts":   _BASE_DIR / "cache" / "tts",
    "stickers": _BASE_DIR / "cache" / "stickers",
}


def _dir_size(p: Path) -> int:
    """返回目录占用字节数（不存在时返回 0）。"""
    if not p.exists():
        return 0
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _clear_dir(p: Path) -> int:
    """删除目录下所有文件（保留目录本身），返回删除文件数。"""
    if not p.exists():
        return 0
    count = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                f.unlink()
                count += 1
            except OSError:
                pass
    return count


@settings_bp.route("/settings/cache/info", methods=["GET"])
async def cache_info():
    """返回各缓存目录的占用大小（字节）。"""
    sizes = {}
    for name, path in _CACHE_DIRS.items():
        sizes[name] = await asyncio.to_thread(_dir_size, path)
    return jsonify({"sizes": sizes})


@settings_bp.route("/settings/cache/clear", methods=["POST"])
async def cache_clear():
    """清理指定缓存目录。body: {"targets": ["image", "tts", "stickers"]}"""
    data = await request.get_json() or {}
    targets = data.get("targets") or list(_CACHE_DIRS.keys())
    results = {}
    for name in targets:
        if name not in _CACHE_DIRS:
            continue
        results[name] = await asyncio.to_thread(_clear_dir, _CACHE_DIRS[name])
    return jsonify({"success": True, "deleted": results})


# ── 表情包管理 ────────────────────────────────────────────────────────────────

@settings_bp.route("/stickers")
async def stickers_page():
    return await render_template("stickers.html")


@settings_bp.route("/api/stickers/list", methods=["GET"])
async def stickers_list():
    """返回所有表情包元数据列表。"""
    from llm.media.sticker_collection import list_all
    items = await asyncio.to_thread(list_all)
    return jsonify({"stickers": items})


@settings_bp.route("/api/stickers/upload", methods=["POST"])
async def stickers_upload():
    """上传新表情包。multipart: file=<图片>, description=<描述>"""
    from llm.media.sticker_collection import save_sticker
    files = await request.files
    form = await request.form
    file = files.get("file")
    if not file:
        return jsonify({"success": False, "error": "未提供文件"}), 400
    description = (form.get("description") or "").strip()
    raw = file.read()
    mime = file.content_type or "image/jpeg"
    # 仅允许图片类型
    if not mime.startswith("image/"):
        return jsonify({"success": False, "error": "仅支持图片文件"}), 400
    result = await asyncio.to_thread(save_sticker, raw, mime, description)
    if result is None:
        return jsonify({"success": False, "error": "已达表情包数量上限"}), 400
    sid, is_dup = result
    return jsonify({"success": True, "id": sid, "duplicate": is_dup})


@settings_bp.route("/api/stickers/<sticker_id>", methods=["PATCH"])
async def stickers_update(sticker_id: str):
    """修改表情包描述。body: {"description": "..."}"""
    from llm.media.sticker_collection import update_sticker_description
    if not sticker_id.isalnum():
        return jsonify({"success": False, "error": "invalid id"}), 400
    data = await request.get_json() or {}
    description = str(data.get("description") or "")
    ok = await asyncio.to_thread(update_sticker_description, sticker_id, description)
    if not ok:
        return jsonify({"success": False, "error": "表情包不存在"}), 404
    return jsonify({"success": True})


@settings_bp.route("/api/stickers/<sticker_id>", methods=["DELETE"])
async def stickers_delete(sticker_id: str):
    """删除指定表情包。"""
    from llm.media.sticker_collection import delete_sticker
    if not sticker_id.isalnum():
        return jsonify({"success": False, "error": "invalid id"}), 400
    ok = await asyncio.to_thread(delete_sticker, sticker_id)
    if not ok:
        return jsonify({"success": False, "error": "表情包不存在"}), 404
    return jsonify({"success": True})


@settings_bp.route("/api/stickers/reconcile", methods=["POST"])
async def stickers_reconcile():
    """全量检查并修复表情包收藏（去重、补编号、清理孤儿文件）。"""
    from llm.media.sticker_collection import reconcile_stickers
    stats = await asyncio.to_thread(reconcile_stickers)
    return jsonify({"success": True, "stats": stats})
