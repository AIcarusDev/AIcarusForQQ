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
import io
import logging
import mimetypes
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from PIL import Image as PILImage, UnidentifiedImageError
from quart import Blueprint, render_template, request, jsonify, send_file

import app_state
from config_loader import (
    normalize_guardian_info,
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
from llm.core.provider import (
    create_adapter,
    build_tool_execution_guard_adapter_cfg,
    build_event_extraction_adapter_cfg,
    build_memory_processing_adapter_cfg,
    build_slow_thinking_adapter_cfg,
    build_compression_adapter_cfg,
)
from llm.compression.config import normalize_generation_config
from llm.core.duplicate_response_guard import normalize_duplicate_model_response_guard_config
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
from platforms import PlatformRegistry
from platforms.core import CoreRuntime
from platforms.registry import get_platform
from platforms.qq import QQRuntime
from platforms.qq.adapter.config import normalize_qq_platform_config
from browser.config import normalize_browser_control_config
from browser.session import launch_isolated_login_browser
from runtime.cache_maintenance import CacheMaintenanceError, cache_maintenance_service
from skills import load_skill_user_body, save_skill_user_body

logger = logging.getLogger("AICQ.web.settings")

settings_bp = Blueprint("settings", __name__)
ROOT_DIR = Path(__file__).resolve().parents[2]
LEGACY_MEMORY_CONFIG_KEY = "v" + "2"

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
    memory_cfg.pop("max_active", None)
    memory_cfg.pop("max_passive", None)
    auto_archive = memory_cfg.get("auto_archive")
    if isinstance(auto_archive, dict):
        auto_archive = dict(auto_archive)
    else:
        auto_archive = {}
    auto_archive.setdefault("enabled", True)
    memory_cfg["auto_archive"] = auto_archive
    processing = memory_cfg.get("processing")
    if isinstance(processing, dict):
        processing = dict(processing)
    else:
        processing = {}
    processing.setdefault("enabled", False)
    processing.setdefault("event_structuring_enabled", False)
    processing.setdefault("algorithmic_storyline_enabled", False)
    processing.setdefault("dry_run", True)
    processing.setdefault("solidify", False)
    processing.setdefault("max_candidate_storylines_per_maintenance", 100)
    processing.setdefault("maintenance_timeout_seconds", 300)
    processing.setdefault("storyline_synthesis_max_inputs_per_maintenance", 32)
    processing.setdefault("storyline_synthesis_max_retries", 3)
    processing.setdefault("provider", "")
    processing.setdefault("model", "")
    processing_gen = processing.get("generation")
    if isinstance(processing_gen, dict):
        processing_gen = dict(processing_gen)
    else:
        processing_gen = {}
    processing_gen.setdefault("temperature", 0.2)
    processing_gen.setdefault("max_output_tokens", 4000)
    processing_gen.setdefault("enable_thinking", False)
    processing["generation"] = processing_gen
    memory_cfg["processing"] = processing
    legacy = memory_cfg.pop(LEGACY_MEMORY_CONFIG_KEY, None)
    if isinstance(legacy, dict):
        memory_cfg.setdefault(
            "memory_predicate_similarity_threshold",
            legacy.get("memory_predicate_similarity_threshold", 0.8),
        )
        memory_cfg.setdefault(
            "memory_recall_max_results",
            legacy.get("memory_recall_max_results", 8),
        )
        memory_cfg.setdefault(
            "memory_recall_recent_fallback",
            legacy.get("memory_recall_recent_fallback", True),
        )
        if "embedding" not in memory_cfg and isinstance(legacy.get("embedding"), dict):
            memory_cfg["embedding"] = dict(legacy["embedding"])
    memory_cfg.setdefault("memory_predicate_similarity_threshold", 0.8)
    memory_cfg.setdefault("memory_recall_max_results", 8)
    memory_cfg.setdefault("memory_recall_recent_fallback", True)
    embedding = memory_cfg.get("embedding")
    if isinstance(embedding, dict):
        embedding = dict(embedding)
    else:
        embedding = {}
    embedding.setdefault("provider", "hash")
    embedding.setdefault("model", "")
    embedding.setdefault("dim", 128)
    memory_cfg["embedding"] = embedding
    events = memory_cfg.get("events")
    if isinstance(events, dict):
        events = dict(events)
    else:
        events = {}
    events.setdefault("recall_limit", 6)
    events.setdefault("world_query_chunks", 6)
    events.setdefault("cognition_query_chunks", 3)
    memory_cfg["events"] = events
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


def _normalize_send_message_shape(value: object) -> str | None:
    shape = str(value or "").strip().lower().replace("-", "_")
    if shape in {"array", "messages", "multi", "multi_message", "batch"}:
        return "array"
    if shape in {"single", "single_message", "message", "segments"}:
        return "single"
    return None


def _default_tools_cfg(cfg: dict) -> dict:
    """Return tool config with UI-visible defaults without mutating saved config."""
    raw = cfg.get("tools", {}) if isinstance(cfg, dict) else {}
    tools = deepcopy(raw) if isinstance(raw, dict) else {}
    send_raw = tools.get("send_message", {})
    send_message = deepcopy(send_raw) if isinstance(send_raw, dict) else {}
    shape = _normalize_send_message_shape(send_message.get("message_shape"))
    send_message["message_shape"] = shape or "array"
    tools["send_message"] = send_message
    return tools


def _qq_platform_runtime_signature(cfg: dict) -> tuple[bool, str, str, int]:
    adapter = cfg.get("adapter") if isinstance(cfg.get("adapter"), dict) else {}
    reverse_ws = adapter.get("reverse_ws") if isinstance(adapter.get("reverse_ws"), dict) else {}
    try:
        port = int(reverse_ws.get("port", 8078))
    except (TypeError, ValueError):
        port = 8078
    return (
        bool(cfg.get("enabled", False)),
        str(adapter.get("type", "auto") or "auto"),
        str(reverse_ws.get("host", "127.0.0.1") or "127.0.0.1"),
        port,
    )


async def _reload_qq_platform_client(
    new_cfg: dict,
    old_cfg: dict | None,
) -> None:
    """Apply QQ platform enable/adapter/host/port changes without requiring a restart."""
    from platforms.qq.handler import register_qq_platform_handlers
    from web.debug_server import broadcast_platform_status, init_debug

    if app_state.platform_registry is None:
        app_state.platform_registry = PlatformRegistry()
    if get_platform("core") is None:
        app_state.platform_registry.register(
            CoreRuntime((app_state.config or {}).get("platforms", {}).get("core", {}) or {})
        )
    runtime = get_platform("qq")
    if runtime is None:
        runtime = QQRuntime(new_cfg)
        app_state.platform_registry.register(runtime)
    old_client = runtime.client
    old_sig = _qq_platform_runtime_signature(old_cfg or {})
    new_sig = _qq_platform_runtime_signature(new_cfg)
    runtime.config = new_cfg

    if not new_sig[0]:
        if old_client is not None:
            await old_client.stop()
        runtime.client = None
        init_debug(app_state.TIMEZONE, None)
        await broadcast_platform_status()
        return

    if old_client is not None and old_sig == new_sig:
        old_client.adapter = new_sig[1]
        adapter_cfg = new_cfg.get("adapter") if isinstance(new_cfg.get("adapter"), dict) else {}
        old_client.adapter_name = str(adapter_cfg.get("name", "") or new_sig[1])
        old_client.set_status_change_handler(broadcast_platform_status)
        init_debug(app_state.TIMEZONE, old_client)
        await broadcast_platform_status()
        return

    if old_client is not None:
        await old_client.stop()

    client = runtime.ensure_client(bot_name=app_state.SELF_NAME)
    register_qq_platform_handlers(runtime)
    client.set_status_change_handler(broadcast_platform_status)
    init_debug(app_state.TIMEZONE, client)
    await broadcast_platform_status()
    await client.start(new_sig[2], new_sig[3])


@settings_bp.route("/settings")
async def settings_page():
    return await render_template("settings.html")


def _browser_login_profile_dir(raw_value: object) -> Path:
    browser_cfg = normalize_browser_control_config(app_state.config.get("browser_control"))
    configured = str(raw_value or browser_cfg.get("profile_dir") or "").strip()
    if not configured:
        configured = str(browser_cfg.get("profile_dir") or "cache/browser_profile/default")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def _browser_login_url(raw_value: object) -> str:
    url = str(raw_value or "").strip() or "https://accounts.google.com/"
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("登录地址必须以 http:// 或 https:// 开头")
    return url


@settings_bp.route("/settings/browser/login", methods=["POST"])
async def browser_login_launch():
    """Launch a visible system browser using browser_control's persistent profile."""
    data = await request.get_json() or {}
    try:
        profile_dir = _browser_login_profile_dir(data.get("profile_dir"))
        login_url = _browser_login_url(data.get("url"))
        chrome_path, process_id = launch_isolated_login_browser(
            profile_dir=profile_dir,
            url=login_url,
        )
    except Exception as exc:
        logger.warning("[settings] 启动浏览器登录失败", exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 400

    return jsonify(
        {
            "success": True,
            "profile_dir": str(profile_dir),
            "url": login_url,
            "browser": chrome_path,
            "process_id": process_id,
            "network_isolation": "agent_gateway",
        }
    )


@settings_bp.route("/settings/full", methods=["GET"])
async def settings_get():
    """返回完整配置供前端填充表单。"""
    cfg = deepcopy(app_state.config)
    cfg.pop("is", None)
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
            "final_reminder": gen_cfg.get("final_reminder", True),
            "enable_thinking": gen_cfg.get("enable_thinking", True),
            "duplicate_model_response_guard": normalize_duplicate_model_response_guard_config(
                gen_cfg.get("duplicate_model_response_guard")
            ),
        },
        "max_calls_per_minute": cfg.get("max_calls_per_minute", 15),
        "self_name": cfg.get("self_name", ""),
        "guardian": normalize_guardian_info(cfg.get("guardian")),
        "timezone": cfg.get("timezone", "Asia/Shanghai"),
        "skills": {
            "qq_social_style": load_skill_user_body("qq-social-style"),
        },
        "platforms": cfg.get("platforms", {}),
        "tts": cfg.get("tts", {
            "enabled": False,
            "host": "127.0.0.1",
            "port": 8765,
            "secret_token": "",
            "max_concurrent_tasks_per_plugin": 8,
        }),
        "web_search": _default_web_search_cfg(cfg),
        "tools": _default_tools_cfg(cfg),
        "browser_control": normalize_browser_control_config(cfg.get("browser_control")),
        "alerting": cfg.get("alerting", {
            "enabled": False,
            "heartbeat_timeout": 120,
            "cooldown": 600,
            "subject_prefix": "[AIcarus 告警]",
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
        "tool_execution_guard": cfg.get("tool_execution_guard", {}),
        "cognition_compression": _default_compression_cfg(cfg, gen_cfg),
        "memory": _default_memory_cfg(cfg),
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
    new_cfg.pop("is", None)
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

    def _apply_generation_controls(
        current: dict,
        incoming: dict,
        *,
        min_tokens: int,
        default_temperature: float = 0.3,
    ) -> dict:
        new_gen = dict(current or {})
        if "temperature" in incoming:
            new_gen["temperature"] = max(
                0.0,
                min(2.0, float(incoming.get("temperature", default_temperature))),
            )
        if "max_output_tokens" in incoming:
            new_gen["max_output_tokens"] = max(min_tokens, int(incoming["max_output_tokens"]))
        if "enable_thinking" in incoming:
            new_gen["enable_thinking"] = bool(incoming["enable_thinking"])
        for key, value in incoming.items():
            if key not in ("temperature", "max_output_tokens", "enable_thinking") and value is not None:
                new_gen[key] = value
        return new_gen

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
        if "duplicate_model_response_guard" in data["generation"]:
            new_gen["duplicate_model_response_guard"] = (
                normalize_duplicate_model_response_guard_config(
                    data["generation"].get("duplicate_model_response_guard")
                )
            )
        new_gen = normalize_generation_config(new_gen)
        new_cfg["generation"] = new_gen
    if "max_calls_per_minute" in data:
        new_cfg["max_calls_per_minute"] = int(data["max_calls_per_minute"])
    if "typing_speed" in data:
        speed_val = float(data["typing_speed"])
        new_cfg["typing_speed"] = speed_val if speed_val > 0 else 1.0
    if "self_name" in data:
        new_cfg["self_name"] = data["self_name"]
    if "timezone" in data:
        tz_val = (data.get("timezone") or "").strip() or "Asia/Shanghai"
        new_cfg["timezone"] = tz_val
    if "platforms" in data and isinstance(data["platforms"], dict):
        new_platforms = dict(new_cfg.get("platforms", {}))
        for platform_name, platform_cfg in data["platforms"].items():
            if isinstance(platform_cfg, dict):
                new_platforms[str(platform_name)] = platform_cfg
        new_cfg["platforms"] = new_platforms
    if "tools" in data and isinstance(data["tools"], dict):
        tools_data = data["tools"]
        current_tools = new_cfg.get("tools", {})
        new_tools = dict(current_tools) if isinstance(current_tools, dict) else {}
        send_data = tools_data.get("send_message")
        if isinstance(send_data, dict):
            current_send = new_tools.get("send_message", {})
            new_send = dict(current_send) if isinstance(current_send, dict) else {}
            if "message_shape" in send_data:
                shape = _normalize_send_message_shape(send_data.get("message_shape"))
                if shape is None:
                    return jsonify({
                        "success": False,
                        "error": "send_message.message_shape 只能是 array 或 single",
                    }), 400
                new_send["message_shape"] = shape
            new_tools["send_message"] = new_send
        new_cfg["tools"] = new_tools
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
        # QQ 平台自动重启子节点
        if "qq_adapter_restart" in ad and isinstance(ad["qq_adapter_restart"], dict):
            nr_in = ad["qq_adapter_restart"]
            platforms_cfg = dict(new_cfg.get("platforms", {}))
            qq_cfg = dict(platforms_cfg.get("qq", {}))
            nr_out = dict(qq_cfg.get("supervisor", {}))
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
            qq_cfg["supervisor"] = nr_out
            platforms_cfg["qq"] = qq_cfg
            new_cfg["platforms"] = platforms_cfg
            new_alerting.pop("qq_adapter_restart", None)
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
    if "tool_execution_guard" in data and isinstance(data["tool_execution_guard"], dict):
        guard_data = data["tool_execution_guard"]
        new_guard = dict(new_cfg.get("tool_execution_guard", {}))
        if "enabled" in guard_data:
            new_guard["enabled"] = bool(guard_data["enabled"])
        for key in ("model",):
            if key in guard_data:
                if guard_data[key]:
                    new_guard[key] = guard_data[key]
                else:
                    new_guard.pop(key, None)
        if "provider" in guard_data:
            provider = guard_data.get("provider")
            if provider:
                new_guard["provider"] = provider
            else:
                new_guard.pop("provider", None)
        new_guard.pop("profile", None)
        new_guard.pop("base_url", None)
        new_guard.pop("api_key_env", None)
        if "generation" in guard_data and isinstance(guard_data["generation"], dict):
            new_guard["generation"] = _apply_generation_controls(
                new_guard.get("generation", {}),
                guard_data["generation"],
                min_tokens=64,
            )
        if "vision" in guard_data:
            new_guard["vision"] = bool(guard_data["vision"])
        new_cfg["tool_execution_guard"] = new_guard
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
            new_cc["generation"] = _apply_generation_controls(
                new_cc.get("generation", {}),
                cc_data["generation"],
                min_tokens=256,
            )
        new_cfg["cognition_compression"] = new_cc
    if "memory" in data and isinstance(data["memory"], dict):
        mem_data = data["memory"]
        new_mem = dict(new_cfg.get("memory", {}))
        new_mem.pop("max_entries", None)
        new_mem.pop("max_active", None)
        new_mem.pop("max_passive", None)
        recall_data = mem_data
        if LEGACY_MEMORY_CONFIG_KEY in mem_data and isinstance(mem_data[LEGACY_MEMORY_CONFIG_KEY], dict):
            recall_data = {**mem_data[LEGACY_MEMORY_CONFIG_KEY], **mem_data}
        if "memory_predicate_similarity_threshold" in recall_data:
            new_mem["memory_predicate_similarity_threshold"] = max(
                0.5,
                min(0.95, float(recall_data["memory_predicate_similarity_threshold"])),
            )
        if "memory_recall_max_results" in recall_data:
            new_mem["memory_recall_max_results"] = max(
                1,
                min(30, int(recall_data["memory_recall_max_results"])),
            )
        if "memory_recall_recent_fallback" in recall_data:
            new_mem["memory_recall_recent_fallback"] = bool(recall_data["memory_recall_recent_fallback"])
        if "embedding" in recall_data and isinstance(recall_data["embedding"], dict):
            embedding_data = recall_data["embedding"]
            new_embedding = dict(new_mem.get("embedding", {}))
            for key in ("provider", "model"):
                if key in embedding_data:
                    new_embedding[key] = str(embedding_data[key] or "")
            if "dim" in embedding_data:
                new_embedding["dim"] = max(1, int(embedding_data["dim"]))
            new_mem["embedding"] = new_embedding
        new_mem.pop(LEGACY_MEMORY_CONFIG_KEY, None)
        if "events" in mem_data and isinstance(mem_data["events"], dict):
            events_data = mem_data["events"]
            new_events = dict(new_mem.get("events", {}))
            if "recall_limit" in events_data:
                new_events["recall_limit"] = max(1, min(30, int(events_data["recall_limit"])))
            if "world_query_chunks" in events_data:
                new_events["world_query_chunks"] = max(0, min(20, int(events_data["world_query_chunks"])))
            if "cognition_query_chunks" in events_data:
                new_events["cognition_query_chunks"] = max(0, min(10, int(events_data["cognition_query_chunks"])))
            new_mem["events"] = new_events
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
                new_aa["generation"] = _apply_generation_controls(
                    new_aa.get("generation", {}),
                    aa_data["generation"],
                    min_tokens=256,
                )
            new_mem["auto_archive"] = new_aa
        if "processing" in mem_data and isinstance(mem_data["processing"], dict):
            mp_data = mem_data["processing"]
            new_mp = dict(new_mem.get("processing", {}))
            if "enabled" in mp_data:
                new_mp["enabled"] = bool(mp_data["enabled"])
            if "event_structuring_enabled" in mp_data:
                new_mp["event_structuring_enabled"] = bool(mp_data["event_structuring_enabled"])
            if "algorithmic_storyline_enabled" in mp_data:
                new_mp["algorithmic_storyline_enabled"] = bool(
                    mp_data["algorithmic_storyline_enabled"]
                )
            if "dry_run" in mp_data:
                new_mp["dry_run"] = bool(mp_data["dry_run"])
            if "solidify" in mp_data:
                new_mp["solidify"] = bool(mp_data["solidify"])
            if "max_candidate_storylines_per_maintenance" in mp_data:
                new_mp["max_candidate_storylines_per_maintenance"] = max(
                    1,
                    min(1000, int(mp_data["max_candidate_storylines_per_maintenance"])),
                )
            if "maintenance_timeout_seconds" in mp_data:
                new_mp["maintenance_timeout_seconds"] = max(
                    0,
                    min(3600, int(mp_data["maintenance_timeout_seconds"])),
                )
            if "storyline_synthesis_max_inputs_per_maintenance" in mp_data:
                new_mp["storyline_synthesis_max_inputs_per_maintenance"] = max(
                    1,
                    min(500, int(mp_data["storyline_synthesis_max_inputs_per_maintenance"])),
                )
            if "storyline_synthesis_max_retries" in mp_data:
                new_mp["storyline_synthesis_max_retries"] = max(
                    1,
                    min(10, int(mp_data["storyline_synthesis_max_retries"])),
                )
            for key in ("model",):
                if key in mp_data:
                    if mp_data[key]:
                        new_mp[key] = mp_data[key]
                    else:
                        new_mp.pop(key, None)
            if "provider" in mp_data:
                provider = mp_data.get("provider")
                if provider:
                    new_mp["provider"] = provider
                else:
                    new_mp.pop("provider", None)
            new_mp.pop("profile", None)
            new_mp.pop("base_url", None)
            new_mp.pop("api_key_env", None)
            if "generation" in mp_data and isinstance(mp_data["generation"], dict):
                new_mp["generation"] = _apply_generation_controls(
                    new_mp.get("generation", {}),
                    mp_data["generation"],
                    min_tokens=512,
                    default_temperature=0.2,
                )
            new_mem["processing"] = new_mp
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
            new_st["generation"] = _apply_generation_controls(
                new_st.get("generation", {}),
                st_data["generation"],
                min_tokens=64,
                default_temperature=1.0,
            )
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
        if "enable_thinking" in vb_data:
            new_vb["enable_thinking"] = bool(vb_data["enable_thinking"])
        if "temperature" in vb_data:
            new_vb["temperature"] = float(vb_data["temperature"])
        if "max_output_tokens" in vb_data:
            new_vb["max_output_tokens"] = int(vb_data["max_output_tokens"])
        if "similarity_threshold" in vb_data:
            new_vb["similarity_threshold"] = int(vb_data["similarity_threshold"])
        new_vb.pop("whitelist", None)
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

    normalize_qq_platform_config(new_cfg, remove_legacy=True)

    raw_auto_archive = (
        data.get("memory", {}).get("auto_archive", {})
        if isinstance(data.get("memory"), dict)
        else {}
    )
    raw_memory_processing = (
        data.get("memory", {}).get("processing", {})
        if isinstance(data.get("memory"), dict)
        else {}
    )
    auto_archive_required = _section_enabled(raw_auto_archive, True)
    memory_processing_required = _section_enabled(raw_memory_processing, False)

    for error in (
        _payload_binding_error("主模型", data),
        _payload_binding_error("工具执行前守门模型", data.get("tool_execution_guard", {}), bool(data.get("tool_execution_guard", {}).get("enabled", False))) if isinstance(data.get("tool_execution_guard"), dict) else None,
        _payload_binding_error("上下文压缩模型", data.get("cognition_compression", {})) if isinstance(data.get("cognition_compression"), dict) else None,
        _payload_binding_error(
            "记忆事件提取模型",
            raw_auto_archive if isinstance(raw_auto_archive, dict) else {},
            auto_archive_required,
        ) if isinstance(data.get("memory"), dict) else None,
        _payload_binding_error(
            "记忆处理模型",
            raw_memory_processing if isinstance(raw_memory_processing, dict) else {},
            memory_processing_required,
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
    new_memory_processing = new_cfg.get("memory", {}).get("processing", {})

    for error in (
        _validate_model_binding("主模型", new_cfg),
        _validate_model_binding("工具执行前守门模型", new_cfg.get("tool_execution_guard", {}), bool(new_cfg.get("tool_execution_guard", {}).get("enabled", False))),
        _validate_model_binding("上下文压缩模型", new_cfg.get("cognition_compression", {}), bool(new_cfg.get("cognition_compression", {}))),
        _validate_model_binding(
            "记忆事件提取模型",
            new_auto_archive if isinstance(new_auto_archive, dict) else {},
            _section_enabled(new_auto_archive, True),
        ),
        _validate_model_binding(
            "记忆处理模型",
            new_memory_processing if isinstance(new_memory_processing, dict) else {},
            _section_enabled(new_memory_processing, False),
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
        app_state.tool_execution_guard_cfg = new_cfg.get("tool_execution_guard", {})
        app_state.tool_execution_guard_adapter = None
        app_state.memory_processing_cfg = new_cfg.get("memory", {}).get("processing", {})
        app_state.memory_processing_adapter = None
        return jsonify({"success": True, "applied": False})

    # ── 热重载 adapter + 写 config（全部在线程池，避免阻塞事件循环）──────────
    # create_adapter / VisionBridge 会初始化 httpx.Client，属于慢同步操作
    def _create_and_save():
        adapter = create_adapter(new_cfg)
        guard_cfg_ = new_cfg.get("tool_execution_guard", {})
        guard_adapter_ = None
        if guard_cfg_.get("enabled", False) and guard_cfg_.get("provider") and guard_cfg_.get("model"):
            guard_adapter_ = create_adapter(
                build_tool_execution_guard_adapter_cfg(new_cfg, guard_cfg_)
            )
        event_extraction_cfg_ = new_cfg.get("memory", {}).get("auto_archive", {})
        event_extraction_adapter_ = None
        if (
            event_extraction_cfg_.get("enabled", True)
            and event_extraction_cfg_.get("provider")
            and event_extraction_cfg_.get("model")
        ):
            event_extraction_adapter_ = create_adapter(
                build_event_extraction_adapter_cfg(new_cfg, event_extraction_cfg_)
            )
        memory_processing_cfg_ = new_cfg.get("memory", {}).get("processing", {})
        memory_processing_adapter_ = None
        if (
            memory_processing_cfg_.get("enabled", False)
            and memory_processing_cfg_.get("provider")
            and memory_processing_cfg_.get("model")
        ):
            memory_processing_adapter_ = create_adapter(
                build_memory_processing_adapter_cfg(new_cfg, memory_processing_cfg_)
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
            guard_cfg_,
            guard_adapter_,
            event_extraction_cfg_,
            event_extraction_adapter_,
            memory_processing_cfg_,
            memory_processing_adapter_,
            compression_cfg_,
            compression_adapter_,
            st_cfg_,
            st_adapter_,
            vb,
        )

    try:
        (
            new_adapter,
            new_guard_cfg,
            new_guard_adapter,
            new_event_extraction_cfg,
            new_event_extraction_adapter,
            new_memory_processing_cfg,
            new_memory_processing_adapter,
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
    # ── 热重载工具执行前守门 adapter ─────────────────────
    app_state.tool_execution_guard_cfg = new_guard_cfg
    app_state.tool_execution_guard_adapter = new_guard_adapter
    # ── 热重载 event extraction adapter ───────────────────
    app_state.event_extraction_cfg = new_event_extraction_cfg
    app_state.event_extraction_adapter = new_event_extraction_adapter
    # ── 热重载 memory processing adapter ──────────────────
    app_state.memory_processing_cfg = new_memory_processing_cfg
    app_state.memory_processing_adapter = new_memory_processing_adapter
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
    app_state.SELF_NAME = new_cfg.get("self_name", app_state.SELF_NAME)
    qq_runtime = get_platform("qq")
    old_qq_platform_cfg = dict(getattr(qq_runtime, "config", {}) or {})
    new_qq_platform_cfg = new_cfg.get("platforms", {}).get("qq", {}) or {}
    app_state.tts_cfg = new_cfg.get("tts", {}) or {}
    app_state.rate_limiter = MinuteRateLimiter(app_state.MAX_CALLS_PER_MINUTE)
    app_state.vision_bridge = new_vision_bridge
    update_session_model_name(app_state.MODEL_NAME)
    init_session_globals(
        max_context=app_state.MAX_CONTEXT,
        timezone=ZoneInfo(new_cfg["timezone"]),
        persona=app_state.persona,
        self_name=app_state.SELF_NAME,
        model_name=app_state.MODEL_NAME,
        guardian_info=new_cfg.get("guardian"),
    )

    try:
        await _reload_qq_platform_client(new_qq_platform_cfg, old_qq_platform_cfg)
    except Exception as exc:
        logger.exception("热重载 QQ adapter 失败")
        return jsonify({"success": False, "error": f"QQ 平台热重载失败: {exc}"}), 400

    # ── 热重载 AlertManager 与 QQAdapterClient 心跳监视 ──────
    try:
        from alerting import AlertManager
        from platforms.qq.supervisor import QQAdapterSupervisor
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
        # QQ 平台监管器热重载
        qq_runtime = get_platform("qq")
        qq_client = getattr(qq_runtime, "client", None)
        new_supervisor = QQAdapterSupervisor(
            new_qq_platform_cfg.get("supervisor", {}) or {},
            client=qq_client,
            alert=new_alert,
        )
        if qq_runtime is not None:
            qq_runtime.supervisor = new_supervisor
        if qq_client is not None:
            if new_alert.enabled:
                qq_client.set_alert_manager(
                    new_alert,
                    heartbeat_timeout=float(new_alerting_cfg.get("heartbeat_timeout", 120)),
                )
            else:
                # 关闭告警：解绑 alert，watchdog 仍在跑但不会发邮件
                qq_client.set_alert_manager(None, heartbeat_timeout=120.0)
            # 同步重启能力
            qq_client.set_supervisor(
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
        self_name=app_state.SELF_NAME,
        model_name=app_state.MODEL_NAME,
        guardian_info=cfg.get("guardian"),
    )
    return jsonify({"success": True})


@settings_bp.route("/settings/guardian", methods=["POST"])
async def guardian_save():
    """独立保存监护人介绍，避免被整页设置快照覆盖。"""
    data = await request.get_json() or {}
    raw_guardian = data.get("guardian")
    if raw_guardian is not None and not isinstance(raw_guardian, str):
        return jsonify({"success": False, "error": "guardian 必须是字符串或 null"}), 400

    guardian_info = normalize_guardian_info(raw_guardian)
    new_cfg = deepcopy(app_state.config)
    new_cfg["guardian"] = guardian_info
    await asyncio.to_thread(
        save_config,
        new_cfg,
        preserve_latest_guardian=False,
    )
    app_state.config = new_cfg
    init_session_globals(
        max_context=app_state.MAX_CONTEXT,
        timezone=app_state.TIMEZONE,
        persona=app_state.persona,
        self_name=app_state.SELF_NAME,
        model_name=app_state.MODEL_NAME,
        guardian_info=guardian_info,
    )
    return jsonify({"success": True, "guardian": guardian_info})


@settings_bp.route("/settings/skills/qq-social-style", methods=["POST"])
async def qq_social_style_save():
    """Save ignored user copy of qq-social-style skill body."""
    data = await request.get_json() or {}
    body = str(data.get("body", ""))
    ok = await asyncio.to_thread(save_skill_user_body, "qq-social-style", body)
    if not ok:
        return jsonify({"success": False, "error": "保存 QQ 社交风格失败"}), 500
    return jsonify({"success": True, "body": load_skill_user_body("qq-social-style")})


# ── Self Image 上传 / 列出 / 删除 / 查看 ──────────────────────────────────────

_SELF_IMAGE_DIR = Path(__file__).resolve().parents[2] / "config" / "self_image"
_ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_SELF_IMAGE_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
_IMAGE_MIME_BY_FORMAT = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
}
_MAX_IMAGE_UPLOAD_BYTES = 8 * 1024 * 1024
_MAX_IMAGE_PIXELS = 40_000_000


class _ImageUploadError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _inspect_image_upload(raw: bytes, *, allowed_mimes: set[str]) -> str:
    if not raw:
        raise _ImageUploadError("图片文件不能为空")
    if len(raw) > _MAX_IMAGE_UPLOAD_BYTES:
        raise _ImageUploadError("单张图片不能超过 8 MiB", 413)
    try:
        with PILImage.open(io.BytesIO(raw)) as image:
            mime = _IMAGE_MIME_BY_FORMAT.get(str(image.format or "").upper(), "")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
                raise _ImageUploadError("图片像素尺寸超出限制")
            image.verify()
    except _ImageUploadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, PILImage.DecompressionBombError) as exc:
        raise _ImageUploadError("文件内容不是有效图片") from exc
    if mime not in allowed_mimes:
        raise _ImageUploadError(f"不支持的图片格式: {mime or 'unknown'}")
    return mime


def _self_image_target(filename: str) -> Path | None:
    base = _SELF_IMAGE_DIR.resolve()
    target = (_SELF_IMAGE_DIR / filename).resolve()
    return target if target.parent == base else None


def _available_self_image_path(filename: str, raw: bytes) -> tuple[Path, bool]:
    target = _SELF_IMAGE_DIR / filename
    if not target.exists():
        return target, False
    try:
        if target.read_bytes() == raw:
            return target, True
    except OSError:
        pass
    stem = target.stem
    suffix = target.suffix
    for index in range(2, 10_000):
        candidate = target.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate, False
    raise _ImageUploadError("无法生成安全的图片文件名", 409)


def _safe_image_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    invalid_chars = '<>:"/\\|?*'
    reserved_stems = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if (
        not name
        or len(name) > 180
        or any(ord(char) < 32 or char in invalid_chars for char in name)
        or Path(name).stem.upper() in reserved_stems
    ):
        raise _ImageUploadError("图片文件名不安全")
    return name


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
    target = _self_image_target(filename)
    if target is None:
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
    file_items = list(files.items(multi=True))
    if not file_items:
        return jsonify({"success": False, "error": "未提供图片文件"}), 400
    if len(file_items) > 12:
        return jsonify({"success": False, "error": "单次最多上传 12 张图片"}), 400

    prepared: list[tuple[str, bytes]] = []
    for _field, file in file_items:
        ext = Path(file.filename).suffix.lower() if file.filename else ""
        if ext not in _ALLOWED_IMAGE_EXTS:
            return jsonify({"success": False, "error": f"不支持的文件类型: {ext}"}), 400
        data = file.read()
        try:
            detected_mime = _inspect_image_upload(
                data,
                allowed_mimes=set(_SELF_IMAGE_MIME_BY_EXT.values()),
            )
        except _ImageUploadError as exc:
            return jsonify({"success": False, "error": str(exc)}), exc.status_code
        if detected_mime != _SELF_IMAGE_MIME_BY_EXT[ext]:
            return jsonify({"success": False, "error": "文件扩展名与图片内容不一致"}), 400
        try:
            safe_name = _safe_image_filename(file.filename)
        except _ImageUploadError as exc:
            return jsonify({"success": False, "error": str(exc)}), exc.status_code
        prepared.append((safe_name, data))

    uploaded = []
    for safe_name, data in prepared:
        try:
            destination, duplicate = await asyncio.to_thread(
                _available_self_image_path,
                safe_name,
                data,
            )
        except _ImageUploadError as exc:
            return jsonify({"success": False, "error": str(exc)}), exc.status_code
        if not duplicate:
            await asyncio.to_thread(destination.write_bytes, data)
        uploaded.append(destination.name)
    return jsonify({"success": True, "uploaded": uploaded})


@settings_bp.route("/settings/self_image/<path:filename>", methods=["DELETE"])
async def self_image_delete(filename: str):
    """删除 config/self_image/ 下的指定文件（防路径穿越）。"""
    target = _self_image_target(filename)
    if target is None:
        return jsonify({"error": "forbidden"}), 403
    if not target.is_file():
        return jsonify({"error": "not found"}), 404
    await asyncio.to_thread(target.unlink)
    return jsonify({"success": True})


# ── 缓存管理 ────────────────────────────────────────────────────────────────


@settings_bp.route("/settings/cache/info", methods=["GET"])
async def cache_info():
    """返回各缓存目录的占用大小（字节）。"""
    overview = await asyncio.to_thread(cache_maintenance_service.overview)
    sizes = {name: int(item["bytes"]) for name, item in overview.items()}
    return jsonify({"sizes": sizes})


@settings_bp.route("/settings/cache/clear", methods=["POST"])
async def cache_clear():
    """清理指定缓存目录。body: {"targets": ["image", "tts", "stickers"]}"""
    data = await request.get_json() or {}
    targets = data.get("targets") or list(cache_maintenance_service.paths)
    results = {}
    try:
        for name in targets:
            if name not in cache_maintenance_service.paths:
                continue
            result = await asyncio.to_thread(cache_maintenance_service.clear_target, name)
            results[name] = result["deleted_files"]
    except CacheMaintenanceError as exc:
        return jsonify({"success": False, "error": str(exc), **exc.details}), exc.status_code
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
    if len(description) > 200:
        return jsonify({"success": False, "error": "描述不能超过 200 个字符"}), 400
    raw = file.read()
    try:
        mime = _inspect_image_upload(
            raw,
            allowed_mimes={"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"},
        )
    except _ImageUploadError as exc:
        return jsonify({"success": False, "error": str(exc)}), exc.status_code
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
    if len(description) > 200:
        return jsonify({"success": False, "error": "描述不能超过 200 个字符"}), 400
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


