"""Domain-scoped settings storage for the versioned vNext API.

The legacy settings endpoint intentionally remains untouched.  This module
provides a smaller contract: each request reads or updates one domain, carries
an opaque revision, and represents secrets as explicit commands instead of
masked strings.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import threading
from typing import Any, Callable

from dotenv import load_dotenv
import yaml

import app_state
from browser.config import normalize_browser_control_config
from config_loader import (
    read_env_imap,
    read_env_smtp,
    save_config,
    save_env_imap,
    save_env_smtp,
    save_env_value,
    save_persona,
)
from llm.compression.config import normalize_generation_config
from llm.core.profiles import normalize_profile_config_inplace, sanitize_model_providers
from platforms.qq.adapter.config import normalize_qq_platform_config
from skills import load_skill_user_body, save_skill_user_body


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "config_user.yaml"
DEFAULT_ENV_PATH = ROOT_DIR / ".env"
DEFAULT_PERSONA_PATH = ROOT_DIR / "config" / "persona.md"

SCHEMA_VERSION = "settings-v1"
SUPPORTED_DOMAINS = frozenset({
    "providers",
    "main-model",
    "specialized-models",
    "persona",
    "qq-adapter",
    "tts",
    "services",
    "alerts",
    "advanced",
})

_PROVIDER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_REVISION_KEY = secrets.token_bytes(32)
_MISSING = object()


class SettingsDomainError(ValueError):
    """Base class for domain contract failures."""


class SettingsValidationError(SettingsDomainError):
    """The submitted domain values or secret commands are invalid."""


class SettingsConflict(SettingsDomainError):
    """The submitted revision no longer matches persisted state."""

    def __init__(self, latest: dict[str, Any]) -> None:
        super().__init__("设置已被其他页面或外部编辑器修改")
        self.latest = latest


def _mapping(value: object) -> dict[str, Any]:
    return deepcopy(value) if isinstance(value, dict) else {}


def _path_get(mapping: dict[str, Any], path: str, default: object = _MISSING) -> object:
    current: object = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _path_set(mapping: dict[str, Any], path: str, value: object) -> None:
    parts = path.split(".")
    current = mapping
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _text(
    value: object,
    *,
    label: str,
    default: str = "",
    required: bool = False,
    maximum: int = 2048,
) -> str:
    result = str(value if value is not None else default).strip()
    if required and not result:
        raise SettingsValidationError(f"{label}不能为空")
    if len(result) > maximum:
        raise SettingsValidationError(f"{label}不能超过 {maximum} 个字符")
    return result


def _integer(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise SettingsValidationError(f"{label}必须是整数") from exc
    if result < minimum or result > maximum:
        raise SettingsValidationError(f"{label}必须在 {minimum} 到 {maximum} 之间")
    return result


def _number(
    value: object,
    *,
    label: str,
    minimum: float,
    maximum: float,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SettingsValidationError(f"{label}必须是数字") from exc
    if result < minimum or result > maximum:
        raise SettingsValidationError(f"{label}必须在 {minimum:g} 到 {maximum:g} 之间")
    return result


def _boolean(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise SettingsValidationError(f"{label}必须是布尔值")


Converter = Callable[[object], object]


def _as_text(label: str, *, required: bool = False, maximum: int = 2048) -> Converter:
    return lambda value: _text(
        value,
        label=label,
        required=required,
        maximum=maximum,
    )


def _as_int(label: str, minimum: int, maximum: int) -> Converter:
    return lambda value: _integer(
        value,
        label=label,
        minimum=minimum,
        maximum=maximum,
    )


def _as_number(label: str, minimum: float, maximum: float) -> Converter:
    return lambda value: _number(
        value,
        label=label,
        minimum=minimum,
        maximum=maximum,
    )


def _as_bool(label: str) -> Converter:
    return lambda value: _boolean(value, label=label)


FIELD_SPECS: dict[str, tuple[tuple[str, str, Converter], ...]] = {
    "main-model": (
        ("provider", "provider", _as_text("供应商", required=True, maximum=64)),
        ("model", "model", _as_text("模型 ID", required=True, maximum=256)),
        ("model_name", "model_name", _as_text("显示名称", maximum=256)),
        (
            "generation.temperature",
            "generation.temperature",
            _as_number("Temperature", 0.0, 2.0),
        ),
        (
            "generation.max_output_tokens",
            "generation.max_output_tokens",
            _as_int("最大输出 Token", 64, 262144),
        ),
        (
            "generation.enable_thinking",
            "generation.enable_thinking",
            _as_bool("思考模式"),
        ),
        (
            "max_calls_per_minute",
            "max_calls_per_minute",
            _as_int("每分钟最大调用", 1, 10000),
        ),
    ),
    "specialized-models": (
        (
            "tool_execution_guard.enabled",
            "tool_execution_guard.enabled",
            _as_bool("工具守门"),
        ),
        (
            "tool_execution_guard.provider",
            "tool_execution_guard.provider",
            _as_text("工具守门供应商", maximum=64),
        ),
        (
            "tool_execution_guard.model",
            "tool_execution_guard.model",
            _as_text("工具守门模型", maximum=256),
        ),
        (
            "tool_execution_guard.generation.temperature",
            "tool_execution_guard.generation.temperature",
            _as_number("工具守门 Temperature", 0.0, 2.0),
        ),
        (
            "tool_execution_guard.generation.max_output_tokens",
            "tool_execution_guard.generation.max_output_tokens",
            _as_int("工具守门 Token", 64, 262144),
        ),
        (
            "cognition_compression.provider",
            "cognition_compression.provider",
            _as_text("上下文压缩供应商", maximum=64),
        ),
        (
            "cognition_compression.model",
            "cognition_compression.model",
            _as_text("上下文压缩模型", maximum=256),
        ),
        (
            "cognition_compression.generation.temperature",
            "cognition_compression.generation.temperature",
            _as_number("压缩 Temperature", 0.0, 2.0),
        ),
        (
            "cognition_compression.generation.max_output_tokens",
            "cognition_compression.generation.max_output_tokens",
            _as_int("压缩 Token", 256, 262144),
        ),
        ("vision_bridge.enabled", "vision_bridge.enabled", _as_bool("Vision Bridge")),
        (
            "vision_bridge.provider",
            "vision_bridge.provider",
            _as_text("Vision 供应商", maximum=64),
        ),
        (
            "vision_bridge.model",
            "vision_bridge.model",
            _as_text("Vision 模型", maximum=256),
        ),
        ("slow_thinking.enabled", "slow_thinking.enabled", _as_bool("慢思考")),
        (
            "slow_thinking.provider",
            "slow_thinking.provider",
            _as_text("慢思考供应商", maximum=64),
        ),
        (
            "slow_thinking.model",
            "slow_thinking.model",
            _as_text("慢思考模型", maximum=256),
        ),
        (
            "slow_thinking.generation.temperature",
            "slow_thinking.generation.temperature",
            _as_number("慢思考 Temperature", 0.0, 2.0),
        ),
        (
            "slow_thinking.generation.max_output_tokens",
            "slow_thinking.generation.max_output_tokens",
            _as_int("慢思考 Token", 64, 262144),
        ),
    ),
    "persona": (
        ("self_name", "self_name", _as_text("自身名称", required=True, maximum=128)),
        ("guardian.name", "guardian.name", _as_text("监护人名称", maximum=128)),
        ("guardian.id", "guardian.id", _as_text("监护人 ID", maximum=128)),
        ("timezone", "timezone", _as_text("时区", required=True, maximum=128)),
    ),
    "qq-adapter": (
        ("enabled", "platforms.qq.enabled", _as_bool("QQ 接入")),
        (
            "adapter.type",
            "platforms.qq.adapter.type",
            _as_text("Adapter 类型", required=True, maximum=64),
        ),
        (
            "adapter.name",
            "platforms.qq.adapter.name",
            _as_text("Adapter 名称", maximum=128),
        ),
        (
            "adapter.debug_only",
            "platforms.qq.adapter.debug_only",
            _as_bool("仅调试"),
        ),
        (
            "adapter.reverse_ws.host",
            "platforms.qq.adapter.reverse_ws.host",
            _as_text("反向 WebSocket 主机", required=True, maximum=255),
        ),
        (
            "adapter.reverse_ws.port",
            "platforms.qq.adapter.reverse_ws.port",
            _as_int("反向 WebSocket 端口", 1, 65535),
        ),
        (
            "access.whitelist.enabled",
            "platforms.qq.access.whitelist.enabled",
            _as_bool("白名单"),
        ),
        (
            "attention.respond_to_self_name",
            "platforms.qq.attention.respond_to_self_name",
            _as_bool("响应自身名称"),
        ),
        ("recovery.enabled", "platforms.qq.recovery.enabled", _as_bool("会话恢复")),
        (
            "recovery.backfill_history",
            "platforms.qq.recovery.backfill_history",
            _as_bool("补齐历史消息"),
        ),
    ),
    "tts": (
        ("enabled", "tts.enabled", _as_bool("TTS")),
        ("host", "tts.host", _as_text("TTS 主机", required=True, maximum=255)),
        ("port", "tts.port", _as_int("TTS 端口", 1, 65535)),
        (
            "max_concurrent_tasks_per_plugin",
            "tts.max_concurrent_tasks_per_plugin",
            _as_int("并发任务", 1, 128),
        ),
    ),
    "services": (
        (
            "web_search.searxng.enabled",
            "web_search.searxng.enabled",
            _as_bool("SearXNG"),
        ),
        (
            "web_search.searxng.base_url",
            "web_search.searxng.base_url",
            _as_text("SearXNG 地址", required=True, maximum=2048),
        ),
        (
            "web_search.searxng.language",
            "web_search.searxng.language",
            _as_text("搜索语言", required=True, maximum=32),
        ),
        (
            "web_search.searxng.safesearch",
            "web_search.searxng.safesearch",
            _as_int("安全搜索", 0, 2),
        ),
        (
            "browser_control.profile_dir",
            "browser_control.profile_dir",
            _as_text("浏览器 Profile", required=True, maximum=2048),
        ),
        (
            "browser_control.multimodal_image_limit",
            "browser_control.multimodal_image_limit",
            _as_int("浏览器图像预算", 0, 64),
        ),
        (
            "browser_control.annotate_screenshots",
            "browser_control.annotate_screenshots",
            _as_bool("截图标注"),
        ),
    ),
    "alerts": (
        ("alerting.enabled", "alerting.enabled", _as_bool("告警")),
        (
            "alerting.heartbeat_timeout",
            "alerting.heartbeat_timeout",
            _as_int("心跳超时", 30, 86400),
        ),
        (
            "alerting.cooldown",
            "alerting.cooldown",
            _as_int("告警冷却", 0, 604800),
        ),
        (
            "alerting.subject_prefix",
            "alerting.subject_prefix",
            _as_text("邮件主题前缀", required=True, maximum=128),
        ),
        (
            "alerting.email_control.enabled",
            "alerting.email_control.enabled",
            _as_bool("邮件远程指令"),
        ),
        (
            "alerting.email_control.poll_interval",
            "alerting.email_control.poll_interval",
            _as_int("邮件轮询间隔", 10, 600),
        ),
        (
            "alerting.email_control.token_ttl_seconds",
            "alerting.email_control.token_ttl_seconds",
            _as_int("邮件指令 Token 有效期", 60, 604800),
        ),
        (
            "alerting.email_control.reuse_smtp_credentials",
            "alerting.email_control.reuse_smtp_credentials",
            _as_bool("复用 SMTP 凭据"),
        ),
    ),
    "advanced": (
        (
            "tools.send_message.message_shape",
            "tools.send_message.message_shape",
            _as_text("消息发送形态", required=True, maximum=16),
        ),
    ),
}


def _copy_config_fields(domain: str, incoming: dict[str, Any], target: dict[str, Any]) -> None:
    for source_path, target_path, converter in FIELD_SPECS.get(domain, ()):
        value = _path_get(incoming, source_path)
        if value is _MISSING:
            continue
        _path_set(target, target_path, converter(value))


def _provider_env_name(provider_id: str) -> str:
    suffix = re.sub(r"[^A-Z0-9]+", "_", provider_id.upper()).strip("_")
    return f"MODEL_PROVIDER_{suffix}_API_KEY"


def _sanitize_provider_values(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise SettingsValidationError("model_providers 必须是对象")
    if not value:
        raise SettingsValidationError("至少需要保留一个模型供应商")
    if len(value) > 32:
        raise SettingsValidationError("模型供应商不能超过 32 个")

    cleaned: dict[str, dict[str, Any]] = {}
    for raw_id, raw_provider in value.items():
        provider_id = str(raw_id).strip()
        if not _PROVIDER_ID_RE.fullmatch(provider_id):
            raise SettingsValidationError(f"无效的供应商 ID: {provider_id or '(空)'}")
        if not isinstance(raw_provider, dict):
            raise SettingsValidationError(f"供应商 {provider_id} 的配置必须是对象")
        provider = deepcopy(raw_provider)
        provider["name"] = _text(
            provider.get("name"),
            label=f"供应商 {provider_id} 的名称",
            required=True,
            maximum=128,
        )
        base_url = _text(
            provider.get("base_url"),
            label=f"供应商 {provider_id} 的 API 地址",
            required=True,
            maximum=2048,
        )
        if not base_url.startswith(("http://", "https://")):
            raise SettingsValidationError(f"供应商 {provider_id} 的 API 地址必须以 http:// 或 https:// 开头")
        provider["base_url"] = base_url.rstrip("/")
        api_key_env = str(provider.get("api_key_env") or _provider_env_name(provider_id)).strip()
        if not _ENV_NAME_RE.fullmatch(api_key_env):
            raise SettingsValidationError(f"供应商 {provider_id} 的 API Key 环境变量名无效")
        provider["api_key_env"] = api_key_env
        provider["requires_api_key"] = bool(provider.get("requires_api_key", True))
        cleaned[provider_id] = provider

    return sanitize_model_providers(cleaned, dedupe_display_names=True)


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return f"{'*' * min(8, len(value) - 4)}{value[-4:]}"


def _read_text_file(path: Path, *, fallback: str, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return fallback
    except OSError as exc:
        raise SettingsDomainError(f"无法读取{label}") from exc


def _default_domain_values(
    domain: str,
    config: dict[str, Any],
    *,
    env_path: Path,
    persona_path: Path,
) -> dict[str, Any]:
    if domain == "providers":
        providers = _mapping(config.get("model_providers"))
        return {
            "model_providers": _sanitize_provider_values(providers) if providers else {}
        }

    if domain == "main-model":
        generation = normalize_generation_config(config.get("generation"))
        return {
            "provider": str(config.get("provider") or ""),
            "model": str(config.get("model") or ""),
            "model_name": str(config.get("model_name") or config.get("model") or ""),
            "generation": {
                "temperature": float(generation.get("temperature", 0.7)),
                "max_output_tokens": int(generation.get("max_output_tokens", 4096)),
                "enable_thinking": bool(generation.get("enable_thinking", True)),
            },
            "max_calls_per_minute": int(config.get("max_calls_per_minute", 15)),
        }

    if domain == "specialized-models":
        guard = _mapping(config.get("tool_execution_guard"))
        compression = _mapping(config.get("cognition_compression"))
        vision = _mapping(config.get("vision_bridge"))
        slow = _mapping(config.get("slow_thinking"))
        return {
            "tool_execution_guard": {
                "enabled": bool(guard.get("enabled", False)),
                "provider": str(guard.get("provider") or ""),
                "model": str(guard.get("model") or ""),
                "generation": {
                    "temperature": float(_mapping(guard.get("generation")).get("temperature", 0.2)),
                    "max_output_tokens": int(_mapping(guard.get("generation")).get("max_output_tokens", 512)),
                },
            },
            "cognition_compression": {
                "provider": str(compression.get("provider") or config.get("provider") or ""),
                "model": str(compression.get("model") or config.get("model") or ""),
                "generation": {
                    "temperature": float(_mapping(compression.get("generation")).get("temperature", 0.3)),
                    "max_output_tokens": int(_mapping(compression.get("generation")).get("max_output_tokens", 2000)),
                },
            },
            "vision_bridge": {
                "enabled": bool(vision.get("enabled", False)),
                "provider": str(vision.get("provider") or ""),
                "model": str(vision.get("model") or ""),
            },
            "slow_thinking": {
                "enabled": bool(slow.get("enabled", False)),
                "provider": str(slow.get("provider") or ""),
                "model": str(slow.get("model") or ""),
                "generation": {
                    "temperature": float(_mapping(slow.get("generation")).get("temperature", 1.0)),
                    "max_output_tokens": int(_mapping(slow.get("generation")).get("max_output_tokens", 8192)),
                },
            },
        }

    if domain == "persona":
        return {
            "self_name": str(
                config.get("self_name")
                or getattr(app_state, "SELF_NAME", "")
                or "AIcarus"
            ),
            "guardian": {
                "name": str(_mapping(config.get("guardian")).get("name") or ""),
                "id": str(_mapping(config.get("guardian")).get("id") or ""),
            },
            "timezone": str(config.get("timezone") or "Asia/Shanghai"),
            "persona": _read_text_file(
                persona_path,
                fallback=str(getattr(app_state, "persona", "") or ""),
                label="Persona 文件",
            ),
            "qq_social_style": load_skill_user_body("qq-social-style"),
        }

    if domain == "qq-adapter":
        normalized = deepcopy(config)
        normalize_qq_platform_config(normalized, remove_legacy=True)
        qq = _mapping(_mapping(normalized.get("platforms")).get("qq"))
        adapter = _mapping(qq.get("adapter"))
        reverse_ws = _mapping(adapter.get("reverse_ws"))
        whitelist = _mapping(_mapping(qq.get("access")).get("whitelist"))
        attention = _mapping(qq.get("attention"))
        recovery = _mapping(qq.get("recovery"))
        return {
            "enabled": bool(qq.get("enabled", False)),
            "adapter": {
                "type": str(adapter.get("type") or "auto"),
                "name": str(adapter.get("name") or ""),
                "debug_only": bool(adapter.get("debug_only", False)),
                "reverse_ws": {
                    "host": str(reverse_ws.get("host") or "127.0.0.1"),
                    "port": int(reverse_ws.get("port", 8078)),
                },
            },
            "access": {"whitelist": {"enabled": bool(whitelist.get("enabled", False))}},
            "attention": {"respond_to_self_name": bool(attention.get("respond_to_self_name", True))},
            "recovery": {
                "enabled": bool(recovery.get("enabled", True)),
                "backfill_history": bool(recovery.get("backfill_history", True)),
            },
        }

    if domain == "tts":
        tts = _mapping(config.get("tts"))
        return {
            "enabled": bool(tts.get("enabled", False)),
            "host": str(tts.get("host") or "127.0.0.1"),
            "port": int(tts.get("port", 8765)),
            "max_concurrent_tasks_per_plugin": int(tts.get("max_concurrent_tasks_per_plugin", 8)),
        }

    if domain == "services":
        search = _mapping(config.get("web_search"))
        searxng = _mapping(search.get("searxng"))
        browser = normalize_browser_control_config(config.get("browser_control"))
        env = _read_env_raw({"QWEATHER_API_HOST"}, env_path)
        return {
            "web_search": {
                "searxng": {
                    "enabled": bool(searxng.get("enabled", False)),
                    "base_url": str(searxng.get("base_url") or "http://127.0.0.1:8888"),
                    "language": str(searxng.get("language") or "zh-CN"),
                    "safesearch": int(searxng.get("safesearch", 0)),
                }
            },
            "browser_control": browser,
            "service_env": {"QWEATHER_API_HOST": env.get("QWEATHER_API_HOST", "")},
        }

    if domain == "alerts":
        alerting = _mapping(config.get("alerting"))
        email_control = _mapping(alerting.get("email_control"))
        smtp = read_env_smtp(str(env_path))
        imap = read_env_imap(str(env_path))
        smtp.pop("AICQ_SMTP_PASSWORD", None)
        imap.pop("AICQ_IMAP_PASSWORD", None)
        return {
            "alerting": {
                "enabled": bool(alerting.get("enabled", False)),
                "heartbeat_timeout": int(alerting.get("heartbeat_timeout", 120)),
                "cooldown": int(alerting.get("cooldown", 600)),
                "subject_prefix": str(alerting.get("subject_prefix") or "[AIcarus 告警]"),
                "email_control": {
                    "enabled": bool(email_control.get("enabled", False)),
                    "poll_interval": int(email_control.get("poll_interval", 30)),
                    "token_ttl_seconds": int(email_control.get("token_ttl_seconds", 600)),
                    "reuse_smtp_credentials": bool(email_control.get("reuse_smtp_credentials", True)),
                },
            },
            "smtp": smtp,
            "imap": imap,
        }

    if domain == "advanced":
        tools = _mapping(config.get("tools"))
        send_message = _mapping(tools.get("send_message"))
        shape = str(send_message.get("message_shape") or "array").strip().lower()
        if shape not in {"array", "single"}:
            shape = "array"
        return {"tools": {"send_message": {"message_shape": shape}}}

    raise SettingsValidationError(f"不支持的设置领域: {domain}")


def _read_env_raw(names: set[str], path: Path) -> dict[str, str]:
    result = {name: "" for name in names}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return result
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name = name.strip()
        if name in result:
            result[name] = value.strip().strip('"').strip("'")
    return result


def _secret_targets(domain: str, config: dict[str, Any]) -> dict[str, tuple[str, str]]:
    if domain == "providers":
        result: dict[str, tuple[str, str]] = {}
        for provider_id, provider in _mapping(config.get("model_providers")).items():
            if not isinstance(provider, dict):
                continue
            env_name = str(provider.get("api_key_env") or "").strip()
            if _ENV_NAME_RE.fullmatch(env_name):
                result[f"provider_api_key::{provider_id}"] = ("env", env_name)
        return result
    if domain == "tts":
        return {"secret_token": ("config", "tts.secret_token")}
    if domain == "services":
        return {
            "tavily_api_key": ("env", "TAVILY_API_KEY"),
            "weather_api_key": ("env", "QWEATHER_API_KEY"),
        }
    if domain == "alerts":
        return {
            "smtp_password": ("env", "AICQ_SMTP_PASSWORD"),
            "imap_password": ("env", "AICQ_IMAP_PASSWORD"),
        }
    if domain == "advanced":
        return {
            "openai_proxy": ("env", "OPENAI_PROXY"),
            "tavily_proxy": ("env", "TAVILY_PROXY"),
        }
    return {}


def _secret_values(domain: str, config: dict[str, Any], env_path: Path) -> dict[str, str]:
    targets = _secret_targets(domain, config)
    env_names = {name for kind, name in targets.values() if kind == "env"}
    env_values = _read_env_raw(env_names, env_path)
    result: dict[str, str] = {}
    for secret_id, (kind, location) in targets.items():
        raw = _path_get(config, location, "") if kind == "config" else env_values.get(location, "")
        result[secret_id] = str(raw or "")
    return result


def _secret_states(domain: str, config: dict[str, Any], env_path: Path) -> dict[str, dict[str, Any]]:
    return {
        secret_id: {
            "configured": bool(value),
            "masked_hint": _mask_secret(value),
        }
        for secret_id, value in _secret_values(domain, config, env_path).items()
    }


def _domain_options(domain: str, config: dict[str, Any]) -> dict[str, Any]:
    providers = [
        {"id": str(provider_id), "label": str(provider.get("name") or provider_id)}
        for provider_id, provider in _mapping(config.get("model_providers")).items()
        if isinstance(provider, dict)
    ]
    if domain in {"main-model", "specialized-models"}:
        return {"providers": providers}
    if domain == "qq-adapter":
        return {
            "adapter_types": [
                {"id": "auto", "label": "自动检测"},
                {"id": "napcat", "label": "NapCat"},
                {"id": "llonebot", "label": "LLOneBot"},
            ]
        }
    return {}


def _validate_model_bindings(config: dict[str, Any], domain: str) -> None:
    providers = set(_mapping(config.get("model_providers")))

    def validate(label: str, value: dict[str, Any], *, required: bool) -> None:
        provider = str(value.get("provider") or "").strip()
        model = str(value.get("model") or "").strip()
        if not required and not provider and not model:
            return
        if not provider or not model:
            raise SettingsValidationError(f"{label}必须同时选择供应商并填写模型 ID")
        if provider not in providers:
            raise SettingsValidationError(f"{label}选择了未定义的供应商: {provider}")

    if domain == "main-model":
        validate("主模型", config, required=True)
    elif domain == "specialized-models":
        guard = _mapping(config.get("tool_execution_guard"))
        compression = _mapping(config.get("cognition_compression"))
        vision = _mapping(config.get("vision_bridge"))
        slow = _mapping(config.get("slow_thinking"))
        validate("工具守门模型", guard, required=bool(guard.get("enabled", False)))
        validate("上下文压缩模型", compression, required=True)
        validate("Vision Bridge", vision, required=bool(vision.get("enabled", False)))
        validate("慢思考模型", slow, required=bool(slow.get("enabled", False)))


class SettingsDomainStore:
    """Read and compare-and-swap domain settings from persistent storage."""

    def __init__(
        self,
        *,
        config_path: Path = DEFAULT_CONFIG_PATH,
        env_path: Path = DEFAULT_ENV_PATH,
        persona_path: Path = DEFAULT_PERSONA_PATH,
    ) -> None:
        self.config_path = Path(config_path)
        self.env_path = Path(env_path)
        self.persona_path = Path(persona_path)
        self._lock = threading.RLock()

    def _load_config(self) -> dict[str, Any]:
        try:
            loaded = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError:
            loaded = deepcopy(getattr(app_state, "config", {}) or {})
        if not isinstance(loaded, dict):
            raise SettingsValidationError("配置文件根节点必须是对象")
        normalize_profile_config_inplace(loaded)
        return loaded

    def _revision(
        self,
        domain: str,
        config: dict[str, Any],
        values: dict[str, Any],
    ) -> str:
        material = {
            "domain": domain,
            "values": values,
            "secrets": _secret_values(domain, config, self.env_path),
        }
        encoded = json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hmac.new(_REVISION_KEY, encoded, hashlib.sha256).hexdigest()[:32]

    def _snapshot(self, domain: str, config: dict[str, Any]) -> dict[str, Any]:
        values = _default_domain_values(
            domain,
            config,
            env_path=self.env_path,
            persona_path=self.persona_path,
        )
        return {
            "domain": domain,
            "schema_version": SCHEMA_VERSION,
            "revision": self._revision(domain, config, values),
            "values": values,
            "secrets": _secret_states(domain, config, self.env_path),
            "options": _domain_options(domain, config),
        }

    def read(self, domain: str) -> dict[str, Any]:
        if domain not in SUPPORTED_DOMAINS:
            raise SettingsValidationError(f"不支持的设置领域: {domain}")
        with self._lock:
            return self._snapshot(domain, self._load_config())

    def update(
        self,
        domain: str,
        *,
        revision: str,
        values: object,
        secret_commands: object,
    ) -> dict[str, Any]:
        if domain not in SUPPORTED_DOMAINS:
            raise SettingsValidationError(f"不支持的设置领域: {domain}")
        if not isinstance(values, dict):
            raise SettingsValidationError("values 必须是对象")
        if not isinstance(secret_commands, dict):
            raise SettingsValidationError("secrets 必须是对象")

        with self._lock:
            current = self._load_config()
            current_snapshot = self._snapshot(domain, current)
            if not revision or not hmac.compare_digest(revision, current_snapshot["revision"]):
                raise SettingsConflict(current_snapshot)

            updated = deepcopy(current)
            side_effects: dict[str, Any] = {}
            self._apply_values(domain, values, updated, side_effects)
            env_operations = self._prepare_secret_commands(
                domain,
                updated,
                secret_commands,
            )
            _validate_model_bindings(updated, domain)
            normalize_profile_config_inplace(updated)

            save_config(updated, str(self.config_path), preserve_latest_workspace=True)
            self._apply_side_effects(domain, side_effects)
            touched_env_names = {
                env_name
                for env_name, _value in env_operations
            } | self._side_effect_env_names(domain, side_effects)
            for env_name, value in env_operations:
                save_env_value(env_name, value, str(self.env_path))
            if env_operations or domain in {"providers", "services", "alerts", "advanced"}:
                load_dotenv(dotenv_path=self.env_path, override=True)
            self._sync_process_env(touched_env_names)

            app_state.config = deepcopy(updated)
            self._apply_runtime_view(domain, updated, side_effects)
            snapshot = self._snapshot(domain, updated)
            restart_required = domain in {
                "providers",
                "main-model",
                "specialized-models",
                "qq-adapter",
                "tts",
                "services",
                "alerts",
            }
            snapshot.update({
                "saved": True,
                "applied": domain in {"persona", "advanced"},
                "restart_required": restart_required,
                "warnings": (
                    ["配置已保存；重启 Core 后该领域会完全生效。"]
                    if restart_required
                    else []
                ),
            })
            return snapshot

    def _apply_values(
        self,
        domain: str,
        values: dict[str, Any],
        config: dict[str, Any],
        side_effects: dict[str, Any],
    ) -> None:
        if domain == "providers":
            if "model_providers" not in values:
                raise SettingsValidationError("缺少 model_providers")
            config.pop("profiles", None)
            config.pop("openai_profiles", None)
            config["model_providers"] = _sanitize_provider_values(values["model_providers"])
            selected = str(config.get("provider") or "")
            if selected not in config["model_providers"]:
                config["provider"] = next(iter(config["model_providers"]))
            return

        _copy_config_fields(domain, values, config)

        if domain == "main-model":
            config["generation"] = normalize_generation_config(config.get("generation"))
            config["model_name"] = str(config.get("model_name") or config.get("model") or "")
        elif domain == "persona":
            persona = _text(
                values.get("persona", getattr(app_state, "persona", "")),
                label="Persona",
                maximum=200_000,
            )
            qq_social_style = _text(
                values.get("qq_social_style", load_skill_user_body("qq-social-style")),
                label="QQ 社交风格",
                maximum=100_000,
            )
            side_effects.update({"persona": persona, "qq_social_style": qq_social_style})
        elif domain == "qq-adapter":
            normalize_qq_platform_config(config, remove_legacy=True)
        elif domain == "services":
            config["browser_control"] = normalize_browser_control_config(config.get("browser_control"))
            service_env = values.get("service_env")
            if service_env is not None:
                if not isinstance(service_env, dict):
                    raise SettingsValidationError("service_env 必须是对象")
                side_effects["service_env"] = {
                    "QWEATHER_API_HOST": _text(
                        service_env.get("QWEATHER_API_HOST", ""),
                        label="天气 API Host",
                        maximum=2048,
                    )
                }
        elif domain == "alerts":
            smtp = values.get("smtp")
            imap = values.get("imap")
            if smtp is not None:
                if not isinstance(smtp, dict):
                    raise SettingsValidationError("smtp 必须是对象")
                side_effects["smtp"] = self._clean_mail_values(smtp, "SMTP")
            if imap is not None:
                if not isinstance(imap, dict):
                    raise SettingsValidationError("imap 必须是对象")
                side_effects["imap"] = self._clean_mail_values(imap, "IMAP")
        elif domain == "advanced":
            shape = str(_path_get(config, "tools.send_message.message_shape", "array"))
            if shape not in {"array", "single"}:
                raise SettingsValidationError("消息发送形态只能是 array 或 single")

    @staticmethod
    def _clean_mail_values(values: dict[str, Any], label: str) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for key, value in values.items():
            name = str(key)
            if "PASSWORD" in name:
                raise SettingsValidationError(f"{label} 密码必须通过 Secret 命令提交")
            if not _ENV_NAME_RE.fullmatch(name):
                raise SettingsValidationError(f"{label} 字段名无效: {name}")
            cleaned[name] = _text(value, label=f"{label} {name}", maximum=4096)
        return cleaned

    def _prepare_secret_commands(
        self,
        domain: str,
        config: dict[str, Any],
        commands: dict[str, Any],
    ) -> list[tuple[str, str]]:
        targets = _secret_targets(domain, config)
        unknown = sorted(set(commands) - set(targets))
        if unknown:
            raise SettingsValidationError(f"未知的 Secret 字段: {', '.join(unknown)}")

        env_operations: list[tuple[str, str]] = []
        for secret_id, payload in commands.items():
            if not isinstance(payload, dict):
                raise SettingsValidationError(f"Secret {secret_id} 的命令必须是对象")
            command = str(payload.get("command") or "").strip().lower()
            if command not in {"keep", "replace", "clear"}:
                raise SettingsValidationError(f"Secret {secret_id} 仅支持 keep、replace 或 clear")
            if command == "keep":
                continue
            value = ""
            if command == "replace":
                value = str(payload.get("value") or "")
                if not value:
                    raise SettingsValidationError(f"Secret {secret_id} 的替换值不能为空")
                if len(value) > 8192:
                    raise SettingsValidationError(f"Secret {secret_id} 不能超过 8192 个字符")
            kind, location = targets[secret_id]
            if kind == "config":
                _path_set(config, location, value)
            else:
                env_operations.append((location, value))
        return env_operations

    def _apply_side_effects(self, domain: str, side_effects: dict[str, Any]) -> None:
        if domain == "persona":
            save_persona(str(side_effects["persona"]), str(self.persona_path))
            if not save_skill_user_body("qq-social-style", str(side_effects["qq_social_style"])):
                raise SettingsDomainError("QQ 社交风格保存失败")
        elif domain == "services":
            for name, value in _mapping(side_effects.get("service_env")).items():
                save_env_value(name, str(value), str(self.env_path))
        elif domain == "alerts":
            if "smtp" in side_effects:
                save_env_smtp(side_effects["smtp"], str(self.env_path))
            if "imap" in side_effects:
                save_env_imap(side_effects["imap"], str(self.env_path))

    @staticmethod
    def _side_effect_env_names(
        domain: str,
        side_effects: dict[str, Any],
    ) -> set[str]:
        if domain == "services":
            return set(_mapping(side_effects.get("service_env")))
        if domain == "alerts":
            return (
                set(_mapping(side_effects.get("smtp")))
                | set(_mapping(side_effects.get("imap")))
            )
        return set()

    def _sync_process_env(self, names: set[str]) -> None:
        """Mirror touched .env keys into this process, including deletions."""

        persisted = _read_env_raw(names, self.env_path)
        for name in names:
            value = persisted.get(name, "")
            if value:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)

    @staticmethod
    def _apply_runtime_view(
        domain: str,
        config: dict[str, Any],
        side_effects: dict[str, Any],
    ) -> None:
        if domain == "main-model":
            app_state.MODEL = str(config.get("model") or "")
            app_state.MODEL_NAME = str(config.get("model_name") or app_state.MODEL)
            app_state.GEN = _mapping(config.get("generation"))
            app_state.MAX_CALLS_PER_MINUTE = int(config.get("max_calls_per_minute", 15))
        elif domain == "persona":
            app_state.SELF_NAME = str(config.get("self_name") or "")
            app_state.persona = str(side_effects.get("persona") or "")
        elif domain == "tts":
            app_state.tts_cfg = _mapping(config.get("tts"))
        elif domain == "specialized-models":
            app_state.tool_execution_guard_cfg = _mapping(config.get("tool_execution_guard"))
            app_state.cognition_compression_cfg = _mapping(config.get("cognition_compression"))
            app_state.slow_thinking_cfg = _mapping(config.get("slow_thinking"))


__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_DOMAINS",
    "SettingsConflict",
    "SettingsDomainError",
    "SettingsDomainStore",
    "SettingsValidationError",
]
