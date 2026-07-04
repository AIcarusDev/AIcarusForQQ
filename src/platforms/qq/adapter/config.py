"""QQ platform configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_ADAPTERS: dict[str, str] = {
    "napcat": "NapCat",
    "llonebot": "LLoneBot",
}


DEFAULT_QQ_PLATFORM_CONFIG: dict[str, Any] = {
    "enabled": False,
    "adapter": {
        "type": "napcat",
        "name": "NapCat",
        "debug_only": False,
        "reverse_ws": {
            "host": "127.0.0.1",
            "port": 8078,
        },
    },
    "access": {
        "whitelist": {
            "enabled": True,
            "private_users": [],
            "group_ids": [],
        },
    },
    "attention": {
        "respond_to_self_name": True,
    },
    "recovery": {
        "enabled": True,
        "page_size": 50,
        "max_pages_per_session": 0,
        "backfill_history": True,
        "seed_from_whitelist": True,
    },
    "supervisor": {
        "enabled": False,
    },
}


def normalize_adapter_name(value: Any) -> str:
    adapter = str(value or "").strip().lower()
    if adapter in SUPPORTED_ADAPTERS:
        return adapter
    return "napcat"


def _merge_legacy_qq_adapter(config: dict[str, Any], raw_cfg: dict[str, Any]) -> None:
    platforms = config.setdefault("platforms", {})
    if not isinstance(platforms, dict):
        platforms = {}
        config["platforms"] = platforms
    qq_cfg = platforms.setdefault("qq", {})
    if not isinstance(qq_cfg, dict):
        qq_cfg = {}
        platforms["qq"] = qq_cfg

    adapter_cfg = dict(qq_cfg.get("adapter") or {})
    reverse_ws = dict(adapter_cfg.get("reverse_ws") or {})
    if "enabled" in raw_cfg and "enabled" not in qq_cfg:
        qq_cfg["enabled"] = raw_cfg.get("enabled")
    if "adapter" in raw_cfg and "type" not in adapter_cfg:
        adapter_cfg["type"] = raw_cfg.get("adapter")
    if "name" in raw_cfg and "name" not in adapter_cfg:
        adapter_cfg["name"] = raw_cfg.get("name")
    if "debug_only" in raw_cfg and "debug_only" not in adapter_cfg:
        adapter_cfg["debug_only"] = raw_cfg.get("debug_only")
    if "host" in raw_cfg and "host" not in reverse_ws:
        reverse_ws["host"] = raw_cfg.get("host")
    if "port" in raw_cfg and "port" not in reverse_ws:
        reverse_ws["port"] = raw_cfg.get("port")
    adapter_cfg["reverse_ws"] = reverse_ws
    qq_cfg["adapter"] = adapter_cfg

    attention = dict(qq_cfg.get("attention") or {})
    if "respond_to_self_name" in raw_cfg and "respond_to_self_name" not in attention:
        attention["respond_to_self_name"] = raw_cfg.get("respond_to_self_name")
    qq_cfg["attention"] = attention

    access = dict(qq_cfg.get("access") or {})
    if isinstance(raw_cfg.get("whitelist"), dict) and "whitelist" not in access:
        access["whitelist"] = deepcopy(raw_cfg["whitelist"])
    qq_cfg["access"] = access

    if isinstance(raw_cfg.get("recovery"), dict) and "recovery" not in qq_cfg:
        qq_cfg["recovery"] = deepcopy(raw_cfg["recovery"])


def _merge_legacy_supervisor(config: dict[str, Any]) -> None:
    alerting = config.get("alerting")
    if not isinstance(alerting, dict):
        return
    legacy = alerting.get("qq_adapter_restart")
    if not isinstance(legacy, dict):
        return
    platforms = config.setdefault("platforms", {})
    if not isinstance(platforms, dict):
        platforms = {}
        config["platforms"] = platforms
    qq_cfg = platforms.setdefault("qq", {})
    if not isinstance(qq_cfg, dict):
        qq_cfg = {}
        platforms["qq"] = qq_cfg
    qq_cfg.setdefault("supervisor", deepcopy(legacy))


def normalize_qq_platform_config(config: dict[str, Any], *, remove_legacy: bool = False) -> dict[str, Any]:
    """Normalize the QQ platform config in-place under ``platforms.qq``."""
    legacy_raw = config.get("qq_adapter")
    if isinstance(legacy_raw, dict):
        _merge_legacy_qq_adapter(config, legacy_raw)
    _merge_legacy_supervisor(config)

    platforms = config.setdefault("platforms", {})
    if not isinstance(platforms, dict):
        platforms = {}
        config["platforms"] = platforms
    raw = platforms.get("qq")
    raw_cfg = raw if isinstance(raw, dict) else {}

    cfg = deepcopy(DEFAULT_QQ_PLATFORM_CONFIG)
    cfg.update({k: v for k, v in raw_cfg.items() if k not in {"adapter", "access", "attention", "recovery", "supervisor"}})

    adapter_raw = raw_cfg.get("adapter") if isinstance(raw_cfg.get("adapter"), dict) else {}
    adapter = deepcopy(DEFAULT_QQ_PLATFORM_CONFIG["adapter"])
    adapter.update({k: v for k, v in adapter_raw.items() if k != "reverse_ws"})
    adapter["type"] = normalize_adapter_name(adapter_raw.get("type", adapter_raw.get("adapter", adapter["type"])))
    adapter["name"] = str(adapter_raw.get("name") or SUPPORTED_ADAPTERS[adapter["type"]])
    adapter["debug_only"] = bool(adapter.get("debug_only", False))
    reverse_ws_raw = adapter_raw.get("reverse_ws") if isinstance(adapter_raw.get("reverse_ws"), dict) else {}
    reverse_ws = deepcopy(DEFAULT_QQ_PLATFORM_CONFIG["adapter"]["reverse_ws"])
    reverse_ws.update(reverse_ws_raw)
    try:
        reverse_ws["port"] = max(1, min(65535, int(reverse_ws.get("port", 8078))))
    except (TypeError, ValueError):
        reverse_ws["port"] = 8078
    reverse_ws["host"] = str(reverse_ws.get("host") or "127.0.0.1").strip() or "127.0.0.1"
    adapter["reverse_ws"] = reverse_ws
    cfg["adapter"] = adapter

    access_raw = raw_cfg.get("access") if isinstance(raw_cfg.get("access"), dict) else {}
    access = deepcopy(DEFAULT_QQ_PLATFORM_CONFIG["access"])
    access.update({k: v for k, v in access_raw.items() if k != "whitelist"})
    whitelist = access_raw.get("whitelist")
    if isinstance(whitelist, dict):
        merged = deepcopy(DEFAULT_QQ_PLATFORM_CONFIG["access"]["whitelist"])
        merged.update(whitelist)
        merged["private_users"] = [str(x).strip() for x in merged.get("private_users", []) if str(x).strip()]
        merged["group_ids"] = [str(x).strip() for x in merged.get("group_ids", []) if str(x).strip()]
        access["whitelist"] = merged
    cfg["access"] = access

    attention_raw = raw_cfg.get("attention") if isinstance(raw_cfg.get("attention"), dict) else {}
    attention = deepcopy(DEFAULT_QQ_PLATFORM_CONFIG["attention"])
    attention.update(attention_raw)
    attention["respond_to_self_name"] = bool(attention.get("respond_to_self_name", True))
    cfg["attention"] = attention

    recovery = raw_cfg.get("recovery")
    if isinstance(recovery, dict):
        merged = deepcopy(DEFAULT_QQ_PLATFORM_CONFIG["recovery"])
        merged.update(recovery)
        cfg["recovery"] = merged

    supervisor = raw_cfg.get("supervisor")
    if isinstance(supervisor, dict):
        merged = deepcopy(DEFAULT_QQ_PLATFORM_CONFIG["supervisor"])
        merged.update(supervisor)
        cfg["supervisor"] = merged

    cfg["enabled"] = bool(cfg.get("enabled", False))

    platforms["qq"] = cfg
    if remove_legacy:
        config.pop("qq_adapter", None)
        alerting = config.get("alerting")
        if isinstance(alerting, dict):
            alerting.pop("qq_adapter_restart", None)
    return cfg


def runtime_adapter_config(qq_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the flat adapter config expected by lower-level adapter helpers."""
    adapter = qq_cfg.get("adapter") if isinstance(qq_cfg.get("adapter"), dict) else {}
    reverse_ws = adapter.get("reverse_ws") if isinstance(adapter.get("reverse_ws"), dict) else {}
    access = qq_cfg.get("access") if isinstance(qq_cfg.get("access"), dict) else {}
    attention = qq_cfg.get("attention") if isinstance(qq_cfg.get("attention"), dict) else {}
    return {
        "enabled": bool(qq_cfg.get("enabled", False)),
        "adapter": str(adapter.get("type") or "napcat"),
        "name": str(adapter.get("name") or SUPPORTED_ADAPTERS.get(str(adapter.get("type") or "napcat"), "NapCat")),
        "host": str(reverse_ws.get("host") or "127.0.0.1"),
        "port": int(reverse_ws.get("port") or 8078),
        "debug_only": bool(adapter.get("debug_only", False)),
        "respond_to_self_name": bool(attention.get("respond_to_self_name", True)),
        "whitelist": deepcopy(access.get("whitelist") or DEFAULT_QQ_PLATFORM_CONFIG["access"]["whitelist"]),
        "recovery": deepcopy(qq_cfg.get("recovery") or DEFAULT_QQ_PLATFORM_CONFIG["recovery"]),
    }


def get_qq_adapter_config(app_config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(app_config, dict):
        return runtime_adapter_config({})
    platforms = app_config.get("platforms")
    qq_cfg = platforms.get("qq") if isinstance(platforms, dict) else {}
    return runtime_adapter_config(qq_cfg if isinstance(qq_cfg, dict) else {})

