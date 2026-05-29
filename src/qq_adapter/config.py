"""QQ adapter configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_ADAPTERS: dict[str, str] = {
    "napcat": "NapCat",
    "llonebot": "LLoneBot",
}


DEFAULT_QQ_ADAPTER_CONFIG: dict[str, Any] = {
    "enabled": False,
    "adapter": "napcat",
    "host": "127.0.0.1",
    "port": 8078,
    "debug_only": False,
    "whitelist": {
        "enabled": True,
        "private_users": [],
        "group_ids": [],
    },
    "recovery": {
        "enabled": True,
        "page_size": 50,
        "max_pages_per_session": 0,
        "backfill_history": True,
        "seed_from_whitelist": True,
    },
}


def normalize_adapter_name(value: Any) -> str:
    adapter = str(value or "").strip().lower()
    if adapter in SUPPORTED_ADAPTERS:
        return adapter
    return "napcat"


def normalize_qq_adapter_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize the single QQ adapter config surface in-place."""
    raw = config.get("qq_adapter")
    raw_cfg = raw if isinstance(raw, dict) else {}

    cfg = deepcopy(DEFAULT_QQ_ADAPTER_CONFIG)
    cfg.update({
        k: v
        for k, v in raw_cfg.items()
        if k not in {"whitelist", "recovery", "config_dir", "adapters"}
    })
    cfg["adapter"] = normalize_adapter_name(raw_cfg.get("adapter", cfg["adapter"]))

    whitelist = raw_cfg.get("whitelist")
    if isinstance(whitelist, dict):
        merged = deepcopy(DEFAULT_QQ_ADAPTER_CONFIG["whitelist"])
        merged.update(whitelist)
        merged["private_users"] = [str(x).strip() for x in merged.get("private_users", []) if str(x).strip()]
        merged["group_ids"] = [str(x).strip() for x in merged.get("group_ids", []) if str(x).strip()]
        cfg["whitelist"] = merged

    recovery = raw_cfg.get("recovery")
    if isinstance(recovery, dict):
        merged = deepcopy(DEFAULT_QQ_ADAPTER_CONFIG["recovery"])
        merged.update(recovery)
        cfg["recovery"] = merged

    cfg["name"] = SUPPORTED_ADAPTERS[cfg["adapter"]]

    try:
        cfg["port"] = max(1, min(65535, int(cfg.get("port", 8078))))
    except (TypeError, ValueError):
        cfg["port"] = 8078
    cfg["host"] = str(cfg.get("host") or "127.0.0.1").strip() or "127.0.0.1"
    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["debug_only"] = bool(cfg.get("debug_only", False))

    config["qq_adapter"] = cfg
    return cfg
