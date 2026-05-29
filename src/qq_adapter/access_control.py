"""Shared QQ adapter conversation access rules."""

from __future__ import annotations

from typing import Any


def get_whitelist_config(qq_adapter_cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(qq_adapter_cfg, dict):
        return {}
    whitelist = qq_adapter_cfg.get("whitelist", {})
    return whitelist if isinstance(whitelist, dict) else {}


def is_whitelist_mode_enabled(qq_adapter_cfg: dict[str, Any] | None) -> bool:
    """Return whether the whitelist is authoritative.

    Missing config defaults to whitelist mode. In this mode an empty list means
    no sessions of that type are allowed.
    """
    whitelist = get_whitelist_config(qq_adapter_cfg)
    return bool(whitelist.get("enabled", True))


def get_whitelist_ids(
    qq_adapter_cfg: dict[str, Any] | None,
    conv_type: str,
) -> set[str]:
    whitelist = get_whitelist_config(qq_adapter_cfg)
    if conv_type == "private":
        raw_ids = whitelist.get("private_users", [])
    elif conv_type == "group":
        raw_ids = whitelist.get("group_ids", [])
    else:
        raw_ids = []
    return {str(item).strip() for item in raw_ids or [] if str(item).strip()}


def is_session_allowed_by_config(
    qq_adapter_cfg: dict[str, Any] | None,
    conv_type: str,
    conv_id: str,
) -> bool:
    if conv_type not in {"private", "group"}:
        return False
    if not is_whitelist_mode_enabled(qq_adapter_cfg):
        return True
    return str(conv_id).strip() in get_whitelist_ids(qq_adapter_cfg, conv_type)


def whitelist_rejection_reason(
    qq_adapter_cfg: dict[str, Any] | None,
    conv_type: str,
    conv_id: str,
) -> str | None:
    if is_session_allowed_by_config(qq_adapter_cfg, conv_type, conv_id):
        return None
    if not is_whitelist_mode_enabled(qq_adapter_cfg):
        return None
    if conv_type == "private":
        return f"私聊用户 {conv_id} 不在白名单中"
    if conv_type == "group":
        return f"群聊 {conv_id} 不在白名单中"
    return f"未知会话类型 {conv_type!r}"
