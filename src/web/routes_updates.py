"""WebUI update announcement endpoints."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from quart import Blueprint, jsonify, request

import app_state
from config_loader import save_config


updates_bp = Blueprint("updates", __name__)

_MANIFEST_PATH = Path(__file__).with_name("update_manifest.json")
_FALLBACK_UPDATE = {
    "version": "2026.06-webui-auth",
    "date": "2026-06-29",
    "level": "breaking",
    "title": "面板通知和访问保护",
    "summary": "这次更新让面板能提醒你重要变化，也可以给 WebUI 设置访问密码。",
    "changes": [
        "以后有重要更新时，面板会直接提示你。",
        "如果你把面板放到局域网、服务器或内网穿透环境，可以设置访问密码。",
        "如果某次更新需要你处理，面板会单独列出来。",
    ],
    "config_changes": [
        {
            "old_path": "napcat.*",
            "new_path": "qq_adapter.*",
            "required": False,
            "action": "旧版 QQ 连接配置可以整理到新版 QQ 连接设置里。",
        },
        {
            "old_path": "",
            "new_path": "webui_auth",
            "required": False,
            "action": "需要保护面板时，可以在“面板安全”里设置访问密码。",
        },
    ],
}


def _normalize_update(item: object) -> dict | None:
    if not isinstance(item, dict):
        return None
    version = str(item.get("version") or "").strip()
    if not version:
        return None
    changes = item.get("changes")
    if not isinstance(changes, list):
        changes = []
    config_changes = item.get("config_changes")
    if not isinstance(config_changes, list):
        config_changes = []
    return {
        "version": version,
        "date": str(item.get("date") or ""),
        "level": str(item.get("level") or "info"),
        "title": str(item.get("title") or version),
        "summary": str(item.get("summary") or ""),
        "changes": [str(change) for change in changes],
        "config_changes": [
            change for change in config_changes if isinstance(change, dict)
        ],
    }


def _load_update_items() -> list[dict]:
    try:
        raw = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        items = raw.get("updates", []) if isinstance(raw, dict) else []
    except Exception:
        items = []
    normalized = [
        item for item in (_normalize_update(raw_item) for raw_item in items)
        if item is not None
    ]
    if not normalized:
        normalized = [deepcopy(_FALLBACK_UPDATE)]
    return sorted(
        normalized,
        key=lambda item: (item.get("date", ""), item.get("version", "")),
        reverse=True,
    )


def _current_update_version() -> str:
    return _load_update_items()[0]["version"]


def _updates_config() -> dict:
    raw = getattr(app_state, "config", {}).get("webui_updates")
    return dict(raw or {}) if isinstance(raw, dict) else {}


def _pending_config_warnings() -> list[dict]:
    cfg = getattr(app_state, "config", {}) or {}
    warnings = []
    napcat_cfg = cfg.get("napcat")
    if isinstance(napcat_cfg, dict):
        detected_fields = []
        field_map = {
            "enabled": "platforms.qq.enabled",
            "host": "platforms.qq.adapter.reverse_ws.host",
            "port": "platforms.qq.adapter.reverse_ws.port",
            "debug_only": "platforms.qq.adapter.debug_only",
        }
        for old_key, new_path in field_map.items():
            if old_key in napcat_cfg:
                detected_fields.append({
                    "old_path": f"napcat.{old_key}",
                    "new_path": new_path,
                    "value_preview": _preview_config_value(napcat_cfg.get(old_key)),
                })
        whitelist = napcat_cfg.get("whitelist")
        if isinstance(whitelist, dict):
            for old_key, new_path in {
                "enabled": "platforms.qq.access.whitelist.enabled",
                "private_users": "platforms.qq.access.whitelist.private_users",
                "group_ids": "platforms.qq.access.whitelist.group_ids",
            }.items():
                if old_key in whitelist:
                    detected_fields.append({
                        "old_path": f"napcat.whitelist.{old_key}",
                        "new_path": new_path,
                        "value_preview": _preview_config_value(whitelist.get(old_key)),
                    })
        warnings.append({
            "level": "warning",
            "title": "旧版 QQ 配置可整理",
            "message": "发现旧版 QQ 配置。整理时会备份旧配置，并保留当前 QQ 连接设置。",
            "old_path": "napcat",
            "new_path": "platforms.qq",
            "detected_fields": detected_fields,
            "actions": [
                "当前 QQ 连接正常时，可以先不处理。",
            ],
        })
    return warnings


def _preview_config_value(value: object) -> str:
    if isinstance(value, list):
        return f"{len(value)} 项"
    if isinstance(value, dict):
        return f"{len(value)} 个字段"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    text = str(value)
    return text if len(text) <= 48 else text[:45] + "..."


def _has_path(cfg: dict, path: str) -> bool:
    cursor: object = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(cursor, dict) or part not in cursor:
            return False
        cursor = cursor[part]
    return isinstance(cursor, dict) and parts[-1] in cursor


def _set_path(cfg: dict, path: str, value: object) -> None:
    cursor = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = deepcopy(value)


def _napcat_migration_pairs(napcat_cfg: dict) -> list[tuple[str, str, object]]:
    pairs: list[tuple[str, str, object]] = []
    for old_key, new_path in {
        "enabled": "platforms.qq.enabled",
        "host": "platforms.qq.adapter.reverse_ws.host",
        "port": "platforms.qq.adapter.reverse_ws.port",
        "debug_only": "platforms.qq.adapter.debug_only",
    }.items():
        if old_key in napcat_cfg:
            pairs.append((f"napcat.{old_key}", new_path, napcat_cfg[old_key]))
    whitelist = napcat_cfg.get("whitelist")
    if isinstance(whitelist, dict):
        for old_key, new_path in {
            "enabled": "platforms.qq.access.whitelist.enabled",
            "private_users": "platforms.qq.access.whitelist.private_users",
            "group_ids": "platforms.qq.access.whitelist.group_ids",
        }.items():
            if old_key in whitelist:
                pairs.append((f"napcat.whitelist.{old_key}", new_path, whitelist[old_key]))
    return pairs


def _get_path(cfg: dict, path: str) -> object:
    cursor: object = cfg
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _build_napcat_migration_plan(cfg: dict) -> dict:
    napcat_cfg = cfg.get("napcat")
    if not isinstance(napcat_cfg, dict):
        return {
            "available": False,
            "error": "未检测到旧版 QQ 配置",
            "migratable": [],
            "same": [],
            "conflicts": [],
            "unsupported": [],
            "backup_key": "",
        }

    backup_key = "napcat_legacy_backup"
    suffix = 2
    while backup_key in cfg:
        backup_key = f"napcat_legacy_backup_{suffix}"
        suffix += 1

    plan = {
        "available": True,
        "backup_key": backup_key,
        "migratable": [],
        "same": [],
        "conflicts": [],
        "unsupported": [],
    }
    supported_old_paths = set()
    for old_path, new_path, old_value in _napcat_migration_pairs(napcat_cfg):
        supported_old_paths.add(old_path)
        if not _has_path(cfg, new_path):
            plan["migratable"].append({
                "old_path": old_path,
                "new_path": new_path,
                "value_preview": _preview_config_value(old_value),
            })
            continue
        new_value = _get_path(cfg, new_path)
        target = plan["same"] if new_value == old_value else plan["conflicts"]
        target.append({
            "old_path": old_path,
            "new_path": new_path,
            "old_value_preview": _preview_config_value(old_value),
            "new_value_preview": _preview_config_value(new_value),
        })

    for key in napcat_cfg:
        old_path = f"napcat.{key}"
        if key == "whitelist":
            whitelist = napcat_cfg.get("whitelist")
            if isinstance(whitelist, dict):
                for subkey in whitelist:
                    sub_path = f"napcat.whitelist.{subkey}"
                    if sub_path not in supported_old_paths:
                        plan["unsupported"].append({
                            "old_path": sub_path,
                            "value_preview": _preview_config_value(whitelist[subkey]),
                        })
            continue
        if old_path not in supported_old_paths:
            plan["unsupported"].append({
                "old_path": old_path,
                "value_preview": _preview_config_value(napcat_cfg[key]),
            })
    return plan


@updates_bp.route("/api/updates/current")
async def current_updates():
    cfg = _updates_config()
    acknowledged = str(cfg.get("ack_version") or "")
    items = _load_update_items()
    latest = items[0]
    return jsonify({
        "current_version": latest["version"],
        "ack_version": acknowledged,
        "needs_popup": acknowledged != latest["version"],
        "has_breaking": any(item.get("level") == "breaking" for item in items),
        "items": items,
        "config_warnings": _pending_config_warnings(),
    })


@updates_bp.route("/api/updates/ack", methods=["POST"])
async def ack_updates():
    data = await request.get_json(silent=True) or {}
    version = str(data.get("version") or _current_update_version())
    new_cfg = deepcopy(app_state.config)
    webui_updates = dict(new_cfg.get("webui_updates") or {})
    webui_updates["ack_version"] = version
    new_cfg["webui_updates"] = webui_updates
    save_config(new_cfg)
    app_state.config = new_cfg
    return jsonify({"success": True, "ack_version": version})


@updates_bp.route("/api/updates/migrations/napcat-to-qq-adapter", methods=["POST"])
async def migrate_napcat_to_qq_adapter():
    cfg = getattr(app_state, "config", {}) or {}
    plan = _build_napcat_migration_plan(cfg)
    if not plan["available"]:
        return jsonify({
            "success": False,
            "error": plan["error"],
        }), 404
    data = await request.get_json(silent=True) or {}
    if data.get("dry_run", False):
        return jsonify({"success": True, "plan": plan})

    napcat_cfg = cfg["napcat"]
    new_cfg = deepcopy(cfg)
    migrated = []
    skipped = []
    for item in plan["migratable"]:
        old_path = item["old_path"]
        new_path = item["new_path"]
        value = _get_path(cfg, old_path)
        _set_path(new_cfg, new_path, value)
        migrated.append(item)
    for item in plan["same"]:
        skipped.append({
            **item,
            "reason": "目标字段已存在且值一致",
        })
    for item in plan["conflicts"]:
        skipped.append({
            **item,
            "reason": "目标字段已存在且值不同，未覆盖",
        })
    for item in plan["unsupported"]:
        skipped.append({
            **item,
            "reason": "暂不支持自动整理",
        })

    backup_key = plan["backup_key"]
    new_cfg[backup_key] = deepcopy(napcat_cfg)
    new_cfg.pop("napcat", None)
    migration_state = dict(new_cfg.get("webui_updates") or {})
    migrations = dict(migration_state.get("migrations") or {})
    migrations["napcat_to_qq_adapter"] = {
        "completed": True,
        "backup_key": backup_key,
        "migrated_count": len(migrated),
        "skipped_count": len(skipped),
    }
    migration_state["migrations"] = migrations
    new_cfg["webui_updates"] = migration_state
    save_config(new_cfg)
    app_state.config = new_cfg
    return jsonify({
        "success": True,
        "plan": plan,
        "migrated": migrated,
        "skipped": skipped,
        "backup_key": backup_key,
        "message": f"旧版 QQ 配置已备份到 {backup_key}",
    })


