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
    "title": "WebUI 更新公告与面板安全规划",
    "summary": "WebUI 开始引入公告弹窗、配置迁移提示和可选登录密码保护。",
    "changes": [
        "新增结构化更新公告接口，后续重要更新会在面板中提示。",
        "新增 WebUI 登录密码方案，用于局域网、云服务器或内网穿透部署。",
        "破坏性配置变更会显示需要修改的配置路径和处理建议。",
    ],
    "config_changes": [
        {
            "old_path": "napcat.*",
            "new_path": "qq_adapter.*",
            "required": False,
            "action": "旧 napcat 配置段仍可能存在于个人配置中；建议迁移到 qq_adapter 配置段。",
        },
        {
            "old_path": "",
            "new_path": "webui_auth",
            "required": False,
            "action": "如需把面板暴露到局域网或公网，请在面板安全中设置访问密码。",
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
            "enabled": "qq_adapter.enabled",
            "host": "qq_adapter.host",
            "port": "qq_adapter.port",
            "debug_only": "qq_adapter.debug_only",
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
                "enabled": "qq_adapter.whitelist.enabled",
                "private_users": "qq_adapter.whitelist.private_users",
                "group_ids": "qq_adapter.whitelist.group_ids",
            }.items():
                if old_key in whitelist:
                    detected_fields.append({
                        "old_path": f"napcat.whitelist.{old_key}",
                        "new_path": new_path,
                        "value_preview": _preview_config_value(whitelist.get(old_key)),
                    })
        warnings.append({
            "level": "warning",
            "title": "检测到旧 napcat 配置段",
            "message": "当前推荐使用 qq_adapter 配置段管理 NapCat / LLoneBot 连接。旧字段不会自动覆盖现有 qq_adapter 配置，请确认是否需要手动迁移。",
            "old_path": "napcat",
            "new_path": "qq_adapter",
            "detected_fields": detected_fields,
            "actions": [
                "打开 config/config_user.yaml。",
                "将仍需要的 napcat.* 字段迁移到对应的 qq_adapter.* 字段。",
                "确认 WebUI 设置页的 QQ / QQ adapter 配置正确后，可删除旧 napcat 配置段。",
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
