"""Versioned, domain-scoped settings routes for WebUI vNext."""

from __future__ import annotations

import asyncio
from typing import Any

from quart import Blueprint, jsonify, request

from web.settings_domains import (
    SCHEMA_VERSION,
    SUPPORTED_DOMAINS,
    SettingsConflict,
    SettingsDomainError,
    SettingsDomainStore,
    SettingsValidationError,
)


ui_v1_settings_bp = Blueprint("ui_v1_settings", __name__)
settings_store = SettingsDomainStore()


def _response(data: Any, *, status: int = 200):
    response = jsonify({"ok": True, "api_version": "1", "data": data})
    response.status_code = status
    response.headers["Cache-Control"] = "no-store"
    if isinstance(data, dict) and data.get("revision"):
        response.headers["ETag"] = f'"{data["revision"]}"'
    return response


def _error(code: str, message: str, status: int, **extra: Any):
    payload = {
        "ok": False,
        "api_version": "1",
        "error": {"code": code, "message": message},
        **extra,
    }
    response = jsonify(payload)
    response.status_code = status
    response.headers["Cache-Control"] = "no-store"
    return response


def _normalize_revision(value: object) -> str:
    revision = str(value or "").strip()
    if revision.startswith("W/"):
        revision = revision[2:].strip()
    if len(revision) >= 2 and revision[0] == revision[-1] == '"':
        revision = revision[1:-1]
    return revision


@ui_v1_settings_bp.route("/api/ui/v1/settings", methods=["GET"])
async def settings_domains():
    return _response({
        "schema_version": SCHEMA_VERSION,
        "domains": sorted(SUPPORTED_DOMAINS),
        "concurrency": "revision",
        "secret_commands": ["keep", "replace", "clear"],
    })


@ui_v1_settings_bp.route("/api/ui/v1/settings/<domain>", methods=["GET"])
async def settings_domain_get(domain: str):
    try:
        snapshot = await asyncio.to_thread(settings_store.read, domain)
    except SettingsValidationError as exc:
        return _error("settings_domain_not_found", str(exc), 404)
    except SettingsDomainError as exc:
        return _error("settings_unavailable", str(exc), 500)
    return _response(snapshot)


@ui_v1_settings_bp.route("/api/ui/v1/settings/<domain>", methods=["PATCH"])
async def settings_domain_patch(domain: str):
    data = await request.get_json(silent=True)
    if not isinstance(data, dict):
        return _error("invalid_settings_payload", "请求正文必须是 JSON 对象", 422)

    revision = _normalize_revision(
        request.headers.get("If-Match") or data.get("revision")
    )
    if not revision:
        return _error(
            "settings_revision_required",
            "保存设置必须携带 If-Match 或 revision",
            428,
        )

    try:
        snapshot = await asyncio.to_thread(
            settings_store.update,
            domain,
            revision=revision,
            values=data.get("values", {}),
            secret_commands=data.get("secrets", {}),
        )
    except SettingsConflict as exc:
        return _error(
            "settings_revision_conflict",
            str(exc),
            409,
            latest=exc.latest,
        )
    except SettingsValidationError as exc:
        return _error("invalid_settings", str(exc), 422)
    except SettingsDomainError as exc:
        return _error("settings_save_failed", str(exc), 500)
    return _response(snapshot)


__all__ = ["settings_store", "ui_v1_settings_bp"]
