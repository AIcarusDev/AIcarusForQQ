"""Auditable v1 maintenance contracts for WebUI vNext."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from quart import Blueprint, jsonify, request

import app_state
from runtime.cache_maintenance import (
    CacheMaintenanceError,
    cache_maintenance_service,
)
from runtime.maintenance import MaintenanceError, maintenance_service
from web import routes_workspace
from workspace.config import WorkspaceProvisionConfig
from workspace.control import WorkspaceControlError


logger = logging.getLogger("AICQ.web.ui_v1.maintenance")
ui_v1_maintenance_bp = Blueprint("ui_v1_maintenance", __name__)


def _success(data: Any, status: int = 200):
    return jsonify({"ok": True, "api_version": "1", "data": data}), status


def _error(
    code: str,
    message: str,
    status: int,
    *,
    details: dict[str, Any] | None = None,
):
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return jsonify({"ok": False, "api_version": "1", "error": error}), status


def _workspace_config() -> WorkspaceProvisionConfig:
    return WorkspaceProvisionConfig.from_root_config(app_state.config, environ=os.environ)


async def _data_domain() -> dict[str, Any]:
    overview = await maintenance_service.overview()
    return {
        "status": "ready",
        "overview": {
            "total_rows": sum(max(0, int(value)) for value in overview.values()),
            "tables": overview,
        },
        "actions": maintenance_service.describe_actions(),
    }


async def _cache_domain() -> dict[str, Any]:
    overview = await asyncio.to_thread(cache_maintenance_service.overview)
    actions = await asyncio.to_thread(
        cache_maintenance_service.describe_actions,
        overview=overview,
    )
    return {
        "status": "ready",
        "overview": {
            "total_bytes": sum(int(item["bytes"]) for item in overview.values()),
            "total_files": sum(int(item["files"]) for item in overview.values()),
            "targets": overview,
        },
        "actions": actions,
    }


async def _workspace_domain() -> dict[str, Any]:
    config = _workspace_config()
    control = routes_workspace.workspace_control
    observed = await asyncio.to_thread(control.probe, config)
    job = await asyncio.to_thread(control.current_job)
    status = await asyncio.to_thread(
        control.status_payload,
        config,
        observed=observed,
        job=job,
    )
    actions = await asyncio.to_thread(
        control.describe_actions,
        config,
        observed=observed,
        current_job=status.get("job"),
    )
    return {"status": "ready", "overview": status, "actions": actions}


async def _domain_or_error(name: str, loader) -> tuple[str, dict[str, Any]]:
    try:
        return name, await loader()
    except Exception:
        logger.exception("加载 vNext %s 维护描述失败", name)
        return name, {
            "status": "error",
            "overview": {},
            "actions": [],
            "error": "该维护领域暂时不可用",
        }


@ui_v1_maintenance_bp.route("/api/ui/v1/maintenance", methods=["GET"])
async def maintenance_overview():
    domains = dict(await asyncio.gather(
        _domain_or_error("data", _data_domain),
        _domain_or_error("cache", _cache_domain),
        _domain_or_error("workspace", _workspace_domain),
    ))
    return _success({"generated_at": int(time.time() * 1000), "domains": domains})


@ui_v1_maintenance_bp.route("/api/ui/v1/maintenance/cache", methods=["GET"])
async def maintenance_cache_overview():
    try:
        data = await _cache_domain()
    except Exception:
        logger.exception("加载 vNext 缓存维护描述失败")
        return _error("cache_maintenance_unavailable", "缓存状态暂时不可用", 500)
    return _success(data)


@ui_v1_maintenance_bp.route(
    "/api/ui/v1/maintenance/actions/<domain>/<action>",
    methods=["POST"],
)
async def maintenance_action(domain: str, action: str):
    data = await request.get_json(silent=True)
    if not isinstance(data, dict):
        return _error("invalid_request", "请求正文必须是 JSON 对象", 400)
    confirmation = data.get("confirmation")
    if not isinstance(confirmation, str):
        return _error("invalid_request", "confirmation 必须是字符串", 400)

    try:
        if domain == "data":
            expected = maintenance_service.expected_confirmation(action)
            if confirmation != expected:
                return _error(
                    "confirmation_mismatch",
                    "确认字符串不匹配",
                    400,
                    details={"expected_confirmation": expected},
                )
            result = await maintenance_service.perform(action)
            return _success({"domain": domain, "result": result.to_dict()})

        if domain == "cache":
            result = await asyncio.to_thread(
                cache_maintenance_service.perform,
                action,
                confirmation=confirmation,
            )
            return _success({"domain": domain, "result": result})

        if domain == "workspace":
            config = _workspace_config()
            job = await asyncio.to_thread(
                routes_workspace.workspace_control.start_job,
                action,
                config,
                confirmation=confirmation,
            )
            return _success({"domain": domain, "result": {"job": job}}, 202)

        return _error("maintenance_domain_not_found", "未知维护领域", 404)
    except MaintenanceError as exc:
        return _error("maintenance_action_rejected", str(exc), exc.status_code)
    except CacheMaintenanceError as exc:
        code = "confirmation_mismatch" if "确认字符串" in str(exc) else "cache_action_failed"
        return _error(code, str(exc), exc.status_code, details=exc.details)
    except WorkspaceControlError as exc:
        code = "confirmation_mismatch" if "确认字符串" in str(exc) else "workspace_action_rejected"
        return _error(code, str(exc), exc.status_code)
    except (TypeError, ValueError) as exc:
        return _error("invalid_request", str(exc), 400)
    except Exception:
        logger.exception("执行 vNext 维护动作失败 domain=%s action=%s", domain, action)
        return _error("maintenance_action_failed", "维护动作执行失败", 500)


@ui_v1_maintenance_bp.route(
    "/api/ui/v1/maintenance/workspace/jobs/<job_id>",
    methods=["GET"],
)
async def maintenance_workspace_job(job_id: str):
    try:
        cursor = max(0, int(request.args.get("cursor", "0") or 0))
        job = await asyncio.to_thread(
            routes_workspace.workspace_control.get_job,
            job_id,
            cursor=cursor,
        )
        return _success({"job": job})
    except (TypeError, ValueError):
        return _error("invalid_request", "cursor 必须是非负整数", 400)
    except WorkspaceControlError as exc:
        return _error("workspace_job_unavailable", str(exc), exc.status_code)
    except Exception:
        logger.exception("加载 vNext 工作区任务失败 job_id=%s", job_id)
        return _error("workspace_job_unavailable", "工作区任务暂时不可用", 500)


__all__ = ["ui_v1_maintenance_bp"]
