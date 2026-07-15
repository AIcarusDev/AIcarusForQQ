"""Workspace settings, status, and user-owned lifecycle APIs."""

from __future__ import annotations

import asyncio
import os
from copy import deepcopy

from quart import Blueprint, jsonify, request

import app_state
from config_loader import save_workspace_config
from workspace.config import WorkspaceProvisionConfig
from workspace.control import WorkspaceControlError, WorkspaceControlPlane


workspace_bp = Blueprint("workspace_control", __name__)
workspace_control = WorkspaceControlPlane()


def _current_workspace_config() -> WorkspaceProvisionConfig:
    return WorkspaceProvisionConfig.from_root_config(
        app_state.config,
        environ=os.environ,
    )


@workspace_bp.route("/api/workspace", methods=["GET"])
async def workspace_status():
    try:
        config = _current_workspace_config()
        payload = await asyncio.to_thread(workspace_control.status_payload, config)
        return jsonify(payload)
    except (ValueError, WorkspaceControlError) as exc:
        status = exc.status_code if isinstance(exc, WorkspaceControlError) else 400
        return jsonify({"ok": False, "error": str(exc)}), status


@workspace_bp.route("/api/workspace/config", methods=["PUT"])
async def workspace_config_save():
    data = await request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "工作区配置必须是对象"}), 400
    try:
        current = _current_workspace_config()
        candidate_root = {"workspace": {
            "enabled": data.get("enabled") is True,
            "install_root": data.get("install_root", current.install_root),
            "resources": data.get("resources", {}),
        }}
        candidate = WorkspaceProvisionConfig.from_root_config(
            candidate_root,
            environ=os.environ,
        )
        observed = await asyncio.to_thread(workspace_control.probe, current)
        if observed.path_locked and candidate.install_root.casefold() != current.install_root.casefold():
            raise WorkspaceControlError("工作区建成后路径已锁定；请先完全卸载工作区")
        if observed.installed_resources and candidate.disk_gib < observed.installed_resources.get("disk_gib", 0):
            raise WorkspaceControlError("工作区磁盘只支持扩容；缩容需要完全卸载后重建")

        new_config = await asyncio.to_thread(
            save_workspace_config,
            candidate.to_config_dict(),
            base_config=deepcopy(app_state.config),
        )
        app_state.config = new_config

        if not candidate.enabled:
            state = getattr(app_state, "namespace_runtime_state", None)
            if state is not None:
                from tools.namespaces import load_namespace_registry

                state.close("workspace", load_namespace_registry())
        return jsonify(await asyncio.to_thread(workspace_control.status_payload, candidate))
    except (ValueError, WorkspaceControlError) as exc:
        status = exc.status_code if isinstance(exc, WorkspaceControlError) else 400
        return jsonify({"ok": False, "error": str(exc)}), status


@workspace_bp.route("/api/workspace/jobs", methods=["POST"])
async def workspace_job_start():
    data = await request.get_json(silent=True) or {}
    try:
        config = _current_workspace_config()
        job = await asyncio.to_thread(
            workspace_control.start_job,
            str(data.get("action") or ""),
            config,
            confirmation=str(data.get("confirmation") or ""),
        )
        return jsonify({"ok": True, "job": job}), 202
    except (ValueError, WorkspaceControlError) as exc:
        status = exc.status_code if isinstance(exc, WorkspaceControlError) else 400
        return jsonify({"ok": False, "error": str(exc)}), status


@workspace_bp.route("/api/workspace/jobs/<job_id>", methods=["GET"])
async def workspace_job_get(job_id: str):
    try:
        cursor = max(0, int(request.args.get("cursor", "0") or 0))
        return jsonify({"ok": True, "job": await asyncio.to_thread(workspace_control.get_job, job_id, cursor=cursor)})
    except (ValueError, WorkspaceControlError) as exc:
        status = exc.status_code if isinstance(exc, WorkspaceControlError) else 400
        return jsonify({"ok": False, "error": str(exc)}), status


__all__ = ["workspace_bp"]
