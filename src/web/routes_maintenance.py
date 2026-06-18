"""Maintenance page and dangerous action APIs."""

from __future__ import annotations

from quart import Blueprint, jsonify, render_template, request

from runtime.maintenance import MaintenanceError, maintenance_service

maintenance_bp = Blueprint("maintenance", __name__)


@maintenance_bp.route("/maintenance")
async def maintenance_page():
    return await render_template(
        "maintenance.html",
        active_page="maintenance",
        actions=maintenance_service.describe_actions(),
    )


@maintenance_bp.route("/api/maintenance/actions", methods=["GET"])
async def api_maintenance_actions():
    return jsonify({
        "ok": True,
        "actions": maintenance_service.describe_actions(),
        "overview": await maintenance_service.overview(),
    })


@maintenance_bp.route("/api/maintenance/actions/<action>", methods=["POST"])
async def api_maintenance_action(action: str):
    data = await request.get_json(silent=True) or {}
    confirmation = str(data.get("confirmation") or "")
    try:
        expected = maintenance_service.expected_confirmation(action)
        if confirmation != expected:
            return jsonify({
                "ok": False,
                "error": "确认字符串不匹配",
                "expected_confirmation": expected,
            }), 400
        result = await maintenance_service.perform(action)
        return jsonify(result.to_dict())
    except MaintenanceError as exc:
        return jsonify({"ok": False, "error": str(exc)}), exc.status_code


__all__ = ["maintenance_bp"]
