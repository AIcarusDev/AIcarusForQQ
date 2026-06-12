"""Runtime control routes."""

from __future__ import annotations

from quart import Blueprint, jsonify, request

import app_state
from runtime.emergency_reset import expected_confirmation, perform_emergency_reset

runtime_bp = Blueprint("runtime", __name__)


@runtime_bp.route("/api/runtime/emergency-reset", methods=["POST"])
async def api_emergency_reset():
    """Clear current consciousness runtime state after exact confirmation."""
    if getattr(app_state, "webui_only", False) or app_state.consciousness_flow is None:
        return jsonify({"ok": False, "error": "核心未运行，无法执行紧急恢复"}), 400

    data = await request.get_json(silent=True) or {}
    confirmation = str(data.get("confirmation") or "")
    expected = expected_confirmation()
    if confirmation != expected:
        return jsonify({
            "ok": False,
            "error": "确认字符串不匹配",
            "expected_confirmation": expected,
        }), 400

    result = await perform_emergency_reset()
    payload = result.to_dict()
    payload.update({
        "ok": True,
        "current_focus": app_state.current_focus,
    })
    return jsonify(payload)


__all__ = ["runtime_bp"]
