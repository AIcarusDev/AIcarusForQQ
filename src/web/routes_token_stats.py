"""Token usage statistics routes."""

import logging

from quart import Blueprint, jsonify, render_template, request

from token_usage_stats import TokenUsageStatsService

logger = logging.getLogger("AICQ.web.token_stats")

token_stats_bp = Blueprint("token_stats", __name__)
_service = TokenUsageStatsService()


@token_stats_bp.route("/token-stats")
async def token_stats_page():
    return await render_template("token_stats.html", active_page="token_stats")


@token_stats_bp.route("/api/token-stats", methods=["GET"])
async def token_stats_api():
    try:
        view = (request.args.get("view") or "summary").strip().lower()
        if view == "timeline":
            return jsonify(await _timeline_payload())
        return jsonify(await _service.snapshot())
    except Exception as exc:
        logger.warning("加载 token 统计失败: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 500


async def _timeline_payload():
    return await _service.timeline(
        granularity=request.args.get("granularity") or "day",
        range_preset=request.args.get("range") or "all",
        start_ms=request.args.get("start_ms", type=int),
        end_ms=request.args.get("end_ms", type=int),
        tz_offset_minutes=request.args.get("tz_offset_minutes", 480),
    )
