"""Token usage statistics routes."""

import logging

from quart import Blueprint, jsonify, render_template

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
        return jsonify(await _service.snapshot())
    except Exception as exc:
        logger.warning("加载 token 统计失败: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 500
