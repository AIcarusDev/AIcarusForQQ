"""Tool usage statistics routes."""

import logging

from quart import Blueprint, jsonify, render_template

from tool_usage_stats import ToolUsageStatsService

logger = logging.getLogger("AICQ.web.tool_stats")

tool_stats_bp = Blueprint("tool_stats", __name__)
_service = ToolUsageStatsService()


@tool_stats_bp.route("/tool-stats")
async def tool_stats_page():
    return await render_template("tool_stats.html", active_page="tool_stats")


@tool_stats_bp.route("/api/tool-stats", methods=["GET"])
async def tool_stats_api():
    try:
        return jsonify(await _service.snapshot())
    except Exception as exc:
        logger.warning("加载工具统计失败: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 500
