"""Tool usage statistics routes."""

import logging

from quart import Blueprint, jsonify, render_template, request

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
        view = (request.args.get("view") or "summary").strip().lower()
        if view == "timeline":
            return jsonify(await _timeline_payload())
        if view == "bucket":
            bucket_start = request.args.get("bucket_start", type=int)
            if bucket_start is None:
                return jsonify({"success": False, "error": "bucket_start is required"}), 400
            return jsonify(await _bucket_payload(bucket_start))
        return jsonify(await _service.snapshot())
    except Exception as exc:
        logger.warning("加载工具统计失败: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 500


async def _timeline_payload():
    tools = [
        name.strip()
        for name in (request.args.get("tools") or "").split(",")
        if name.strip()
    ]
    return await _service.timeline(
        granularity=request.args.get("granularity") or "day",
        range_preset=request.args.get("range") or "all",
        tool_names=tools,
        limit=request.args.get("limit", 6),
        start_ms=request.args.get("start_ms", type=int),
        end_ms=request.args.get("end_ms", type=int),
        tz_offset_minutes=request.args.get("tz_offset_minutes", 480),
    )


async def _bucket_payload(bucket_start: int):
    return await _service.bucket_detail(
        granularity=request.args.get("granularity") or "day",
        bucket_start=bucket_start,
        tool_name=request.args.get("tool") or None,
        tz_offset_minutes=request.args.get("tz_offset_minutes", 480),
        limit=request.args.get("limit", 30),
    )


@tool_stats_bp.route("/api/tool-stats/timeline", methods=["GET"])
async def tool_stats_timeline_api():
    try:
        return jsonify(await _timeline_payload())
    except Exception as exc:
        logger.warning("加载工具趋势失败: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 500


@tool_stats_bp.route("/api/tool-stats/bucket", methods=["GET"])
async def tool_stats_bucket_api():
    try:
        bucket_start = request.args.get("bucket_start", type=int)
        if bucket_start is None:
            return jsonify({"success": False, "error": "bucket_start is required"}), 400
        return jsonify(await _bucket_payload(bucket_start))
    except Exception as exc:
        logger.warning("加载工具趋势详情失败: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 500
