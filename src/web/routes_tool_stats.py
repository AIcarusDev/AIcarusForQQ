"""Tool usage statistics routes."""

import logging

from quart import Blueprint, jsonify, render_template, request

import app_state
from tool_usage_stats import ToolUsageStatsService

logger = logging.getLogger("AICQ.web.tool_stats")

tool_stats_bp = Blueprint("tool_stats", __name__)
_service = ToolUsageStatsService()

_DOWNLOAD_FAILURE_LABELS = {
    "source_unavailable": "下载源不可用",
    "file_too_large": "文件超过大小限制",
    "insufficient_disk_space": "磁盘空间不足",
    "transport_error": "下载连接异常",
    "write_failed": "文件写入失败",
    "size_mismatch": "文件大小不一致",
    "verification_failed": "文件保存校验失败",
    "download_interrupted": "下载被中断",
    "filesystem_unavailable": "文件空间不可用",
    "runtime_unavailable": "运行环境不可用",
    "internal_error": "内部错误",
}
_LEGACY_GENERIC_DOWNLOAD_FAILURE_MESSAGES = {
    "下载失败",
    "QQ 文件下载失败",
    "QQ 文件传输中断",
    "QQ 文件下载内容校验失败",
}


def _qq_runtime():
    registry = getattr(app_state, "platform_registry", None)
    return registry.get("qq") if registry is not None else None


def _download_monitor_job(job: dict) -> dict:
    failure = job.get("failure") if isinstance(job.get("failure"), dict) else None
    display_failure = None
    if failure is not None:
        code = str(failure.get("code") or "internal_error")
        message = str(failure.get("message") or "").strip()
        display_failure = {
            "code": code,
            "display_message": (
                message
                if message and message not in _LEGACY_GENERIC_DOWNLOAD_FAILURE_MESSAGES
                else _DOWNLOAD_FAILURE_LABELS.get(code, "未知原因")
            ),
        }
    return {
        "conversation": job.get("conversation"),
        "original_filename": job.get("original_filename"),
        "status": job.get("status"),
        "bytes_downloaded": job.get("bytes_downloaded"),
        "total_bytes": job.get("total_bytes"),
        "progress_percent": job.get("progress_percent"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "finished_at": job.get("finished_at"),
        "failure": display_failure,
    }


def _download_response(payload: dict, status: int = 200):
    response = jsonify(payload)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.status_code = status
    return response


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


@tool_stats_bp.route("/api/tool-stats/downloads", methods=["GET"])
async def tool_stats_downloads_api():
    limit = max(1, min(request.args.get("limit", 20, type=int) or 20, 100))
    try:
        runtime = _qq_runtime()
        client = getattr(runtime, "client", None)
        account = getattr(runtime, "account", None)
        account_id = str(getattr(account, "account_id", "") or "").strip()
        connected = bool(getattr(runtime, "connected", False))
        configured_adapter = str(getattr(client, "configured_adapter", "auto") or "auto") if client else "auto"
        detected_adapter = str(getattr(client, "detected_adapter", "") or "") if client else ""
        download_adapter = detected_adapter or (configured_adapter if configured_adapter != "auto" else "")
        if runtime is None or client is None or not account_id:
            return _download_response({
                "success": True,
                "available": False,
                "connected": False,
                "adapter": download_adapter,
                "download_capability": "unknown",
                "active": [],
                "history": [],
            })
        from platforms.qq.files.service import get_qq_file_service

        download_service = get_qq_file_service(client, app_state.workspace_service)
        if download_service.agent_qq() != account_id:
            logger.warning("下载任务账号上下文与 QQ runtime 不一致")
            return _download_response({
                "success": True,
                "available": False,
                "connected": connected,
                "adapter": download_adapter,
                "download_capability": "available" if download_adapter in {"napcat", "llonebot"} else "unknown",
                "active": [],
                "history": [],
            })
        result = await download_service.list_downloads(None, 0, limit)
        return _download_response({
            "success": True,
            "available": True,
            "connected": connected,
            "adapter": download_adapter,
            "download_capability": "available" if download_adapter in {"napcat", "llonebot"} else "unknown",
            "active": [_download_monitor_job(job) for job in result["active"]],
            "history": [_download_monitor_job(job) for job in result["terminal"]],
        })
    except Exception as exc:
        logger.warning("加载下载任务失败: %s", exc, exc_info=True)
        return _download_response({"success": False, "error": "下载任务暂时不可用"}, 500)


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
