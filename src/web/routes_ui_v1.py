"""Versioned UI contracts used by WebUI vNext."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from quart import Blueprint, jsonify, request

import app_state
from memory.semantic_query import (
    HARD_DEPTH_LIMIT,
    HARD_EDGE_LIMIT,
    HARD_NODE_LIMIT,
    HARD_ROW_LIMIT,
    LANGUAGE_VERSION,
    MemoryQLSyntaxError,
    MemoryQLValidationError,
    MemoryQueryTimeout,
    MemoryQueryUnavailable,
    SemanticMemoryService,
)
from token_usage_stats import TokenUsageStatsService
from tool_usage_stats import ToolUsageStatsService


logger = logging.getLogger("AICQ.web.ui_v1")
ui_v1_bp = Blueprint("ui_v1", __name__)
tool_stats_service = ToolUsageStatsService()
token_stats_service = TokenUsageStatsService()
semantic_memory_service = SemanticMemoryService()


def _optional_int(name: str) -> int | None:
    raw = request.args.get(name)
    if raw in {None, ""}:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = _optional_int(name)
    if value is None:
        value = default
    return max(minimum, min(maximum, value))


def _success(data: Any):
    return jsonify({"ok": True, "api_version": "1", "data": data})


def _error(code: str, message: str, status: int):
    return jsonify({
        "ok": False,
        "api_version": "1",
        "error": {"code": code, "message": message},
    }), status


def _positive_int(data: dict[str, Any], name: str, default: int) -> int:
    value = data.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须是大于 0 的整数")
    return value


def _non_negative_int(data: dict[str, Any], name: str, default: int) -> int:
    value = data.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数")
    return value


def _runtime_mode() -> tuple[str, bool]:
    if bool(getattr(app_state, "webui_only", False)):
        return "webui_only", False
    if bool(getattr(app_state, "webui_standalone", False)):
        return "standalone", False
    return "full", True


@ui_v1_bp.route("/api/ui/v1/capabilities", methods=["GET"])
async def capabilities():
    mode, core_available = _runtime_mode()
    return jsonify({
        "ok": True,
        "api_version": "1",
        "runtime": {
            "mode": mode,
            "core_available": core_available,
        },
        "migration": {
            "legacy_path": "/",
            "vnext_path": "/new/",
        },
        "capabilities": {
            "overview": {"read": True},
            "observability": {
                "tools": {
                    "timeline": True,
                    "latency": ["avg", "p50", "p95", "max"],
                },
                "tokens": {
                    "timeline": True,
                    "group_by": ["feature", "model"],
                },
            },
            "settings": {
                "domain_writes": True,
                "revision_conflicts": True,
                "secret_commands": ["keep", "replace", "clear"],
            },
            "memory": {
                "schema": True,
                "query": True,
                "read_only": True,
                "language": {"name": "MemoryQL", "version": LANGUAGE_VERSION},
            },
            "maintenance": {
                "server_described_actions": True,
                "exact_confirmation": True,
                "domains": ["data", "cache", "workspace"],
            },
        },
    })


@ui_v1_bp.route("/api/ui/v1/memory/schema", methods=["GET"])
async def memory_schema():
    try:
        data = await asyncio.to_thread(semantic_memory_service.schema)
    except Exception:
        logger.exception("加载 vNext 记忆语义 schema 失败")
        return _error("memory_schema_unavailable", "记忆语义结构暂时不可用", 500)
    return _success(data)


@ui_v1_bp.route("/api/ui/v1/memory/query", methods=["POST"])
async def memory_query():
    data = await request.get_json(silent=True)
    if not isinstance(data, dict):
        return _error("invalid_request", "请求正文必须是 JSON 对象", 400)

    query = data.get("query")
    language_version = data.get("language_version")
    if not isinstance(query, str) or not query.strip():
        return _error("invalid_request", "query 必须是非空字符串", 400)
    if not isinstance(language_version, str) or not language_version.strip():
        return _error("invalid_request", "language_version 必须是非空字符串", 400)

    try:
        limits = {
            "node_limit": _positive_int(data, "node_limit", HARD_NODE_LIMIT),
            "edge_limit": _positive_int(data, "edge_limit", HARD_EDGE_LIMIT),
            "row_limit": _positive_int(data, "row_limit", HARD_ROW_LIMIT),
            "max_depth": _non_negative_int(data, "max_depth", HARD_DEPTH_LIMIT),
        }
        result = await asyncio.to_thread(
            semantic_memory_service.query,
            query,
            language_version=language_version,
            **limits,
        )
    except MemoryQueryTimeout as exc:
        return _error(exc.code, str(exc), 408)
    except MemoryQueryUnavailable as exc:
        return _error(exc.code, str(exc), 503)
    except (MemoryQLSyntaxError, MemoryQLValidationError) as exc:
        return _error(exc.code, str(exc), 422)
    except ValueError as exc:
        return _error("invalid_request", str(exc), 400)
    except Exception:
        logger.exception("执行 vNext MemoryQL 失败")
        return _error("memory_query_unavailable", "记忆查询暂时不可用", 500)
    return _success(result)


@ui_v1_bp.route("/api/ui/v1/observability/tools", methods=["GET"])
async def observability_tools():
    tool_names = [
        name.strip()
        for name in (request.args.get("tools") or "").split(",")
        if name.strip()
    ]
    try:
        data = await tool_stats_service.timeline(
            granularity=request.args.get("granularity") or "day",
            range_preset=request.args.get("range") or "all",
            tool_names=tool_names,
            limit=_bounded_int("limit", 6, 1, 12),
            start_ms=_optional_int("start_ms"),
            end_ms=_optional_int("end_ms"),
            tz_offset_minutes=_bounded_int(
                "tz_offset_minutes",
                480,
                -14 * 60,
                14 * 60,
            ),
        )
    except Exception:
        logger.exception("加载 vNext 工具统计失败")
        return _error("tool_stats_unavailable", "工具统计暂时不可用", 500)
    return _success(data)


@ui_v1_bp.route("/api/ui/v1/observability/tokens", methods=["GET"])
async def observability_tokens():
    group_by = (request.args.get("group_by") or "feature").strip().lower()
    if group_by not in {"feature", "model"}:
        return _error(
            "invalid_group_by",
            "group_by 仅支持 feature 或 model",
            400,
        )
    try:
        data = await token_stats_service.timeline(
            granularity=request.args.get("granularity") or "day",
            range_preset=request.args.get("range") or "all",
            start_ms=_optional_int("start_ms"),
            end_ms=_optional_int("end_ms"),
            tz_offset_minutes=_bounded_int(
                "tz_offset_minutes",
                480,
                -14 * 60,
                14 * 60,
            ),
            provider=request.args.get("provider"),
            model=request.args.get("model"),
            feature=request.args.get("feature"),
            group_by=group_by,
        )
    except Exception:
        logger.exception("加载 vNext Token 统计失败")
        return _error("token_stats_unavailable", "Token 统计暂时不可用", 500)
    return _success(data)


__all__ = ["ui_v1_bp"]
