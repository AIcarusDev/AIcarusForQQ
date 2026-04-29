"""工具参数处理流水线。"""

import logging
from typing import Any

from .models import ToolArgumentFailure, ToolArgumentProcessingResult
from .parser import parse_argument_object
from .schema import SchemaRepairer, repair_arguments_by_declaration, validate_arguments_by_declaration
from .semantic import SemanticSanitizer, sanitize_tool_arguments

logger = logging.getLogger("AICQ.llm.tool_calling")


def _merge_detail_lists(*value_lists: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    merged: list[str] = []
    for values in value_lists:
        for value in values:
            if value and value not in merged:
                merged.append(value)
    return tuple(merged)


def _log_processing_result(provider_name: str, raw_arguments: str | None, result: ToolArgumentProcessingResult) -> None:
    if result.repaired_json:
        logger.warning(
            "[%s] 工具参数已按 tool-call 规则自动恢复: %s",
            provider_name,
            result.fn_name,
        )
        if result.repaired_source is not None and result.repaired_source != raw_arguments:
            logger.debug(
                "[%s] 工具参数恢复前后不同: %s raw=%r repaired=%r",
                provider_name,
                result.fn_name,
                (raw_arguments or "")[:200],
                result.repaired_source[:200],
            )
    if result.parse_changes:
        logger.warning(
            "[%s] 工具参数已在解析阶段静默修复: %s changes=%s",
            provider_name,
            result.fn_name,
            "; ".join(result.parse_changes),
        )
    if result.schema_changes:
        logger.warning(
            "[%s] 工具参数已按 schema 自动修复: %s changes=%s",
            provider_name,
            result.fn_name,
            "; ".join(result.schema_changes),
        )
    if result.semantic_changes:
        logger.warning(
            "[%s] 工具参数已按语义规则静默清洗: %s changes=%s",
            provider_name,
            result.fn_name,
            "; ".join(result.semantic_changes),
        )
    if not result.ok and result.failure is not None:
        logger.warning(
            "[%s] 工具参数%s阶段失败: %s — %s",
            provider_name,
            result.failure.stage,
            result.fn_name,
            result.failure.message,
        )
        if result.failure.details:
            logger.warning(
                "[%s] 工具参数%s阶段详情: %s details=%s",
                provider_name,
                result.failure.stage,
                result.fn_name,
                "; ".join(result.failure.details),
            )


def process_tool_arguments(
    raw_arguments: str | None,
    fn_name: str,
    provider_name: str,
    tool_declaration: dict[str, Any] | None = None,
    schema_repairer: SchemaRepairer | None = None,
    semantic_sanitizer: SemanticSanitizer | None = None,
) -> ToolArgumentProcessingResult:
    """统一执行解析、schema 修复/校验、语义清洗。"""
    parsed_args, repaired_json, repaired_source, parse_changes = parse_argument_object(
        raw_arguments,
        fn_name,
    )
    if parsed_args is None:
        result = ToolArgumentProcessingResult(
            fn_name=fn_name,
            args={},
            ok=False,
            failure=ToolArgumentFailure(
                stage="parse",
                message="arguments is not a valid JSON object",
            ),
        )
        _log_processing_result(provider_name, raw_arguments, result)
        return result

    current_args = parsed_args
    current_args, schema_changes = repair_arguments_by_declaration(
        current_args,
        tool_declaration,
        schema_repairer,
    )
    valid_schema, schema_errors, schema_message = validate_arguments_by_declaration(
        current_args,
        tool_declaration,
    )

    if not valid_schema:
        result = ToolArgumentProcessingResult(
            fn_name=fn_name,
            args=current_args,
            ok=False,
            failure=ToolArgumentFailure(
                stage="schema",
                message=schema_message or "arguments do not satisfy schema",
                details=_merge_detail_lists(schema_changes, schema_errors),
            ),
            repaired_json=repaired_json,
            repaired_source=repaired_source,
            parse_changes=tuple(parse_changes),
            schema_changes=tuple(schema_changes),
        )
        _log_processing_result(provider_name, raw_arguments, result)
        return result

    semantic_args, semantic_changes, semantic_error = sanitize_tool_arguments(
        current_args,
        semantic_sanitizer,
    )
    if semantic_error is not None:
        result = ToolArgumentProcessingResult(
            fn_name=fn_name,
            args=semantic_args,
            ok=False,
            failure=ToolArgumentFailure(
                stage="semantic",
                message=semantic_error,
                details=tuple(semantic_changes),
            ),
            repaired_json=repaired_json,
            repaired_source=repaired_source,
            parse_changes=tuple(parse_changes),
            schema_changes=tuple(schema_changes),
            semantic_changes=tuple(semantic_changes),
        )
        _log_processing_result(provider_name, raw_arguments, result)
        return result

    result = ToolArgumentProcessingResult(
        fn_name=fn_name,
        args=semantic_args,
        ok=True,
        repaired_json=repaired_json,
        repaired_source=repaired_source,
        parse_changes=tuple(parse_changes),
        schema_changes=tuple(schema_changes),
        semantic_changes=tuple(semantic_changes),
    )
    _log_processing_result(provider_name, raw_arguments, result)
    return result


def parse_tool_arguments(
    raw_arguments: str | None,
    fn_name: str,
    provider_name: str,
    tool_declaration: dict[str, Any] | None = None,
    schema_repairer: SchemaRepairer | None = None,
    semantic_sanitizer: SemanticSanitizer | None = None,
) -> tuple[dict[str, Any], bool]:
    """兼容旧调用点，只返回 (args, ok)。"""
    result = process_tool_arguments(
        raw_arguments,
        fn_name,
        provider_name,
        tool_declaration,
        schema_repairer,
        semantic_sanitizer,
    )
    return result.args, result.ok


def build_tool_argument_error(result: ToolArgumentProcessingResult) -> dict[str, Any]:
    """将处理失败结果转为回传模型的结构化错误。"""
    if result.ok or result.failure is None:
        return {}

    payload: dict[str, Any] = {
        "error": f"TOOL_ARGUMENT_{result.failure.stage.upper()}_FAILED: {result.failure.message}",
        "stage": result.failure.stage,
    }
    if result.failure.details:
        payload["details"] = list(result.failure.details)

    repairs = [*result.parse_changes, *result.schema_changes, *result.semantic_changes]
    if repairs:
        payload["repairs"] = repairs
    if result.args:
        payload["arguments"] = result.args
    return payload