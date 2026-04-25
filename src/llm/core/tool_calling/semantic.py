"""语义层清洗与验证。"""

from typing import Any, Callable

SemanticSanitizer = Callable[[dict[str, Any]], tuple[dict[str, Any], list[str], str | None]]


def sanitize_tool_arguments(
    args: dict[str, Any],
    semantic_sanitizer: SemanticSanitizer | None = None,
) -> tuple[dict[str, Any], list[str], str | None]:
    """执行工具级语义清洗；返回清洗结果、变更记录和失败信息。"""
    if semantic_sanitizer is None:
        return args, [], None
    return semantic_sanitizer(args)