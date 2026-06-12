"""IS 的结构化决策工具协议。"""

from typing import Any

from ..core.internal_tool import InternalToolSpec


DECLARATION: dict[str, Any] = {
    "name": "decide_continuation",
    "description": "做出你的决策：是否继续发送余下消息。",
    "parameters": {
        "type": "object",
        "properties": {
            "continue": {
                "type": "boolean",
                "description": "你是否继续发送余下消息，true 表示继续，false 表示不继续。",
            },
            "reason": {
                "type": "string",
                "description": "做这个决策的原因。",
            },
        },
        "required": ["continue", "reason"],
    },
}


def sanitize_semantic_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str], str | None]:
    """清理 decision reason 中的首尾空白。"""
    reason = args.get("reason")
    if not isinstance(reason, str):
        return args, [], None

    normalized_reason = reason.strip()
    if normalized_reason == reason:
        return args, [], None

    repaired_args = dict(args)
    repaired_args["reason"] = normalized_reason
    return repaired_args, ["trimmed reason whitespace"], None


TOOL = InternalToolSpec(
    declaration=DECLARATION,
    semantic_sanitizer=sanitize_semantic_args,
)


def read_result(args: dict[str, Any]) -> tuple[bool, str]:
    """将工具参数转换为 IS 的 continue/reason 决策。"""
    should_continue = bool(args.get("continue", True))
    reason = str(args.get("reason", ""))
    return should_continue, reason