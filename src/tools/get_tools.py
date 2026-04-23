"""get_tools.py — 渐进式披露：激活潜伏工具

调用此工具后，provider 循环在同轮内立即注入指定工具的 schema，
使其在本次回复的后续工具调用中可直接使用。
"""

from typing import Any

ALWAYS_AVAILABLE: bool = True

DECLARATION: dict = {
    "name": "get_tools",
    "description": (
        "激活一个或多个隐藏工具。"
        "请根据 <function_tools> 中的 <hidden> 列表选择需要的工具名称。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tool_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要激活的工具名称列表，例如 [\"get_qq_signature\", \"get_user_avatar\"]",
            }
        },
        "required": ["tool_names"],
    },
}


def execute(tool_names: list) -> dict:
    """返回激活信号；provider 循环读取 _inject_tools 并完成注入。"""
    names = [str(n) for n in (tool_names or [])]
    return {
        "activated": names,
        "_inject_tools": names,
    }


def sanitize_semantic_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str], str | None]:
    """去重并清理 tool_names 中的空白项。"""
    tool_names = args.get("tool_names")
    if not isinstance(tool_names, list):
        return args, [], None

    normalized_names: list[str] = []
    seen_names: set[str] = set()
    changes: list[str] = []
    for index, raw_name in enumerate(tool_names):
        name = str(raw_name).strip()
        if not name:
            changes.append(f"tool_names[{index}]: removed blank tool name")
            continue
        if name in seen_names:
            changes.append(f"tool_names[{index}]: removed duplicate tool name {name!r}")
            continue
        seen_names.add(name)
        normalized_names.append(name)
        if name != raw_name:
            changes.append(f"tool_names[{index}]: trimmed surrounding whitespace")

    if normalized_names == tool_names:
        return args, [], None

    repaired_args = dict(args)
    repaired_args["tool_names"] = normalized_names
    if not normalized_names:
        return repaired_args, changes, "tool_names is empty after semantic normalization"
    return repaired_args, changes, None
