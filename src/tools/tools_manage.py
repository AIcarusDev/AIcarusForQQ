"""tools_manage.py — 渐进式披露：发现与激活潜伏工具。"""

from __future__ import annotations

import re
from typing import Any

ALWAYS_AVAILABLE: bool = True

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

DECLARATION: dict = {
    "name": "tools_manage",
    "description": (
        "函数工具管理；作用于 `<tools><hidden>` 中尚未激活的工具。"
        "`get` 激活一个或多个隐藏工具；"
        "`preview` 预览一个或多个隐藏工具的顶层 description；当基于隐藏工具的名称，不确定其作用时可使用。"
        "`search` 用中文关键词只读搜索隐藏工具的顶层 description，最多返回 5 个匹配；在你有意图，但是不确定自己有没有相关工具时可使用。"
        "`search`与`preview`不直接激活工具，只做只读。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "get": {
                "type": "array",
                "items": {"type": "string"},
                "description": "激活工具，写入要激活的工具名称列表，例如 [\"get_qq_signature\", \"get_user_avatar\"]",
            },
            "preview": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要预览 description 的隐藏工具名称列表。",
            },
            "search": {
                "type": "string",
                "description": "用于匹配隐藏工具顶层 description 的中文关键词。",
            },
        },
        "anyOf": [
            {"required": ["get"]},
            {"required": ["preview"]},
            {"required": ["search"]},
        ],
    },
}


def execute(
    get: list | None = None,
    preview: list | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """返回管理请求；provider 负责按当前 ToolCollection 补充真实结果。"""
    result: dict[str, Any] = {"ok": True}
    if get is not None:
        names = [str(n) for n in (get or [])]
        result["activated"] = names
        result["_inject_tools"] = names
    return result


def repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """兼容旧 get_tools 参数名，将 tool_names 映射为 get。"""
    if "tool_names" not in args or "get" in args:
        return args, []

    repaired = dict(args)
    repaired["get"] = repaired.pop("tool_names")
    return repaired, ["tool_names -> get (legacy get_tools args)"]


def sanitize_semantic_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str], str | None]:
    """去重/清理动作参数，并要求 search 使用中文关键词。"""
    repaired_args = dict(args)
    changes: list[str] = []

    for key in ("get", "preview"):
        if key not in repaired_args:
            continue
        normalized_names, name_changes = _normalize_name_list(key, repaired_args.get(key))
        changes.extend(name_changes)
        if normalized_names:
            repaired_args[key] = normalized_names
        else:
            repaired_args.pop(key, None)

    if "search" in repaired_args:
        raw_search = repaired_args.get("search")
        search = str(raw_search).strip()
        if search != raw_search:
            changes.append("search: trimmed surrounding whitespace")
        if not search:
            repaired_args.pop("search", None)
        elif _CJK_RE.search(search) is None:
            return repaired_args, changes, "search must contain a Chinese keyword"
        else:
            repaired_args["search"] = search

    if not any(
        key in repaired_args
        for key in ("get", "preview", "search")
    ):
        return repaired_args, changes, "tools_manage requires at least one non-empty action"

    return repaired_args, changes, None


def _normalize_name_list(key: str, value: Any) -> tuple[list[str], list[str]]:
    if not isinstance(value, list):
        return [], []

    normalized_names: list[str] = []
    seen_names: set[str] = set()
    changes: list[str] = []
    for index, raw_name in enumerate(value):
        name = str(raw_name).strip()
        if not name:
            changes.append(f"{key}[{index}]: removed blank tool name")
            continue
        if name in seen_names:
            changes.append(f"{key}[{index}]: removed duplicate tool name {name!r}")
            continue
        seen_names.add(name)
        normalized_names.append(name)
        if name != raw_name:
            changes.append(f"{key}[{index}]: trimmed surrounding whitespace")

    return normalized_names, changes
