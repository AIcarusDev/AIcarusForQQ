"""namespace_manage.py - manage prompt-facing tool namespaces."""

from __future__ import annotations

from typing import Any

DECLARATION: dict = {
    "name": "namespace_manage",
    "description": (
        "管理工具 namespace 的展开、折叠、预览和搜索。"
        "只影响工具 schema 是否进入 prompt，不直接执行业务工具。"
        "open 只在下一轮生效；本轮不要继续调用刚打开 namespace 内的工具。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "open": {
                "type": "array",
                "items": {"type": "string"},
                "description": "打开一个或多个 namespace，使其内部工具在下一轮可用。",
            },
            "close": {
                "type": "array",
                "items": {"type": "string"},
                "description": "关闭一个或多个 namespace。core 不能关闭。",
            },
            "preview": {
                "type": "array",
                "items": {"type": "string"},
                "description": "预览 namespace 内的工具名称和简短介绍，不展开 schema。",
            },
            "search": {
                "type": "string",
                "description": "用关键词搜索当前未展开 namespace 内部工具的 description。",
            },
        },
        "anyOf": [
            {"required": ["open"]},
            {"required": ["close"]},
            {"required": ["preview"]},
            {"required": ["search"]},
        ],
    },
}


def execute(**_: Any) -> dict[str, Any]:
    return {"ok": True}


def sanitize_semantic_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str], str | None]:
    repaired = dict(args)
    changes: list[str] = []
    for key in ("open", "close", "preview"):
        if key not in repaired:
            continue
        names, name_changes = _normalize_name_list(key, repaired.get(key))
        changes.extend(name_changes)
        if names:
            repaired[key] = names
        else:
            repaired.pop(key, None)

    if "search" in repaired:
        search = str(repaired.get("search") or "").strip()
        if search != repaired.get("search"):
            changes.append("search: trimmed surrounding whitespace")
        if search:
            repaired["search"] = search
        else:
            repaired.pop("search", None)

    if not any(key in repaired for key in ("open", "close", "preview", "search")):
        return repaired, changes, "namespace_manage requires at least one non-empty action"

    return repaired, changes, None


def _normalize_name_list(key: str, value: Any) -> tuple[list[str], list[str]]:
    if not isinstance(value, list):
        return [], []
    names: list[str] = []
    seen: set[str] = set()
    changes: list[str] = []
    for index, raw in enumerate(value):
        name = str(raw or "").strip()
        if not name:
            changes.append(f"{key}[{index}]: removed blank namespace name")
            continue
        if name in seen:
            changes.append(f"{key}[{index}]: removed duplicate namespace {name!r}")
            continue
        seen.add(name)
        names.append(name)
        if name != raw:
            changes.append(f"{key}[{index}]: trimmed surrounding whitespace")
    return names, changes
