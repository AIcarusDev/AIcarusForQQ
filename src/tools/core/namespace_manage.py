"""namespace_manage.py - manage prompt-facing tool namespaces."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from tools.contract import ToolArgsModel, ToolContract


class NamespaceManageArgs(ToolArgsModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "anyOf": [
                {"required": ["open"]},
                {"required": ["close"]},
                {"required": ["preview"]},
                {"required": ["search"]},
            ],
        },
    )

    open: list[str] | None = Field(
        default=None,
        description="打开一个或多个 namespace，使其包含的能力可用。",
    )
    close: list[str] | None = Field(
        default=None,
        description="关闭一个或多个已经用不到的 namespace。无法关闭 core。",
    )
    preview: list[str] | None = Field(
        default=None,
        description="预览 namespace 内的工具名称和简短介绍，而不展开。",
    )
    search: str | None = Field(
        default=None,
        description="用中文关键词搜索当前未展开 namespace 内部工具的 description。",
    )


TOOL_CONTRACT = ToolContract(
    name="namespace_manage",
    description=(
        "核心的能力管理工具，处理 namespace 的展开、关闭、预览和搜索。"
        "至少填写 open、close、preview、search 中的一个字段。"
    ),
    args_model=NamespaceManageArgs,
)


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
