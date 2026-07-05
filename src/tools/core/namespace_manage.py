"""namespace_manage.py - manage prompt-facing tool namespaces."""

from __future__ import annotations

from typing import Any, Callable

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

PARALLEL_SAFE = True
PARALLEL_KEY = "namespace_state"
REQUIRES_CONTEXT: list[str] = ["tool_collection"]


def execute(**_: Any) -> dict[str, Any]:
    return {"ok": True}


def make_handler(tool_collection) -> Callable[..., dict[str, Any]]:
    def _execute(**kwargs: Any) -> dict[str, Any]:
        return _namespace_manage_result(kwargs, tool_collection)

    return _execute


def repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    repaired = dict(args)
    changes: list[str] = []
    for key in ("open", "close", "preview"):
        value = repaired.get(key)
        if isinstance(value, str):
            repaired[key] = [value]
            changes.append(f"{key}: string -> single-item array")
    return repaired, changes


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


def _namespace_manage_result(args: dict[str, Any], tool_collection) -> dict[str, Any]:
    registry = getattr(tool_collection, "namespace_registry", None)
    state = getattr(tool_collection, "namespace_state", None)
    if registry is None or state is None:
        return {"ok": False, "error": "namespace registry is unavailable"}

    result: dict[str, Any] = {"ok": True}
    lifecycle: dict[str, list[str]] = {"opened": [], "closed": []}
    opened_or_available: list[str] = []
    closed: list[str] = []
    already_closed: list[str] = []
    protected: list[str] = []
    not_found: list[str] = []

    for name in _namespace_name_list(args.get("open")):
        if name not in getattr(tool_collection, "namespace_specs", {}):
            not_found.append(name)
            continue
        status = state.open(name, registry, getattr(tool_collection, "round_index", 0))
        if status in {"opened", "already_open"}:
            opened_or_available.append(name)
            if status == "opened":
                lifecycle["opened"].append(name)
        else:
            not_found.append(name)

    for name in _namespace_name_list(args.get("close")):
        status = state.close(name, registry)
        if status == "closed":
            closed.append(name)
            lifecycle["closed"].append(name)
        elif status == "protected":
            protected.append(name)
        elif status == "already_closed":
            already_closed.append(name)
        else:
            not_found.append(name)

    previews: list[dict[str, Any]] = []
    preview_warnings: list[dict[str, Any]] = []
    for name in _namespace_name_list(args.get("preview")):
        preview = tool_collection.preview_namespace(name)
        if preview is None:
            preview_warnings.append({"name": name, "warning": "未找到 namespace。"})
        else:
            previews.append(preview)
    _set_non_empty(result, "closed", closed)
    _set_non_empty(result, "already_closed", already_closed)
    _set_non_empty(result, "protected", protected)
    _set_non_empty(result, "not_found", not_found)
    _set_non_empty(result, "preview", previews)
    _set_non_empty(result, "warnings", preview_warnings)

    search = args.get("search")
    if isinstance(search, str):
        _set_non_empty(result, "search", tool_collection.search_inactive_namespaces(search))

    active_namespaces = tool_collection.active_namespace_names()
    _set_non_empty(result, "tools", _namespace_tools_for_namespaces(opened_or_available, tool_collection))
    _set_non_empty(
        result,
        "attached_tools",
        _namespace_attached_tools_for_namespaces(opened_or_available, active_namespaces, tool_collection),
    )
    _set_non_empty(result, "skills", _loaded_skills_for_namespaces(opened_or_available, registry))
    if lifecycle["opened"] or lifecycle["closed"]:
        result["_namespace_lifecycle"] = lifecycle
    return result


def _set_non_empty(result: dict[str, Any], key: str, value: Any) -> None:
    if value not in (None, "", [], {}):
        result[key] = value


def _namespace_name_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for item in value:
        name = str(item or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _namespace_tools_for_namespaces(
    namespaces: list[str],
    tool_collection,
) -> list[dict[str, Any]]:
    registry = getattr(tool_collection, "namespace_registry", None)
    all_specs = getattr(tool_collection, "all_specs", {}) or {}
    entries: list[dict[str, Any]] = []
    for namespace in namespaces:
        spec = registry.get(namespace) if registry is not None else None
        if spec is None:
            continue
        if not getattr(spec, "visible", True) or not getattr(spec, "discoverable", True):
            continue
        tools = [tool for tool in getattr(spec, "tools", ()) or () if tool in all_specs]
        if tools:
            entries.append({"namespace": namespace, "tools": tools})
    return entries


def _namespace_attached_tools_for_namespaces(
    namespaces: list[str],
    active_namespaces: list[str],
    tool_collection,
) -> list[dict[str, Any]]:
    registry = getattr(tool_collection, "namespace_registry", None)
    all_specs = getattr(tool_collection, "all_specs", {}) or {}
    active = set(active_namespaces)
    attached: list[dict[str, Any]] = []
    for namespace in namespaces:
        spec = registry.get(namespace) if registry is not None else None
        if spec is None:
            continue
        if not getattr(spec, "visible", True):
            continue
        for attach in getattr(spec, "attach", ()) or ():
            if attach.namespace in active:
                continue
            if attach.tool not in all_specs:
                continue
            attached.append({
                "host_namespace": namespace,
                "source_namespace": attach.namespace,
                "tools": [attach.tool],
            })
    return attached


def _loaded_skills_for_namespaces(namespaces: list[str], registry) -> list[dict[str, str]]:
    try:
        from skills import load_skill_body
    except Exception:
        load_skill_body = None

    loaded: list[dict[str, str]] = []
    seen: set[str] = set()
    for namespace in namespaces:
        spec = registry.get(namespace) if registry is not None else None
        skill = str(getattr(spec, "skill", "") or "").strip()
        if not skill or skill in seen:
            continue
        if load_skill_body is not None and not load_skill_body(skill).strip():
            continue
        seen.add(skill)
        loaded.append({"namespace": namespace, "skill": skill})
    return loaded
