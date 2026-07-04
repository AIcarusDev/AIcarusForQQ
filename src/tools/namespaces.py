"""Namespace registry and runtime state for prompt-facing tool visibility."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("AICQ.tools.namespaces")

_REGISTRY_PATH = Path(__file__).with_name("namespaces.yaml")
_MODULE_REGISTRY_PATH = Path(__file__).with_name("modules.yaml")
_PLATFORMS_DIR = Path(__file__).resolve().parents[1] / "platforms"
_PLATFORM_TOOLS_MANIFEST = "tools_manifest.yaml"
CORE_NAMESPACE = "core"


@dataclass(frozen=True)
class NamespaceAttachSpec:
    namespace: str
    tool: str
    reason: str = ""


@dataclass(frozen=True)
class NamespaceCloseOnSpec:
    tool: str
    action: str = ""
    ok: bool | None = None


@dataclass(frozen=True)
class NamespaceLifecycleSpec:
    keep_open_while: str = ""
    close_on: tuple[NamespaceCloseOnSpec, ...] = ()


@dataclass(frozen=True)
class NamespaceSpec:
    name: str
    description: str = ""
    permanent: bool = False
    closeable: bool = True
    visible: bool = True
    openable: bool = True
    discoverable: bool = True
    path: str = ""
    import_path: str = ""
    ttl_rounds: int | None = None
    skill: str = ""
    tools: tuple[str, ...] = ()
    attach: tuple[NamespaceAttachSpec, ...] = ()
    lifecycle: NamespaceLifecycleSpec = field(default_factory=NamespaceLifecycleSpec)


@dataclass(frozen=True)
class NamespaceRegistry:
    namespaces: dict[str, NamespaceSpec]
    order: tuple[str, ...]
    tool_to_namespace: dict[str, str]

    def get(self, name: str) -> NamespaceSpec | None:
        return self.namespaces.get(str(name or "").strip())

    def namespace_for_tool(self, tool_name: str) -> str:
        return self.tool_to_namespace.get(str(tool_name or "").strip(), "")

    def known_namespace_names(self) -> set[str]:
        return set(self.namespaces)

    def is_prompt_visible(self, name: str) -> bool:
        spec = self.get(name)
        return bool(spec and spec.visible)

    def is_openable(self, name: str) -> bool:
        spec = self.get(name)
        return bool(spec and spec.visible and spec.openable)


@dataclass(frozen=True)
class ModuleMountSpec:
    source_namespace: str
    target_namespace: str
    tools: tuple[str, ...] = ()
    when: str = ""


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    path: str = ""
    always_active: bool = False
    active_when: str = ""
    namespaces: tuple[str, ...] = ()
    mounts: tuple[ModuleMountSpec, ...] = ()


@dataclass(frozen=True)
class ModuleRegistry:
    modules: dict[str, ModuleSpec]
    order: tuple[str, ...]


@dataclass
class NamespaceRuntimeState:
    """Unique global namespace state.

    ``open_order`` stores non-permanent namespaces in prompt order. Permanent
    namespaces are derived from the registry and never stored here.
    """

    open_order: list[str] = field(default_factory=list)
    last_active_round: dict[str, int] = field(default_factory=dict)
    recovered_from_flow: bool = False

    def is_open(self, namespace: str, registry: NamespaceRegistry) -> bool:
        spec = registry.get(namespace)
        if spec is None:
            return False
        return bool(spec.permanent or namespace in self.open_order)

    def active_namespaces(self, registry: NamespaceRegistry) -> list[str]:
        names: list[str] = []
        for name in registry.order:
            spec = registry.get(name)
            if spec is not None and spec.permanent:
                names.append(name)
        for name in self.open_order:
            if name in registry.namespaces and name not in names:
                names.append(name)
        return names

    def open(self, namespace: str, registry: NamespaceRegistry, round_index: int) -> str:
        spec = registry.get(namespace)
        if spec is None:
            return "not_found"
        if not spec.visible or not spec.openable:
            return "not_found"
        if spec.permanent:
            self.last_active_round[namespace] = round_index
            return "already_open"
        if namespace not in self.open_order:
            self.open_order.append(namespace)
            self.last_active_round[namespace] = round_index
            return "opened"
        self.last_active_round[namespace] = round_index
        return "already_open"

    def close(self, namespace: str, registry: NamespaceRegistry) -> str:
        spec = registry.get(namespace)
        if spec is None:
            return "not_found"
        if not spec.visible or not spec.openable:
            return "not_found"
        if spec.permanent or not spec.closeable:
            return "protected"
        if namespace in self.open_order:
            self.open_order = [name for name in self.open_order if name != namespace]
            self.last_active_round.pop(namespace, None)
            return "closed"
        return "already_closed"

    def mark_active(self, namespace: str, registry: NamespaceRegistry, round_index: int) -> None:
        spec = registry.get(namespace)
        if spec is None:
            return
        if spec.permanent:
            self.last_active_round[namespace] = round_index
            return
        if namespace in self.open_order:
            self.last_active_round[namespace] = round_index

    def apply_ttl(
        self,
        registry: NamespaceRegistry,
        *,
        current_round: int,
        default_ttl_rounds: int | None,
    ) -> list[str]:
        expired: list[str] = []
        retained: list[str] = []
        for namespace in self.open_order:
            spec = registry.get(namespace)
            if spec is None:
                expired.append(namespace)
                self.last_active_round.pop(namespace, None)
                continue
            if _lifecycle_keep_open(spec):
                retained.append(namespace)
                continue
            ttl = spec.ttl_rounds if spec.ttl_rounds is not None else default_ttl_rounds
            if ttl is None:
                retained.append(namespace)
                continue
            last_active = self.last_active_round.get(namespace, current_round)
            if current_round - last_active > int(ttl):
                expired.append(namespace)
                self.last_active_round.pop(namespace, None)
            else:
                retained.append(namespace)
        self.open_order = retained
        return expired

    def replace_with(self, other: "NamespaceRuntimeState") -> None:
        self.open_order = list(other.open_order)
        self.last_active_round = dict(other.last_active_round)
        self.recovered_from_flow = other.recovered_from_flow


def load_namespace_registry(path: Path | None = None) -> NamespaceRegistry:
    registry_path = path or _REGISTRY_PATH
    namespaces: dict[str, NamespaceSpec] = {}
    order: list[str] = []
    tool_to_namespace: dict[str, str] = {}
    for source_path, raw_namespaces in _iter_namespace_sources(registry_path, include_platforms=path is None):
        for raw_name, raw_spec in raw_namespaces.items():
            name = str(raw_name or "").strip()
            if not name or not isinstance(raw_spec, dict):
                continue
            if name in namespaces:
                raise ValueError(f"Duplicate namespace {name!r} in {source_path}")
            attach_specs = tuple(
                NamespaceAttachSpec(
                    namespace=str(item.get("namespace") or "").strip(),
                    tool=str(item.get("tool") or "").strip(),
                    reason=str(item.get("reason") or "").strip(),
                )
                for item in raw_spec.get("attach") or []
                if isinstance(item, dict) and item.get("namespace") and item.get("tool")
            )
            lifecycle_value = raw_spec.get("lifecycle")
            lifecycle_raw: dict[str, Any] = lifecycle_value if isinstance(lifecycle_value, dict) else {}
            close_on = tuple(
                NamespaceCloseOnSpec(
                    tool=str(item.get("tool") or "").strip(),
                    action=str(item.get("action") or "").strip(),
                    ok=bool(item["ok"]) if "ok" in item else None,
                )
                for item in lifecycle_raw.get("close_on") or []
                if isinstance(item, dict) and item.get("tool")
            )
            lifecycle = NamespaceLifecycleSpec(
                keep_open_while=str(lifecycle_raw.get("keep_open_while") or "").strip(),
                close_on=close_on,
            )
            ttl_raw = raw_spec.get("ttl_rounds")
            ttl_rounds = int(ttl_raw) if ttl_raw is not None else None
            tools = tuple(str(tool or "").strip() for tool in raw_spec.get("tools") or [] if str(tool or "").strip())
            spec = NamespaceSpec(
                name=name,
                description=str(raw_spec.get("description") or ""),
                permanent=bool(raw_spec.get("permanent", False)),
                closeable=bool(raw_spec.get("closeable", True)),
                visible=bool(raw_spec.get("visible", True)),
                openable=bool(raw_spec.get("openable", True)),
                discoverable=bool(raw_spec.get("discoverable", True)),
                path=str(raw_spec.get("path") or name).strip(),
                import_path=str(raw_spec.get("import_path") or "").strip(),
                ttl_rounds=ttl_rounds,
                skill=str(raw_spec.get("skill") or "").strip(),
                tools=tools,
                attach=attach_specs,
                lifecycle=lifecycle,
            )
            namespaces[name] = spec
            order.append(name)
            for tool in tools:
                if tool in tool_to_namespace:
                    raise ValueError(
                        f"Duplicate tool {tool!r} in namespaces "
                        f"{tool_to_namespace[tool]!r} and {name!r}"
                    )
                tool_to_namespace[tool] = name

    return NamespaceRegistry(
        namespaces=namespaces,
        order=tuple(order),
        tool_to_namespace=tool_to_namespace,
    )


def load_module_registry(path: Path | None = None) -> ModuleRegistry:
    registry_path = path or _MODULE_REGISTRY_PATH
    modules: dict[str, ModuleSpec] = {}
    order: list[str] = []
    for source_path, raw_modules in _iter_module_sources(registry_path, include_platforms=path is None):
        for raw_name, raw_spec in raw_modules.items():
            name = str(raw_name or "").strip()
            if not name or not isinstance(raw_spec, dict):
                continue
            if name in modules:
                raise ValueError(f"Duplicate tool module {name!r} in {source_path}")
            mounts = tuple(
                ModuleMountSpec(
                    source_namespace=str(item.get("from") or item.get("source") or "").strip(),
                    target_namespace=str(item.get("to") or item.get("target") or "").strip(),
                    tools=tuple(str(tool or "").strip() for tool in item.get("tools") or [] if str(tool or "").strip()),
                    when=str(item.get("when") or "").strip(),
                )
                for item in raw_spec.get("mounts") or []
                if isinstance(item, dict) and (item.get("from") or item.get("source")) and (item.get("to") or item.get("target"))
            )
            namespaces = tuple(
                str(namespace or "").strip()
                for namespace in raw_spec.get("namespaces") or []
                if str(namespace or "").strip()
            )
            spec = ModuleSpec(
                name=name,
                path=str(raw_spec.get("path") or name).strip(),
                always_active=bool(raw_spec.get("always_active", False)),
                active_when=str(raw_spec.get("active_when") or "").strip(),
                namespaces=namespaces,
                mounts=mounts,
            )
            modules[name] = spec
            order.append(name)

    registry = ModuleRegistry(modules=modules, order=tuple(order))
    if path is None:
        _validate_module_registry(load_namespace_registry(), registry)
    return registry


def recover_namespace_state_from_flow(
    flow: Any,
    registry: NamespaceRegistry,
    *,
    max_rounds: int | None,
    current_round: int,
) -> NamespaceRuntimeState:
    state = NamespaceRuntimeState(recovered_from_flow=True)
    if flow is None or not hasattr(flow, "recent_rounds"):
        return state
    rounds = list(flow.recent_rounds(max_rounds or 20))
    for rnd in rounds:
        seq = int(getattr(rnd, "seq", current_round) or current_round)
        calls = list(getattr(rnd, "calls", []) or [])
        responses = list(getattr(rnd, "responses", []) or [])
        for index, call in enumerate(calls):
            name = str(getattr(call, "name", "") or "")
            args = getattr(call, "args", {}) if isinstance(getattr(call, "args", {}), dict) else {}
            response = responses[index].response if index < len(responses) else None
            if name == "namespace_manage":
                _replay_namespace_manage(state, registry, args, seq)
                continue
            namespace = registry.namespace_for_tool(name)
            if not namespace or namespace == CORE_NAMESPACE:
                continue
            if _response_kept_tool_reachable(response):
                state.open(namespace, registry, seq)
    state.apply_ttl(registry, current_round=current_round, default_ttl_rounds=max_rounds)
    return state


def _replay_namespace_manage(
    state: NamespaceRuntimeState,
    registry: NamespaceRegistry,
    args: dict[str, Any],
    seq: int,
) -> None:
    for name in _name_list(args.get("open")):
        state.open(name, registry, seq)
    for name in _name_list(args.get("close")):
        state.close(name, registry)


def _name_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        name = str(item or "").strip()
        if name and name not in result:
            result.append(name)
    return result


def _response_kept_tool_reachable(response: object) -> bool:
    if not isinstance(response, dict):
        return True
    if response.get("tool_not_executed") is True and response.get("namespace_opened_next_round") is not True:
        return False
    error = str(response.get("error") or "")
    if "closed earlier in this same action" in error:
        return False
    if error.startswith("未知工具:"):
        return False
    return True


def _lifecycle_keep_open(spec: NamespaceSpec) -> bool:
    if spec.lifecycle.keep_open_while != "browser_world_active":
        return False
    try:
        from browser.session import browser_world_view_state

        state = browser_world_view_state()
        return bool(state.get("active"))
    except Exception:
        logger.debug("[tools] browser lifecycle keep_open check failed", exc_info=True)
        return False


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return raw


def _iter_platform_manifest_paths() -> tuple[Path, ...]:
    if not _PLATFORMS_DIR.is_dir():
        return ()
    return tuple(
        path
        for path in sorted(_PLATFORMS_DIR.glob(f"*/{_PLATFORM_TOOLS_MANIFEST}"))
        if path.is_file()
    )


def _iter_namespace_sources(
    registry_path: Path,
    *,
    include_platforms: bool,
) -> list[tuple[Path, dict[str, Any]]]:
    sources: list[tuple[Path, dict[str, Any]]] = []
    raw = _read_yaml_mapping(registry_path)
    raw_namespaces = raw.get("namespaces")
    if raw_namespaces is None and not registry_path.exists():
        raw_namespaces = {}
    if not isinstance(raw_namespaces, dict):
        raise ValueError(f"Invalid namespace registry: {registry_path}")
    if include_platforms:
        core_namespaces, remaining_namespaces = _split_mapping_after_key(raw_namespaces, CORE_NAMESPACE)
        sources.append((registry_path, core_namespaces))
    else:
        remaining_namespaces = raw_namespaces
        sources.append((registry_path, raw_namespaces))

    if include_platforms:
        for manifest_path in _iter_platform_manifest_paths():
            manifest = _read_yaml_mapping(manifest_path)
            raw_platform_namespaces = manifest.get("namespaces") or {}
            if not isinstance(raw_platform_namespaces, dict):
                raise ValueError(f"Invalid platform namespace manifest: {manifest_path}")
            sources.append((manifest_path, raw_platform_namespaces))
        if remaining_namespaces:
            sources.append((registry_path, remaining_namespaces))
    return sources


def _iter_module_sources(
    registry_path: Path,
    *,
    include_platforms: bool,
) -> list[tuple[Path, dict[str, Any]]]:
    sources: list[tuple[Path, dict[str, Any]]] = []
    raw = _read_yaml_mapping(registry_path)
    raw_modules = raw.get("modules")
    if raw_modules is None and not registry_path.exists():
        raw_modules = {}
    if not isinstance(raw_modules, dict):
        raise ValueError(f"Invalid module registry: {registry_path}")
    if include_platforms:
        core_modules, remaining_modules = _split_mapping_after_key(raw_modules, CORE_NAMESPACE)
        sources.append((registry_path, core_modules))
    else:
        remaining_modules = raw_modules
        sources.append((registry_path, raw_modules))

    if include_platforms:
        for manifest_path in _iter_platform_manifest_paths():
            manifest = _read_yaml_mapping(manifest_path)
            platform_modules: dict[str, Any] = {}
            raw_modules_block = manifest.get("modules")
            if raw_modules_block is not None:
                if not isinstance(raw_modules_block, dict):
                    raise ValueError(f"Invalid platform module manifest: {manifest_path}")
                platform_modules.update(raw_modules_block)
            raw_module = manifest.get("module")
            if raw_module is not None:
                if not isinstance(raw_module, dict):
                    raise ValueError(f"Invalid platform module manifest: {manifest_path}")
                name = str(raw_module.get("name") or manifest_path.parent.name).strip()
                if not name:
                    raise ValueError(f"Platform module manifest missing name: {manifest_path}")
                spec = dict(raw_module)
                spec.pop("name", None)
                platform_modules[name] = spec
            sources.append((manifest_path, platform_modules))
        if remaining_modules:
            sources.append((registry_path, remaining_modules))
    return sources


def _split_mapping_after_key(mapping: dict[str, Any], key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    before: dict[str, Any] = {}
    after: dict[str, Any] = {}
    found = False
    for item_key, value in mapping.items():
        if found:
            after[item_key] = value
            continue
        before[item_key] = value
        if item_key == key:
            found = True
    if not found:
        return mapping, {}
    return before, after


def _validate_module_registry(namespace_registry: NamespaceRegistry, module_registry: ModuleRegistry) -> None:
    for module_name in module_registry.order:
        module = module_registry.modules[module_name]
        for namespace in module.namespaces:
            if namespace not in namespace_registry.namespaces:
                raise ValueError(
                    f"Module {module_name!r} references unknown namespace {namespace!r}"
                )
        for mount in module.mounts:
            source = namespace_registry.get(mount.source_namespace)
            target = namespace_registry.get(mount.target_namespace)
            if source is None:
                raise ValueError(
                    f"Module {module_name!r} mount references unknown source namespace "
                    f"{mount.source_namespace!r}"
                )
            if target is None:
                raise ValueError(
                    f"Module {module_name!r} mount references unknown target namespace "
                    f"{mount.target_namespace!r}"
                )
            if source.visible:
                raise ValueError(
                    f"Module {module_name!r} mount source {source.name!r} must be hidden"
                )
            if not target.visible:
                raise ValueError(
                    f"Module {module_name!r} mount target {target.name!r} must be visible"
                )
            missing_tools = [tool for tool in mount.tools if tool not in source.tools]
            if missing_tools:
                raise ValueError(
                    f"Module {module_name!r} mount from {source.name!r} references "
                    f"tools not owned by that namespace: {', '.join(missing_tools)}"
                )
