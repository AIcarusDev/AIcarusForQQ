"""工具规格、namespace 规格与工具集合。"""

from dataclasses import dataclass, field
from typing import Any, Callable

from .namespaces import NamespaceRegistry, NamespaceRuntimeState, NamespaceSpec

ToolHandler = Callable[..., dict[str, Any]]
SchemaRepairer = Callable[[dict[str, Any]], tuple[dict[str, Any], list[str]]]
SemanticSanitizer = Callable[[dict[str, Any]], tuple[dict[str, Any], list[str], str | None]]


@dataclass(frozen=True)
class ToolEffect:
    """A coarse external surface effect used by pre-execution guard routing."""

    surface: str = ""
    kind: str = ""


@dataclass(frozen=True)
class ToolExecutionPolicy:
    """Executor scheduling metadata for one tool.

    Tools are serial by default. A tool must explicitly opt in before the
    executor may batch it with other eligible tools.
    """

    parallel_safe: bool = False
    parallel_key: str = ""


@dataclass(frozen=True)
class ToolSpec:
    """单个工具的统一规格。"""

    name: str
    declaration: dict[str, Any]
    handler: ToolHandler
    module_name: str
    description: str = ""
    prompt_signature: str = ""
    result_cdata: bool = False
    externally_perceptible: bool = False
    always_available: bool = True
    schema_repairer: SchemaRepairer | None = None
    semantic_sanitizer: SemanticSanitizer | None = None
    namespace: str = ""
    visible_namespace: str = ""
    attached_to: str = ""
    mounted_to: str = ""
    mounted_by_module: str = ""
    visibility: str = "visible"
    tool_kind: str = ""
    effect: ToolEffect | None = None
    execution: ToolExecutionPolicy = field(default_factory=ToolExecutionPolicy)

    @property
    def call_namespace(self) -> str:
        """Namespace the model uses to address this tool in the current view."""
        return self.mounted_to or self.attached_to or self.visible_namespace or self.namespace

    @property
    def call_name(self) -> str:
        namespace = self.call_namespace
        return f"{namespace}.{self.name}" if namespace else self.name


@dataclass
class ToolCollection:
    """运行时工具集合，按 namespace 区分可见与不可见工具。"""

    active_specs: dict[str, ToolSpec] = field(default_factory=dict)
    latent_specs: dict[str, ToolSpec] = field(default_factory=dict)
    all_specs: dict[str, ToolSpec] = field(default_factory=dict)
    namespace_specs: dict[str, NamespaceSpec] = field(default_factory=dict)
    namespace_registry: NamespaceRegistry | None = None
    namespace_state: NamespaceRuntimeState | None = None
    active_namespace_order: list[str] = field(default_factory=list)
    round_index: int = 0

    @staticmethod
    def route_key(namespace: str, name: str) -> str:
        namespace = str(namespace or "").strip()
        name = str(name or "").strip()
        return f"{namespace}.{name}" if namespace else name

    @staticmethod
    def spec_key(spec: ToolSpec) -> str:
        return ToolCollection.route_key(spec.call_namespace, spec.name)

    def clone(self) -> "ToolCollection":
        return ToolCollection(
            active_specs=dict(self.active_specs),
            latent_specs=dict(self.latent_specs),
            all_specs=dict(self.all_specs),
            namespace_specs=dict(self.namespace_specs),
            namespace_registry=self.namespace_registry,
            namespace_state=self.namespace_state,
            active_namespace_order=list(self.active_namespace_order),
            round_index=self.round_index,
        )

    def active_names(self) -> list[str]:
        if self.namespace_registry is None:
            return list(self.active_specs)
        names: list[str] = []
        for namespace in self.active_namespace_order:
            spec = self.namespace_specs.get(namespace)
            if spec is None:
                continue
            for tool_name in spec.tools:
                key = self.route_key(namespace, tool_name)
                if key in self.active_specs and key not in names:
                    names.append(key)
            for key, tool_spec in self.active_specs.items():
                if tool_spec.attached_to == namespace and key not in names:
                    names.append(key)
            for key, tool_spec in self.active_specs.items():
                if tool_spec.mounted_to == namespace and key not in names:
                    names.append(key)
        remaining = [name for name in self.active_specs if name not in names]
        names.extend(remaining)
        return names

    def latent_names(self) -> list[str]:
        if self.namespace_registry is None:
            return list(self.latent_specs)
        names: list[str] = []
        active = set(self.active_namespace_names())
        for namespace in self.namespace_registry.order:
            if namespace in active:
                continue
            spec = self.namespace_specs.get(namespace)
            if spec is None:
                continue
            if not spec.visible or not spec.discoverable:
                continue
            for tool_name in spec.tools:
                key = self.route_key(namespace, tool_name)
                if key in self.latent_specs and key not in names:
                    names.append(key)
        names.extend(name for name in self.latent_specs if name not in names)
        return names

    def active_declarations(self) -> list[dict[str, Any]]:
        return [
            self.active_specs[name].declaration
            for name in self.active_names()
        ]

    def active_prompt_signatures(self) -> list[str]:
        return [
            self.active_specs[name].prompt_signature
            for name in self.active_names()
            if self.active_specs[name].prompt_signature
        ]

    def has_active_tools(self) -> bool:
        return bool(self.active_specs)

    def _find_by_short_name(self, specs: dict[str, ToolSpec], name: str) -> ToolSpec | None:
        matches = [spec for spec in specs.values() if spec.name == name]
        return matches[0] if len(matches) == 1 else None

    def matching_active(self, name: str) -> list[ToolSpec]:
        return [spec for spec in self.active_specs.values() if spec.name == name]

    def matching_latent(self, name: str) -> list[ToolSpec]:
        return [spec for spec in self.latent_specs.values() if spec.name == name]

    def matching_any(self, name: str) -> list[ToolSpec]:
        seen: set[int] = set()
        matches: list[ToolSpec] = []
        for spec in [*self.active_specs.values(), *self.latent_specs.values(), *self.all_specs.values()]:
            if spec.name != name or id(spec) in seen:
                continue
            seen.add(id(spec))
            matches.append(spec)
        return matches

    def get_active(self, name: str, namespace: str = "") -> ToolSpec | None:
        if namespace:
            return self.active_specs.get(self.route_key(namespace, name))
        if "." in str(name or ""):
            return self.active_specs.get(str(name or "").strip())
        return self._find_by_short_name(self.active_specs, name)

    def get_latent(self, name: str, namespace: str = "") -> ToolSpec | None:
        if namespace:
            return self.latent_specs.get(self.route_key(namespace, name))
        if "." in str(name or ""):
            return self.latent_specs.get(str(name or "").strip())
        return self._find_by_short_name(self.latent_specs, name)

    def get_any(self, name: str, namespace: str = "") -> ToolSpec | None:
        if namespace:
            key = self.route_key(namespace, name)
            return self.active_specs.get(key) or self.latent_specs.get(key) or self.all_specs.get(key)
        if "." in str(name or ""):
            key = str(name or "").strip()
            return self.active_specs.get(key) or self.latent_specs.get(key) or self.all_specs.get(key)
        return (
            self._find_by_short_name(self.active_specs, name)
            or self._find_by_short_name(self.latent_specs, name)
            or self._find_by_short_name(self.all_specs, name)
        )

    def namespace_for_tool(self, name: str, namespace: str = "") -> str:
        spec = self.get_any(name, namespace)
        if spec is not None and spec.namespace:
            return spec.namespace
        if self.namespace_registry is None:
            return ""
        return self.namespace_registry.namespace_for_tool(name, namespace=namespace)

    def is_namespace_active(self, namespace: str) -> bool:
        if self.namespace_registry is None or self.namespace_state is None:
            return namespace in self.active_namespace_order
        return self.namespace_state.is_open(namespace, self.namespace_registry)

    def active_namespace_names(self) -> list[str]:
        if self.namespace_registry is None or self.namespace_state is None:
            return list(self.active_namespace_order)
        return [
            name
            for name in self.namespace_state.active_namespaces(self.namespace_registry)
            if name in self.namespace_specs
            and self.namespace_specs[name].visible
        ]

    def inactive_namespace_summaries(self) -> list[dict[str, str]]:
        if self.namespace_registry is None:
            return []
        active = set(self.active_namespace_names())
        entries: list[dict[str, str]] = []
        for name in self.namespace_registry.order:
            if name in active:
                continue
            spec = self.namespace_specs.get(name)
            if spec is None:
                continue
            if not spec.visible or not spec.discoverable:
                continue
            if not any(self.route_key(name, tool) in self.all_specs for tool in spec.tools):
                continue
            entries.append({"name": name, "description": spec.description})
        return entries

    def namespace_prompt_blocks(self) -> list[dict[str, Any]]:
        if self.namespace_registry is None:
            return [{
                "name": "core",
                "active": True,
                "declarations": self.active_declarations(),
                "signatures": self.active_prompt_signatures(),
            }]
        blocks: list[dict[str, Any]] = []
        active = set(self.active_namespace_names())
        for namespace in self.active_namespace_names():
            spec = self.namespace_specs.get(namespace)
            if spec is None:
                continue
            if not spec.visible:
                continue
            declarations: list[dict[str, Any]] = []
            signatures: list[str] = []
            for tool_name in spec.tools:
                tool_spec = self.active_specs.get(self.route_key(namespace, tool_name))
                if tool_spec is not None and not tool_spec.attached_to and not tool_spec.mounted_to:
                    declarations.append(tool_spec.declaration)
                    signatures.append(tool_spec.prompt_signature)
            for key in self.active_names():
                tool_spec = self.active_specs.get(key)
                if tool_spec is not None and tool_spec.attached_to == namespace:
                    declarations.append(tool_spec.declaration)
                    signatures.append(tool_spec.prompt_signature)
            for key in self.active_names():
                tool_spec = self.active_specs.get(key)
                if tool_spec is not None and tool_spec.mounted_to == namespace:
                    declarations.append(tool_spec.declaration)
                    signatures.append(tool_spec.prompt_signature)
            blocks.append({
                "name": namespace,
                "active": True,
                "declarations": declarations,
                "signatures": [signature for signature in signatures if signature],
            })
        for namespace in self.namespace_registry.order:
            if namespace in active:
                continue
            spec = self.namespace_specs.get(namespace)
            if spec is None or not any(self.route_key(namespace, tool) in self.all_specs for tool in spec.tools):
                continue
            if not spec.visible or not spec.discoverable:
                continue
            blocks.append({
                "name": namespace,
                "description": spec.description,
                "active": False,
            })
        return blocks

    def namespace_tool_names(self, namespace: str, *, only_inactive: bool = False) -> list[str]:
        spec = self.namespace_specs.get(namespace)
        if spec is None:
            return []
        if not spec.visible or not spec.discoverable:
            return []
        names = [name for name in spec.tools if self.route_key(namespace, name) in self.all_specs]
        if only_inactive:
            names = [name for name in names if self.route_key(namespace, name) not in self.active_specs]
        return names

    def preview_namespace(self, namespace: str) -> dict[str, Any] | None:
        spec = self.namespace_specs.get(namespace)
        if spec is None:
            return None
        tools: list[dict[str, str]] = []
        for tool_name in self.namespace_tool_names(namespace):
            tool_spec = self.all_specs.get(self.route_key(namespace, tool_name))
            if tool_spec is None:
                continue
            declaration = tool_spec.declaration or {}
            tools.append({
                "name": tool_name,
                "description": tool_spec.description or str(declaration.get("description") or ""),
            })
        return {"name": namespace, "tools": tools}

    def search_inactive_namespaces(self, query: str, *, limit: int = 5) -> list[dict[str, str]]:
        keyword = str(query or "").strip()
        if not keyword:
            return []
        active = set(self.active_namespace_names())
        matches: list[dict[str, str]] = []
        namespace_order = self.namespace_registry.order if self.namespace_registry else ()
        for namespace in namespace_order:
            if namespace in active:
                continue
            for tool_name in self.namespace_tool_names(namespace):
                tool_spec = self.all_specs.get(self.route_key(namespace, tool_name))
                if tool_spec is None:
                    continue
                description = tool_spec.description or str((tool_spec.declaration or {}).get("description") or "")
                if keyword not in description:
                    continue
                matches.append({
                    "namespace": namespace,
                    "name": tool_name,
                    "description": description,
                })
                if len(matches) >= limit:
                    return matches
        return matches

    def apply_lifecycle_after_tool(self, tool_name: str, args: dict[str, Any], result: object) -> None:
        if self.namespace_registry is None or self.namespace_state is None:
            return
        if not isinstance(result, dict):
            return
        for namespace, spec in self.namespace_specs.items():
            for hook in spec.lifecycle.close_on:
                if hook.tool != tool_name:
                    continue
                if hook.action and str(args.get("action") or "") != hook.action:
                    continue
                if hook.ok is not None and bool(result.get("ok")) is not hook.ok:
                    continue
                self.namespace_state.close(namespace, self.namespace_registry)
