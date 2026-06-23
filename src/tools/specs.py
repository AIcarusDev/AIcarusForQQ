"""工具规格、namespace 规格与工具集合。"""

from dataclasses import dataclass, field
from typing import Any, Callable

from .namespaces import NamespaceRegistry, NamespaceRuntimeState, NamespaceSpec

ToolHandler = Callable[..., dict[str, Any]]
SchemaRepairer = Callable[[dict[str, Any]], tuple[dict[str, Any], list[str]]]
SemanticSanitizer = Callable[[dict[str, Any]], tuple[dict[str, Any], list[str], str | None]]


@dataclass(frozen=True)
class ToolSpec:
    """单个工具的统一规格。"""

    name: str
    declaration: dict[str, Any]
    handler: ToolHandler
    module_name: str
    externally_perceptible: bool = False
    always_available: bool = True
    schema_repairer: SchemaRepairer | None = None
    semantic_sanitizer: SemanticSanitizer | None = None
    namespace: str = ""
    attached_to: str = ""


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
                if tool_name in self.active_specs and tool_name not in names:
                    names.append(tool_name)
            for tool_name, tool_spec in self.active_specs.items():
                if tool_spec.attached_to == namespace and tool_name not in names:
                    names.append(tool_name)
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
            for tool_name in spec.tools:
                if tool_name in self.latent_specs and tool_name not in names:
                    names.append(tool_name)
        names.extend(name for name in self.latent_specs if name not in names)
        return names

    def active_declarations(self) -> list[dict[str, Any]]:
        return [
            self.active_specs[name].declaration
            for name in self.active_names()
        ]

    def has_active_tools(self) -> bool:
        return bool(self.active_specs)

    def get_active(self, name: str) -> ToolSpec | None:
        return self.active_specs.get(name)

    def get_latent(self, name: str) -> ToolSpec | None:
        return self.latent_specs.get(name)

    def get_any(self, name: str) -> ToolSpec | None:
        return self.active_specs.get(name) or self.latent_specs.get(name) or self.all_specs.get(name)

    def namespace_for_tool(self, name: str) -> str:
        spec = self.get_any(name)
        if spec is not None and spec.namespace:
            return spec.namespace
        if self.namespace_registry is None:
            return ""
        return self.namespace_registry.namespace_for_tool(name)

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
            if not any(tool in self.all_specs for tool in spec.tools):
                continue
            entries.append({"name": name, "description": spec.description})
        return entries

    def namespace_prompt_blocks(self) -> list[dict[str, Any]]:
        if self.namespace_registry is None:
            return [{
                "name": "core",
                "active": True,
                "declarations": self.active_declarations(),
            }]
        blocks: list[dict[str, Any]] = []
        active = set(self.active_namespace_names())
        for namespace in self.active_namespace_names():
            spec = self.namespace_specs.get(namespace)
            if spec is None:
                continue
            declarations: list[dict[str, Any]] = []
            for tool_name in spec.tools:
                tool_spec = self.active_specs.get(tool_name)
                if tool_spec is not None and not tool_spec.attached_to:
                    declarations.append(tool_spec.declaration)
            for tool_name in self.active_names():
                tool_spec = self.active_specs.get(tool_name)
                if tool_spec is not None and tool_spec.attached_to == namespace:
                    declarations.append(tool_spec.declaration)
            blocks.append({
                "name": namespace,
                "active": True,
                "declarations": declarations,
            })
        for namespace in self.namespace_registry.order:
            if namespace in active:
                continue
            spec = self.namespace_specs.get(namespace)
            if spec is None or not any(tool in self.all_specs for tool in spec.tools):
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
        names = [name for name in spec.tools if name in self.all_specs]
        if only_inactive:
            names = [name for name in names if name not in self.active_specs]
        return names

    def preview_namespace(self, namespace: str) -> dict[str, Any] | None:
        spec = self.namespace_specs.get(namespace)
        if spec is None:
            return None
        tools: list[dict[str, str]] = []
        for tool_name in self.namespace_tool_names(namespace):
            tool_spec = self.all_specs.get(tool_name)
            if tool_spec is None:
                continue
            declaration = tool_spec.declaration or {}
            tools.append({
                "name": tool_name,
                "description": str(declaration.get("description") or ""),
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
                tool_spec = self.all_specs.get(tool_name)
                if tool_spec is None:
                    continue
                description = str((tool_spec.declaration or {}).get("description") or "")
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
