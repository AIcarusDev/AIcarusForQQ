"""工具规格与工具集合。"""

from dataclasses import dataclass, field
from typing import Any, Callable

from .ordering import tool_order_key

ToolHandler = Callable[..., dict[str, Any]]
SchemaRepairer = Callable[[dict[str, Any]], tuple[dict[str, Any], list[str]]]
SemanticSanitizer = Callable[[dict[str, Any]], tuple[dict[str, Any], list[str], str | None]]


@dataclass(frozen=True)
class ToolGroupSpec:
    """Prompt-facing latent tool group."""

    name: str
    description: str
    keywords: tuple[str, ...] = ()


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
    group: str = ""


@dataclass
class ToolCollection:
    """运行时工具集合，区分当前可用与潜伏工具。"""

    active_specs: dict[str, ToolSpec] = field(default_factory=dict)
    latent_specs: dict[str, ToolSpec] = field(default_factory=dict)
    group_specs: dict[str, ToolGroupSpec] = field(default_factory=dict)

    def clone(self) -> "ToolCollection":
        return ToolCollection(
            active_specs=dict(self.active_specs),
            latent_specs=dict(self.latent_specs),
            group_specs=dict(self.group_specs),
        )

    def active_names(self) -> list[str]:
        return sorted(self.active_specs.keys(), key=tool_order_key)

    def latent_names(self) -> list[str]:
        return sorted(self.latent_specs.keys(), key=tool_order_key)

    def hidden_groups(self) -> list[dict[str, str]]:
        """Return latent tool groups that still have hidden members."""
        groups = {
            spec.group
            for spec in self.latent_specs.values()
            if spec.group
        }
        entries: list[dict[str, str]] = []
        for name in sorted(groups):
            group = self.group_specs.get(name)
            entries.append({
                "name": name,
                "description": group.description if group is not None else name,
            })
        return entries

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

    def get_group(self, name: str) -> ToolGroupSpec | None:
        return self.group_specs.get(name)

    def group_for_name(self, name: str) -> str:
        if name in self.group_specs:
            return name
        spec = self.active_specs.get(name) or self.latent_specs.get(name)
        return spec.group if spec is not None else ""

    def group_tool_names(self, group: str) -> list[str]:
        names = [
            name
            for name, spec in {
                **self.active_specs,
                **self.latent_specs,
            }.items()
            if spec.group == group
        ]
        return sorted(names, key=tool_order_key)

    def latent_activation_names(self, name: str) -> list[str]:
        group = self.group_for_name(name)
        if group:
            names = [
                candidate
                for candidate, spec in self.latent_specs.items()
                if spec.group == group
            ]
            return sorted(names, key=tool_order_key)
        return [name] if name in self.latent_specs else []

    def activate(self, name: str) -> ToolSpec | None:
        activated = self.activate_related(name)
        if activated:
            return self.active_specs.get(name) or activated[0]
        return None

    def activate_related(self, name: str) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for candidate in self.latent_activation_names(name):
            spec = self.latent_specs.pop(candidate, None)
            if spec is None:
                continue
            self.active_specs[candidate] = spec
            specs.append(spec)
        return specs

    def remove_active(self, name: str) -> ToolSpec | None:
        return self.active_specs.pop(name, None)
