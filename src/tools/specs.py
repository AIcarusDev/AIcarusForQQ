"""工具规格与工具集合。"""

from dataclasses import dataclass, field
from typing import Any, Callable

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
    always_available: bool = True
    schema_repairer: SchemaRepairer | None = None
    semantic_sanitizer: SemanticSanitizer | None = None


@dataclass
class ToolCollection:
    """运行时工具集合，区分当前可用与潜伏工具。"""

    active_specs: dict[str, ToolSpec] = field(default_factory=dict)
    latent_specs: dict[str, ToolSpec] = field(default_factory=dict)

    def clone(self) -> "ToolCollection":
        return ToolCollection(
            active_specs=dict(self.active_specs),
            latent_specs=dict(self.latent_specs),
        )

    def active_names(self) -> list[str]:
        return list(self.active_specs.keys())

    def latent_names(self) -> list[str]:
        return list(self.latent_specs.keys())

    def active_declarations(self) -> list[dict[str, Any]]:
        return [spec.declaration for spec in self.active_specs.values()]

    def has_active_tools(self) -> bool:
        return bool(self.active_specs)

    def get_active(self, name: str) -> ToolSpec | None:
        return self.active_specs.get(name)

    def get_latent(self, name: str) -> ToolSpec | None:
        return self.latent_specs.get(name)

    def activate(self, name: str) -> ToolSpec | None:
        spec = self.latent_specs.pop(name, None)
        if spec is None:
            return None
        self.active_specs[name] = spec
        return spec

    def remove_active(self, name: str) -> ToolSpec | None:
        return self.active_specs.pop(name, None)