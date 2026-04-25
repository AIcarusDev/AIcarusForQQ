"""内部结构化单工具调用协议。"""

from dataclasses import dataclass
from typing import Any

from .tool_calling.schema import SchemaRepairer
from .tool_calling.semantic import SemanticSanitizer


@dataclass(frozen=True)
class InternalToolSpec:
    """描述一个仅供内部子模型使用的结构化输出工具。"""

    declaration: dict[str, Any]
    schema_repairer: SchemaRepairer | None = None
    semantic_sanitizer: SemanticSanitizer | None = None

    @property
    def name(self) -> str:
        return str(self.declaration.get("name", ""))