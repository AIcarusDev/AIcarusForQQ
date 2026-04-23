"""工具参数处理流水线的数据模型。"""

from dataclasses import dataclass, field
from typing import Any, Literal

ToolArgumentStage = Literal["parse", "schema", "semantic"]


@dataclass(frozen=True)
class ToolArgumentFailure:
    """描述某一阶段的失败原因。"""

    stage: ToolArgumentStage
    message: str
    details: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ToolArgumentProcessingResult:
    """工具参数处理的统一结果。"""

    fn_name: str
    args: dict[str, Any]
    ok: bool
    failure: ToolArgumentFailure | None = None
    repaired_json: bool = False
    repaired_source: str | None = None
    parse_changes: tuple[str, ...] = field(default_factory=tuple)
    schema_changes: tuple[str, ...] = field(default_factory=tuple)
    semantic_changes: tuple[str, ...] = field(default_factory=tuple)