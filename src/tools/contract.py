"""Python-first tool contract helpers.

This module lets a tool define its parameter contract once as a Pydantic model.
The loader can then derive both the backend JSON Schema declaration and the
model-facing TypeScript-like signature from that same source.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from .prompt_signatures import build_prompt_signature


class ToolArgsModel(BaseModel):
    """Base class for tool argument models.

    Tool calls should not silently accept stray model-generated fields. Individual
    tools can still override this config when they intentionally accept extras.
    """

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class ToolContract:
    name: str
    description: str
    args_model: type[BaseModel]
    result_model: type[BaseModel] | None = None

    def declaration(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": _normalize_schema(self.args_model.model_json_schema()),
        }

    def prompt_signature(self) -> str:
        return build_prompt_signature(self.declaration())

    def validate_args(self, args: dict[str, Any]) -> BaseModel:
        return self.args_model.model_validate(args)


F = TypeVar("F", bound=Callable[..., Any])


def tool(
    *,
    name: str,
    description: str,
    args_model: type[BaseModel],
    result_model: type[BaseModel] | None = None,
) -> Callable[[F], Callable[..., Any]]:
    """Attach a Python-first tool contract to a legacy-compatible handler.

    The returned wrapper still accepts ``**kwargs`` so existing ToolExecutor code
    can call it unchanged. The wrapped implementation receives one validated
    Pydantic argument object.
    """

    contract = ToolContract(
        name=name,
        description=description,
        args_model=args_model,
        result_model=result_model,
    )

    def decorate(func: F) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(**kwargs: Any) -> Any:
            try:
                args = contract.validate_args(kwargs)
            except ValidationError as exc:
                return {
                    "error": "工具参数不符合定义",
                    "details": exc.errors(include_url=False),
                }
            return func(args)

        wrapper.__tool_contract__ = contract  # type: ignore[attr-defined]
        return cast(Callable[..., Any], wrapper)

    return decorate


def get_contract_from_module(mod: Any) -> ToolContract | None:
    contract = getattr(mod, "TOOL_CONTRACT", None)
    if isinstance(contract, ToolContract):
        return contract
    execute = getattr(mod, "execute", None)
    contract = getattr(execute, "__tool_contract__", None)
    return contract if isinstance(contract, ToolContract) else None


def _normalize_schema(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if key == "title":
                continue
            normalized[key] = _normalize_schema(child)
        if (
            (normalized.get("type") == "object" or "properties" in normalized)
            and "additionalProperties" not in normalized
        ):
            normalized["additionalProperties"] = False
        return normalized
    if isinstance(value, list):
        return [_normalize_schema(item) for item in value]
    return value
