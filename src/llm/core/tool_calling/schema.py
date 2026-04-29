"""schema 修复与严格校验。"""

import json
import re
from typing import Any, Callable, Iterable

from jsonschema import exceptions as jsonschema_exceptions
from jsonschema.validators import validator_for

_INTEGER_LITERAL_RE = re.compile(r"^[+-]?\d+$")

SchemaRepairer = Callable[[dict[str, Any]], tuple[dict[str, Any], list[str]]]


def _get_parameters_schema(tool_declaration: dict[str, Any] | None) -> dict[str, Any] | None:
    if not tool_declaration:
        return None
    parameters = tool_declaration.get("parameters")
    return parameters if isinstance(parameters, dict) else None


def _json_path(parts: Iterable[Any]) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return path


def _error_sort_key(error: jsonschema_exceptions.ValidationError) -> tuple[str, str]:
    return (
        _json_path(error.path),
        error.message,
    )


def _format_validation_error(error: jsonschema_exceptions.ValidationError) -> str:
    path = _json_path(error.path)
    return f"{path}: {error.message}"


def validate_arguments_by_declaration(
    args: dict[str, Any],
    tool_declaration: dict[str, Any] | None,
) -> tuple[bool, list[str], str | None]:
    """对 arguments 执行严格 schema 校验。"""
    parameters = _get_parameters_schema(tool_declaration)
    if parameters is None:
        return True, [], None

    try:
        validator_cls = validator_for(parameters)
        validator_cls.check_schema(parameters)
        validator = validator_cls(parameters)
    except jsonschema_exceptions.SchemaError as exc:
        return False, [f"$: tool declaration schema invalid: {exc.message}"], "tool declaration schema invalid"

    errors = sorted(validator.iter_errors(args), key=_error_sort_key)
    if not errors:
        return True, [], None

    return False, [_format_validation_error(error) for error in errors], "arguments do not satisfy schema"


def _coerce_integer_value(value: Any) -> tuple[Any, bool]:
    """按保守规则将整数参数恢复为 int。"""
    if isinstance(value, bool):
        return value, False
    if isinstance(value, int):
        return value, False
    if isinstance(value, str):
        stripped = value.strip()
        if _INTEGER_LITERAL_RE.fullmatch(stripped):
            return int(stripped), True
    return value, False


def _repair_value_by_schema(
    value: Any,
    schema: dict[str, Any],
    path: str,
) -> tuple[Any, list[str]]:
    """按 JSON schema 的有限子集修复参数值。"""
    changes: list[str] = []
    schema_type = schema.get("type")

    if (schema_type == "object" or "properties" in schema) and isinstance(value, dict):
        props = schema.get("properties") or {}
        repaired = value
        for key, child_schema in props.items():
            if key not in value or not isinstance(child_schema, dict):
                continue
            child_path = f"{path}.{key}" if path else str(key)
            repaired_child, child_changes = _repair_value_by_schema(
                value[key],
                child_schema,
                child_path,
            )
            if child_changes:
                if repaired is value:
                    repaired = dict(value)
                repaired[key] = repaired_child
                changes.extend(child_changes)
        return repaired, changes

    if (schema_type == "array" or "items" in schema) and isinstance(value, list):
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return value, changes
        repaired = value
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]" if path else f"[{index}]"
            repaired_item, item_changes = _repair_value_by_schema(item, item_schema, item_path)
            if item_changes:
                if repaired is value:
                    repaired = list(value)
                repaired[index] = repaired_item
                changes.extend(item_changes)
        return repaired, changes

    if schema_type == "array" and isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return value, changes
        if isinstance(parsed, list):
            repaired, inner_changes = _repair_value_by_schema(parsed, schema, path)
            changes.append(f"{path}: string -> array (double-serialized JSON)")
            changes.extend(inner_changes)
            return repaired, changes
        return value, changes

    if schema_type == "string" and schema.get("x-coerce-integer") and not isinstance(value, bool) and isinstance(value, int):
        coerced = str(value)
        changes.append(f"{path}: {value!r} -> {coerced!r} (string id)")
        return coerced, changes

    if schema_type != "integer":
        return value, changes

    repaired_value, coerced = _coerce_integer_value(value)
    if coerced:
        changes.append(f"{path}: {value!r} -> {repaired_value!r} (int)")

    if isinstance(repaired_value, bool) or not isinstance(repaired_value, int):
        return repaired_value, changes

    clamped = repaired_value
    lo = schema.get("minimum")
    hi = schema.get("maximum")
    if lo is not None:
        clamped = max(clamped, lo)
    if hi is not None:
        clamped = min(clamped, hi)
    if clamped != repaired_value:
        lo_str = str(lo) if lo is not None else "-∞"
        hi_str = str(hi) if hi is not None else "+∞"
        changes.append(f"{path}: {repaired_value!r} -> {clamped!r} (range [{lo_str}, {hi_str}])")
        repaired_value = clamped

    return repaired_value, changes


def repair_arguments_by_declaration(
    args: dict[str, Any],
    tool_declaration: dict[str, Any] | None,
    schema_repairer: SchemaRepairer | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """在严格 schema 校验前执行可证明安全的结构修复。"""
    repaired = args
    changes: list[str] = []

    parameters = _get_parameters_schema(tool_declaration)
    if parameters is not None:
        repaired, generic_changes = _repair_value_by_schema(repaired, parameters, "")
        if isinstance(repaired, dict):
            changes.extend(generic_changes)
        else:
            repaired = args

    if callable(schema_repairer):
        repaired, tool_changes = schema_repairer(repaired)
        changes.extend(tool_changes)

    return repaired, changes