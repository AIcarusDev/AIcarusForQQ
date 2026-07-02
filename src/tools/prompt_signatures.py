"""Model-facing tool signature helpers.

The backend keeps JSON Schema for validation. The prompt gets a readable
TypeScript-like signature so the model sees a compact calling contract without
duplicating schema-only structure.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any


def normalize_prompt_signature(value: str) -> str:
    """Normalize a handwritten prompt signature without minifying it."""
    return textwrap.dedent(str(value or "")).strip()


def strip_schema_descriptions(obj: object) -> object:
    """Remove model-facing descriptions from a validation schema/declaration."""
    return _strip_schema_descriptions(obj, in_properties=False)


def _strip_schema_descriptions(obj: object, *, in_properties: bool) -> object:
    if isinstance(obj, dict):
        stripped: dict[str, object] = {}
        for key, value in obj.items():
            if key == "description" and not in_properties:
                continue
            stripped[key] = _strip_schema_descriptions(
                value,
                in_properties=(key == "properties"),
            )
        return stripped
    if isinstance(obj, list):
        return [_strip_schema_descriptions(item, in_properties=False) for item in obj]
    return obj


def build_prompt_signature(declaration: dict[str, Any]) -> str:
    """Build a readable TypeScript-like signature from a JSON Schema declaration."""
    name = str(declaration.get("name") or "tool").strip()
    description = _clean_description(declaration.get("description"))
    parameters = declaration.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}

    lines: list[str] = []
    if description:
        lines.extend(f"// {line}" for line in description.splitlines())
    lines.append(f"{name}(args: {_schema_to_ts(parameters, 0, parameters)})")
    return "\n".join(lines)


def _clean_description(value: object) -> str:
    text = textwrap.dedent(str(value or "")).strip()
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _literal(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _resolve_ref(root: dict[str, Any], ref: str) -> dict[str, Any] | None:
    if not ref.startswith("#/"):
        return None
    current: Any = root
    for part in ref[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, dict) else None


def _schema_to_ts(
    schema: dict[str, Any],
    indent: int,
    root: dict[str, Any],
    *,
    omit_null: bool = False,
) -> str:
    if not isinstance(schema, dict):
        return "unknown"
    if "$ref" in schema:
        resolved = _resolve_ref(root, str(schema["$ref"]))
        if resolved is not None:
            merged = {**resolved, **{k: v for k, v in schema.items() if k != "$ref"}}
            return _schema_to_ts(merged, indent, root, omit_null=omit_null)
    if "const" in schema:
        return _literal(schema["const"])
    if "enum" in schema and isinstance(schema["enum"], list):
        return " | ".join(_literal(item) for item in schema["enum"])

    for key in ("anyOf", "oneOf"):
        variants = schema.get(key)
        if isinstance(variants, list) and variants:
            return " | ".join(
                _schema_to_ts(item, indent, root, omit_null=omit_null)
                for item in variants
                if isinstance(item, dict) and not (omit_null and item.get("type") == "null")
            )

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return " | ".join(
            _schema_to_ts({"type": item}, indent, root, omit_null=omit_null)
            for item in schema_type
            if not (omit_null and item == "null")
        )
    if schema_type == "string":
        return "string"
    if schema_type == "null":
        return "null"
    if schema_type in {"integer", "number"}:
        return "number"
    if schema_type == "boolean":
        return "boolean"
    if schema_type == "array":
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            item_type = _schema_to_ts(item_schema, indent, root)
        else:
            item_type = "unknown"
        if "\n" in item_type or "|" in item_type:
            return f"({item_type})[]"
        return f"{item_type}[]"
    if schema_type == "object" or "properties" in schema:
        return _object_to_ts(schema, indent, root)

    return "unknown"


def _object_to_ts(schema: dict[str, Any], indent: int, root: dict[str, Any]) -> str:
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        return "{}"
    required = set(schema.get("required") or [])
    pad = "  " * indent
    child_pad = "  " * (indent + 1)
    lines = ["{"]
    for key, child_schema in props.items():
        if not isinstance(child_schema, dict):
            child_schema = {}
        optional = "" if key in required else "?"
        field_type = _schema_to_ts(
            child_schema,
            indent + 1,
            root,
            omit_null=key not in required,
        )
        description = _field_comment(child_schema)
        if "\n" in field_type:
            line = f"{child_pad}{key}{optional}: {field_type};"
        else:
            line = f"{child_pad}{key}{optional}: {field_type};"
        if description:
            line += " // " + " ".join(description.splitlines())
        lines.append(line)
    lines.append(f"{pad}}}")
    return "\n".join(lines)


def _field_comment(schema: dict[str, Any]) -> str:
    parts: list[str] = []
    description = _clean_description(schema.get("description"))
    if description:
        parts.append(" ".join(description.splitlines()))
    constraints = _schema_constraints(schema)
    if constraints:
        text = "；".join(constraints)
        if text not in "；".join(parts):
            parts.append(text)
    return _join_comment_parts(parts)


def _join_comment_parts(parts: list[str]) -> str:
    if not parts:
        return ""
    comment = parts[0]
    for part in parts[1:]:
        if not part:
            continue
        if comment and comment[-1] in "。.!！?？":
            comment += part if part.startswith(" ") else " " + part
        else:
            comment += "；" + part
    return comment


def _schema_constraints(schema: dict[str, Any]) -> list[str]:
    schema = _constraint_schema(schema)
    constraints: list[str] = []
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if minimum is not None and maximum is not None:
        constraints.append(f"范围 {minimum}~{maximum}")
    elif minimum is not None and minimum not in (0, 1):
        constraints.append(f"最小 {minimum}")
    elif maximum is not None:
        constraints.append(f"最大 {maximum}")

    min_length = schema.get("minLength")
    max_length = schema.get("maxLength")
    if min_length is not None and max_length is not None:
        if min_length in (0, 1):
            constraints.append(f"最多 {max_length} 个字符")
        else:
            constraints.append(f"长度 {min_length}~{max_length} 个字符")
    elif min_length is not None and min_length not in (0, 1):
        constraints.append(f"至少 {min_length} 个字符")
    elif max_length is not None:
        constraints.append(f"最多 {max_length} 个字符")

    min_items = schema.get("minItems")
    max_items = schema.get("maxItems")
    if min_items is not None and max_items is not None:
        if min_items in (0, 1):
            constraints.append(f"最多 {max_items} 项")
        else:
            constraints.append(f"数组长度 {min_items}~{max_items}")
    elif min_items is not None and min_items not in (0, 1):
        constraints.append(f"至少 {min_items} 项")
    elif max_items is not None:
        constraints.append(f"最多 {max_items} 项")
    if schema.get("uniqueItems") is True:
        constraints.append("数组项不可重复")
    return constraints


def _constraint_schema(schema: dict[str, Any]) -> dict[str, Any]:
    direct_keys = {
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
    if any(key in schema for key in direct_keys):
        return schema

    for union_key in ("anyOf", "oneOf"):
        variants = schema.get(union_key)
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, dict) or variant.get("type") == "null":
                continue
            if any(key in variant for key in direct_keys):
                return variant
    return schema
