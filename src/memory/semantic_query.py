"""Bounded, read-only semantic memory queries for WebUI vNext.

MemoryQL is deliberately not SQL.  The parser accepts a small versioned
grammar, validates every semantic identifier against this module's schema,
and only then builds parameterized statements from static templates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Callable, Iterable
import uuid

from database import DB_PATH


SCHEMA_VERSION = "memory-semantic-v1"
LANGUAGE_VERSION = "1.0"
HARD_NODE_LIMIT = 80
HARD_EDGE_LIMIT = 120
HARD_ROW_LIMIT = 100
HARD_DEPTH_LIMIT = 2
HARD_QUERY_LENGTH = 8_000
HARD_TIMEOUT_MS = 500


class MemoryQueryError(ValueError):
    """Base class for safe query failures."""

    code = "memory_query_invalid"


class MemoryQLSyntaxError(MemoryQueryError):
    code = "memoryql_syntax_error"


class MemoryQLValidationError(MemoryQueryError):
    code = "memoryql_validation_error"


class MemoryQueryTimeout(MemoryQueryError):
    code = "memory_query_budget_exceeded"


class MemoryQueryUnavailable(MemoryQueryError):
    code = "memory_query_unavailable"


@dataclass(frozen=True)
class NodePattern:
    variable: str
    type_name: str


@dataclass(frozen=True)
class RelationPattern:
    source: str
    type_name: str
    target: str


@dataclass(frozen=True)
class Condition:
    connector: str
    variable: str
    property_name: str
    operator: str
    value: Any


@dataclass(frozen=True)
class ExpandClause:
    variable: str
    depth: int


@dataclass(frozen=True)
class LimitClause:
    nodes: int | None = None
    edges: int | None = None
    rows: int | None = None


@dataclass(frozen=True)
class QueryAst:
    nodes: tuple[NodePattern, ...]
    relations: tuple[RelationPattern, ...]
    conditions: tuple[Condition, ...]
    expand: ExpandClause | None
    return_kind: str
    limit: LimitClause


PROPERTY_SCHEMA: dict[str, dict[str, dict[str, Any]]] = {
    "Event": {
        "id": {"type": "id", "operators": ["=", "!=", ">", ">=", "<", "<="]},
        "summary": {"type": "string", "operators": ["=", "!=", "~="]},
        "event_type": {"type": "string", "operators": ["=", "!=", "~="]},
        "occurred_at": {"type": "datetime", "operators": ["=", ">", ">=", "<", "<="]},
        "confidence": {"type": "number", "operators": ["=", "!=", ">", ">=", "<", "<="]},
        "status": {"type": "string", "operators": ["=", "!=", "~="]},
    },
    "CanonicalEntity": {
        "id": {"type": "id", "operators": ["=", "!="]},
        "name": {"type": "string", "operators": ["=", "!=", "~="]},
        "kind": {"type": "string", "operators": ["=", "!=", "~="]},
        "confidence": {"type": "number", "operators": ["=", "!=", ">", ">=", "<", "<="]},
        "status": {"type": "string", "operators": ["=", "!="]},
        "updated_at": {"type": "datetime", "operators": ["=", ">", ">=", "<", "<="]},
    },
    "Storyline": {
        "id": {"type": "id", "operators": ["=", "!="]},
        "scope": {"type": "string", "operators": ["=", "!=", "~="]},
        "summary": {"type": "string", "operators": [], "projected_only": True},
        "origin_type": {"type": "string", "operators": ["=", "!="]},
        "status": {"type": "string", "operators": ["=", "!="]},
        "score": {"type": "number", "operators": ["=", "!=", ">", ">=", "<", "<="]},
        "member_count": {"type": "number", "operators": ["=", "!=", ">", ">=", "<", "<="]},
        "updated_at": {"type": "datetime", "operators": ["=", ">", ">=", "<", "<="]},
    },
    "Source": {
        "id": {"type": "string", "operators": ["=", "!=", "~="]},
        "kind": {"type": "string", "operators": ["=", "!=", "~="]},
        "timestamp": {"type": "string", "operators": ["=", "!=", ">", ">=", "<", "<="]},
    },
}

TYPE_META: dict[str, dict[str, Any]] = {
    "Event": {
        "label": "事件",
        "description": "从对话与认知过程抽取的语义事件。",
        "table": "MemoryEvents",
    },
    "CanonicalEntity": {
        "label": "规范实体",
        "description": "经实体解析后稳定指向的角色、对象或概念。",
        "table": "MemoryCanonicalEntities",
    },
    "Storyline": {
        "label": "故事线",
        "description": "相关事件形成的演进脉络。",
        "table": "MemoryStorylines",
    },
    "Source": {
        "label": "来源",
        "description": "事件能够追溯到的会话或认知来源。",
        "table": "MemoryEventSources",
    },
}

RELATION_META: dict[str, dict[str, Any]] = {
    "INVOLVES": {
        "label": "事件参与",
        "description": "事件涉及一个规范实体。",
        "source": "Event",
        "target": "CanonicalEntity",
    },
    "PART_OF": {
        "label": "故事线归属",
        "description": "事件属于一条故事线。",
        "source": "Event",
        "target": "Storyline",
    },
    "DERIVED_FROM": {
        "label": "来源追溯",
        "description": "事件由一个稳定来源记录派生。",
        "source": "Event",
        "target": "Source",
    },
    "RELATES_TO": {
        "label": "事件关系",
        "description": "结构化处理得到的事件间关系。",
        "source": "Event",
        "target": "Event",
    },
}

PROPERTY_COLUMNS: dict[str, dict[str, str]] = {
    "Event": {
        "id": "event_id",
        "summary": "summary",
        "event_type": "event_type_norm",
        "occurred_at": "occurred_at",
        "confidence": "confidence",
        "status": "status",
    },
    "CanonicalEntity": {
        "id": "entity_id",
        "name": "canonical_name",
        "kind": "entity_type",
        "confidence": "confidence",
        "status": "status",
        "updated_at": "updated_at",
    },
    "Storyline": {
        "id": "storyline_id",
        "scope": "scope",
        "origin_type": "origin_type",
        "status": "status",
        "score": "score",
        "member_count": "member_count",
        "updated_at": "updated_at",
    },
    "Source": {
        "id": "source_id",
        "kind": "source_kind",
        "timestamp": "source_timestamp",
    },
}

TYPE_REQUIRED_COLUMNS: dict[str, dict[str, set[str]]] = {
    "Event": {
        "MemoryEvents": {
            "event_id", "summary", "event_type_norm", "occurred_at",
            "confidence", "status", "is_deleted",
        },
    },
    "CanonicalEntity": {
        "MemoryCanonicalEntities": {
            "entity_id", "canonical_name", "entity_type", "confidence",
            "status", "updated_at",
        },
    },
    "Storyline": {
        "MemoryStorylines": {
            "storyline_id", "scope", "origin_type", "status", "score",
            "member_count", "updated_at",
        },
    },
    "Source": {
        "MemoryEventSources": {
            "source_kind", "source_id", "source_timestamp", "created_at",
        },
    },
}

STORYLINE_SUMMARY_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "MemoryStorylineSummaryTasks": {"task_id", "storyline_id"},
    "MemorySummaryCache": {"task_id", "status", "summary", "updated_at_ms"},
}

RELATION_REQUIRED_COLUMNS: dict[str, dict[str, set[str]]] = {
    "INVOLVES": {
        "MemoryEntityMentions": {"event_id", "entity_id", "role", "confidence"},
    },
    "PART_OF": {
        "MemoryStorylineMembers": {"event_id", "storyline_id", "score", "rank", "status"},
    },
    "DERIVED_FROM": {
        "MemoryEventSources": {
            "event_source_id", "event_id", "source_kind", "source_id",
            "source_seq", "source_timestamp",
        },
    },
    "RELATES_TO": {
        "MemoryEventRelations": {
            "relation_id", "source_event_id", "target_event_id",
            "relation_type", "confidence", "status", "revision", "updated_at_ms",
        },
    },
}

_CLAUSE_RE = re.compile(r"(?im)^\s*(MATCH|WHERE|EXPAND|RETURN|LIMIT)\b")
_NODE_RE = re.compile(r"^\$(?P<var>[A-Za-z][A-Za-z0-9_]*)\s+ISA\s+(?P<type>[A-Za-z][A-Za-z0-9_]*)$", re.I)
_RELATION_RE = re.compile(
    r"^\(\s*\$(?P<src>[A-Za-z][A-Za-z0-9_]*)\s*\)\s*-\s*\[\s*(?P<rel>[A-Za-z][A-Za-z0-9_]*)\s*\]\s*->\s*\(\s*\$(?P<dst>[A-Za-z][A-Za-z0-9_]*)\s*\)$",
    re.I,
)
_CONDITION_RE = re.compile(
    r"^\$(?P<var>[A-Za-z][A-Za-z0-9_]*)\.(?P<prop>[A-Za-z][A-Za-z0-9_]*)\s*(?P<op>~=|!=|>=|<=|=|>|<)\s*(?P<value>.+)$",
    re.S,
)
_EXPAND_RE = re.compile(r"^\$(?P<var>[A-Za-z][A-Za-z0-9_]*)\s+DEPTH\s+(?P<depth>\d+)$", re.I)
_GRAPH_LIMIT_RE = re.compile(r"^NODES\s+(?P<nodes>\d+)\s+EDGES\s+(?P<edges>\d+)$", re.I)
_ROW_LIMIT_RE = re.compile(r"^ROWS\s+(?P<rows>\d+)$", re.I)
_FORBIDDEN_RE = re.compile(
    r"\b(CREATE|UPDATE|DELETE|INSERT|UPSERT|DROP|ALTER|ATTACH|DETACH|PRAGMA|REPLACE|TRUNCATE)\b",
    re.I,
)


def _strip_quoted(text: str) -> str:
    output: list[str] = []
    quoted = False
    escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            output.append(" ")
        else:
            if char == '"':
                quoted = True
                output.append(" ")
            else:
                output.append(char)
    return "".join(output)


def _split_clauses(query: str) -> dict[str, str]:
    matches = list(_CLAUSE_RE.finditer(query))
    if not matches:
        raise MemoryQLSyntaxError("查询必须按行声明 MATCH、RETURN 与 LIMIT")
    clauses: dict[str, str] = {}
    order: list[str] = []
    for index, match in enumerate(matches):
        name = match.group(1).upper()
        if name in clauses:
            raise MemoryQLSyntaxError(f"{name} 子句不能重复")
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(query)
        value = query[start:end].strip()
        if not value:
            raise MemoryQLSyntaxError(f"{name} 子句不能为空")
        clauses[name] = value
        order.append(name)
    expected_order = [name for name in ("MATCH", "WHERE", "EXPAND", "RETURN", "LIMIT") if name in clauses]
    if order != expected_order:
        raise MemoryQLSyntaxError("子句顺序必须是 MATCH、WHERE、EXPAND、RETURN、LIMIT")
    for required in ("MATCH", "RETURN", "LIMIT"):
        if required not in clauses:
            raise MemoryQLSyntaxError(f"缺少必填的 {required} 子句")
    return clauses


def _split_conditions(text: str) -> tuple[list[str], list[str]]:
    conditions: list[str] = []
    connectors: list[str] = []
    start = 0
    quoted = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            index += 1
            continue
        if char == '"':
            quoted = True
            index += 1
            continue
        match = re.match(r"\s+(AND|OR)\s+", text[index:], re.I)
        if match:
            conditions.append(text[start:index].strip())
            connectors.append(match.group(1).upper())
            index += match.end()
            start = index
            continue
        index += 1
    conditions.append(text[start:].strip())
    if any(not condition for condition in conditions):
        raise MemoryQLSyntaxError("WHERE 中存在空条件")
    return conditions, connectors


def _literal(raw: str) -> Any:
    value = raw.strip()
    if value.startswith('"'):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise MemoryQLSyntaxError("字符串必须使用有效的双引号 JSON 形式") from exc
        if not isinstance(parsed, str):
            raise MemoryQLSyntaxError("双引号值必须是字符串")
        return parsed
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", value):
        return float(value)
    raise MemoryQLSyntaxError(f"无法识别的条件值: {value[:40]}")


def parse_memoryql(query: str) -> QueryAst:
    if not isinstance(query, str) or not query.strip():
        raise MemoryQLSyntaxError("查询不能为空")
    if len(query) > HARD_QUERY_LENGTH:
        raise MemoryQLValidationError(f"查询不能超过 {HARD_QUERY_LENGTH} 个字符")
    if _FORBIDDEN_RE.search(_strip_quoted(query)):
        raise MemoryQLValidationError("MemoryQL 是只读语言，不支持写入或维护关键字")

    clauses = _split_clauses(query.strip())
    nodes: list[NodePattern] = []
    relations: list[RelationPattern] = []
    for raw_line in clauses["MATCH"].splitlines():
        line = raw_line.strip()
        if not line:
            continue
        node_match = _NODE_RE.fullmatch(line)
        if node_match:
            nodes.append(NodePattern(node_match.group("var"), node_match.group("type")))
            continue
        relation_match = _RELATION_RE.fullmatch(line)
        if relation_match:
            relations.append(RelationPattern(
                relation_match.group("src"),
                relation_match.group("rel").upper(),
                relation_match.group("dst"),
            ))
            continue
        raise MemoryQLSyntaxError(f"无法解析 MATCH 行: {line[:120]}")
    if not nodes:
        raise MemoryQLValidationError("MATCH 至少需要一个节点声明")
    if len(nodes) > 8 or len(relations) > 1:
        raise MemoryQLValidationError("MemoryQL 1.0 最多声明 8 个节点和 1 条关系")

    node_by_variable: dict[str, NodePattern] = {}
    for node in nodes:
        if node.variable in node_by_variable:
            raise MemoryQLValidationError(f"变量 ${node.variable} 重复声明")
        canonical_type = next((name for name in TYPE_META if name.lower() == node.type_name.lower()), None)
        if canonical_type is None:
            raise MemoryQLValidationError(f"未知的节点类型: {node.type_name}")
        canonical = NodePattern(node.variable, canonical_type)
        node_by_variable[node.variable] = canonical
    nodes = list(node_by_variable.values())

    for relation in relations:
        if relation.source not in node_by_variable or relation.target not in node_by_variable:
            raise MemoryQLValidationError("关系两端变量必须先在 MATCH 中声明")
        meta = RELATION_META.get(relation.type_name)
        if meta is None:
            raise MemoryQLValidationError(f"未知的关系类型: {relation.type_name}")
        source_type = node_by_variable[relation.source].type_name
        target_type = node_by_variable[relation.target].type_name
        if source_type != meta["source"] or target_type != meta["target"]:
            raise MemoryQLValidationError(
                f"{relation.type_name} 的方向必须是 {meta['source']} → {meta['target']}"
            )
    connected = {variable for relation in relations for variable in (relation.source, relation.target)}
    if relations and connected != set(node_by_variable):
        raise MemoryQLValidationError("MemoryQL 1.0 不支持与关系断开的节点声明")
    if not relations and len(nodes) != 1:
        raise MemoryQLValidationError("没有关系时只能查询一个节点变量")

    conditions: list[Condition] = []
    if "WHERE" in clauses:
        raw_conditions, connectors = _split_conditions(clauses["WHERE"])
        if len(raw_conditions) > 12:
            raise MemoryQLValidationError("WHERE 最多包含 12 个条件")
        for index, raw_condition in enumerate(raw_conditions):
            match = _CONDITION_RE.fullmatch(raw_condition)
            if not match:
                raise MemoryQLSyntaxError(f"无法解析 WHERE 条件: {raw_condition[:120]}")
            variable = match.group("var")
            if variable not in node_by_variable:
                raise MemoryQLValidationError(f"条件引用了未声明变量: ${variable}")
            property_name = match.group("prop")
            properties = PROPERTY_SCHEMA[node_by_variable[variable].type_name]
            if property_name not in properties:
                raise MemoryQLValidationError(
                    f"{node_by_variable[variable].type_name} 没有属性 {property_name}"
                )
            operator = match.group("op")
            if operator not in properties[property_name]["operators"]:
                raise MemoryQLValidationError(f"属性 {property_name} 不支持运算符 {operator}")
            conditions.append(Condition(
                "" if index == 0 else connectors[index - 1],
                variable,
                property_name,
                operator,
                _literal(match.group("value")),
            ))

    expand = None
    if "EXPAND" in clauses:
        match = _EXPAND_RE.fullmatch(clauses["EXPAND"])
        if not match:
            raise MemoryQLSyntaxError("EXPAND 格式应为 $变量 DEPTH 1|2")
        variable = match.group("var")
        depth = int(match.group("depth"))
        if variable not in node_by_variable:
            raise MemoryQLValidationError(f"EXPAND 引用了未声明变量: ${variable}")
        if node_by_variable[variable].type_name != "Event":
            raise MemoryQLValidationError("MemoryQL 1.0 仅支持从 Event 变量展开")
        if depth < 1 or depth > HARD_DEPTH_LIMIT:
            raise MemoryQLValidationError(f"EXPAND 深度必须在 1 到 {HARD_DEPTH_LIMIT} 之间")
        expand = ExpandClause(variable, depth)

    return_kind = clauses["RETURN"].strip().upper()
    if return_kind not in {"GRAPH", "TABLE", "RAW"}:
        raise MemoryQLValidationError("RETURN 仅支持 GRAPH、TABLE 或 RAW")
    graph_limit_match = _GRAPH_LIMIT_RE.fullmatch(clauses["LIMIT"])
    row_limit_match = _ROW_LIMIT_RE.fullmatch(clauses["LIMIT"])
    if graph_limit_match:
        limit = LimitClause(
            nodes=int(graph_limit_match.group("nodes")),
            edges=int(graph_limit_match.group("edges")),
        )
        if return_kind == "TABLE":
            raise MemoryQLValidationError("RETURN TABLE 必须使用 LIMIT ROWS")
    elif row_limit_match:
        limit = LimitClause(rows=int(row_limit_match.group("rows")))
        if return_kind == "GRAPH":
            raise MemoryQLValidationError("RETURN GRAPH 必须使用 LIMIT NODES ... EDGES ...")
    else:
        raise MemoryQLSyntaxError("LIMIT 格式应为 NODES n EDGES n 或 ROWS n")
    if any(value is not None and value < 1 for value in (limit.nodes, limit.edges, limit.rows)):
        raise MemoryQLValidationError("LIMIT 中的预算必须大于 0")

    return QueryAst(tuple(nodes), tuple(relations), tuple(conditions), expand, return_kind, limit)


def _iso_from_epoch(value: object) -> str | None:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    seconds = numeric / 1000 if numeric > 10_000_000_000 else numeric
    try:
        return datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace("+00:00", "Z")
    except (OSError, OverflowError, ValueError):
        return None


def _epoch_from_iso(value: str) -> int:
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise MemoryQLValidationError(f"无效的 ISO 8601 时间: {value}") from exc
    return int(parsed.timestamp() * 1000)


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha1("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _event_node(row: sqlite3.Row, prefix: str) -> dict[str, Any]:
    event_id = int(row[f"{prefix}event_id"])
    summary = str(row[f"{prefix}summary"] or "")
    return {
        "id": f"event:{event_id}",
        "type": "Event",
        "label": summary or f"Event {event_id}",
        "properties": {
            "id": event_id,
            "summary": summary,
            "event_type": str(row[f"{prefix}event_type"] or ""),
            "occurred_at": _iso_from_epoch(row[f"{prefix}occurred_at"]),
            "confidence": float(row[f"{prefix}confidence"] or 0),
            "status": str(row[f"{prefix}status"] or ""),
        },
        "provenance": {"source_kind": "memory_event", "record_id": event_id},
    }


def _entity_node(row: sqlite3.Row, prefix: str) -> dict[str, Any]:
    entity_id = str(row[f"{prefix}entity_id"] or "")
    name = str(row[f"{prefix}name"] or entity_id)
    return {
        "id": f"entity:{entity_id}",
        "type": "CanonicalEntity",
        "label": name,
        "properties": {
            "id": entity_id,
            "name": name,
            "kind": str(row[f"{prefix}kind"] or ""),
            "confidence": float(row[f"{prefix}confidence"] or 0),
            "status": str(row[f"{prefix}status"] or ""),
            "updated_at": _iso_from_epoch(row[f"{prefix}updated_at"]),
        },
        "provenance": {"source_kind": "canonical_entity", "record_id": entity_id},
    }


def _storyline_node(row: sqlite3.Row, prefix: str) -> dict[str, Any]:
    storyline_id = str(row[f"{prefix}storyline_id"] or "")
    scope = str(row[f"{prefix}scope"] or "")
    return {
        "id": f"storyline:{storyline_id}",
        "type": "Storyline",
        "label": scope or storyline_id,
        "properties": {
            "id": storyline_id,
            "scope": scope,
            "origin_type": str(row[f"{prefix}origin_type"] or ""),
            "status": str(row[f"{prefix}status"] or ""),
            "score": float(row[f"{prefix}score"] or 0),
            "member_count": int(row[f"{prefix}member_count"] or 0),
            "summary": str(row[f"{prefix}summary"] or ""),
            "updated_at": _iso_from_epoch(row[f"{prefix}updated_at"]),
        },
        "provenance": {"source_kind": "memory_storyline", "record_id": storyline_id},
    }


def _source_node(row: sqlite3.Row, prefix: str) -> dict[str, Any]:
    source_kind = str(row[f"{prefix}source_kind"] or "")
    source_id = str(row[f"{prefix}source_id"] or "")
    node_id = _stable_id("source", source_kind, source_id)
    return {
        "id": node_id,
        "type": "Source",
        "label": f"{source_kind} · {source_id}" if source_kind else source_id,
        "properties": {
            "id": source_id,
            "kind": source_kind,
            "timestamp": str(row[f"{prefix}timestamp"] or ""),
        },
        "provenance": {"source_kind": "memory_source", "record_id": source_id},
    }


@dataclass
class MatchRecord:
    bindings: dict[str, dict[str, Any]]
    edges: list[dict[str, Any]]
    table_row: dict[str, Any]


def _condition_value(condition: Condition, type_name: str) -> Any:
    property_type = PROPERTY_SCHEMA[type_name][condition.property_name]["type"]
    value = condition.value
    if property_type == "datetime":
        if not isinstance(value, str):
            raise MemoryQLValidationError(f"{condition.property_name} 必须使用 ISO 8601 字符串")
        return _epoch_from_iso(value)
    if property_type == "number" and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        raise MemoryQLValidationError(f"{condition.property_name} 必须使用数值")
    if property_type == "id" and (
        isinstance(value, bool)
        or not isinstance(value, (int, float, str))
        or (isinstance(value, str) and not value)
    ):
        raise MemoryQLValidationError(f"{condition.property_name} 必须使用数字或字符串标识符")
    if property_type == "string" and not isinstance(value, str):
        raise MemoryQLValidationError(f"{condition.property_name} 必须使用字符串")
    return value


def _compile_conditions(
    conditions: Iterable[Condition],
    variable_context: dict[str, tuple[str, str]],
) -> tuple[str, list[Any]]:
    fragments: list[str] = []
    params: list[Any] = []
    for condition in conditions:
        if condition.variable not in variable_context:
            raise MemoryQLValidationError(f"执行计划缺少变量 ${condition.variable}")
        type_name, alias = variable_context[condition.variable]
        column = PROPERTY_COLUMNS[type_name][condition.property_name]
        expression = f"{alias}.{column}"
        value = _condition_value(condition, type_name)
        if condition.operator == "~=":
            fragment = f"instr(COALESCE({expression}, ''), ?) > 0"
        else:
            fragment = f"{expression} {condition.operator} ?"
        connector = condition.connector or "AND"
        fragments.append((f"{connector} " if fragments else "") + fragment)
        params.append(value)
    return " ".join(fragments), params


EVENT_SELECT = """
{alias}.event_id AS {prefix}event_id,
{alias}.summary AS {prefix}summary,
{alias}.event_type_norm AS {prefix}event_type,
{alias}.occurred_at AS {prefix}occurred_at,
{alias}.confidence AS {prefix}confidence,
{alias}.status AS {prefix}status
"""

ENTITY_SELECT = """
{alias}.entity_id AS {prefix}entity_id,
{alias}.canonical_name AS {prefix}name,
{alias}.entity_type AS {prefix}kind,
{alias}.confidence AS {prefix}confidence,
{alias}.status AS {prefix}status,
{alias}.updated_at AS {prefix}updated_at
"""

def _storyline_select(alias: str, prefix: str, layout: dict[str, set[str]]) -> str:
    summary_expression = "''"
    if _requirements_available(STORYLINE_SUMMARY_REQUIRED_COLUMNS, layout):
        summary_expression = f"""
        COALESCE((
            SELECT cache.summary
            FROM MemoryStorylineSummaryTasks task
            JOIN MemorySummaryCache cache ON cache.task_id=task.task_id
            WHERE task.storyline_id={alias}.storyline_id
              AND cache.status IN ('ready', 'stale')
              AND TRIM(cache.summary)<>''
            ORDER BY CASE cache.status WHEN 'ready' THEN 0 ELSE 1 END,
                     cache.updated_at_ms DESC
            LIMIT 1
        ), '')
        """
    return f"""
    {alias}.storyline_id AS {prefix}storyline_id,
    {alias}.scope AS {prefix}scope,
    {alias}.origin_type AS {prefix}origin_type,
    {alias}.status AS {prefix}status,
    {alias}.score AS {prefix}score,
    {alias}.member_count AS {prefix}member_count,
    {alias}.updated_at AS {prefix}updated_at,
    {summary_expression} AS {prefix}summary
    """

SOURCE_SELECT = """
{alias}.source_kind AS {prefix}source_kind,
{alias}.source_id AS {prefix}source_id,
{alias}.source_timestamp AS {prefix}timestamp
"""


def _relation_query(
    pattern: RelationPattern,
    node_by_var: dict[str, NodePattern],
    conditions: tuple[Condition, ...],
    limit: int,
    layout: dict[str, set[str]],
) -> tuple[str, list[Any], Callable[[sqlite3.Row], MatchRecord]]:
    relation = pattern.type_name
    source_var = pattern.source
    target_var = pattern.target
    if relation == "INVOLVES":
        context = {source_var: ("Event", "e"), target_var: ("CanonicalEntity", "c")}
        condition_sql, params = _compile_conditions(conditions, context)
        sql = f"""
            SELECT {EVENT_SELECT.format(alias='e', prefix='src_')},
                   {ENTITY_SELECT.format(alias='c', prefix='dst_')},
                   m.role AS rel_role, m.confidence AS rel_confidence,
                   m.evidence_json AS rel_evidence
            FROM MemoryEntityMentions m
            JOIN MemoryEvents e ON e.event_id=m.event_id
            JOIN MemoryCanonicalEntities c ON c.entity_id=m.entity_id
            WHERE e.is_deleted=0 AND c.status='active'
              {f'AND ({condition_sql})' if condition_sql else ''}
            ORDER BY e.occurred_at DESC, e.event_id DESC
            LIMIT ?
        """

        def transform(row: sqlite3.Row) -> MatchRecord:
            source = _event_node(row, "src_")
            target = _entity_node(row, "dst_")
            role = str(row["rel_role"] or "")
            edge = {
                "id": _stable_id("edge", relation, source["id"], target["id"], role),
                "type": relation,
                "source": source["id"],
                "target": target["id"],
                "label": role or relation,
                "properties": {"role": role, "confidence": float(row["rel_confidence"] or 0)},
                "provenance": {"source_kind": "entity_mention"},
            }
            return MatchRecord(
                {source_var: source, target_var: target},
                [edge],
                {f"{source_var}.id": source["properties"]["id"], f"{source_var}.summary": source["label"], f"{target_var}.id": target["properties"]["id"], f"{target_var}.name": target["label"], "relation": relation, "role": role},
            )

    elif relation == "PART_OF":
        context = {source_var: ("Event", "e"), target_var: ("Storyline", "s")}
        condition_sql, params = _compile_conditions(conditions, context)
        sql = f"""
            SELECT {EVENT_SELECT.format(alias='e', prefix='src_')},
                   {_storyline_select('s', 'dst_', layout)},
                   m.score AS rel_score, m.rank AS rel_rank
            FROM MemoryStorylineMembers m
            JOIN MemoryEvents e ON e.event_id=m.event_id
            JOIN MemoryStorylines s ON s.storyline_id=m.storyline_id
            WHERE e.is_deleted=0 AND m.status='active' AND s.status='active'
              {f'AND ({condition_sql})' if condition_sql else ''}
            ORDER BY s.updated_at DESC, m.score DESC
            LIMIT ?
        """

        def transform(row: sqlite3.Row) -> MatchRecord:
            source = _event_node(row, "src_")
            target = _storyline_node(row, "dst_")
            edge = {
                "id": _stable_id("edge", relation, source["id"], target["id"]),
                "type": relation,
                "source": source["id"],
                "target": target["id"],
                "label": relation,
                "properties": {"score": float(row["rel_score"] or 0), "rank": int(row["rel_rank"] or 0)},
                "provenance": {"source_kind": "storyline_membership"},
            }
            return MatchRecord(
                {source_var: source, target_var: target},
                [edge],
                {f"{source_var}.id": source["properties"]["id"], f"{source_var}.summary": source["label"], f"{target_var}.id": target["properties"]["id"], f"{target_var}.scope": target["properties"]["scope"], "relation": relation, "score": edge["properties"]["score"]},
            )

    elif relation == "DERIVED_FROM":
        context = {source_var: ("Event", "e"), target_var: ("Source", "s")}
        condition_sql, params = _compile_conditions(conditions, context)
        sql = f"""
            SELECT {EVENT_SELECT.format(alias='e', prefix='src_')},
                   {SOURCE_SELECT.format(alias='s', prefix='dst_')},
                   s.event_source_id AS rel_id, s.source_seq AS rel_seq
            FROM MemoryEventSources s
            JOIN MemoryEvents e ON e.event_id=s.event_id
            WHERE e.is_deleted=0
              {f'AND ({condition_sql})' if condition_sql else ''}
            ORDER BY e.occurred_at DESC, s.event_source_id DESC
            LIMIT ?
        """

        def transform(row: sqlite3.Row) -> MatchRecord:
            source = _event_node(row, "src_")
            target = _source_node(row, "dst_")
            edge = {
                "id": f"edge:source:{int(row['rel_id'])}",
                "type": relation,
                "source": source["id"],
                "target": target["id"],
                "label": relation,
                "properties": {"sequence": row["rel_seq"]},
                "provenance": {"source_kind": "event_source", "record_id": int(row["rel_id"])},
            }
            return MatchRecord(
                {source_var: source, target_var: target},
                [edge],
                {f"{source_var}.id": source["properties"]["id"], f"{source_var}.summary": source["label"], f"{target_var}.kind": target["properties"]["kind"], f"{target_var}.id": target["properties"]["id"], "relation": relation},
            )

    elif relation == "RELATES_TO":
        # Bound the narrow relation rows before projecting the potentially large
        # event payloads. Sorting fully joined event rows made the first query
        # after startup exceed the hard timeout on production-sized databases.
        context = {
            source_var: ("Event", "source_filter"),
            target_var: ("Event", "target_filter"),
        }
        condition_sql, params = _compile_conditions(conditions, context)
        sql = f"""
            SELECT {EVENT_SELECT.format(alias='a', prefix='src_')},
                   {EVENT_SELECT.format(alias='b', prefix='dst_')},
                   r.relation_id AS rel_id, r.relation_type AS rel_type,
                   r.confidence AS rel_confidence, r.revision AS rel_revision
            FROM (
                SELECT r.relation_id, r.source_event_id, r.target_event_id,
                       r.relation_type, r.confidence, r.revision, r.updated_at_ms
                FROM MemoryEventRelations r
                JOIN MemoryEvents source_filter
                  ON source_filter.event_id=r.source_event_id
                JOIN MemoryEvents target_filter
                  ON target_filter.event_id=r.target_event_id
                WHERE source_filter.is_deleted=0
                  AND target_filter.is_deleted=0
                  AND r.status='active'
                  {f'AND ({condition_sql})' if condition_sql else ''}
                ORDER BY r.updated_at_ms DESC, r.relation_id DESC
                LIMIT ?
            ) r
            JOIN MemoryEvents a ON a.event_id=r.source_event_id
            JOIN MemoryEvents b ON b.event_id=r.target_event_id
            ORDER BY r.updated_at_ms DESC, r.relation_id DESC
        """

        def transform(row: sqlite3.Row) -> MatchRecord:
            source = _event_node(row, "src_")
            target = _event_node(row, "dst_")
            relation_label = str(row["rel_type"] or relation)
            edge = {
                "id": f"edge:relation:{row['rel_id']}",
                "type": relation,
                "source": source["id"],
                "target": target["id"],
                "label": relation_label,
                "properties": {"relation_type": relation_label, "confidence": float(row["rel_confidence"] or 0), "revision": int(row["rel_revision"] or 0)},
                "provenance": {"source_kind": "event_relation", "record_id": str(row["rel_id"])},
            }
            return MatchRecord(
                {source_var: source, target_var: target},
                [edge],
                {f"{source_var}.id": source["properties"]["id"], f"{source_var}.summary": source["label"], f"{target_var}.id": target["properties"]["id"], f"{target_var}.summary": target["label"], "relation": relation_label, "confidence": edge["properties"]["confidence"]},
            )
    else:
        raise MemoryQLValidationError(f"未实现的关系类型: {relation}")
    return sql, [*params, limit], transform


def _node_query(
    pattern: NodePattern,
    conditions: tuple[Condition, ...],
    limit: int,
    layout: dict[str, set[str]],
) -> tuple[str, list[Any], Callable[[sqlite3.Row], dict[str, Any]]]:
    variable = pattern.variable
    type_name = pattern.type_name
    context = {variable: (type_name, "n")}
    condition_sql, params = _compile_conditions(conditions, context)
    if type_name == "Event":
        sql = f"""
            SELECT {EVENT_SELECT.format(alias='n', prefix='node_')}
            FROM MemoryEvents n
            WHERE n.is_deleted=0 {f'AND ({condition_sql})' if condition_sql else ''}
            ORDER BY n.occurred_at DESC, n.event_id DESC LIMIT ?
        """
        transform = lambda row: _event_node(row, "node_")
    elif type_name == "CanonicalEntity":
        sql = f"""
            SELECT {ENTITY_SELECT.format(alias='n', prefix='node_')}
            FROM MemoryCanonicalEntities n
            WHERE n.status='active' {f'AND ({condition_sql})' if condition_sql else ''}
            ORDER BY n.updated_at DESC, n.entity_id LIMIT ?
        """
        transform = lambda row: _entity_node(row, "node_")
    elif type_name == "Storyline":
        sql = f"""
            SELECT {_storyline_select('n', 'node_', layout)}
            FROM MemoryStorylines n
            WHERE n.status='active' {f'AND ({condition_sql})' if condition_sql else ''}
            ORDER BY n.updated_at DESC, n.storyline_id LIMIT ?
        """
        transform = lambda row: _storyline_node(row, "node_")
    elif type_name == "Source":
        sql = f"""
            SELECT n.source_kind AS node_source_kind,
                   n.source_id AS node_source_id,
                   MAX(n.source_timestamp) AS node_timestamp
            FROM MemoryEventSources n
            WHERE 1=1 {f'AND ({condition_sql})' if condition_sql else ''}
            GROUP BY n.source_kind, n.source_id
            ORDER BY MAX(n.created_at) DESC LIMIT ?
        """
        transform = lambda row: _source_node(row, "node_")
    else:
        raise MemoryQLValidationError(f"未实现的节点类型: {type_name}")
    return sql, [*params, limit], transform


class _ResultCollector:
    def __init__(self, node_limit: int, edge_limit: int, row_limit: int) -> None:
        self.node_limit = node_limit
        self.edge_limit = edge_limit
        self.row_limit = row_limit
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}
        self.rows: list[dict[str, Any]] = []
        self.truncated = False

    def add(self, record: MatchRecord, *, include_row: bool = True) -> bool:
        new_nodes = [node for node in record.bindings.values() if node["id"] not in self.nodes]
        new_edges = [edge for edge in record.edges if edge["id"] not in self.edges]
        if len(self.nodes) + len(new_nodes) > self.node_limit or len(self.edges) + len(new_edges) > self.edge_limit:
            self.truncated = True
            return False
        for node in new_nodes:
            self.nodes[node["id"]] = node
        for edge in new_edges:
            self.edges[edge["id"]] = edge
        if include_row:
            if len(self.rows) < self.row_limit:
                self.rows.append(record.table_row)
            else:
                self.truncated = True
        return True


def _table_layout(connection: sqlite3.Connection) -> dict[str, set[str]]:
    expected_tables = {
        table
        for requirements in [
            *TYPE_REQUIRED_COLUMNS.values(),
            *RELATION_REQUIRED_COLUMNS.values(),
            STORYLINE_SUMMARY_REQUIRED_COLUMNS,
        ]
        for table in requirements
    }
    existing_tables = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    return {
        table: {
            str(row[1])
            for row in connection.execute(f'PRAGMA table_info("{table}")')
        }
        for table in expected_tables & existing_tables
    }


def _requirements_available(
    requirements: dict[str, set[str]],
    layout: dict[str, set[str]],
) -> bool:
    return all(columns <= layout.get(table, set()) for table, columns in requirements.items())


def _type_available(type_name: str, layout: dict[str, set[str]]) -> bool:
    return _requirements_available(TYPE_REQUIRED_COLUMNS[type_name], layout)


def _relation_available(relation: str, layout: dict[str, set[str]]) -> bool:
    meta = RELATION_META[relation]
    return (
        _type_available(meta["source"], layout)
        and _type_available(meta["target"], layout)
        and _requirements_available(RELATION_REQUIRED_COLUMNS[relation], layout)
    )


def _layout_gaps(layout: dict[str, set[str]]) -> tuple[list[str], dict[str, list[str]]]:
    required: dict[str, set[str]] = {}
    for requirements in [*TYPE_REQUIRED_COLUMNS.values(), *RELATION_REQUIRED_COLUMNS.values()]:
        for table, columns in requirements.items():
            required.setdefault(table, set()).update(columns)
    missing_tables = sorted(table for table in required if table not in layout)
    missing_columns = {
        table: sorted(columns - layout.get(table, set()))
        for table, columns in required.items()
        if table in layout and columns - layout[table]
    }
    return missing_tables, missing_columns


def _assert_query_available(ast: QueryAst, layout: dict[str, set[str]]) -> None:
    for node in ast.nodes:
        if not _type_available(node.type_name, layout):
            raise MemoryQueryUnavailable(f"当前记忆 schema 尚未提供 {node.type_name}")
    for relation in ast.relations:
        if not _relation_available(relation.type_name, layout):
            raise MemoryQueryUnavailable(f"当前记忆 schema 尚未提供 {relation.type_name}")


def _query_rows(connection: sqlite3.Connection, sql: str, params: list[Any]) -> list[sqlite3.Row]:
    try:
        return connection.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise MemoryQueryTimeout("查询超过服务端时间预算") from exc
        raise


def _expand_event_neighbors(
    connection: sqlite3.Connection,
    event_ids: set[int],
    collector: _ResultCollector,
    layout: dict[str, set[str]],
) -> set[int]:
    if not event_ids or len(collector.edges) >= collector.edge_limit:
        return set()
    placeholders = ",".join("?" for _ in event_ids)
    remaining = collector.edge_limit - len(collector.edges)
    discovered_events: set[int] = set()

    expansion_specs: list[tuple[str, str, Callable[[sqlite3.Row], MatchRecord]]] = []
    if _relation_available("INVOLVES", layout):
        sql = f"""
            SELECT {EVENT_SELECT.format(alias='e', prefix='src_')},
                   {ENTITY_SELECT.format(alias='c', prefix='dst_')},
                   m.role AS rel_role, m.confidence AS rel_confidence
            FROM MemoryEntityMentions m
            JOIN MemoryEvents e ON e.event_id=m.event_id
            JOIN MemoryCanonicalEntities c ON c.entity_id=m.entity_id
            WHERE e.is_deleted=0 AND c.status='active' AND e.event_id IN ({placeholders})
            ORDER BY e.occurred_at DESC LIMIT ?
        """

        def mention_record(row: sqlite3.Row) -> MatchRecord:
            event = _event_node(row, "src_")
            entity = _entity_node(row, "dst_")
            role = str(row["rel_role"] or "")
            edge = {"id": _stable_id("edge", "INVOLVES", event["id"], entity["id"], role), "type": "INVOLVES", "source": event["id"], "target": entity["id"], "label": role or "INVOLVES", "properties": {"role": role, "confidence": float(row["rel_confidence"] or 0)}, "provenance": {"source_kind": "entity_mention"}}
            return MatchRecord({"event": event, "entity": entity}, [edge], {})

        expansion_specs.append(("INVOLVES", sql, mention_record))
    if _relation_available("PART_OF", layout):
        sql = f"""
            SELECT {EVENT_SELECT.format(alias='e', prefix='src_')},
                   {_storyline_select('s', 'dst_', layout)},
                   m.score AS rel_score, m.rank AS rel_rank
            FROM MemoryStorylineMembers m
            JOIN MemoryEvents e ON e.event_id=m.event_id
            JOIN MemoryStorylines s ON s.storyline_id=m.storyline_id
            WHERE e.is_deleted=0 AND m.status='active' AND s.status='active'
              AND e.event_id IN ({placeholders})
            ORDER BY m.score DESC LIMIT ?
        """

        def membership_record(row: sqlite3.Row) -> MatchRecord:
            event = _event_node(row, "src_")
            storyline = _storyline_node(row, "dst_")
            edge = {"id": _stable_id("edge", "PART_OF", event["id"], storyline["id"]), "type": "PART_OF", "source": event["id"], "target": storyline["id"], "label": "PART_OF", "properties": {"score": float(row["rel_score"] or 0), "rank": int(row["rel_rank"] or 0)}, "provenance": {"source_kind": "storyline_membership"}}
            return MatchRecord({"event": event, "storyline": storyline}, [edge], {})

        expansion_specs.append(("PART_OF", sql, membership_record))
    if _relation_available("DERIVED_FROM", layout):
        sql = f"""
            SELECT {EVENT_SELECT.format(alias='e', prefix='src_')},
                   {SOURCE_SELECT.format(alias='s', prefix='dst_')},
                   s.event_source_id AS rel_id, s.source_seq AS rel_seq
            FROM MemoryEventSources s
            JOIN MemoryEvents e ON e.event_id=s.event_id
            WHERE e.is_deleted=0 AND e.event_id IN ({placeholders})
            ORDER BY e.occurred_at DESC LIMIT ?
        """

        def source_record(row: sqlite3.Row) -> MatchRecord:
            event = _event_node(row, "src_")
            source = _source_node(row, "dst_")
            edge = {"id": f"edge:source:{int(row['rel_id'])}", "type": "DERIVED_FROM", "source": event["id"], "target": source["id"], "label": "DERIVED_FROM", "properties": {"sequence": row["rel_seq"]}, "provenance": {"source_kind": "event_source", "record_id": int(row["rel_id"])}}
            return MatchRecord({"event": event, "source": source}, [edge], {})

        expansion_specs.append(("DERIVED_FROM", sql, source_record))
    if _relation_available("RELATES_TO", layout):
        sql = f"""
            SELECT {EVENT_SELECT.format(alias='a', prefix='src_')},
                   {EVENT_SELECT.format(alias='b', prefix='dst_')},
                   r.relation_id AS rel_id, r.relation_type AS rel_type,
                   r.confidence AS rel_confidence, r.revision AS rel_revision
            FROM MemoryEventRelations r
            JOIN MemoryEvents a ON a.event_id=r.source_event_id
            JOIN MemoryEvents b ON b.event_id=r.target_event_id
            WHERE a.is_deleted=0 AND b.is_deleted=0 AND r.status='active'
              AND (a.event_id IN ({placeholders}) OR b.event_id IN ({placeholders}))
            ORDER BY r.updated_at_ms DESC LIMIT ?
        """

        def relation_record(row: sqlite3.Row) -> MatchRecord:
            source = _event_node(row, "src_")
            target = _event_node(row, "dst_")
            discovered_events.update({source["properties"]["id"], target["properties"]["id"]})
            label = str(row["rel_type"] or "RELATES_TO")
            edge = {"id": f"edge:relation:{row['rel_id']}", "type": "RELATES_TO", "source": source["id"], "target": target["id"], "label": label, "properties": {"relation_type": label, "confidence": float(row["rel_confidence"] or 0), "revision": int(row["rel_revision"] or 0)}, "provenance": {"source_kind": "event_relation", "record_id": str(row["rel_id"])} }
            return MatchRecord({"source": source, "target": target}, [edge], {})

        params_prefix = [*event_ids, *event_ids]
        rows = _query_rows(connection, sql, [*params_prefix, remaining + 1])
        if len(rows) > remaining:
            collector.truncated = True
        for row in rows[:remaining]:
            if not collector.add(relation_record(row), include_row=False):
                break

    for _name, sql, transform in expansion_specs:
        remaining = collector.edge_limit - len(collector.edges)
        if remaining <= 0:
            collector.truncated = True
            break
        rows = _query_rows(connection, sql, [*event_ids, remaining + 1])
        if len(rows) > remaining:
            collector.truncated = True
        for row in rows[:remaining]:
            if not collector.add(transform(row), include_row=False):
                break
    return discovered_events - event_ids


def _column_list(rows: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for row in rows:
        for key in row:
            if key not in result:
                result.append(key)
    return result


class SemanticMemoryService:
    def __init__(self, db_path: str | Path = DB_PATH) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.is_file():
            raise MemoryQueryUnavailable("记忆数据库尚未创建")
        connection = sqlite3.connect(
            f"file:{self.db_path.resolve().as_posix()}?mode=ro",
            uri=True,
            timeout=0.25,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        return connection

    def schema(self) -> dict[str, Any]:
        try:
            connection = self._connect()
        except MemoryQueryUnavailable:
            connection = None
        try:
            layout = _table_layout(connection) if connection is not None else {}

            def count(sql: str, available: bool) -> int:
                if connection is None or not available:
                    return 0
                return int(connection.execute(sql).fetchone()[0])

            type_counts = {
                "Event": count("SELECT COUNT(*) FROM MemoryEvents WHERE is_deleted=0", _type_available("Event", layout)),
                "CanonicalEntity": count("SELECT COUNT(*) FROM MemoryCanonicalEntities WHERE status='active'", _type_available("CanonicalEntity", layout)),
                "Storyline": count("SELECT COUNT(*) FROM MemoryStorylines WHERE status='active'", _type_available("Storyline", layout)),
                "Source": count("SELECT COUNT(*) FROM (SELECT 1 FROM MemoryEventSources GROUP BY source_kind, source_id)", _type_available("Source", layout)),
            }
            relation_counts = {
                "INVOLVES": count("SELECT COUNT(*) FROM MemoryEntityMentions m JOIN MemoryEvents e ON e.event_id=m.event_id JOIN MemoryCanonicalEntities c ON c.entity_id=m.entity_id WHERE e.is_deleted=0 AND c.status='active'", _relation_available("INVOLVES", layout)),
                "PART_OF": count("SELECT COUNT(*) FROM MemoryStorylineMembers m JOIN MemoryEvents e ON e.event_id=m.event_id JOIN MemoryStorylines s ON s.storyline_id=m.storyline_id WHERE m.status='active' AND e.is_deleted=0 AND s.status='active'", _relation_available("PART_OF", layout)),
                "DERIVED_FROM": count("SELECT COUNT(*) FROM MemoryEventSources s JOIN MemoryEvents e ON e.event_id=s.event_id WHERE e.is_deleted=0", _relation_available("DERIVED_FROM", layout)),
                "RELATES_TO": count("SELECT COUNT(*) FROM MemoryEventRelations r JOIN MemoryEvents a ON a.event_id=r.source_event_id JOIN MemoryEvents b ON b.event_id=r.target_event_id WHERE r.status='active' AND a.is_deleted=0 AND b.is_deleted=0", _relation_available("RELATES_TO", layout)),
            }
            types = [
                {
                    "name": name,
                    "label": meta["label"],
                    "description": meta["description"],
                    "count": type_counts[name],
                    "available": _type_available(name, layout),
                    "properties": [
                        {"name": prop, **definition}
                        for prop, definition in PROPERTY_SCHEMA[name].items()
                    ],
                }
                for name, meta in TYPE_META.items()
            ]
            relations = [
                {
                    "name": name,
                    "label": meta["label"],
                    "description": meta["description"],
                    "source": meta["source"],
                    "target": meta["target"],
                    "count": relation_counts[name],
                    "available": _relation_available(name, layout),
                }
                for name, meta in RELATION_META.items()
            ]
            missing = [item["name"] for item in [*types, *relations] if not item["available"]]
            missing_tables, missing_columns = _layout_gaps(layout)
            return {
                "schema_version": SCHEMA_VERSION,
                "language": {
                    "name": "MemoryQL",
                    "version": LANGUAGE_VERSION,
                    "read_only": True,
                    "clauses": ["MATCH", "WHERE", "EXPAND", "RETURN", "LIMIT"],
                },
                "types": types,
                "relations": relations,
                "limits": {
                    "nodes": HARD_NODE_LIMIT,
                    "edges": HARD_EDGE_LIMIT,
                    "rows": HARD_ROW_LIMIT,
                    "depth": HARD_DEPTH_LIMIT,
                    "timeout_ms": HARD_TIMEOUT_MS,
                    "query_characters": HARD_QUERY_LENGTH,
                    "relation_patterns": 1,
                },
                "compatibility": {
                    "status": "compatible" if not missing else "degraded",
                    "missing": missing,
                    "missing_tables": missing_tables,
                    "missing_columns": missing_columns,
                    "message": (
                        "语义层可完整查询。"
                        if not missing
                        else "部分处理层尚未生成；不可用类型会保持只读说明，不会猜测字段。"
                    ),
                },
            }
        finally:
            if connection is not None:
                connection.close()

    def query(
        self,
        query: str,
        *,
        language_version: str,
        node_limit: int = HARD_NODE_LIMIT,
        edge_limit: int = HARD_EDGE_LIMIT,
        row_limit: int = HARD_ROW_LIMIT,
        max_depth: int = HARD_DEPTH_LIMIT,
    ) -> dict[str, Any]:
        if language_version != LANGUAGE_VERSION:
            raise MemoryQLValidationError(
                f"不支持的 MemoryQL 版本 {language_version!r}；当前版本为 {LANGUAGE_VERSION}"
            )
        ast = parse_memoryql(query)
        if ast.expand and ast.expand.depth > max(0, min(HARD_DEPTH_LIMIT, int(max_depth))):
            raise MemoryQLValidationError("查询 EXPAND 深度超过请求预算 max_depth")

        requested = {
            "nodes": ast.limit.nodes if ast.limit.nodes is not None else int(node_limit),
            "edges": ast.limit.edges if ast.limit.edges is not None else int(edge_limit),
            "rows": ast.limit.rows if ast.limit.rows is not None else int(row_limit),
            "depth": ast.expand.depth if ast.expand else 0,
        }
        effective = {
            "nodes": max(1, min(HARD_NODE_LIMIT, int(node_limit), int(requested["nodes"]))),
            "edges": max(1, min(HARD_EDGE_LIMIT, int(edge_limit), int(requested["edges"]))),
            "rows": max(1, min(HARD_ROW_LIMIT, int(row_limit), int(requested["rows"]))),
            "depth": requested["depth"],
            "timeout_ms": HARD_TIMEOUT_MS,
        }
        started = time.monotonic()
        deadline = started + HARD_TIMEOUT_MS / 1000
        connection = self._connect()
        connection.set_progress_handler(lambda: 1 if time.monotonic() >= deadline else 0, 1_000)
        collector = _ResultCollector(effective["nodes"], effective["edges"], effective["rows"])
        seed_event_ids: set[int] = set()
        try:
            layout = _table_layout(connection)
            _assert_query_available(ast, layout)
            node_by_var = {node.variable: node for node in ast.nodes}
            if ast.relations:
                relation = ast.relations[0]
                fetch_limit = min(effective["edges"], effective["rows"]) + 1
                sql, params, transform = _relation_query(
                    relation,
                    node_by_var,
                    ast.conditions,
                    fetch_limit,
                    layout,
                )
                records = [transform(row) for row in _query_rows(connection, sql, params)]
                if len(records) >= fetch_limit:
                    collector.truncated = True
                for record in records[: fetch_limit - 1]:
                    if not collector.add(record):
                        break
                    if ast.expand and ast.expand.variable in record.bindings:
                        node = record.bindings[ast.expand.variable]
                        if node["type"] == "Event":
                            seed_event_ids.add(int(node["properties"]["id"]))
            else:
                pattern = ast.nodes[0]
                seed_limit = min(effective["nodes"], 12) if ast.expand else effective["nodes"]
                sql, params, transform = _node_query(pattern, ast.conditions, seed_limit + 1, layout)
                rows = _query_rows(connection, sql, params)
                if len(rows) > seed_limit:
                    collector.truncated = True
                for row in rows[:seed_limit]:
                    node = transform(row)
                    collector.add(MatchRecord({pattern.variable: node}, [], node["properties"]))
                    if ast.expand and node["type"] == "Event":
                        seed_event_ids.add(int(node["properties"]["id"]))

            frontier = seed_event_ids
            visited = set(seed_event_ids)
            for _depth in range(effective["depth"]):
                if not frontier:
                    break
                frontier = _expand_event_neighbors(connection, frontier, collector, layout) - visited
                visited.update(frontier)
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise MemoryQueryTimeout("查询超过服务端时间预算") from exc
            raise MemoryQueryUnavailable("语义查询暂时不可用") from exc
        finally:
            connection.close()

        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        nodes = list(collector.nodes.values())
        edges = list(collector.edges.values())
        table_columns = _column_list(collector.rows)
        plan = [
            f"校验 MemoryQL {LANGUAGE_VERSION} 只读 AST",
            f"匹配 {len(ast.nodes)} 个节点声明与 {len(ast.relations)} 条关系声明",
        ]
        if ast.conditions:
            plan.append(f"应用 {len(ast.conditions)} 个参数化语义过滤条件")
        if ast.expand:
            plan.append(f"从 ${ast.expand.variable} 有界展开 {ast.expand.depth} 跳")
        plan.append("投影为本次请求独享的隔离结果集")
        clamped = any(
            int(requested[key]) > int(effective[key])
            for key in ("nodes", "edges", "rows")
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "language_version": LANGUAGE_VERSION,
            "query_id": f"mq_{uuid.uuid4().hex[:12]}",
            "return_kind": ast.return_kind.lower(),
            "budget": {
                "requested": requested,
                "effective": effective,
                "consumed": {
                    "nodes": len(nodes),
                    "edges": len(edges),
                    "rows": len(collector.rows),
                    "elapsed_ms": elapsed_ms,
                },
                "clamped": clamped,
            },
            "truncated": collector.truncated or clamped,
            "nodes": nodes,
            "edges": edges,
            "table": {"columns": table_columns, "rows": collector.rows},
            "provenance": {
                "read_only": True,
                "isolation": "per_query",
                "schema_version": SCHEMA_VERSION,
                "node_records": len(nodes),
                "edge_records": len(edges),
            },
            "explain": {
                "ast": asdict(ast),
                "plan": plan,
                "warnings": (["请求预算已被服务端硬上限收紧。"] if clamped else []),
            },
        }


__all__ = [
    "HARD_DEPTH_LIMIT",
    "HARD_EDGE_LIMIT",
    "HARD_NODE_LIMIT",
    "HARD_ROW_LIMIT",
    "LANGUAGE_VERSION",
    "MemoryQLSyntaxError",
    "MemoryQLValidationError",
    "MemoryQueryError",
    "MemoryQueryTimeout",
    "MemoryQueryUnavailable",
    "SCHEMA_VERSION",
    "SemanticMemoryService",
    "parse_memoryql",
]
