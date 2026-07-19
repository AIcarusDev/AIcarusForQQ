import { ApiError, requestV1Data } from "./http.js";

export const MEMORYQL_LANGUAGE_VERSION = "1.0";

function list(value) {
  return Array.isArray(value) ? value : [];
}

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function normalizeSchema(payload) {
  if (!payload || typeof payload !== "object") {
    throw new ApiError("记忆语义结构为空。", { code: "incompatible_contract" });
  }

  return {
    schemaVersion: String(payload.schema_version || "unknown"),
    language: {
      name: String(payload.language?.name || "MemoryQL"),
      version: String(payload.language?.version || MEMORYQL_LANGUAGE_VERSION),
      readOnly: payload.language?.read_only !== false,
      clauses: list(payload.language?.clauses).map(String),
    },
    types: list(payload.types).map((item) => ({
      name: String(item?.name || "Unknown"),
      label: String(item?.label || item?.name || "未知类型"),
      description: String(item?.description || ""),
      count: Math.max(0, finiteNumber(item?.count)),
      available: item?.available !== false,
      properties: list(item?.properties).map((property) => ({
        name: String(property?.name || ""),
        type: String(property?.type || "unknown"),
        operators: list(property?.operators).map(String),
      })),
    })),
    relations: list(payload.relations).map((item) => ({
      name: String(item?.name || "Unknown"),
      label: String(item?.label || item?.name || "未知关系"),
      description: String(item?.description || ""),
      source: String(item?.source || "Unknown"),
      target: String(item?.target || "Unknown"),
      count: Math.max(0, finiteNumber(item?.count)),
      available: item?.available !== false,
    })),
    limits: {
      nodes: Math.max(1, finiteNumber(payload.limits?.nodes, 80)),
      edges: Math.max(1, finiteNumber(payload.limits?.edges, 120)),
      rows: Math.max(1, finiteNumber(payload.limits?.rows, 100)),
      depth: Math.max(0, finiteNumber(payload.limits?.depth, 2)),
      timeoutMs: Math.max(1, finiteNumber(payload.limits?.timeout_ms, 500)),
      queryCharacters: Math.max(1, finiteNumber(payload.limits?.query_characters, 8000)),
    },
    compatibility: {
      status: String(payload.compatibility?.status || "unknown"),
      missing: list(payload.compatibility?.missing).map(String),
      message: String(payload.compatibility?.message || ""),
    },
  };
}

function normalizeQueryResult(payload) {
  if (!payload || typeof payload !== "object") {
    throw new ApiError("记忆查询结果为空。", { code: "incompatible_contract" });
  }

  const nodes = list(payload.nodes).filter((node) => node && typeof node === "object");
  const nodeIds = new Set(nodes.map((node) => String(node.id)));
  const edges = list(payload.edges).filter((edge) => (
    edge
    && typeof edge === "object"
    && nodeIds.has(String(edge.source))
    && nodeIds.has(String(edge.target))
  ));

  return {
    schemaVersion: String(payload.schema_version || "unknown"),
    languageVersion: String(payload.language_version || MEMORYQL_LANGUAGE_VERSION),
    queryId: String(payload.query_id || "unknown"),
    returnKind: String(payload.return_kind || "graph"),
    budget: payload.budget && typeof payload.budget === "object" ? payload.budget : {},
    truncated: payload.truncated === true,
    nodes,
    edges,
    table: {
      columns: list(payload.table?.columns).map(String),
      rows: list(payload.table?.rows).filter((row) => row && typeof row === "object"),
    },
    provenance: payload.provenance && typeof payload.provenance === "object" ? payload.provenance : {},
    explain: payload.explain && typeof payload.explain === "object" ? payload.explain : {},
  };
}

export async function loadMemorySchema({ signal } = {}) {
  const payload = await requestV1Data("/api/ui/v1/memory/schema", { signal });
  return normalizeSchema(payload);
}

export async function runMemoryQuery(query, {
  signal,
  languageVersion = MEMORYQL_LANGUAGE_VERSION,
  nodeLimit = 80,
  edgeLimit = 120,
  rowLimit = 100,
  maxDepth = 2,
} = {}) {
  const payload = await requestV1Data("/api/ui/v1/memory/query", {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      language_version: languageVersion,
      node_limit: nodeLimit,
      edge_limit: edgeLimit,
      row_limit: rowLimit,
      max_depth: maxDepth,
    }),
  });
  return normalizeQueryResult(payload);
}
