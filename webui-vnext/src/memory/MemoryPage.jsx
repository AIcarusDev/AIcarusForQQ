import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  ReactFlow,
  useNodesState,
} from "@xyflow/react";
import {
  AlertCircle,
  BarChart3,
  BookOpen,
  Box,
  Brain,
  Clock3,
  Code2,
  Database,
  Eye,
  Layers3,
  Maximize2,
  Minimize2,
  Network,
  RefreshCw,
  ScrollText,
  ShieldCheck,
} from "lucide-react";
import { loadMemorySchema, runMemoryQuery } from "../api/memoryApi.js";
import { MemoryQueryGuide } from "./MemoryQueryGuide.jsx";
import { MEMORY_QUERY_EXAMPLES } from "./memoryQueryGuide.js";

const SCHEMA_ICONS = {
  Event: Box,
  CanonicalEntity: Brain,
  Storyline: Layers3,
  Source: Database,
  INVOLVES: Network,
  PART_OF: Layers3,
  DERIVED_FROM: Database,
  RELATES_TO: Network,
};

const NODE_COLORS = {
  Event: "#b87721",
  CanonicalEntity: "#2d746d",
  Storyline: "#78669d",
  Source: "#60707a",
};

const NODE_CLASS_NAMES = {
  Event: "event-node",
  CanonicalEntity: "entity-node",
  Storyline: "storyline-node",
  Source: "source-node",
};

const GRAPH_NODE_WIDTH = 190;
const GRAPH_NODE_HEIGHT = 78;
const GRAPH_COLUMN_GAP = 72;
const GRAPH_ROW_GAP = 58;

function formatCount(value) {
  return new Intl.NumberFormat("zh-CN").format(Number(value) || 0);
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function nodeMeta(node) {
  const properties = node?.properties || {};
  if (node?.type === "Event") {
    return [properties.event_type, properties.occurred_at].filter(Boolean).join(" · ") || "语义事件";
  }
  if (node?.type === "CanonicalEntity") {
    const confidence = Number.isFinite(Number(properties.confidence))
      ? `置信度 ${Number(properties.confidence).toFixed(2)}`
      : "";
    return [properties.kind, confidence].filter(Boolean).join(" · ") || "规范实体";
  }
  if (node?.type === "Storyline") {
    const summary = String(properties.summary || "").trim();
    const members = Number.isFinite(Number(properties.member_count))
      ? `${Number(properties.member_count)} 个成员`
      : "";
    return summary || [properties.status, members].filter(Boolean).join(" · ") || "故事线";
  }
  if (node?.type === "Source") {
    return [properties.kind, properties.timestamp].filter(Boolean).join(" · ") || "记忆来源";
  }
  return "语义节点";
}

function compareGraphIds(left, right) {
  return String(left).localeCompare(String(right), "zh-CN", { numeric: true });
}

function graphTraversalOrder(nodes, edges) {
  const nodeById = new Map(nodes.map((node) => [String(node.id), node]));
  const adjacency = new Map([...nodeById.keys()].map((id) => [id, new Set()]));

  for (const edge of edges) {
    const source = String(edge.source);
    const target = String(edge.target);
    if (source === target || !adjacency.has(source) || !adjacency.has(target)) continue;
    adjacency.get(source).add(target);
    adjacency.get(target).add(source);
  }

  const degree = (id) => adjacency.get(id)?.size || 0;
  const byConnectivity = (left, right) => degree(right) - degree(left) || compareGraphIds(left, right);
  const roots = [...nodeById.keys()].sort(byConnectivity);
  const visited = new Set();
  const ordered = [];

  for (const root of roots) {
    if (visited.has(root)) continue;
    const queue = [root];
    visited.add(root);

    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const current = queue[cursor];
      ordered.push(nodeById.get(current));
      const neighbors = [...(adjacency.get(current) || [])].sort(byConnectivity);
      for (const neighbor of neighbors) {
        if (visited.has(neighbor)) continue;
        visited.add(neighbor);
        queue.push(neighbor);
      }
    }
  }

  return ordered;
}

function layoutResultGraph(nodes, edges) {
  const ordered = graphTraversalOrder(nodes, edges);
  if (!ordered.length) return [];

  const columnStep = GRAPH_NODE_WIDTH + GRAPH_COLUMN_GAP;
  const rowStep = GRAPH_NODE_HEIGHT + GRAPH_ROW_GAP;
  const columns = Math.max(1, Math.ceil(Math.sqrt(ordered.length * (rowStep / columnStep))));
  return ordered.map((node, index) => {
    const row = Math.floor(index / columns);
    const column = index % columns;
    const rowLength = Math.min(columns, ordered.length - row * columns);
    const centeredColumn = column + (columns - rowLength) / 2;
    return {
      id: String(node.id),
      type: "memoryNode",
      position: {
        x: Math.round(48 + centeredColumn * columnStep),
        y: 42 + row * rowStep,
      },
      data: {
        kind: String(node.type || "Unknown"),
        label: String(node.label || node.id || "未命名节点"),
        meta: nodeMeta(node),
        raw: node,
      },
      className: `memory-node ${NODE_CLASS_NAMES[node.type] || ""}`,
    };
  });
}

function buildFlowEdges(edges) {
  const groups = new Map();
  for (const edge of edges) {
    const source = String(edge.source);
    const target = String(edge.target);
    const key = [source, target].sort().join("::");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(String(edge.id));
  }

  return edges.map((edge) => {
    const source = String(edge.source);
    const target = String(edge.target);
    const siblings = groups.get([source, target].sort().join("::")) || [];
    const siblingIndex = siblings.indexOf(String(edge.id));
    const offset = siblingIndex - (siblings.length - 1) / 2;
    return {
      id: String(edge.id),
      source,
      target,
      label: String(edge.label || edge.type || ""),
      type: "default",
      pathOptions: { curvature: 0.22 + Math.abs(offset) * 0.13 },
      labelBgPadding: [7, 4],
      labelBgBorderRadius: 5,
      interactionWidth: 22,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 18,
        height: 18,
        color: "var(--primary)",
      },
    };
  });
}

function MemoryGraphNode({ data, selected }) {
  return (
    <div className={`memory-node-card ${selected ? "is-selected" : ""}`}>
      <Handle type="target" position={Position.Left} />
      <span>{data.kind}</span>
      <strong title={data.label}>{data.label}</strong>
      <small title={data.meta}>{data.meta}</small>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const MEMORY_NODE_TYPES = { memoryNode: MemoryGraphNode };

function ResourceState({ icon: Icon = AlertCircle, title, detail, action, actionLabel = "重试" }) {
  return (
    <div className="memory-resource-state" role="status">
      <Icon size={22} />
      <strong>{title}</strong>
      <span>{detail}</span>
      {action && <button type="button" onClick={action}>{actionLabel}</button>}
    </div>
  );
}

function ResultInspector({ selectedNode, history, onRestoreQuery }) {
  const properties = Object.entries(selectedNode?.properties || {});
  const provenance = selectedNode?.provenance || {};

  return (
    <aside className="memory-inspector">
      <div className="eyebrow">INSPECTOR</div>
      {selectedNode ? (
        <>
          <strong>{selectedNode.label || selectedNode.id}</strong>
          <span>{selectedNode.type || "Unknown"}</span>
          <dl>
            {properties.map(([name, value]) => (
              <div key={name}><dt>{name}</dt><dd>{formatValue(value)}</dd></div>
            ))}
            <div><dt>投影来源</dt><dd>{formatValue(provenance.source_kind)}</dd></div>
            <div><dt>原始记录</dt><dd>{formatValue(provenance.record_id)}</dd></div>
          </dl>
          <p className="memory-inspector-note">节点仅存在于本次隔离结果中；继续探索请修改 MATCH 或添加有界 EXPAND。</p>
        </>
      ) : (
        <p className="memory-inspector-note">在图中选择节点后，可在这里查看真实属性与来源。</p>
      )}
      <div className="memory-history">
        <div className="eyebrow">RECENT</div>
        {history.length ? history.map((item) => (
          <button type="button" key={item.id} onClick={() => onRestoreQuery(item.query)}>
            <span>{item.id}</span>
            <small>{item.summary}</small>
          </button>
        )) : <span>本次页面尚无查询记录</span>}
      </div>
    </aside>
  );
}

export function MemoryPage({ effectiveTheme, onToast }) {
  const [query, setQuery] = useState(MEMORY_QUERY_EXAMPLES[0].query);
  const [schema, setSchema] = useState(null);
  const [schemaLoading, setSchemaLoading] = useState(true);
  const [schemaError, setSchemaError] = useState("");
  const [schemaMode, setSchemaMode] = useState("types");
  const [selectedSchemaName, setSelectedSchemaName] = useState("");
  const [result, setResult] = useState(null);
  const [resultView, setResultView] = useState("graph");
  const [selectedNodeId, setSelectedNodeId] = useState("");
  const [queryError, setQueryError] = useState(null);
  const [running, setRunning] = useState(false);
  const [history, setHistory] = useState([]);
  const [guideOpen, setGuideOpen] = useState(false);
  const [graphExpanded, setGraphExpanded] = useState(false);
  const [flowNodes, setFlowNodes, onNodesChange] = useNodesState([]);
  const schemaControllerRef = useRef(null);
  const queryControllerRef = useRef(null);
  const guideButtonRef = useRef(null);
  const flowInstanceRef = useRef(null);

  const reloadSchema = useCallback(async () => {
    schemaControllerRef.current?.abort();
    const controller = new AbortController();
    schemaControllerRef.current = controller;
    setSchemaLoading(true);
    setSchemaError("");
    try {
      const nextSchema = await loadMemorySchema({ signal: controller.signal });
      if (schemaControllerRef.current !== controller) return;
      setSchema(nextSchema);
      setSelectedSchemaName((current) => (
        [...nextSchema.types, ...nextSchema.relations].some((item) => item.name === current)
          ? current
          : nextSchema.types[0]?.name || nextSchema.relations[0]?.name || ""
      ));
    } catch (error) {
      if (error?.name !== "AbortError" && schemaControllerRef.current === controller) {
        setSchemaError(error?.message || "无法读取记忆语义结构。 ");
      }
    } finally {
      if (schemaControllerRef.current === controller) setSchemaLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    schemaControllerRef.current = controller;
    loadMemorySchema({ signal: controller.signal })
      .then((nextSchema) => {
        if (schemaControllerRef.current !== controller) return;
        setSchema(nextSchema);
        setSelectedSchemaName(nextSchema.types[0]?.name || nextSchema.relations[0]?.name || "");
      })
      .catch((error) => {
        if (error?.name !== "AbortError" && schemaControllerRef.current === controller) {
          setSchemaError(error?.message || "无法读取记忆语义结构。 ");
        }
      })
      .finally(() => {
        if (schemaControllerRef.current === controller) setSchemaLoading(false);
      });
    return () => {
      controller.abort();
      queryControllerRef.current?.abort();
    };
  }, []);

  const schemaItems = schemaMode === "types" ? schema?.types || [] : schema?.relations || [];
  const selectedSchema = [...(schema?.types || []), ...(schema?.relations || [])]
    .find((item) => item.name === selectedSchemaName);

  const selectSchemaMode = (mode) => {
    setSchemaMode(mode);
    const items = mode === "types" ? schema?.types || [] : schema?.relations || [];
    setSelectedSchemaName(items[0]?.name || "");
  };

  const executeQuery = useCallback(async () => {
    const submittedQuery = query.trim();
    if (!submittedQuery || running) return;
    queryControllerRef.current?.abort();
    const controller = new AbortController();
    queryControllerRef.current = controller;
    setRunning(true);
    setQueryError(null);
    setResult(null);
    setSelectedNodeId("");
    setGraphExpanded(false);
    try {
      const nextResult = await runMemoryQuery(submittedQuery, {
        signal: controller.signal,
        languageVersion: schema?.language.version || "1.0",
        nodeLimit: schema?.limits.nodes || 80,
        edgeLimit: schema?.limits.edges || 120,
        rowLimit: schema?.limits.rows || 100,
        maxDepth: schema?.limits.depth ?? 2,
      });
      if (queryControllerRef.current !== controller) return;
      setResult(nextResult);
      setSelectedNodeId(String(nextResult.nodes[0]?.id || ""));
      setResultView(nextResult.returnKind === "table" ? "table" : nextResult.returnKind === "raw" ? "raw" : "graph");
      setHistory((items) => [{
        id: nextResult.queryId,
        query: submittedQuery,
        summary: `${nextResult.nodes.length} 节点 · ${nextResult.edges.length} 关系`,
      }, ...items].slice(0, 4));
      onToast?.(`查询完成：${nextResult.nodes.length} 个节点、${nextResult.edges.length} 条关系`);
    } catch (error) {
      if (error?.name !== "AbortError" && queryControllerRef.current === controller) {
        setQueryError({ code: error?.code || "query_failed", message: error?.message || "记忆查询失败。" });
      }
    } finally {
      if (queryControllerRef.current === controller) setRunning(false);
    }
  }, [onToast, query, running, schema]);

  const applyExample = (example) => {
    setQuery(example.query);
    setQueryError(null);
    const nextView = /RETURN\s+TABLE/i.test(example.query) ? "table" : /RETURN\s+RAW/i.test(example.query) ? "raw" : "graph";
    setResultView(nextView);
    if (nextView !== "graph") setGraphExpanded(false);
  };

  const closeGuide = () => {
    setGuideOpen(false);
    window.requestAnimationFrame(() => guideButtonRef.current?.focus());
  };

  useEffect(() => {
    setFlowNodes(layoutResultGraph(result?.nodes || [], result?.edges || []));
  }, [result, setFlowNodes]);

  useEffect(() => {
    if (!graphExpanded) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setGraphExpanded(false);
    };
    document.body.classList.add("memory-graph-expanded");
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.classList.remove("memory-graph-expanded");
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [graphExpanded]);

  const flowEdges = useMemo(() => buildFlowEdges(result?.edges || []), [result]);
  const resultQueryId = result?.queryId || "";
  useEffect(() => {
    if (!flowNodes.length || resultView !== "graph") return undefined;
    const timer = window.setTimeout(() => {
      flowInstanceRef.current?.fitView({ padding: graphExpanded ? 0.18 : 0.24, duration: 220 });
    }, graphExpanded ? 120 : 40);
    return () => window.clearTimeout(timer);
  }, [flowNodes.length, graphExpanded, resultQueryId, resultView]);

  const selectedNode = result?.nodes.find((node) => String(node.id) === selectedNodeId) || null;
  const consumed = result?.budget?.consumed || {};
  const effective = result?.budget?.effective || {};
  const nodeBudget = schema?.limits.nodes || 80;
  const resultTabs = [
    ["graph", "图", Network],
    ["table", "表格", BarChart3],
    ["raw", "原始", Code2],
    ["explain", "解释", ScrollText],
  ];

  return (
    <div className="wide-workspace memory-workspace">
      <section className="panel-window schema-panel">
        <div className="panel-header">
          <div><div className="eyebrow">SEMANTIC SCHEMA</div><h3>记忆结构</h3></div>
          <span className="count-chip" title={schema?.schemaVersion || "正在读取版本"}>
            {schema?.language.version ? `v${schema.language.version}` : "—"}
          </span>
        </div>
        <div className="schema-mode-tabs" role="tablist" aria-label="Schema 类型">
          <button type="button" role="tab" aria-selected={schemaMode === "types"} className={schemaMode === "types" ? "active" : ""} onClick={() => selectSchemaMode("types")}>节点类型</button>
          <button type="button" role="tab" aria-selected={schemaMode === "relations"} className={schemaMode === "relations" ? "active" : ""} onClick={() => selectSchemaMode("relations")}>关系类型</button>
        </div>

        {schemaLoading ? (
          <ResourceState icon={RefreshCw} title="正在读取语义结构" detail="只检查版本、类型、关系与可用数量。" />
        ) : schemaError ? (
          <ResourceState title="语义结构暂不可用" detail={schemaError} action={reloadSchema} />
        ) : (
          <>
            <div className="schema-list">
              {schemaItems.map((item) => {
                const Icon = SCHEMA_ICONS[item.name] || Network;
                return (
                  <button
                    type="button"
                    className={`${selectedSchemaName === item.name ? "active" : ""} ${item.available ? "" : "is-unavailable"}`}
                    key={item.name}
                    onClick={() => setSelectedSchemaName(item.name)}
                  >
                    <Icon size={18} />
                    <span><strong>{item.name}</strong><small>{item.label}{item.available ? "" : " · 不可用"}</small></span>
                    <code>{formatCount(item.count)}</code>
                  </button>
                );
              })}
            </div>
            {selectedSchema && (
              <div className="schema-detail">
                <div className="eyebrow">SELECTED</div>
                <strong>{selectedSchema.name}</strong>
                <p>{selectedSchema.description}</p>
                {"properties" in selectedSchema ? (
                  <code>{selectedSchema.properties.map((property) => `${property.name}: ${property.type}`).join(" · ") || "当前版本未公开属性"}</code>
                ) : (
                  <code>{selectedSchema.source} → {selectedSchema.target}</code>
                )}
              </div>
            )}
          </>
        )}
        <div className={`schema-note ${schema?.compatibility.status === "degraded" ? "is-degraded" : ""}`}>
          {schema?.compatibility.status === "degraded" ? <AlertCircle size={15} /> : <ShieldCheck size={15} />}
          <span>{schema?.compatibility.message || "只读语义层，与物理 SQLite 表解耦"}</span>
        </div>
      </section>

      <section className="panel-window query-panel">
        <div className="panel-header">
          <div><div className="eyebrow">MEMORYQL · {schema?.language.version || "1.0"}</div><h3>有限子图查询</h3></div>
          <div className="query-panel-actions">
            <button
              ref={guideButtonRef}
              type="button"
              className="memory-doc-trigger"
              aria-haspopup="dialog"
              aria-controls="memoryql-guide-dialog"
              aria-expanded={guideOpen}
              title="打开 MemoryQL 语法说明书"
              onClick={() => setGuideOpen(true)}
            >
              <BookOpen size={15} /> DOC
            </button>
            <span className="online-chip"><span /><b className="budget-label-full">{formatCount(consumed.nodes)} / {formatCount(effective.nodes || nodeBudget)} 节点</b><b className="budget-label-short">{formatCount(consumed.nodes)} / {formatCount(effective.nodes || nodeBudget)}</b></span>
          </div>
        </div>

        <div className="query-presets" aria-label="查询示例">
          <span><Clock3 size={14} /> 示例</span>
          {MEMORY_QUERY_EXAMPLES.map((example) => (
            <button type="button" key={example.label} disabled={running} onClick={() => applyExample(example)}>{example.label}</button>
          ))}
        </div>

        <div className="query-editor">
          <textarea
            value={query}
            maxLength={schema?.limits.queryCharacters || 8000}
            onChange={(event) => setQuery(event.target.value)}
            aria-label="记忆查询"
            spellCheck="false"
          />
          <div className="query-editor-footer">
            <span>只读 · Schema {schema?.schemaVersion || "读取中"} · 最大深度 {schema?.limits.depth ?? 2}</span>
            <button type="button" onClick={executeQuery} disabled={running || !query.trim() || schemaLoading || Boolean(schemaError)}>
              {running ? <RefreshCw className="spin" size={16} /> : <Eye size={16} />}
              {running ? "执行中…" : "运行查询"}
            </button>
          </div>
        </div>

        {graphExpanded && <div className="memory-result-backdrop" aria-hidden="true" />}
        <div className={`memory-result-workbench ${graphExpanded ? "is-expanded" : ""}`}>
          <div className="memory-result-toolbar">
            <div>
              <div className="eyebrow">ISOLATED RESULT</div>
              <strong>{result ? result.queryId : "尚未运行"}</strong>
              <span>{result ? `${result.nodes.length} 节点 · ${result.edges.length} 关系 · ${formatValue(consumed.elapsed_ms)}ms${result.truncated ? " · 已截断" : ""}` : "每次查询都会生成独立结果集"}</span>
            </div>
            <div className="memory-result-toolbar-actions">
              <div className="memory-result-tabs" role="tablist" aria-label="结果视图">
                {resultTabs.map(([id, label, Icon]) => (
                  <button
                    key={id}
                    type="button"
                    role="tab"
                    disabled={!result}
                    aria-selected={resultView === id}
                    className={resultView === id ? "active" : ""}
                    onClick={() => {
                      setResultView(id);
                      if (id !== "graph") setGraphExpanded(false);
                    }}
                  >
                    <Icon size={14} /> {label}
                  </button>
                ))}
              </div>
              {result && resultView === "graph" && (
                <button
                  type="button"
                  className="graph-expand-button"
                  aria-label={graphExpanded ? "还原记忆图窗口" : "放大记忆图窗口"}
                  aria-pressed={graphExpanded}
                  onClick={() => setGraphExpanded((expanded) => !expanded)}
                >
                  {graphExpanded ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
                  {graphExpanded ? "还原" : "放大"}
                </button>
              )}
            </div>
          </div>

          {running ? (
            <div className="memory-result-loading" role="status"><RefreshCw className="spin" size={20} /> 正在校验语法并生成有限结果集…</div>
          ) : queryError ? (
            <ResourceState title="查询未执行" detail={`${queryError.message}（${queryError.code}）`} action={executeQuery} actionLabel="重新运行" />
          ) : !result ? (
            <ResourceState icon={Network} title="等待查询" detail="载入示例或编写 MemoryQL；完整实例图不会在后台自动加载。" />
          ) : (
            <div className="memory-result-body">
              <div className="memory-result-surface">
                {resultView === "graph" && (flowNodes.length ? (
                  <div className="memory-graph" aria-label="查询结果图">
                    <ReactFlow
                      key={result.queryId}
                      nodes={flowNodes}
                      edges={flowEdges}
                      nodeTypes={MEMORY_NODE_TYPES}
                      colorMode={effectiveTheme}
                      onInit={(instance) => { flowInstanceRef.current = instance; }}
                      onNodesChange={onNodesChange}
                      fitView
                      fitViewOptions={{ padding: 0.24 }}
                      minZoom={0.22}
                      maxZoom={2.2}
                      deleteKeyCode={null}
                      nodesConnectable={false}
                      onNodeClick={(_, node) => setSelectedNodeId(node.id)}
                    >
                      <Background gap={22} size={1} />
                      <MiniMap
                        pannable
                        zoomable
                        ariaLabel="记忆查询结果缩略图"
                        bgColor="var(--surface-muted)"
                        maskColor="color-mix(in srgb, var(--surface-solid) 62%, transparent)"
                        maskStrokeColor="var(--border-strong)"
                        nodeBorderRadius={5}
                        nodeStrokeWidth={3}
                        nodeColor={(node) => NODE_COLORS[node.data?.kind] || "#71807d"}
                        nodeStrokeColor={(node) => NODE_COLORS[node.data?.kind] || "#71807d"}
                      />
                      <Controls showInteractive={false} />
                    </ReactFlow>
                  </div>
                ) : <ResourceState icon={Network} title="未命中节点" detail="查询已完成，但当前条件下没有可投影到图中的节点。" />)}

                {resultView === "table" && (result.table.columns.length && result.table.rows.length ? (
                  <div className="memory-table-wrap">
                    <table>
                      <thead><tr>{result.table.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
                      <tbody>{result.table.rows.map((row, rowIndex) => (
                        <tr key={`${result.queryId}-${rowIndex}`}>
                          {result.table.columns.map((column) => <td key={column}>{formatValue(row[column])}</td>)}
                        </tr>
                      ))}</tbody>
                    </table>
                  </div>
                ) : <ResourceState icon={BarChart3} title="没有表格行" detail="查询已完成，但没有命中符合条件的记录。" />)}

                {resultView === "raw" && <pre className="memory-raw">{JSON.stringify(result, null, 2)}</pre>}

                {resultView === "explain" && (
                  <div className="memory-explain">
                    <ol>
                      {(Array.isArray(result.explain.plan) ? result.explain.plan : []).map((step, index) => (
                        <li key={`${index}-${step}`}><strong>步骤 {index + 1}</strong><span>{String(step)}</span></li>
                      ))}
                    </ol>
                    {Array.isArray(result.explain.warnings) && result.explain.warnings.length > 0 && (
                      <div className="memory-explain-warnings"><AlertCircle size={16} /> {result.explain.warnings.join("；")}</div>
                    )}
                    <details>
                      <summary>查看已校验 AST</summary>
                      <pre>{JSON.stringify(result.explain.ast || {}, null, 2)}</pre>
                    </details>
                  </div>
                )}
              </div>

              <ResultInspector
                selectedNode={selectedNode}
                history={history}
                onRestoreQuery={(historyQuery) => {
                  setQuery(historyQuery);
                  setQueryError(null);
                }}
              />
            </div>
          )}
        </div>
      </section>
      <MemoryQueryGuide open={guideOpen} onClose={closeGuide} onUseExample={applyExample} />
    </div>
  );
}
