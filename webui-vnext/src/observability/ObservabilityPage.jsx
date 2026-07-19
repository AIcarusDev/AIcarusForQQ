import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  Check,
  ChevronDown,
  ChevronUp,
  CircleAlert,
  Clock3,
  Gauge,
  Layers3,
  ListFilter,
  RefreshCw,
  Search,
  ShieldCheck,
  Trash2,
  TrendingUp,
} from "lucide-react";
import { loadObservability, subscribeLogs } from "../api/observabilityApi.js";
import { RANGE_OPTIONS } from "./observabilityData.js";

const PAGE_TITLES = {
  logs: "运行日志",
  tools: "工具统计",
  tokens: "Token 用量",
};

const PAGE_CONFIG = {
  tools: {
    eyebrow: "TOOL DIAGNOSTICS",
    chartTitle: "调用频率趋势",
    chartDescription: "按采样窗口堆叠显示真实调用量，可隐藏单个工具检查构成变化。",
    unitLabel: "次调用",
  },
  tokens: {
    eyebrow: "TOKEN DIAGNOSTICS",
    chartTitle: "Token 消耗趋势",
    chartDescription: "按处理环节分解真实消耗，未知 usage 请求不会混入 Token 总量。",
    unitLabel: "Token",
  },
};

function isLongLog(message) {
  const text = String(message || "");
  return text.length > 520 || text.split(/\r?\n/).length > 6;
}

function formatCount(value) {
  return new Intl.NumberFormat("zh-CN").format(Math.round(Number(value) || 0));
}

function formatCompact(value) {
  const number = Number(value) || 0;
  const absolute = Math.abs(number);
  if (absolute >= 1_000_000) {
    return `${(number / 1_000_000).toFixed(absolute >= 10_000_000 ? 1 : 2).replace(/\.0+$/, "")}M`;
  }
  if (absolute >= 1_000) {
    return `${(number / 1_000).toFixed(absolute >= 100_000 ? 1 : 2).replace(/\.0+$/, "")}K`;
  }
  return formatCount(number);
}

function formatMeasure(page, value) {
  return page === "tokens" ? formatCompact(value) : formatCount(value);
}

function formatLatency(value) {
  const milliseconds = Number(value) || 0;
  if (!milliseconds) return "—";
  if (milliseconds < 1000) return `${Math.round(milliseconds)}ms`;
  return `${(milliseconds / 1000).toFixed(milliseconds >= 10_000 ? 1 : 2).replace(/\.0+$/, "")}s`;
}

function formatUpdatedAt(timestamp) {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date(timestamp));
}

function selectedAnalytics(data, visibleSeries) {
  const series = data.series.filter((item) => visibleSeries.includes(item.id));
  if (!data.rows.length || !series.length) return { series, peak: null };
  const peak = data.rows.reduce((currentPeak, row) => {
    const total = series.reduce((sum, item) => sum + Number(row[item.id] || 0), 0);
    return !currentPeak || total > currentPeak.total ? { ...row, total } : currentPeak;
  }, null);
  return { series, peak };
}

function AnalyticsPage({ page }) {
  const config = PAGE_CONFIG[page];
  const [range, setRange] = useState("24h");
  const [reloadKey, setReloadKey] = useState(0);
  const [visibleSeries, setVisibleSeries] = useState([]);
  const [resource, setResource] = useState({
    status: "loading",
    data: null,
    error: null,
  });

  useEffect(() => {
    const controller = new AbortController();
    loadObservability(page, range, { signal: controller.signal })
      .then((data) => {
        setResource({ status: "ready", data, error: null });
        setVisibleSeries(data.series.map((item) => item.id));
      })
      .catch((error) => {
        if (error?.name !== "AbortError") {
          setResource({ status: "error", data: null, error });
        }
      });

    return () => controller.abort();
  }, [page, range, reloadKey]);

  const data = resource.data;
  const analytics = useMemo(
    () => data ? selectedAnalytics(data, visibleSeries) : { series: [], peak: null },
    [data, visibleSeries],
  );

  const toggleSeries = (id) => {
    setVisibleSeries((current) => {
      if (current.includes(id) && current.length === 1) return current;
      return current.includes(id)
        ? current.filter((item) => item !== id)
        : [...current, id];
    });
  };

  const refresh = () => {
    setResource((current) => ({ status: "loading", data: current.data, error: null }));
    setReloadKey((value) => value + 1);
  };
  const selectRange = (nextRange) => {
    if (nextRange === range) return;
    setResource((current) => ({ status: "loading", data: current.data, error: null }));
    setRange(nextRange);
  };
  const empty = resource.status === "ready" && (!data?.series.length || !data?.rows.length);

  return (
    <div className="observability-page">
      <section className="panel-window observability-overview">
        <div className="panel-header observability-page-header">
          <div>
            <div className="eyebrow">OBSERVABILITY</div>
            <h3>{PAGE_TITLES[page]}</h3>
            <p>
              {page === "tokens"
                ? "从时间变化定位消耗高点，再下钻到具体处理环节。"
                : "从调用趋势观察频率、延迟和成功率变化。"}
            </p>
          </div>
          <button
            className="quiet-button"
            type="button"
            onClick={refresh}
            disabled={resource.status === "loading"}
          >
            <RefreshCw className={resource.status === "loading" ? "spin" : ""} size={15} />
            {resource.status === "loading" ? "刷新中" : "刷新"}
          </button>
        </div>
        <div className="observability-controls">
          <div className="range-tabs" aria-label="统计时间范围">
            {RANGE_OPTIONS.map(([key, label]) => (
              <button
                className={range === key ? "active" : ""}
                key={key}
                type="button"
                aria-pressed={range === key}
                onClick={() => selectRange(key)}
              >
                {label}
              </button>
            ))}
          </div>
          <span>
            {data?.generatedAt
              ? `真实数据 · 更新于 ${formatUpdatedAt(data.generatedAt)}`
              : "正在连接统计服务"}
          </span>
        </div>
      </section>

      {resource.status === "error" && (
        <ResourceState
          icon={CircleAlert}
          title="统计数据暂时不可用"
          detail={resource.error?.message || "后端没有返回可用数据。"}
          action="重新加载"
          onAction={refresh}
        />
      )}

      {resource.status === "loading" && !data && (
        <ResourceState
          loading
          icon={RefreshCw}
          title="正在读取统计数据"
          detail="图表会在真实契约返回后显示。"
        />
      )}

      {empty && (
        <ResourceState
          icon={Activity}
          title="当前范围内还没有记录"
          detail="这是正常的空数据状态；产生调用后可在这里查看趋势。"
          action="重新检查"
          onAction={refresh}
        />
      )}

      {data && !empty && resource.status !== "error" && (
        <>
          <SummaryGrid page={page} data={data} />

          <section className="panel-window trend-panel">
            <div className="trend-panel-header">
              <div>
                <div className="eyebrow">{config.eyebrow}</div>
                <h3>{config.chartTitle}</h3>
                <p>{config.chartDescription}</p>
              </div>
            </div>

            <div className="series-toggles" aria-label="图表系列筛选">
              {data.series.map((series) => {
                const active = visibleSeries.includes(series.id);
                const locked = active && visibleSeries.length === 1;
                return (
                  <button
                    className={active ? "active" : ""}
                    key={series.id}
                    type="button"
                    aria-pressed={active}
                    aria-disabled={locked}
                    onClick={() => toggleSeries(series.id)}
                    style={{ "--series-color": series.color }}
                  >
                    <span /> {series.label}
                  </button>
                );
              })}
            </div>

            <div className="trend-workbench">
              <div
                className="usage-chart"
                role="img"
                aria-label={`${config.chartTitle}，当前显示 ${analytics.series.length} 个系列`}
              >
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={data.rows}
                    margin={{ top: 14, right: 16, bottom: 2, left: 0 }}
                    accessibilityLayer
                  >
                    <CartesianGrid vertical={false} stroke="var(--chart-grid)" strokeDasharray="3 5" />
                    <XAxis
                      dataKey="label"
                      axisLine={false}
                      tickLine={false}
                      minTickGap={24}
                      tick={{ fill: "var(--muted)", fontSize: 12, fontFamily: "IBM Plex Mono" }}
                    />
                    <YAxis
                      axisLine={false}
                      tickLine={false}
                      width={52}
                      tickFormatter={(value) => formatMeasure(page, value)}
                      tick={{ fill: "var(--muted)", fontSize: 12, fontFamily: "IBM Plex Mono" }}
                    />
                    <Tooltip
                      cursor={{ stroke: "var(--border-strong)", strokeDasharray: "3 3" }}
                      content={<UsageTooltip page={page} unitLabel={config.unitLabel} />}
                    />
                    {analytics.peak && (
                      <ReferenceLine
                        x={analytics.peak.label}
                        stroke="var(--primary)"
                        strokeDasharray="4 5"
                        strokeOpacity={0.55}
                      />
                    )}
                    {analytics.series.map((series) => (
                      <Area
                        key={series.id}
                        type="monotone"
                        dataKey={series.id}
                        name={series.label}
                        stackId="usage"
                        stroke={series.color}
                        fill={series.color}
                        fillOpacity={0.16}
                        strokeWidth={2}
                        activeDot={{ r: 4, strokeWidth: 2, fill: "var(--surface-solid)" }}
                        isAnimationActive={false}
                      />
                    ))}
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              <PeakInspector page={page} analytics={analytics} />
            </div>
          </section>

          <BreakdownTable page={page} data={data} range={range} />
        </>
      )}
    </div>
  );
}

function ResourceState({ action, detail, icon: Icon, loading = false, onAction, title }) {
  return (
    <section className="panel-window observability-resource-state" role={loading ? "status" : "alert"}>
      <span className="resource-state-icon">
        <Icon className={loading ? "spin" : ""} size={21} />
      </span>
      <div>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>
      {action && (
        <button className="quiet-button" type="button" onClick={onAction}>
          <RefreshCw size={15} /> {action}
        </button>
      )}
    </section>
  );
}

function SummaryGrid({ page, data }) {
  let summaries;
  if (page === "tools") {
    const failedCalls = data.series.reduce((sum, item) => sum + Number(item.raw.failed || 0), 0);
    summaries = [
      { label: "总调用", value: formatCount(data.summary.totalCalls), note: "来自持久化工具事件", Icon: Activity },
      { label: "活跃工具", value: String(data.summary.activeSeries), note: `当前展示 ${data.series.length} 个系列`, Icon: Layers3 },
      { label: "P95 延迟", value: formatLatency(data.summary.p95ElapsedMs), note: "仅统计具有有效耗时的调用", Icon: Clock3 },
      {
        label: "成功率",
        value: data.summary.successRate === null ? "—" : `${(data.summary.successRate * 100).toFixed(1)}%`,
        note: `失败调用 ${formatCount(failedCalls)} 次`,
        Icon: ShieldCheck,
      },
    ];
  } else {
    summaries = [
      { label: "总消耗", value: formatCompact(data.summary.totalTokens), note: `${formatCount(data.summary.knownRequests)} 次已知用量请求`, Icon: Gauge },
      { label: "输入 Token", value: formatCompact(data.summary.inputTokens), note: `缓存输入 ${formatCompact(data.summary.cachedInputTokens)}`, Icon: Layers3 },
      { label: "输出 Token", value: formatCompact(data.summary.outputTokens), note: `推理输出 ${formatCompact(data.summary.reasoningOutputTokens)}`, Icon: Activity },
      { label: "未知用量请求", value: formatCount(data.summary.unknownRequests), note: "未计入 Token 总量", Icon: CircleAlert },
    ];
  }

  return (
    <div className="observability-summary-grid">
      {summaries.map(({ label, value, note, Icon }) => (
        <article className="observability-stat" key={label}>
          <div><span>{label}</span><Icon size={16} /></div>
          <strong>{value}</strong>
          <small>{note}</small>
        </article>
      ))}
    </div>
  );
}

function UsageTooltip({ active, label, page, payload, unitLabel }) {
  if (!active || !payload?.length) return null;
  const total = payload.reduce((sum, item) => sum + Number(item.value || 0), 0);

  return (
    <div className="chart-tooltip">
      <div><strong>{label}</strong><span>{formatMeasure(page, total)} {unitLabel}</span></div>
      {payload.slice().reverse().map((item) => (
        <p key={item.dataKey}>
          <span><i style={{ background: item.color }} />{item.name}</span>
          <strong>{formatMeasure(page, item.value)}</strong>
        </p>
      ))}
    </div>
  );
}

function PeakInspector({ page, analytics }) {
  const peak = analytics.peak;
  const breakdown = peak
    ? analytics.series
      .map((series) => ({ ...series, value: Number(peak[series.id] || 0) }))
      .sort((left, right) => right.value - left.value)
    : [];

  return (
    <aside className="peak-inspector" aria-label="峰值详情">
      <div className="peak-inspector-heading">
        <span><TrendingUp size={15} /></span>
        <div><small>PEAK WINDOW</small><strong>{peak?.label || "暂无"}</strong></div>
      </div>
      <strong>{page === "tokens" ? "Token 用量高点" : "工具调用高点"}</strong>
      <p>
        {peak
          ? `当前筛选系列在此窗口合计 ${formatMeasure(page, peak.total)}。`
          : "当前范围内没有可比较的采样窗口。"}
      </p>
      <div className="peak-breakdown">
        {breakdown.map((series) => (
          <div key={series.id}>
            <span><i style={{ background: series.color }} />{series.label}</span>
            <strong>{formatMeasure(page, series.value)}</strong>
            <div>
              <span
                style={{
                  width: `${peak.total ? (series.value / peak.total) * 100 : 0}%`,
                  background: series.color,
                }}
              />
            </div>
          </div>
        ))}
      </div>
      <small className="diagnostic-hint">
        高点仅表示当前范围内的最大值，不直接代表异常。
      </small>
    </aside>
  );
}

function BreakdownTable({ page, data, range }) {
  return (
    <section className="panel-window breakdown-panel">
      <div className="breakdown-heading">
        <div>
          <div className="eyebrow">BREAKDOWN</div>
          <h3>{page === "tokens" ? "环节用量明细" : "工具健康明细"}</h3>
        </div>
        <span>{RANGE_OPTIONS.find(([key]) => key === range)?.[1]}累计</span>
      </div>
      <div className="breakdown-table-wrap">
        <table>
          <thead>
            <tr>
              <th>{page === "tokens" ? "处理环节" : "工具"}</th>
              {page === "tokens" ? (
                <><th>输入</th><th>输出</th><th>总计</th><th>未知请求</th></>
              ) : (
                <><th>调用次数</th><th>平均耗时</th><th>P95</th><th>成功率</th></>
              )}
            </tr>
          </thead>
          <tbody>
            {data.series.map((series) => {
              const item = series.raw;
              const toolTotal = Number(item.total || 0);
              const toolSuccessRate = toolTotal ? Number(item.success || 0) / toolTotal : null;
              return (
                <tr key={series.id}>
                  <td><i style={{ background: series.color }} />{series.label}</td>
                  {page === "tokens" ? (
                    <>
                      <td>{formatCompact(item.input_tokens)}</td>
                      <td>{formatCompact(item.output_tokens)}</td>
                      <td><strong>{formatCompact(item.total_tokens)}</strong></td>
                      <td>{formatCount(item.unknown_requests)}</td>
                    </>
                  ) : (
                    <>
                      <td><strong>{formatCount(toolTotal)}</strong></td>
                      <td>{formatLatency(item.avg_elapsed_ms)}</td>
                      <td>{formatLatency(item.p95_elapsed_ms)}</td>
                      <td>{toolSuccessRate === null ? "—" : `${(toolSuccessRate * 100).toFixed(1)}%`}</td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function LogPage() {
  const [records, setRecords] = useState([]);
  const [connection, setConnection] = useState("connecting");
  const [level, setLevel] = useState("ALL");
  const [query, setQuery] = useState("");
  const [sourceFilterOpen, setSourceFilterOpen] = useState(false);
  const [selectedSources, setSelectedSources] = useState(() => new Set());
  const [expandedRecords, setExpandedRecords] = useState(() => new Set());
  const [error, setError] = useState(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    const stream = subscribeLogs({
      signal: controller.signal,
      onRecords: (incoming) => setRecords((current) => {
        const map = new Map(current.map((record) => [record.seq, record]));
        for (const record of incoming) map.set(record.seq, record);
        return [...map.values()].sort((left, right) => left.seq - right.seq).slice(-1_200);
      }),
      onStatus: setConnection,
      onError: setError,
    });
    return () => { controller.abort(); stream.close(); };
  }, [reloadKey]);

  const reconnect = () => {
    setConnection("connecting");
    setError(null);
    setReloadKey((value) => value + 1);
  };

  const sourceOptions = useMemo(() => {
    const counts = new Map();
    for (const record of records) {
      const source = String(record.source || record.file || "未标记模块");
      counts.set(source, (counts.get(source) || 0) + 1);
    }
    return [...counts.entries()]
      .map(([source, count]) => ({ source, count }))
      .sort((left, right) => right.count - left.count || left.source.localeCompare(right.source, "zh-CN"));
  }, [records]);

  const visible = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase("zh-CN");
    return records.filter((record) => {
      const levelMatches = level === "ALL"
        || (level === "ERROR" ? ["ERROR", "CRITICAL"].includes(record.level) : record.level === level);
      if (!levelMatches) return false;
      const source = String(record.source || record.file || "未标记模块");
      if (selectedSources.size && !selectedSources.has(source)) return false;
      if (!needle) return true;
      return `${record.source} ${record.message} ${record.file}`.toLocaleLowerCase("zh-CN").includes(needle);
    }).slice().reverse();
  }, [level, query, records, selectedSources]);

  const toggleSource = (source) => {
    setSelectedSources((current) => {
      const next = new Set(current);
      if (next.has(source)) next.delete(source);
      else next.add(source);
      return next;
    });
  };

  const toggleRecord = (seq) => {
    setExpandedRecords((current) => {
      const next = new Set(current);
      if (next.has(seq)) next.delete(seq);
      else next.add(seq);
      return next;
    });
  };

  const clearView = () => {
    setRecords([]);
    setExpandedRecords(new Set());
  };

  const statusLabel = connection === "live" ? "实时" : connection === "reconnecting" ? "正在重连" : "正在连接";
  return (
    <div className="logs-workspace">
      <section className="log-toolbar panel-window">
        <div className="log-toolbar-main">
          <div><div className="eyebrow">LIVE RUNTIME</div><h3>实时日志</h3></div>
          <span className={`connection-chip ${connection === "live" ? "live" : "offline"}`}><span />{statusLabel}</span>
        </div>
        <div className="log-controls">
          <div className="segmented-control" aria-label="日志级别">
            {["ALL", "DEBUG", "INFO", "WARNING", "ERROR"].map((item) => <button key={item} type="button" className={level === item ? "active" : ""} onClick={() => setLevel(item)}>{item === "WARNING" ? "WARN" : item}</button>)}
          </div>
          <button
            type="button"
            className={`log-source-toggle ${sourceFilterOpen ? "is-open" : ""} ${selectedSources.size ? "has-filter" : ""}`}
            aria-expanded={sourceFilterOpen}
            aria-controls="log-source-filter"
            onClick={() => setSourceFilterOpen((open) => !open)}
          >
            <ListFilter size={15} />
            模块
            {selectedSources.size > 0 && <span>{selectedSources.size}</span>}
          </button>
          <label className="search-box"><Search size={15} /><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索模块或消息" /></label>
          <button type="button" className="secondary-button" onClick={clearView}><Trash2 size={14} />清空视图</button>
        </div>
        {sourceFilterOpen && (
          <div className="log-source-filter" id="log-source-filter">
            <div className="log-source-filter-head">
              <div><strong>模块 / 函数</strong><span>可多选；未选择时显示全部来源</span></div>
              <button type="button" disabled={!selectedSources.size} onClick={() => setSelectedSources(new Set())}>重置为全部</button>
            </div>
            <div className="log-source-options">
              {sourceOptions.length ? sourceOptions.map(({ source, count }) => {
                const selected = selectedSources.has(source);
                return (
                  <button key={source} type="button" aria-pressed={selected} className={selected ? "active" : ""} onClick={() => toggleSource(source)}>
                    <span className="source-filter-check">{selected && <Check size={13} />}</span>
                    <strong title={source}>{source}</strong>
                    <small>{count}</small>
                  </button>
                );
              }) : <span className="log-source-empty">收到日志后会在这里列出模块。</span>}
            </div>
          </div>
        )}
        <div className="log-meta"><span>缓冲 {records.length} 条</span><span>当前显示 {visible.length} 条</span>{selectedSources.size > 0 && <span>已筛选 {selectedSources.size} 个模块</span>}{records.length >= 1_200 && <span>仅保留最近 1,200 条</span>}</div>
      </section>
      {error && <div className="inline-resource-state panel-window" role="alert"><CircleAlert size={18} /><div><strong>部分实时数据无法解析</strong><p>{error.message}</p></div><button type="button" className="secondary-button" onClick={reconnect}>重新连接</button></div>}
      <section className="log-stream panel-window" aria-live="polite">
        {visible.length ? visible.map((record) => {
          const message = record.message || "（空日志）";
          const long = isLongLog(message);
          const expanded = expandedRecords.has(record.seq);
          const messageId = `log-message-${record.seq}`;
          return (
            <article className={`log-record level-${record.level}`} key={record.seq}>
              <time>{record.timestamp || "—"}</time>
              <span className="log-level">{record.level === "WARNING" ? "WARN" : record.level}</span>
              <div>
                <header><strong>{record.source}</strong><small>#{record.seq}</small></header>
                <pre id={messageId} className={`log-record-message ${long && !expanded ? "is-collapsed" : ""}`}>{message}</pre>
                {long && (
                  <button
                    type="button"
                    className="log-record-toggle"
                    aria-expanded={expanded}
                    aria-controls={messageId}
                    aria-label={`${expanded ? "收起" : "展开"}日志 #${record.seq}`}
                    onClick={() => toggleRecord(record.seq)}
                  >
                    {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    {expanded ? "收起" : "展开全文"}
                  </button>
                )}
                {record.file && <small>{record.file}{record.line ? `:${record.line}` : ""}</small>}
              </div>
            </article>
          );
        }) : (
          <ResourceState icon={Activity} title={records.length ? "没有匹配的日志" : "等待运行日志"} detail={records.length ? "调整级别或搜索条件后再试。" : "服务端产生新记录后会自动显示。"} />
        )}
      </section>
    </div>
  );
}

export function ObservabilityPage({ page }) {
  return page === "logs" ? <LogPage /> : <AnalyticsPage key={page} page={page} />;
}
