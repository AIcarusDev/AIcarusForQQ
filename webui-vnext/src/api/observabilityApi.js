import { requestV1Data } from "./http.js";
import { openCursorStream } from "./realtime.js";

const RANGE_QUERY = {
  "24h": { range: "24h", granularity: "hour" },
  "7d": { range: "7d", granularity: "day" },
  "30d": { range: "30d", granularity: "day" },
};

const SERIES_COLORS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "#8b6f9f",
  "#b2794e",
  "#5f7f9d",
  "#7f8b61",
];

const FEATURE_LABELS = {
  legacy: "旧版记录",
  memory: "记忆处理",
  main_round: "主模型",
  main_round_retry_no_tool: "主模型重试（无工具）",
  cognition_compression: "认知压缩",
  memory_event_extraction: "记忆提取",
  memory_processing: "记忆处理",
  memory_recall: "记忆召回",
  slow_thinking: "慢思考",
  tool_guard: "工具守门",
  compression: "上下文压缩",
  vision: "视觉理解",
};

function featureLabel(feature) {
  return FEATURE_LABELS[feature] || feature || "unknown";
}

function labelForBucket(timestamp, range) {
  const date = new Date(timestamp);
  if (range === "24h") {
    return new Intl.DateTimeFormat("zh-CN", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(date);
  }
  if (range === "7d") {
    return new Intl.DateTimeFormat("zh-CN", { weekday: "short" }).format(date);
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}

function timezoneOffsetMinutes() {
  return -new Date().getTimezoneOffset();
}

function seriesIdentity(items, kind) {
  return items.map((item, index) => ({
    id: `series_${index}`,
    label: kind === "tokens" ? featureLabel(item.feature) : item.name,
    source: kind === "tokens" ? item.feature : item.name,
    color: SERIES_COLORS[index % SERIES_COLORS.length],
    raw: item,
  }));
}

function buildRows(series, range, valueKey) {
  const bucketStarts = series[0]?.raw?.points?.map((point) => point.bucket_start) || [];
  return bucketStarts.map((bucketStart, pointIndex) => {
    const row = {
      bucketStart,
      label: labelForBucket(bucketStart, range),
    };
    for (const item of series) {
      row[item.id] = Number(item.raw.points?.[pointIndex]?.[valueKey] || 0);
    }
    return row;
  });
}

function peakFromRows(rows, series) {
  if (!rows.length) return null;
  return rows.reduce((peak, row) => {
    const total = series.reduce((sum, item) => sum + Number(row[item.id] || 0), 0);
    return !peak || total > peak.total ? { ...row, total } : peak;
  }, null);
}

function adaptTools(payload, range) {
  const series = seriesIdentity(Array.isArray(payload?.tools) ? payload.tools : [], "tools");
  const rows = buildRows(series, range, "total");
  const totalSuccess = series.reduce((sum, item) => sum + Number(item.raw.success || 0), 0);
  const totalFailed = series.reduce((sum, item) => sum + Number(item.raw.failed || 0), 0);
  const totalCalls = totalSuccess + totalFailed;

  return {
    kind: "tools",
    generatedAt: Number(payload?.generated_at || Date.now()),
    range,
    rows,
    series,
    peak: peakFromRows(rows, series),
    summary: {
      totalCalls: Number(payload?.summary?.total_calls || totalCalls),
      activeSeries: series.filter((item) => Number(item.raw.total || 0) > 0).length,
      p95ElapsedMs: Number(payload?.summary?.p95_elapsed_ms || 0),
      successRate: totalCalls ? totalSuccess / totalCalls : null,
    },
  };
}

function adaptTokens(payload, range) {
  const rawSeries = Array.isArray(payload?.series) ? payload.series : [];
  const series = seriesIdentity(rawSeries, "tokens");
  const rows = buildRows(series, range, "total_tokens");
  const aggregate = rawSeries.reduce((result, item) => ({
    inputTokens: result.inputTokens + Number(item.input_tokens || 0),
    outputTokens: result.outputTokens + Number(item.output_tokens || 0),
    cachedInputTokens: result.cachedInputTokens + Number(item.cached_input_tokens || 0),
    reasoningOutputTokens: result.reasoningOutputTokens + Number(item.reasoning_output_tokens || 0),
  }), {
    inputTokens: 0,
    outputTokens: 0,
    cachedInputTokens: 0,
    reasoningOutputTokens: 0,
  });

  return {
    kind: "tokens",
    generatedAt: Number(payload?.generated_at || Date.now()),
    range,
    rows,
    series,
    peak: peakFromRows(rows, series),
    summary: {
      totalTokens: Number(payload?.summary?.total_tokens || 0),
      totalRequests: Number(payload?.summary?.total_requests || 0),
      knownRequests: Number(payload?.summary?.known_requests || 0),
      unknownRequests: Number(payload?.summary?.unknown_requests || 0),
      ...aggregate,
    },
  };
}

export async function loadObservability(kind, range, { signal } = {}) {
  const query = RANGE_QUERY[range] || RANGE_QUERY["24h"];
  const params = new URLSearchParams({
    range: query.range,
    granularity: query.granularity,
    tz_offset_minutes: String(timezoneOffsetMinutes()),
  });

  if (kind === "tokens") {
    params.set("group_by", "feature");
    const payload = await requestV1Data(`/api/ui/v1/observability/tokens?${params}`, { signal });
    return adaptTokens(payload, range);
  }

  params.set("limit", "8");
  const payload = await requestV1Data(`/api/ui/v1/observability/tools?${params}`, { signal });
  return adaptTools(payload, range);
}

function normalizeLogRecord(record) {
  const level = String(record?.level || "INFO").toUpperCase();
  const source = String(record?.name || record?.module || "runtime");
  return {
    seq: Math.max(0, Number(record?.seq) || 0),
    level,
    source,
    message: String(record?.message || ""),
    timestamp: String(record?.timestamp || record?.time || ""),
    file: String(record?.file || ""),
    line: Math.max(0, Number(record?.lineno) || 0),
  };
}

export function subscribeLogs({ initialCursor = 0, onRecords, onStatus, onError, signal }) {
  return openCursorStream({
    path: "/log/ws/log",
    initialCursor,
    signal,
    selectItems: (envelope) => envelope?.type === "snapshot" ? envelope.records : [envelope],
    onItems: (items) => onRecords?.(items.map(normalizeLogRecord)),
    onStatus,
    onError,
  });
}
