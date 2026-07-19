const TOOL_EVENT_TYPES = new Set([
  "tool_planned",
  "tool_started",
  "tool_progress",
  "tool_guard",
  "tool_blocked",
  "tool_skipped",
  "tool_error",
  "tool_finished",
]);

const COGNITION_EVENT_TYPES = new Set([
  "cognition_start",
  "cognition_delta",
  "cognition_end",
  "cognition_final",
  "cognition_discarded",
]);

const TERMINAL_TOOL_STATES = new Set(["done", "blocked", "skipped", "error"]);

export const AGENT_EVENT_FILTERS = [
  ["all", "全部"],
  ["cognition", "认知"],
  ["tools", "工具"],
  ["runtime", "运行"],
];

export function agentEventCategory(type) {
  if (TOOL_EVENT_TYPES.has(type)) return "tools";
  if (COGNITION_EVENT_TYPES.has(type)) return "cognition";
  return "runtime";
}

export function agentEventLabel(type) {
  return {
    round_start: "轮次开始",
    round_started: "轮次开始",
    world_frame: "世界帧就绪",
    model_request: "请求模型",
    cognition_start: "开始认知",
    cognition_delta: "认知片段",
    cognition_end: "认知完成",
    cognition_final: "认知定稿",
    cognition_discarded: "认知已丢弃",
    action_start: "开始行动",
    action_end: "行动完成",
    tool_planned: "规划工具",
    tool_started: "调用工具",
    tool_progress: "工具进度",
    tool_guard: "执行检查",
    tool_blocked: "工具被阻止",
    tool_skipped: "工具已跳过",
    tool_error: "工具错误",
    tool_finished: "工具完成",
    tools_collected: "工具结果已汇总",
    round_retry: "轮次重试",
    round_done: "轮次完成",
    round_finished: "轮次完成",
    round_error: "轮次异常",
    round_persisted: "轮次已持久化",
  }[type] || String(type || "事件");
}

function text(value) {
  return value === null || value === undefined ? "" : String(value);
}

function finiteNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function earliest(left, right) {
  const a = finiteNumber(left);
  const b = finiteNumber(right);
  if (!a || a <= 0) return b && b > 0 ? b : 0;
  if (!b || b <= 0) return a;
  return Math.min(a, b);
}

function latest(left, right) {
  return Math.max(finiteNumber(left) || 0, finiteNumber(right) || 0);
}

export function summarizeAgentPayload(value, limit = 180) {
  if (value === null || value === undefined || value === "") return "";
  let normalized = "";
  if (typeof value === "string") normalized = value;
  else if (Array.isArray(value)) normalized = `${value.length} 项`;
  else if (typeof value === "object") {
    normalized = Object.entries(value)
      .slice(0, 4)
      .map(([key, item]) => {
        let rendered = item;
        if (typeof item === "object" && item !== null) {
          try {
            rendered = JSON.stringify(item);
          } catch {
            rendered = String(item);
          }
        }
        const compact = text(rendered).replace(/\s+/g, " ").trim();
        return `${key}=${compact.length > 72 ? `${compact.slice(0, 71)}…` : compact}`;
      })
      .join(", ");
  } else normalized = String(value);
  normalized = normalized.replace(/\s+/g, " ").trim();
  return normalized.length > limit ? `${normalized.slice(0, limit - 1)}…` : normalized;
}

export function agentPayloadText(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function agentEventDetail(event) {
  if (!event) return "";
  if (event.type === "world_frame") {
    const chars = finiteNumber(event.world_chars) ?? text(event.world_xml).length;
    return chars ? `${chars} chars` : "未保存 World 内容";
  }
  if (event.type === "model_request") {
    return [event.feature, event.subfeature, event.model].filter(Boolean).join(" · ");
  }
  if (event.type === "tools_collected") {
    const count = finiteNumber(event.tool_count) ?? (Array.isArray(event.tools) ? event.tools.length : 0);
    return `${count} 个工具结果`;
  }
  const value = event.text
    || event.cognition
    || event.message
    || event.result_preview
    || event.args_preview
    || event.error
    || event.reason
    || event.turn_id;
  return summarizeAgentPayload(value);
}

function createRound(id, createdAt = 0) {
  return {
    id,
    turnId: "",
    createdAt: finiteNumber(createdAt) || 0,
    updatedAt: finiteNumber(createdAt) || 0,
    sessionKey: "",
    conversation: "",
    conversationType: "",
    conversationId: "",
    provider: "",
    model: "",
    status: "running",
    stage: "starting",
    persisted: false,
    live: false,
    error: "",
    elapsedMs: null,
    promptTokens: 0,
    outputTokens: 0,
    worldXml: "",
    cognitionDraft: "",
    cognitionFinal: "",
    cognitionState: "idle",
    cognitionDiscardReason: "",
    tools: [],
    events: [],
    retries: 0,
  };
}

function ensureRound(rounds, id, createdAt = 0) {
  if (!rounds.has(id)) rounds.set(id, createRound(id, createdAt));
  return rounds.get(id);
}

function assignContext(round, source) {
  const sessionKey = text(source?.sessionKey ?? source?.session_key);
  const conversation = text(source?.conversation ?? source?.conv_name);
  const conversationType = text(source?.conversationType ?? source?.conv_type);
  const conversationId = text(source?.conversationId ?? source?.conv_id);
  if (sessionKey) round.sessionKey = sessionKey;
  if (conversation) round.conversation = conversation;
  if (conversationType) round.conversationType = conversationType;
  if (conversationId) round.conversationId = conversationId;
  if (source?.focus && !round.sessionKey) round.sessionKey = text(source.focus);
  if (source?.provider) round.provider = text(source.provider);
  if (source?.model) round.model = text(source.model);
}

function toolIdentity(source, fallbackIndex) {
  const callId = text(source?.call_id ?? source?.id);
  const index = finiteNumber(source?.tool_index ?? source?.index) || fallbackIndex || 0;
  const name = text(source?.tool_name ?? source?.name) || "tool";
  return { callId, index, name };
}

function findTool(round, identity) {
  if (identity.callId) {
    const byCall = round.tools.find((tool) => tool.callId === identity.callId);
    if (byCall) return byCall;
  }
  if (identity.index) {
    const byIndexAndName = round.tools.find(
      (tool) => tool.index === identity.index && tool.name === identity.name,
    );
    if (byIndexAndName) return byIndexAndName;
    const byIndex = round.tools.find((tool) => tool.index === identity.index);
    if (byIndex) return byIndex;
  }
  return round.tools.find((tool) => tool.name === identity.name) || null;
}

function setToolStatus(tool, status) {
  if (!status) return;
  if (TERMINAL_TOOL_STATES.has(tool.status)) return;
  tool.status = status;
}

function upsertTool(round, source, fallbackIndex = 0, fromHistory = false) {
  const identity = toolIdentity(source, fallbackIndex);
  let tool = findTool(round, identity);
  if (!tool) {
    tool = {
      key: identity.callId || `${identity.name}:${identity.index || round.tools.length + 1}`,
      callId: identity.callId,
      index: identity.index || round.tools.length + 1,
      name: identity.name,
      module: "",
      arguments: null,
      result: null,
      argsPreview: "",
      resultPreview: "",
      elapsedMs: null,
      status: fromHistory ? text(source?.status) || "done" : "planned",
      progress: "",
      eventCount: 0,
    };
    round.tools.push(tool);
  }

  if (identity.callId) {
    tool.callId = identity.callId;
    tool.key = identity.callId;
  }
  if (identity.index) tool.index = identity.index;
  if (identity.name) tool.name = identity.name;
  if (source?.module) tool.module = text(source.module);
  const args = source?.arguments ?? source?.args;
  if (args !== undefined) tool.arguments = args;
  if (source?.result !== undefined) tool.result = source.result;
  if (source?.args_preview) tool.argsPreview = text(source.args_preview);
  else if (args !== undefined) tool.argsPreview = summarizeAgentPayload(args);
  if (source?.result_preview) tool.resultPreview = text(source.result_preview);
  else if (source?.result !== undefined) tool.resultPreview = summarizeAgentPayload(source.result);
  const elapsed = finiteNumber(source?.elapsed_ms ?? source?.elapsedMs);
  if (elapsed !== null) tool.elapsedMs = elapsed;
  if (source?.message) tool.progress = text(source.message);
  tool.eventCount += fromHistory ? 0 : 1;

  if (fromHistory) setToolStatus(tool, text(source?.status) || "done");
  else if (source.type === "tool_planned") setToolStatus(tool, "planned");
  else if (["tool_started", "tool_progress", "tool_guard"].includes(source.type)) setToolStatus(tool, "running");
  else if (source.type === "tool_blocked") setToolStatus(tool, "blocked");
  else if (source.type === "tool_skipped") setToolStatus(tool, "skipped");
  else if (source.type === "tool_error") setToolStatus(tool, "error");
  else if (source.type === "tool_finished") setToolStatus(tool, source.ok === false ? "error" : "done");
  return tool;
}

function ingestTurn(round, turn) {
  round.turnId = text(turn.id);
  round.createdAt = earliest(round.createdAt, turn.createdAt);
  round.updatedAt = latest(round.updatedAt, turn.createdAt);
  round.persisted = true;
  round.status = "done";
  round.stage = "persisted";
  assignContext(round, turn);
  if (turn.cognition) round.cognitionFinal = text(turn.cognition);
  if (turn.worldXml) round.worldXml = text(turn.worldXml);
  round.promptTokens = finiteNumber(turn.promptTokens) || 0;
  round.outputTokens = finiteNumber(turn.outputTokens) || 0;
  round.elapsedMs = finiteNumber(turn.elapsedMs);
  for (const [index, tool] of (turn.tools || []).entries()) upsertTool(round, tool, index + 1, true);
}

function eventStage(type) {
  return {
    round_start: "starting",
    world_frame: "world",
    model_request: "request",
    cognition_start: "cognition",
    cognition_delta: "cognition",
    cognition_end: "cognition",
    cognition_final: "cognition",
    cognition_discarded: "retry",
    action_start: "action",
    action_end: "action",
    tool_planned: "planning",
    tool_started: "tool",
    tool_progress: "tool",
    tool_guard: "guard",
    tool_blocked: "blocked",
    tool_skipped: "skipped",
    tool_error: "error",
    tool_finished: "result",
    tools_collected: "result",
    round_retry: "retry",
    round_done: "done",
    round_finished: "done",
    round_error: "error",
    round_persisted: "persisted",
  }[type] || text(type) || "event";
}

function ingestEvent(round, event) {
  round.live = true;
  const createdAt = finiteNumber(event.created_at) || Date.now();
  round.createdAt = earliest(round.createdAt, createdAt);
  round.updatedAt = latest(round.updatedAt, createdAt);
  round.stage = eventStage(event.type);
  assignContext(round, event);
  round.events.push(event);

  if (event.type === "world_frame") round.worldXml = text(event.world_xml);
  if (event.type === "cognition_start") {
    if (round.cognitionState === "discarded" && !round.cognitionFinal) round.cognitionDraft = "";
    round.cognitionState = "streaming";
    round.cognitionDiscardReason = "";
  } else if (event.type === "cognition_delta") {
    round.cognitionDraft += text(event.text);
    round.cognitionState = "streaming";
  } else if (event.type === "cognition_final") {
    round.cognitionFinal = text(event.cognition || event.text);
    round.cognitionState = "complete";
  } else if (event.type === "cognition_discarded") {
    if (!round.cognitionFinal) round.cognitionDraft = "";
    round.cognitionState = "discarded";
    round.cognitionDiscardReason = text(event.reason) || "discarded";
  } else if (event.type === "cognition_end" && round.cognitionState !== "discarded") {
    round.cognitionState = "complete";
  }

  if (TOOL_EVENT_TYPES.has(event.type)) upsertTool(round, event);
  if (event.type === "round_retry") {
    round.retries = Math.max(round.retries, finiteNumber(event.retry_count) || round.retries + 1);
    if (!round.persisted) round.status = "running";
  } else if (["round_done", "round_finished"].includes(event.type)) {
    round.status = "done";
    round.error = "";
    round.elapsedMs = finiteNumber(event.elapsed_ms) ?? round.elapsedMs;
    round.promptTokens = finiteNumber(event.prompt_tokens) ?? round.promptTokens;
    round.outputTokens = finiteNumber(event.output_tokens) ?? round.outputTokens;
  } else if (event.type === "round_error") {
    round.status = "error";
    round.error = text(event.error) || "round_error";
    round.elapsedMs = finiteNumber(event.elapsed_ms) ?? round.elapsedMs;
  } else if (event.type === "round_persisted") {
    round.persisted = true;
    round.turnId = text(event.turn_id) || round.turnId;
    if (round.status !== "error") round.status = "done";
  }
}

function eventIdentity(event, index) {
  const seq = finiteNumber(event?.seq);
  if (seq && seq > 0) return `seq:${seq}`;
  return `${text(event?.type)}:${finiteNumber(event?.created_at) || 0}:${text(event?.round_id)}:${index}`;
}

function normalizeEvents(events) {
  const unique = new Map();
  for (const [index, event] of (events || []).entries()) {
    if (!event?.type) continue;
    unique.set(eventIdentity(event, index), event);
  }
  return [...unique.values()].sort((left, right) => {
    const leftSeq = finiteNumber(left.seq);
    const rightSeq = finiteNumber(right.seq);
    if (leftSeq && rightSeq) return leftSeq - rightSeq;
    return (finiteNumber(left.created_at) || 0) - (finiteNumber(right.created_at) || 0);
  });
}

export function projectAgentRounds(events = [], turns = []) {
  const orderedEvents = normalizeEvents(events);
  const roundIdByTurn = new Map();
  for (const event of orderedEvents) {
    if (event.type === "round_persisted" && event.turn_id && event.round_id) {
      roundIdByTurn.set(text(event.turn_id), text(event.round_id));
    }
  }

  const rounds = new Map();
  for (const turn of turns || []) {
    if (!turn?.id) continue;
    const id = roundIdByTurn.get(text(turn.id)) || `turn:${text(turn.id)}`;
    ingestTurn(ensureRound(rounds, id, turn.createdAt), turn);
  }

  for (const event of orderedEvents) {
    const mappedRound = event.turn_id ? roundIdByTurn.get(text(event.turn_id)) : "";
    const id = text(event.round_id) || mappedRound;
    if (!id) continue;
    ingestEvent(ensureRound(rounds, id, event.created_at), event);
  }

  return [...rounds.values()]
    .map((round) => ({
      ...round,
      conversation: round.conversation || round.sessionKey || round.conversationId || "未知会话",
      cognition: (round.cognitionFinal || round.cognitionDraft).trim(),
      cognitionState: round.cognitionFinal ? "complete" : round.cognitionState,
      tools: [...round.tools].sort((left, right) => left.index - right.index || left.name.localeCompare(right.name)),
      events: [...round.events].sort((left, right) => {
        const seqDelta = (finiteNumber(left.seq) || 0) - (finiteNumber(right.seq) || 0);
        return seqDelta || (finiteNumber(left.created_at) || 0) - (finiteNumber(right.created_at) || 0);
      }),
    }))
    .sort((left, right) => left.createdAt - right.createdAt || left.id.localeCompare(right.id));
}

export function roundMatchesAgentFilter(round, filter) {
  if (filter === "cognition") return Boolean(round.cognition || round.cognitionState === "discarded");
  if (filter === "tools") return round.tools.length > 0;
  if (filter === "runtime") return round.events.some((event) => agentEventCategory(event.type) === "runtime") || Boolean(round.worldXml);
  return true;
}

export function agentRoundStatus(round) {
  if (round.status === "error") return { id: "error", label: "运行异常" };
  if (round.status === "done") return { id: "done", label: "已完成" };
  return { id: "running", label: "运行中" };
}
