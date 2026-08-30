import { requestJson } from "./http.js";
import { openCursorStream } from "./realtime.js";

function text(value, fallback = "") {
  return value === null || value === undefined ? fallback : String(value);
}

function list(value) {
  return Array.isArray(value) ? value : [];
}

function imageMap(value) {
  return value && !Array.isArray(value) && typeof value === "object" ? value : {};
}

function dataImage(media) {
  const mime = text(media?.mime).toLowerCase();
  const base64 = text(media?.base64);
  if (!mime.startsWith("image/") || !base64) return null;
  return `data:${mime};base64,${base64}`;
}

function formatFileSize(value) {
  const size = Number(value);
  if (!Number.isFinite(size) || size < 0) return "";
  if (size < 1024) return `${Math.trunc(size)}B`;
  const units = ["KB", "MB", "GB", "TB"];
  let amount = size;
  for (const unit of units) {
    amount /= 1024;
    if (amount < 1024 || unit === "TB") {
      const digits = Number.isInteger(amount) ? 0 : amount >= 10 ? 1 : 2;
      return `${Number(amount.toFixed(digits))}${unit}`;
    }
  }
  return "";
}

function normalizeSegment(segment, images) {
  const type = text(segment?.type, "unknown");
  if (type === "text") return { type, text: text(segment?.text) };
  if (type === "mention") {
    return { type, text: text(segment?.display, `@${text(segment?.uid)}`) };
  }
  if (type === "emoji") {
    return { type, text: `[${text(segment?.name, text(segment?.id, "表情"))}]` };
  }
  if (type === "image" || type === "sticker") {
    const stickerId = text(segment?.sticker_id);
    const reference = text(segment?.image_ref || segment?.ref);
    const media = reference ? images[reference] : null;
    const src = stickerId
      ? `/api/sticker/${encodeURIComponent(stickerId)}`
      : dataImage(media);
    return {
      type,
      src,
      text: text(media?.label, media?.expired ? "图片已过期" : type === "sticker" ? "贴纸" : "图片"),
    };
  }
  if (type === "voice") {
    const duration = Number(segment?.duration);
    return { type, text: Number.isFinite(duration) ? `语音 ${Math.round(duration)}s` : "语音" };
  }
  if (type === "file") {
    const filename = text(segment?.filename, text(segment?.label)).trim();
    const size = formatFileSize(segment?.size_bytes);
    const isDownloaded = segment?.is_downloaded === true;
    const label = filename ? `文件:${filename}` : "文件";
    return {
      type,
      text: [label, size, isDownloaded ? "已下载" : "未下载"].filter(Boolean).join(" · "),
      filename,
      sizeBytes: Number(segment?.size_bytes),
      isDownloaded,
    };
  }
  const labels = { forward: "合并转发", reply: "回复", file: "文件" };
  return { type, text: text(segment?.label, labels[type] || type) };
}

export function normalizeMessage(message, index = 0) {
  const images = imageMap(message?.images);
  const segments = list(message?.content_segments).map((item) => normalizeSegment(item, images));
  const content = text(message?.content) || segments.map((item) => item.text).join("");
  return {
    id: text(message?.message_id, `${text(message?.timestamp, "message")}-${index}`),
    role: text(message?.role, "user"),
    sender: text(message?.sender_name || message?.sender_card || message?.sender_id, "未知发送者"),
    timestamp: text(message?.timestamp),
    content,
    contentType: text(message?.content_type, "text"),
    deliveryState: text(message?.delivery_state),
    deliveryError: text(message?.delivery_error),
    segments,
    raw: message,
  };
}

function normalizeSession(session) {
  const key = text(session?.session_key);
  return {
    key,
    label: text(session?.focus_name || session?.conv_name, key || "未命名会话"),
    platform: text(session?.focus_platform, "qq"),
    type: text(session?.focus_type || session?.conv_type),
    id: text(session?.focus_id || session?.conv_id),
  };
}

function cognitionFromResult(result) {
  if (!result || typeof result !== "object") return "";
  for (const key of ["cognition", "response", "action", "text", "content"]) {
    if (typeof result[key] === "string" && result[key].trim()) return result[key].trim();
  }
  return "";
}

function normalizeToolExecution(tool, index) {
  const namespace = text(tool?.namespace).trim();
  const functionName = text(tool?.function || tool?.name, "tool").trim() || "tool";
  const name = namespace && !functionName.startsWith(`${namespace}.`)
    ? `${namespace}.${functionName}`
    : functionName;
  const result = tool?.result && typeof tool.result === "object" ? tool.result : {};
  const status = result?.tool_not_executed
    ? "blocked"
    : result?.ok === false || result?.error
      ? "error"
      : "done";
  return {
    id: text(tool?.call_id, `history-tool-${index + 1}`),
    index: index + 1,
    name,
    namespace,
    arguments: tool?.arguments ?? tool?.args ?? {},
    result,
    elapsedMs: Number.isFinite(Number(tool?.elapsed_ms)) ? Number(tool.elapsed_ms) : null,
    status,
  };
}

function normalizeTurn(turn) {
  const tools = list(turn?.tool_calls);
  const result = turn?.result && typeof turn.result === "object" ? turn.result : {};
  const tokens = result?.tokens && typeof result.tokens === "object" ? result.tokens : {};
  return {
    id: text(turn?.turn_id),
    createdAt: Number(turn?.created_at) || 0,
    sessionKey: text(turn?.session_key),
    conversation: text(turn?.conv_name, text(turn?.session_key, "未知会话")),
    conversationType: text(turn?.conv_type),
    conversationId: text(turn?.conv_id),
    cognition: cognitionFromResult(result),
    motive: text(result?.motive),
    worldXml: text(turn?.world_xml),
    promptTokens: Number(tokens?.in ?? tokens?.prompt) || 0,
    outputTokens: Number(tokens?.out ?? tokens?.output) || 0,
    elapsedMs: Number.isFinite(Number(result?.elapsed_ms)) ? Number(result.elapsed_ms) : null,
    toolCount: tools.length,
    tools: tools.map(normalizeToolExecution),
  };
}

export async function loadCoreChat({ signal, limit = 100 } = {}) {
  const payload = await requestJson(`/api/core/chat?limit=${Math.max(1, Math.min(200, limit))}`, { signal });
  return {
    sessionKey: text(payload?.session_key),
    messages: list(payload?.messages).map(normalizeMessage),
  };
}

export async function sendCoreChat({ content, clientId, signal }) {
  const payload = await requestJson("/api/core/chat", {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, client_id: clientId }),
  });
  return {
    duplicate: Boolean(payload?.duplicate),
    clientId: text(payload?.client_id, clientId),
    message: normalizeMessage(payload?.message),
  };
}

export async function loadFocusOverview({ signal } = {}) {
  const payload = await requestJson("/api/focus/state", { signal });
  return {
    currentFocus: text(payload?.current_focus),
    sessions: list(payload?.sessions).map(normalizeSession),
    turns: list(payload?.recent_turns).map(normalizeTurn),
  };
}

export async function loadFocusContext(key, { signal } = {}) {
  const params = new URLSearchParams();
  if (key) params.set("key", key);
  const payload = await requestJson(`/api/focus/context?${params}`, { signal });
  return {
    sessionKey: text(payload?.session_key),
    messages: list(payload?.messages).map(normalizeMessage),
  };
}

export async function loadAgentState({ signal } = {}) {
  const payload = await requestJson("/api/agent/state", { signal });
  const events = list(payload?.events);
  return {
    identity: text(payload?.self_name, "AIcarus"),
    provider: text(payload?.provider),
    model: text(payload?.model),
    currentFocus: text(payload?.current_focus),
    sessions: list(payload?.sessions).map(normalizeSession),
    turns: list(payload?.recent_turns).map(normalizeTurn),
    events,
    stats: payload?.stats || {},
    streamId: text(payload?.stats?.stream_id),
    latestSeq: Math.max(0, ...events.map((event) => Number(event?.seq) || 0)),
  };
}

export async function loadAgentTurns({ before = 0, limit = 24, signal } = {}) {
  const params = new URLSearchParams({
    limit: String(Math.max(1, Math.min(100, Number(limit) || 24))),
  });
  if (Number(before) > 0) params.set("before", String(Math.floor(Number(before))));
  const payload = await requestJson(`/api/agent/turns?${params}`, { signal });
  return list(payload?.turns).map(normalizeTurn);
}

export function subscribeAgentEvents({ initialCursor = 0, initialStreamId = "", onEvents, onStats, onStatus, onError, signal }) {
  return openCursorStream({
    path: "/agent/ws/events",
    initialCursor,
    initialStreamId,
    signal,
    selectItems: (envelope) => envelope?.type === "snapshot" ? envelope.events : [envelope],
    onItems: onEvents,
    onEnvelope: (envelope) => {
      if (envelope?.stats) onStats?.(envelope.stats);
    },
    onStatus,
    onError,
  });
}
