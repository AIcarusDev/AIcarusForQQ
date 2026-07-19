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

function normalizeTurn(turn) {
  const tools = list(turn?.tool_calls);
  return {
    id: text(turn?.turn_id),
    createdAt: Number(turn?.created_at) || 0,
    sessionKey: text(turn?.session_key),
    conversation: text(turn?.conv_name, text(turn?.session_key, "未知会话")),
    cognition: cognitionFromResult(turn?.result),
    toolCount: tools.length,
    tools: tools.map((tool) => text(tool?.function || tool?.name, "tool")),
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
