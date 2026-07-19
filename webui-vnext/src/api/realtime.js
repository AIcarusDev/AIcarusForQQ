import { redirectToLogin, requestJson } from "./http.js";

function websocketUrl(path, cursor, streamId) {
  const url = new URL(path, window.location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  if (cursor > 0) url.searchParams.set("since", String(cursor));
  if (streamId) url.searchParams.set("stream_id", streamId);
  return url.toString();
}

/**
 * Open a cursor-based JSON WebSocket with snapshot ingestion, de-duplication,
 * and bounded reconnect backoff. Consumers only receive records newer than
 * the latest observed sequence.
 */
export function openCursorStream({
  path,
  initialCursor = 0,
  initialStreamId = "",
  selectItems,
  onItems,
  onEnvelope,
  onStatus,
  onError,
  signal,
}) {
  let cursor = Math.max(0, Number(initialCursor) || 0);
  let streamId = String(initialStreamId || "");
  let socket = null;
  let reconnectTimer = 0;
  let reconnectAttempt = 0;
  let stopped = false;

  const emitStatus = (status) => onStatus?.(status);

  const stop = () => {
    if (stopped) return;
    stopped = true;
    window.clearTimeout(reconnectTimer);
    if (socket && socket.readyState < WebSocket.CLOSING) socket.close(1000, "view closed");
    socket = null;
  };

  const scheduleReconnect = () => {
    if (stopped || signal?.aborted) return;
    reconnectAttempt += 1;
    emitStatus("reconnecting");
    const delay = Math.min(8_000, 500 * (2 ** Math.min(reconnectAttempt - 1, 4)));
    reconnectTimer = window.setTimeout(connect, delay);

    if (reconnectAttempt === 3) {
      requestJson("/api/auth/status", { redirectOnUnauthorized: false })
        .then((status) => {
          if (status?.enabled && !status?.authenticated) {
            stop();
            redirectToLogin();
          }
        })
        .catch(() => {});
    }
  };

  const ingest = (envelope) => {
    const envelopeStreamId = String(envelope?.stream_id || "");
    if (envelope?.cursor_reset === true
      || (streamId && envelopeStreamId && envelopeStreamId !== streamId)) {
      cursor = 0;
    }
    if (envelopeStreamId) streamId = envelopeStreamId;
    const selected = selectItems?.(envelope);
    const items = Array.isArray(selected) ? selected : [];
    const fresh = [];
    for (const item of items) {
      const sequence = Math.max(0, Number(item?.seq) || 0);
      if (sequence && sequence <= cursor) continue;
      if (sequence) cursor = sequence;
      fresh.push(item);
    }
    if (fresh.length) onItems?.(fresh, envelope);
    onEnvelope?.(envelope, cursor);
  };

  function connect() {
    if (stopped || signal?.aborted) return;
    emitStatus(reconnectAttempt ? "reconnecting" : "connecting");
    socket = new WebSocket(websocketUrl(path, cursor, streamId));

    socket.addEventListener("open", () => {
      reconnectAttempt = 0;
      emitStatus("live");
    });

    socket.addEventListener("message", (event) => {
      try {
        ingest(JSON.parse(event.data));
      } catch {
        onError?.(new Error("实时数据格式无法识别"));
      }
    });

    socket.addEventListener("close", () => {
      socket = null;
      scheduleReconnect();
    });

    socket.addEventListener("error", () => {
      emitStatus("reconnecting");
    });
  }

  if (signal) {
    if (signal.aborted) stopped = true;
    else signal.addEventListener("abort", stop, { once: true });
  }
  if (!stopped) connect();

  return {
    close: stop,
    getCursor: () => cursor,
    getStreamId: () => streamId,
  };
}
