import { requestJson } from "./http.js";

function list(value) {
  return Array.isArray(value) ? value : [];
}

function text(value, fallback = "") {
  return value === null || value === undefined ? fallback : String(value);
}

export async function loadUpdates({ signal } = {}) {
  const payload = await requestJson("/api/updates/current", { signal });
  return {
    currentVersion: text(payload?.current_version),
    acknowledgedVersion: text(payload?.ack_version),
    needsAttention: Boolean(payload?.needs_popup),
    hasBreaking: Boolean(payload?.has_breaking),
    items: list(payload?.items),
    configWarnings: list(payload?.config_warnings),
  };
}

export async function acknowledgeUpdates(version, { signal } = {}) {
  return requestJson("/api/updates/ack", {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ version }),
  });
}

export async function migrateNapcat({ dryRun, signal } = {}) {
  return requestJson("/api/updates/migrations/napcat-to-qq-adapter", {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dry_run: Boolean(dryRun) }),
  });
}

export function stickerImageUrl(id, version = "") {
  const suffix = version ? `?v=${encodeURIComponent(version)}` : "";
  return `/api/sticker/${encodeURIComponent(id)}${suffix}`;
}

export async function loadStickers({ signal } = {}) {
  const payload = await requestJson("/api/stickers/list", { signal });
  return list(payload?.stickers).map((item) => ({
    id: text(item?.id),
    description: text(item?.description),
    createdAt: text(item?.created_at),
    mime: text(item?.mime),
    filename: text(item?.filename),
    imageUrl: stickerImageUrl(text(item?.id), text(item?.sha256).slice(0, 12)),
  }));
}

export async function uploadSticker(file, description, { signal } = {}) {
  const body = new FormData();
  body.append("file", file);
  body.append("description", description);
  return requestJson("/api/stickers/upload", { method: "POST", signal, body });
}

export async function updateSticker(id, description, { signal } = {}) {
  return requestJson(`/api/stickers/${encodeURIComponent(id)}`, {
    method: "PATCH",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ description }),
  });
}

export async function deleteSticker(id, { signal } = {}) {
  return requestJson(`/api/stickers/${encodeURIComponent(id)}`, { method: "DELETE", signal });
}

export async function reconcileStickers({ signal } = {}) {
  return requestJson("/api/stickers/reconcile", { method: "POST", signal });
}

export function selfImageUrl(name, version = "") {
  const suffix = version ? `?v=${encodeURIComponent(version)}` : "";
  return `/settings/self_image/${encodeURIComponent(name)}${suffix}`;
}

export async function loadSelfImages({ signal } = {}) {
  const payload = await requestJson("/settings/self_image", { signal });
  return list(payload?.files).map((file) => ({
    name: text(file?.name),
    size: Math.max(0, Number(file?.size) || 0),
    imageUrl: selfImageUrl(text(file?.name), text(file?.size)),
  }));
}

export async function uploadSelfImages(files, { signal } = {}) {
  const body = new FormData();
  for (const file of files) body.append("files", file);
  return requestJson("/settings/self_image", { method: "POST", signal, body });
}

export async function deleteSelfImage(name, { signal } = {}) {
  return requestJson(`/settings/self_image/${encodeURIComponent(name)}`, {
    method: "DELETE",
    signal,
  });
}

export async function loadWorkspace({ signal } = {}) {
  const payload = await requestJson("/api/computer", { signal });
  const state = text(payload?.state, text(payload?.observed?.state, "unknown"));
  return {
    state,
    stateLabel: {
      ready: "已就绪",
      building: "正在构建",
      not_built: "尚未构建",
      unavailable: "不可用",
      error: "状态异常",
    }[state] || state,
    config: payload?.config && typeof payload.config === "object" ? payload.config : {},
    observed: payload?.observed && typeof payload.observed === "object" ? payload.observed : {},
    job: payload?.job && typeof payload.job === "object" ? payload.job : null,
  };
}
