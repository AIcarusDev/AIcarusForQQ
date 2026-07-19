import { ApiError, requestV1Data } from "./http.js";

function list(value) {
  return Array.isArray(value) ? value : [];
}

function object(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function normalizeAction(action, domain) {
  return {
    id: String(action?.id || "unknown"),
    domain,
    label: String(action?.label || action?.id || "未知动作"),
    danger: String(action?.danger || "medium"),
    available: action?.available !== false,
    disabledReason: String(action?.disabled_reason || ""),
    expectedConfirmation: String(action?.expected_confirmation ?? action?.confirmation ?? ""),
    confirmationRequired: action?.confirmation_required === true,
    target: String(action?.target || "未声明目标"),
    summary: String(action?.summary || ""),
    effects: list(action?.effects).map(String),
    preserves: list(action?.preserves).map(String),
    backup: {
      created: action?.backup?.created === true,
      kind: String(action?.backup?.kind || "none"),
      description: String(action?.backup?.description || "未声明备份策略"),
    },
    metrics: object(action?.metrics),
  };
}

function normalizeDomain(domain, name) {
  const value = object(domain);
  return {
    name,
    status: String(value.status || "error"),
    error: String(value.error || ""),
    overview: object(value.overview),
    actions: list(value.actions).map((action) => normalizeAction(action, name)),
  };
}

function normalizeOverview(payload) {
  const domains = object(payload?.domains);
  return {
    generatedAt: Number(payload?.generated_at || Date.now()),
    domains: {
      data: normalizeDomain(domains.data, "data"),
      cache: normalizeDomain(domains.cache, "cache"),
      workspace: normalizeDomain(domains.workspace, "workspace"),
    },
  };
}

export async function loadMaintenanceOverview({ signal } = {}) {
  const payload = await requestV1Data("/api/ui/v1/maintenance", { signal });
  return normalizeOverview(payload);
}

export async function loadCacheMaintenance({ signal } = {}) {
  const payload = await requestV1Data("/api/ui/v1/maintenance/cache", { signal });
  return normalizeDomain(payload, "cache");
}

export async function executeMaintenanceAction(action, confirmation, { signal } = {}) {
  if (!action?.domain || !action?.id) {
    throw new ApiError("维护动作缺少服务端标识。", { code: "invalid_maintenance_action" });
  }
  return requestV1Data(
    `/api/ui/v1/maintenance/actions/${encodeURIComponent(action.domain)}/${encodeURIComponent(action.id)}`,
    {
      method: "POST",
      signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirmation: String(confirmation || "") }),
    },
  );
}

export async function loadWorkspaceMaintenanceJob(jobId, cursor = 0, { signal } = {}) {
  const params = new URLSearchParams({ cursor: String(Math.max(0, Number(cursor) || 0)) });
  const payload = await requestV1Data(
    `/api/ui/v1/maintenance/workspace/jobs/${encodeURIComponent(jobId)}?${params}`,
    { signal },
  );
  return object(payload.job);
}
