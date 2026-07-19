import { requestJson, requestV1Data } from "./http.js";

export function loadSettingsDomain(domain, { signal } = {}) {
  return requestV1Data(`/api/ui/v1/settings/${encodeURIComponent(domain)}`, { signal });
}

export function saveSettingsDomain(domain, { revision, values, secrets }, { signal } = {}) {
  return requestV1Data(`/api/ui/v1/settings/${encodeURIComponent(domain)}`, {
    method: "PATCH",
    signal,
    headers: {
      "Content-Type": "application/json",
      "If-Match": `"${revision}"`,
    },
    body: JSON.stringify({ values, secrets }),
  });
}

export function loadSecurityStatus({ signal } = {}) {
  return requestJson("/api/auth/status", { signal });
}

export function replaceSecurityPassword(password, { signal } = {}) {
  return requestJson("/api/auth/password", {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "set", password }),
  });
}
