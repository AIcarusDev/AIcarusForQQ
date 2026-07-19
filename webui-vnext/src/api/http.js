export class ApiError extends Error {
  constructor(message, { status = 0, code = "request_failed", payload = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.payload = payload;
  }
}

function errorDetails(payload, fallback) {
  if (payload?.error && typeof payload.error === "object") {
    return {
      code: String(payload.error.code || "request_failed"),
      message: String(payload.error.message || fallback),
    };
  }
  if (typeof payload?.error === "string") {
    return { code: "request_failed", message: payload.error };
  }
  if (typeof payload?.message === "string") {
    return { code: "request_failed", message: payload.message };
  }
  return { code: "request_failed", message: fallback };
}

export function redirectToLogin() {
  if (typeof window === "undefined") return;
  const next = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  const target = `/login?next=${encodeURIComponent(next)}`;
  if (window.location.pathname !== "/login") window.location.assign(target);
}

export async function requestJson(path, options = {}) {
  const {
    headers,
    redirectOnUnauthorized = true,
    signal,
    ...fetchOptions
  } = options;

  let response;
  try {
    response = await fetch(path, {
      credentials: "same-origin",
      ...fetchOptions,
      signal,
      headers: {
        Accept: "application/json",
        ...headers,
      },
    });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new ApiError("无法连接 WebUI 后端，请检查服务是否仍在运行。", {
      code: "network_unavailable",
    });
  }

  const body = await response.text();
  let payload = null;
  if (body) {
    try {
      payload = JSON.parse(body);
    } catch {
      throw new ApiError("后端返回了无法识别的响应。", {
        status: response.status,
        code: "invalid_response",
        payload: body,
      });
    }
  }

  if (response.status === 401 && redirectOnUnauthorized) redirectToLogin();

  if (!response.ok || payload?.ok === false || payload?.success === false) {
    const fallback = response.status === 401 ? "登录状态已失效。" : `请求失败（HTTP ${response.status}）。`;
    const details = errorDetails(payload, fallback);
    throw new ApiError(details.message, {
      status: response.status,
      code: details.code,
      payload,
    });
  }

  return payload;
}

export async function requestV1Data(path, options = {}) {
  const payload = await requestJson(path, options);
  if (payload?.api_version !== "1" || !Object.hasOwn(payload, "data")) {
    throw new ApiError("新版接口返回了不兼容的数据格式。", {
      status: 200,
      code: "incompatible_contract",
      payload,
    });
  }
  return payload.data;
}
