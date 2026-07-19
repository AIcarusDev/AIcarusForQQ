import { requestJson } from "./http.js";
import { loadMemorySchema } from "./memoryApi.js";

function nonNegativeInteger(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? Math.floor(number) : 0;
}

function runtimeState(runtime) {
  if (runtime.core_available) return "running";
  if (runtime.mode === "webui_only") return "stopped";
  return "unavailable";
}

function modeLabel(mode) {
  return {
    full: "完整运行",
    webui_only: "仅 WebUI",
    standalone: "独立面板",
  }[mode] || "未知模式";
}

export async function loadRuntimeOverview({ signal } = {}) {
  const [capabilities, dashboard, core, memorySchema] = await Promise.all([
    requestJson("/api/ui/v1/capabilities", { signal }),
    requestJson("/api/status", { signal }),
    requestJson("/api/core/status", { signal }),
    loadMemorySchema({ signal }).catch(() => null),
  ]);

  const runtime = capabilities?.runtime || {};
  const state = runtimeState(runtime);
  const memoryCounts = dashboard?.memory_counts || {};
  const semanticRelations = memorySchema?.relations
    ?.filter((relation) => relation.available)
    .reduce((total, relation) => total + nonNegativeInteger(relation.count), 0);
  const hasSemanticCounts = Number.isFinite(semanticRelations);

  return {
    generatedAt: Date.now(),
    identity: {
      name: String(dashboard?.self_name || "AIcarus"),
      model: String(dashboard?.model || "未配置"),
    },
    runtime: {
      mode: String(runtime.mode || "unknown"),
      modeLabel: modeLabel(runtime.mode),
      state,
      coreAvailable: Boolean(runtime.core_available),
      launcherManaged: Boolean(core?.launcher_mode),
    },
    activity: {
      currentFocus: dashboard?.current_focus ? String(dashboard.current_focus) : null,
      todayMessages: nonNegativeInteger(dashboard?.today_messages),
      memoryEvents: nonNegativeInteger(memoryCounts.events),
      memoryRelations: hasSemanticCounts
        ? semanticRelations
        : nonNegativeInteger(memoryCounts.relations),
      memoryRelationLabel: hasSemanticCounts ? "语义关系" : "存储关系",
      uptimeSeconds: nonNegativeInteger(dashboard?.uptime_seconds),
    },
    migration: capabilities?.migration || { legacy_path: "/", vnext_path: "/new/" },
    capabilities: capabilities?.capabilities || {},
  };
}

export async function logoutSession({ signal } = {}) {
  return requestJson("/api/auth/logout", {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json" },
  });
}
