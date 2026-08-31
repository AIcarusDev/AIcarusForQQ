(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.ToolStatsRuntime = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function createLatestResource() {
    let requestId = 0;
    return {
      phase: "idle",
      data: null,
      error: null,
      begin() {
        requestId += 1;
        this.phase = "loading";
        this.data = null;
        this.error = null;
        return requestId;
      },
      invalidate() {
        requestId += 1;
        this.phase = "idle";
        this.data = null;
        this.error = null;
        return requestId;
      },
      isCurrent(candidate) {
        return candidate === requestId;
      },
      succeed(candidate, data) {
        if (!this.isCurrent(candidate)) return false;
        this.phase = "ready";
        this.data = data;
        this.error = null;
        return true;
      },
      fail(candidate, error) {
        if (!this.isCurrent(candidate)) return false;
        this.phase = "error";
        this.data = null;
        this.error = error;
        return true;
      },
    };
  }

  function parseGlossary(payload) {
    if (payload?.schema_version !== 1 || payload?.locale !== "zh-CN") {
      return Object.create(null);
    }
    const tools = payload.tools;
    if (!tools || typeof tools !== "object" || Array.isArray(tools)) {
      return Object.create(null);
    }
    const glossary = Object.create(null);
    Object.entries(tools).forEach(([rawName, entry]) => {
      if (!entry || typeof entry !== "object" || typeof entry.zh_name !== "string") return;
      const zhName = entry.zh_name.trim();
      if (!zhName) return;
      glossary[rawName] = {
        zh_name: zhName,
        zh_definition: typeof entry.zh_definition === "string" ? entry.zh_definition.trim() : "",
      };
    });
    return glossary;
  }

  function resolveToolName(glossary, language, rawName) {
    const raw = String(rawName ?? "");
    const entry = language === "zh-CN" ? glossary?.[raw] : null;
    if (!entry?.zh_name) {
      return { raw_name: raw, display_name: raw, definition: "", translated: false };
    }
    return {
      raw_name: raw,
      display_name: entry.zh_name,
      definition: String(entry.zh_definition || "").trim(),
      translated: true,
    };
  }

  return { createLatestResource, parseGlossary, resolveToolName };
});
