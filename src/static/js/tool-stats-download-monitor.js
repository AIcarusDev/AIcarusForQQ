(function (root) {
  "use strict";

  const STATUS_LABELS = {
    queued: "等待下载",
    resolving: "解析文件",
    downloading: "下载中",
    verifying: "保存中",
    completed: "下载成功",
    failed: "下载失败",
    stopped: "已停止",
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function escapeAttr(value) {
    return escapeHtml(value).replace(/`/g, "&#96;");
  }

  function formatTime(value) {
    if (!value) return "时间未知";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "时间未知";
    return date.toLocaleString("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  }

  function formatFileSize(value) {
    const bytes = Number(value);
    if (!Number.isFinite(bytes) || bytes < 0) return "";
    if (bytes < 1024) return `${Math.round(bytes)} B`;
    const units = ["KB", "MB", "GB", "TB"];
    let size = bytes / 1024;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex += 1;
    }
    return `${size.toFixed(size >= 100 ? 0 : size >= 10 ? 1 : 2).replace(/\.0+$/, "")} ${units[unitIndex]}`;
  }

  function conversationLabel(job) {
    const conversation = job?.conversation || {};
    return conversation.type === "private"
      ? `私聊 ${conversation.id || "未知"}`
      : `群 ${conversation.id || "未知"}`;
  }

  function idleMarkup(message) {
    return `<div class="download-idle">
      <span>${escapeHtml(message)}</span>
      <div class="download-progress-track idle" aria-hidden="true"><span></span></div>
    </div>`;
  }

  function updateText(element, value) {
    if (element.textContent !== value) element.textContent = value;
  }

  function create(options = {}) {
    const endpoint = options.endpoint || "/api/tool-stats/downloads?limit=20";
    const status = document.getElementById(options.statusId || "downloadMonitorStatus");
    const currentList = document.getElementById(options.currentId || "downloadCurrentList");
    const historyList = document.getElementById(options.historyId || "downloadHistoryList");
    const state = { controller: null, timer: null, loaded: false, destroyed: false };

    function render(data) {
      const available = data?.available !== false;
      const connected = available && data?.connected !== false;
      const capabilityKnown = data?.download_capability !== "unknown";
      const active = Array.isArray(data?.active) ? data.active : [];
      const history = Array.isArray(data?.history) ? data.history : [];
      status.className = `download-monitor-status${connected && capabilityKnown ? "" : " unavailable"}`;
      updateText(status, !available
        ? "下载状态不可用"
        : connected && !capabilityKnown
          ? "适配器类型未识别，暂不支持下载"
        : connected
          ? (active.length ? `${active.length} 个任务进行中` : "当前无下载任务")
          : "QQ 下载未连接");

      currentList.innerHTML = active.length
        ? `<div class="download-task-list">${active.map(job => {
            const rawTotal = job.total_bytes;
            const total = rawTotal === null || rawTotal === undefined ? null : Number(rawTotal);
            const knownSize = Number.isFinite(total) && total >= 0;
            const downloaded = Math.max(0, Number(job.bytes_downloaded || 0));
            const reported = Number(job.progress_percent);
            const progress = knownSize
              ? Math.min(100, Math.max(0, Number.isFinite(reported) ? reported : total === 0 ? 100 : downloaded * 100 / total))
              : null;
            const progressAttrs = knownSize
              ? `aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(progress)}"`
              : 'aria-valuetext="正在获取文件大小"';
            return `<article class="download-task-card">
              <div class="download-task-meta">
                <span class="download-state-pill ${escapeAttr(job.status)}">${escapeHtml(STATUS_LABELS[job.status] || "处理中")}</span>
                <span>${escapeHtml(conversationLabel(job))}</span>
                <time>${escapeHtml(formatTime(job.created_at))}</time>
                <span class="download-task-name" title="${escapeAttr(job.original_filename || "未命名文件")}">${escapeHtml(job.original_filename || "未命名文件")}</span>
              </div>
              <div class="download-progress-line">
                <div class="download-progress-track ${knownSize ? "" : "indeterminate"}" role="progressbar" aria-label="${escapeAttr(job.original_filename || "文件")} 下载进度" ${progressAttrs}>
                  <span${knownSize ? ` style="width:${progress}%"` : ""}></span>
                </div>
                <span class="download-progress-text">${knownSize
                  ? `${escapeHtml(formatFileSize(downloaded))} / ${escapeHtml(formatFileSize(total))} · ${Math.round(progress)}%`
                  : "正在获取文件大小"}</span>
              </div>
            </article>`;
          }).join("")}</div>`
        : idleMarkup(!available
          ? "暂时无法读取下载任务"
          : connected && !capabilityKnown
            ? "当前适配器类型未识别，暂不支持开始新下载"
          : connected
            ? "当前没有下载任务"
            : "QQ 账号未连接，当前没有实时下载任务");

      historyList.innerHTML = history.length
        ? `<div class="download-history-list">${history.map(job => {
            const resultLabel = STATUS_LABELS[job.status] || "状态未知";
            let detail = "";
            if (job.status === "failed") detail = job.failure?.display_message || "未知原因";
            if (job.status === "completed" && job.total_bytes !== null && job.total_bytes !== undefined) {
              detail = formatFileSize(job.total_bytes);
            }
            return `<article class="download-history-row">
              <div class="download-history-main">
                <time>${escapeHtml(formatTime(job.finished_at || job.updated_at))}</time>
                <span>${escapeHtml(conversationLabel(job))} 下载</span>
                <span class="download-history-name" title="${escapeAttr(job.original_filename || "未命名文件")}">${escapeHtml(job.original_filename || "未命名文件")}</span>
              </div>
              <div class="download-history-result">
                <strong class="${escapeAttr(job.status)}">${escapeHtml(resultLabel)}</strong>
                ${detail ? `<span title="${escapeAttr(detail)}">${escapeHtml(detail)}</span>` : ""}
              </div>
            </article>`;
          }).join("")}</div>`
        : '<div class="download-empty">暂无下载记录</div>';
    }

    function schedule(delay) {
      if (!state.destroyed) state.timer = window.setTimeout(load, delay);
    }

    async function load() {
      window.clearTimeout(state.timer);
      state.controller?.abort();
      const controller = new AbortController();
      state.controller = controller;
      if (!state.loaded) status.className = "download-monitor-status loading";
      try {
        const response = await fetch(endpoint, { signal: controller.signal, cache: "no-store" });
        const payload = await response.json();
        if (!response.ok || payload.success === false) {
          throw new Error(payload.error || `HTTP ${response.status}`);
        }
        render(payload || {});
        state.loaded = true;
        schedule(document.hidden || payload.available === false || payload.connected === false ? 5000 : 1000);
      } catch (error) {
        if (error?.name === "AbortError" || state.destroyed) return;
        status.className = "download-monitor-status error";
        updateText(status, "下载状态暂时不可用");
        schedule(5000);
      }
    }

    function handleVisibilityChange() {
      if (!document.hidden) load();
    }

    document.addEventListener("visibilitychange", handleVisibilityChange);
    return {
      load,
      destroy() {
        state.destroyed = true;
        window.clearTimeout(state.timer);
        state.controller?.abort();
        document.removeEventListener("visibilitychange", handleVisibilityChange);
      },
    };
  }

  root.ToolStatsDownloadMonitor = { create };
})(typeof globalThis !== "undefined" ? globalThis : this);
