import { useEffect, useMemo, useRef, useState } from "react";
import {
  Archive,
  CheckCircle2,
  CircleAlert,
  Database,
  HardDrive,
  RefreshCw,
  ShieldCheck,
  Terminal,
  TriangleAlert,
  X,
} from "lucide-react";
import {
  executeMaintenanceAction,
  loadMaintenanceOverview,
  loadWorkspaceMaintenanceJob,
} from "../api/maintenanceApi.js";

const DOMAIN_META = {
  data: {
    label: "认知与数据",
    eyebrow: "RUNTIME & DATA",
    description: "重置运行态、长期记忆或整个业务数据库。",
    Icon: Database,
  },
  cache: {
    label: "可再生成缓存",
    eyebrow: "REGENERABLE CACHE",
    description: "按明确目录清理图片、语音或表情派生缓存。",
    Icon: Archive,
  },
  workspace: {
    label: "Linux 工作区",
    eyebrow: "ISOLATED WORKSPACE",
    description: "构建、升级、重启、清空或卸载专用隔离环境。",
    Icon: HardDrive,
  },
};

const DANGER_LABELS = {
  medium: "需要确认",
  high: "高风险",
  critical: "不可逆操作",
};

const TERMINAL_JOB_STATES = new Set(["ready", "failed", "waiting_reboot"]);

function formatCount(value) {
  return new Intl.NumberFormat("zh-CN").format(Math.max(0, Number(value) || 0));
}

function formatBytes(value) {
  const bytes = Math.max(0, Number(value) || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

function workspaceStateLabel(state) {
  return {
    ready: "已就绪",
    not_built: "尚未构建",
    needs_upgrade: "需要升级",
    needs_apply: "等待应用配置",
    building: "正在构建",
    applying: "正在应用",
    restarting: "正在重启",
    clearing: "正在清理",
    uninstalling: "正在卸载",
    waiting_reboot: "等待系统重启",
    failed: "任务失败",
  }[state] || state || "未知";
}

function MaintenanceSummary({ resource }) {
  const domains = resource?.domains;
  return (
    <section className="maintenance-summary-grid">
      <article>
        <Database size={18} />
        <span>数据库受管行</span>
        <strong>{domains ? formatCount(domains.data.overview.total_rows) : "—"}</strong>
      </article>
      <article>
        <Archive size={18} />
        <span>可清理缓存</span>
        <strong>{domains ? formatBytes(domains.cache.overview.total_bytes) : "—"}</strong>
      </article>
      <article>
        <HardDrive size={18} />
        <span>工作区</span>
        <strong>{domains ? workspaceStateLabel(domains.workspace.overview.state) : "—"}</strong>
      </article>
    </section>
  );
}

function ActionCard({ action, onSelect }) {
  return (
    <article className={`maintenance-action-card danger-${action.danger} ${action.available ? "" : "is-disabled"}`}>
      <div className="maintenance-action-heading">
        <div>
          <span className="maintenance-danger-chip"><TriangleAlert size={13} /> {DANGER_LABELS[action.danger] || action.danger}</span>
          <h3>{action.label}</h3>
        </div>
        <span className="maintenance-action-status">{action.available ? "可执行" : "不可用"}</span>
      </div>
      <p>{action.summary}</p>
      <dl className="maintenance-action-facts">
        <div><dt>目标</dt><dd>{action.target}</dd></div>
        <div><dt>备份</dt><dd>{action.backup.description}</dd></div>
      </dl>
      <div className="maintenance-action-lists">
        <div><strong>将会发生</strong><ul>{action.effects.map((effect) => <li key={effect}>{effect}</li>)}</ul></div>
        <div><strong>明确保留</strong><ul>{action.preserves.map((item) => <li key={item}>{item}</li>)}</ul></div>
      </div>
      {action.disabledReason && <p className="maintenance-disabled-reason"><CircleAlert size={14} /> {action.disabledReason}</p>}
      <button type="button" disabled={!action.available} onClick={() => onSelect(action)}>
        <ShieldCheck size={15} /> 查看并确认
      </button>
    </article>
  );
}

function WorkspaceJobPanel({ job }) {
  if (!job?.job_id) return null;
  const log = String(job.log || "").trim();
  return (
    <section className={`workspace-job-panel ${job.status === "failed" ? "is-failed" : ""}`}>
      <div>
        <div><Terminal size={16} /><strong>工作区任务 {job.job_id}</strong></div>
        <span>{workspaceStateLabel(job.status)} · {job.stage || "等待阶段信息"}</span>
      </div>
      {log && <pre>{log}</pre>}
      {job.error && <p><CircleAlert size={15} /> {job.error}</p>}
    </section>
  );
}

function ConfirmationDialog({ action, executing, error, onClose, onExecute }) {
  const dialogRef = useRef(null);
  const inputRef = useRef(null);
  const [confirmation, setConfirmation] = useState("");

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog || !action) return;
    if (!dialog.open) dialog.showModal();
    window.requestAnimationFrame(() => inputRef.current?.focus());
  }, [action]);

  if (!action) return null;
  const matches = !action.confirmationRequired || confirmation === action.expectedConfirmation;

  return (
    <dialog
      ref={dialogRef}
      className="maintenance-confirm-dialog"
      aria-labelledby="maintenance-confirm-title"
      onCancel={(event) => {
        event.preventDefault();
        if (!executing) onClose();
      }}
      onClick={(event) => {
        if (event.target === dialogRef.current && !executing) onClose();
      }}
    >
      <div className="maintenance-confirm-shell">
        <header>
          <div><TriangleAlert size={18} /><div><span>{DANGER_LABELS[action.danger]}</span><h2 id="maintenance-confirm-title">确认：{action.label}</h2></div></div>
          <button type="button" aria-label="关闭确认窗口" disabled={executing} onClick={onClose}><X size={18} /></button>
        </header>
        <div className="maintenance-confirm-body">
          <p>{action.summary}</p>
          <dl>
            <div><dt>精确目标</dt><dd>{action.target}</dd></div>
            <div><dt>备份策略</dt><dd>{action.backup.description}</dd></div>
          </dl>
          {action.confirmationRequired ? (
            <label>
              <span>请输入服务端给出的完整确认字符串</span>
              <code>{action.expectedConfirmation}</code>
              <input
                ref={inputRef}
                type="text"
                value={confirmation}
                autoComplete="off"
                spellCheck="false"
                disabled={executing}
                onChange={(event) => setConfirmation(event.target.value)}
              />
            </label>
          ) : (
            <div className="maintenance-no-confirm"><ShieldCheck size={16} /> 此动作无需输入确认词，但服务端仍会在启动前复核目标状态。</div>
          )}
          {error && <div className="maintenance-confirm-error" role="alert"><CircleAlert size={15} /> {error}</div>}
        </div>
        <footer>
          <button type="button" disabled={executing} onClick={onClose}>取消</button>
          <button type="button" className={`danger-${action.danger}`} disabled={!matches || executing} onClick={() => onExecute(confirmation)}>
            {executing ? <RefreshCw className="spin" size={15} /> : <TriangleAlert size={15} />}
            {executing ? "正在提交…" : "执行服务端动作"}
          </button>
        </footer>
      </div>
    </dialog>
  );
}

export function MaintenancePage({ onToast }) {
  const [reloadKey, setReloadKey] = useState(0);
  const [resource, setResource] = useState({ status: "loading", data: null, error: null });
  const [selectedAction, setSelectedAction] = useState(null);
  const [executing, setExecuting] = useState(false);
  const [executeError, setExecuteError] = useState("");
  const [lastResult, setLastResult] = useState(null);
  const [trackedJob, setTrackedJob] = useState(null);

  useEffect(() => {
    const controller = new AbortController();
    loadMaintenanceOverview({ signal: controller.signal })
      .then((data) => {
        setResource({ status: "ready", data, error: null });
        const currentJob = data.domains.workspace.overview.job;
        if (currentJob?.job_id && !TERMINAL_JOB_STATES.has(currentJob.status)) setTrackedJob(currentJob);
      })
      .catch((error) => {
        if (error?.name !== "AbortError") setResource((current) => ({ ...current, status: "error", error }));
      });
    return () => controller.abort();
  }, [reloadKey]);

  useEffect(() => {
    if (!trackedJob?.job_id || TERMINAL_JOB_STATES.has(trackedJob.status)) return undefined;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const nextJob = await loadWorkspaceMaintenanceJob(
          trackedJob.job_id,
          trackedJob.log_cursor || 0,
          { signal: controller.signal },
        );
        setTrackedJob((current) => ({
          ...current,
          ...nextJob,
          log: `${current?.log || ""}${nextJob.log || ""}`,
        }));
        if (TERMINAL_JOB_STATES.has(nextJob.status)) setReloadKey((value) => value + 1);
      } catch (error) {
        if (error?.name !== "AbortError") onToast?.(error?.message || "工作区任务状态读取失败");
      }
    }, 1400);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [onToast, trackedJob]);

  const actionsByDomain = useMemo(() => resource.data?.domains || {}, [resource.data]);

  const reload = () => {
    setResource((current) => ({ ...current, status: "loading", error: null }));
    setReloadKey((value) => value + 1);
  };

  const execute = async (confirmation) => {
    if (!selectedAction || executing) return;
    setExecuting(true);
    setExecuteError("");
    try {
      const response = await executeMaintenanceAction(selectedAction, confirmation);
      setLastResult(response.result || null);
      const job = response.result?.job;
      if (job?.job_id) setTrackedJob(job);
      onToast?.(job?.job_id ? "工作区任务已由服务端启动" : response.result?.message || "维护动作已完成");
      setSelectedAction(null);
      setReloadKey((value) => value + 1);
    } catch (error) {
      setExecuteError(error?.message || "维护动作执行失败");
    } finally {
      setExecuting(false);
    }
  };

  return (
    <div className="page-stack maintenance-page">
      <section className="panel-window maintenance-intro">
        <div>
          <div className="eyebrow">SERVER-GUARDED OPERATIONS</div>
          <h2>维护与恢复</h2>
          <p>目标、影响、备份策略和确认字符串全部来自服务端。页面不会自行推导，也不会在浏览器中保存确认内容。</p>
        </div>
        <button type="button" className="secondary-button" disabled={resource.status === "loading"} onClick={reload}>
          <RefreshCw className={resource.status === "loading" ? "spin" : ""} size={15} />
          {resource.status === "loading" ? "同步中" : "重新检查"}
        </button>
      </section>

      <MaintenanceSummary resource={resource.data} />

      {resource.status === "error" && (
        <section className="panel-window maintenance-resource-error" role="alert">
          <CircleAlert size={20} /><div><strong>无法读取维护契约</strong><p>{resource.error?.message || "请确认后端仍在运行。"}</p></div><button type="button" onClick={reload}>重试</button>
        </section>
      )}

      {resource.status === "loading" && !resource.data && (
        <section className="panel-window maintenance-loading" role="status"><RefreshCw className="spin" size={20} /> 正在读取维护目标与服务端确认规则…</section>
      )}

      {lastResult?.message && (
        <section className="maintenance-last-result"><CheckCircle2 size={17} /><div><strong>{lastResult.message}</strong><span>{lastResult.maintenance_id ? `操作 ID ${lastResult.maintenance_id}` : "服务端已返回执行结果"}</span></div></section>
      )}

      <WorkspaceJobPanel job={trackedJob || actionsByDomain.workspace?.overview?.job} />

      {Object.entries(DOMAIN_META).map(([domainName, meta]) => {
        const domain = actionsByDomain[domainName];
        if (!domain) return null;
        const Icon = meta.Icon;
        return (
          <section className="maintenance-domain" key={domainName}>
            <div className="maintenance-domain-header">
              <span><Icon size={19} /></span>
              <div><div className="eyebrow">{meta.eyebrow}</div><h2>{meta.label}</h2><p>{meta.description}</p></div>
              <code>{domain.status === "ready" ? `${domain.actions.length} actions` : "unavailable"}</code>
            </div>
            {domain.status !== "ready" ? (
              <div className="maintenance-domain-error"><CircleAlert size={17} /> {domain.error || "该维护领域暂时不可用"}</div>
            ) : (
              <div className="maintenance-action-grid">
                {domain.actions.map((action) => <ActionCard key={`${domainName}-${action.id}`} action={action} onSelect={(nextAction) => { setSelectedAction(nextAction); setExecuteError(""); }} />)}
              </div>
            )}
          </section>
        );
      })}

      <ConfirmationDialog
        key={`${selectedAction?.domain || "none"}-${selectedAction?.id || "none"}`}
        action={selectedAction}
        executing={executing}
        error={executeError}
        onClose={() => { if (!executing) setSelectedAction(null); }}
        onExecute={execute}
      />
    </div>
  );
}
