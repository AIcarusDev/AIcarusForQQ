import { useEffect, useRef, useState } from "react";
import { Bell, CircleAlert, CircleCheck, RefreshCw, X } from "lucide-react";
import { acknowledgeUpdates, loadUpdates, migrateNapcat } from "../api/resourceApi.js";

function levelLabel(level) {
  return { breaking: "重要", warning: "注意", info: "信息" }[level] || level;
}

export function UpdateCenter({ onToast }) {
  const [resource, setResource] = useState({ status: "loading", data: null, error: null });
  const [open, setOpen] = useState(false);
  const [migration, setMigration] = useState({ status: "idle", plan: null, result: null, error: null });
  const triggerRef = useRef(null);
  const closeRef = useRef(null);

  const refresh = () => {
    const controller = new AbortController();
    setResource((current) => ({ ...current, status: "loading", error: null }));
    loadUpdates({ signal: controller.signal })
      .then((data) => setResource({ status: "ready", data, error: null }))
      .catch((error) => {
        if (error?.name !== "AbortError") setResource({ status: "error", data: null, error });
      });
    return controller;
  };

  useEffect(() => {
    const controller = refresh();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    window.requestAnimationFrame(() => closeRef.current?.focus());
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        setOpen(false);
        window.requestAnimationFrame(() => triggerRef.current?.focus());
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open]);

  const close = () => {
    setOpen(false);
    window.requestAnimationFrame(() => triggerRef.current?.focus());
  };

  const acknowledge = async () => {
    try {
      await acknowledgeUpdates(resource.data.currentVersion);
      setResource((current) => ({ ...current, data: { ...current.data, needsAttention: false, acknowledgedVersion: current.data.currentVersion } }));
      onToast("更新公告已标记为已读");
    } catch (error) {
      onToast(error?.message || "标记失败");
    }
  };

  const previewMigration = async () => {
    setMigration({ status: "loading", plan: null, result: null, error: null });
    try {
      const payload = await migrateNapcat({ dryRun: true });
      setMigration({ status: "preview", plan: payload.plan, result: null, error: null });
    } catch (error) {
      setMigration({ status: "error", plan: null, result: null, error });
    }
  };

  const applyMigration = async () => {
    setMigration((current) => ({ ...current, status: "loading", error: null }));
    try {
      const result = await migrateNapcat({ dryRun: false });
      setMigration((current) => ({ ...current, status: "done", result, error: null }));
      onToast("旧版 QQ 配置已整理并保留备份");
      const data = await loadUpdates();
      setResource({ status: "ready", data, error: null });
    } catch (error) {
      setMigration((current) => ({ ...current, status: "error", error }));
    }
  };

  const attention = Boolean(resource.data?.needsAttention || resource.data?.configWarnings?.length);
  return (
    <>
      <button ref={triggerRef} className="icon-button notification-button" type="button" aria-label="查看更新公告" title="更新公告" aria-expanded={open} onClick={() => setOpen(true)}>
        <Bell size={18} />
        {attention && <span className="notification-dot" />}
      </button>
      {open && (
        <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) close(); }}>
          <section className="update-dialog panel-window" role="dialog" aria-modal="true" aria-labelledby="update-dialog-title">
            <div className="dialog-header">
              <div><div className="eyebrow">RELEASE NOTES</div><h2 id="update-dialog-title">更新公告</h2></div>
              <button ref={closeRef} type="button" className="icon-button" aria-label="关闭更新公告" onClick={close}><X size={19} /></button>
            </div>
            <div className="update-dialog-body">
              {resource.status === "loading" ? (
                <div className="inline-resource-state"><RefreshCw className="spin" size={18} /><div><strong>正在读取更新信息</strong><p>同步版本说明与配置提醒。</p></div></div>
              ) : resource.status === "error" ? (
                <div className="inline-resource-state" role="alert"><CircleAlert size={18} /><div><strong>更新信息读取失败</strong><p>{resource.error?.message}</p></div><button type="button" className="secondary-button" onClick={refresh}>重试</button></div>
              ) : (
                <>
                  {resource.data.items.map((item) => (
                    <article className="release-note" key={item.version}>
                      <header><span className={`release-level level-${item.level}`}>{levelLabel(item.level)}</span><strong>{item.title}</strong><time>{item.date}</time></header>
                      <p>{item.summary}</p>
                      {item.changes?.length > 0 && <ul>{item.changes.map((change) => <li key={change}>{change}</li>)}</ul>}
                      <small>{item.version}</small>
                    </article>
                  ))}
                  {resource.data.configWarnings.map((warning) => (
                    <article className="config-warning" key={`${warning.old_path}-${warning.new_path}`}>
                      <CircleAlert size={18} /><div><strong>{warning.title}</strong><p>{warning.message}</p>
                        {migration.status === "idle" && <button type="button" className="secondary-button" onClick={previewMigration}>预览整理计划</button>}
                        {migration.status === "loading" && <span className="inline-progress"><RefreshCw className="spin" size={14} />正在处理</span>}
                        {migration.plan && (
                          <div className="migration-plan">
                            <dl><div><dt>可迁移</dt><dd>{migration.plan.migratable?.length || 0}</dd></div><div><dt>冲突</dt><dd>{migration.plan.conflicts?.length || 0}</dd></div><div><dt>不支持</dt><dd>{migration.plan.unsupported?.length || 0}</dd></div></dl>
                            <p>备份键：<code>{migration.plan.backup_key}</code></p>
                            <button type="button" className="primary-button" onClick={applyMigration}>确认整理并备份</button>
                          </div>
                        )}
                        {migration.status === "done" && <span className="migration-done"><CircleCheck size={15} />整理完成，旧配置已备份</span>}
                        {migration.status === "error" && <span className="form-error">{migration.error?.message}</span>}
                      </div>
                    </article>
                  ))}
                </>
              )}
            </div>
            {resource.status === "ready" && (
              <div className="dialog-footer">
                <span>当前版本 {resource.data.currentVersion}</span>
                <button type="button" className="primary-button" disabled={!resource.data.needsAttention} onClick={acknowledge}>{resource.data.needsAttention ? "标记为已读" : "已读"}</button>
              </div>
            )}
          </section>
        </div>
      )}
    </>
  );
}
