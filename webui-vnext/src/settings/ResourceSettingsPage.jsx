import { useEffect, useRef, useState } from "react";
import {
  Archive,
  ChevronRight,
  CircleAlert,
  HardDrive,
  Image as ImageIcon,
  RefreshCw,
  Server,
  Trash2,
  Upload,
} from "lucide-react";
import { loadCacheMaintenance } from "../api/maintenanceApi.js";
import {
  deleteSelfImage,
  loadSelfImages,
  loadWorkspace,
  uploadSelfImages,
} from "../api/resourceApi.js";

function formatBytes(value) {
  const bytes = Math.max(0, Number(value) || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

function SettingsHeader({ eyebrow, title, description, badge }) {
  return (
    <div className="settings-section-header">
      <div><div className="eyebrow">{eyebrow}</div><h2>{title}</h2><p>{description}</p></div>
      {badge && <span className="readonly-badge">{badge}</span>}
    </div>
  );
}

function SelfImageSettings({ onToast }) {
  const [resource, setResource] = useState({ status: "loading", items: [], error: null });
  const [uploading, setUploading] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  const inputRef = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    loadSelfImages({ signal: controller.signal })
      .then((items) => setResource({ status: "ready", items, error: null }))
      .catch((error) => {
        if (error?.name !== "AbortError") setResource((current) => ({ ...current, status: "error", error }));
      });
    return () => controller.abort();
  }, [reloadKey]);

  const upload = async (files) => {
    if (!files.length) return;
    setUploading(true);
    try {
      await uploadSelfImages(files);
      onToast(`已上传 ${files.length} 张自身形象`);
      setReloadKey((value) => value + 1);
    } catch (error) {
      onToast(error?.message || "上传失败");
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const remove = async (item) => {
    if (!window.confirm(`确定删除“${item.name}”？`)) return;
    try {
      await deleteSelfImage(item.name);
      setResource((current) => ({ ...current, items: current.items.filter((candidate) => candidate.name !== item.name) }));
      onToast("自身形象已删除");
    } catch (error) {
      onToast(error?.message || "删除失败");
    }
  };

  return (
    <div className="settings-content resource-settings-content">
      <SettingsHeader eyebrow="IDENTITY RESOURCE" title="自身形象" description="管理 Core 可用于自我表达的本地图片资源。" />
      <section className="form-section">
        <div className="form-section-head"><div><h3>图片资源</h3><p>文件保存在服务端配置目录，浏览器不保存图片内容。</p></div>
          <label className={`secondary-button file-button ${uploading ? "disabled" : ""}`}><input ref={inputRef} type="file" accept="image/png,image/jpeg,image/webp,image/gif" multiple disabled={uploading} onChange={(event) => upload([...event.target.files])} />{uploading ? <RefreshCw className="spin" size={15} /> : <Upload size={15} />}{uploading ? "上传中" : "上传图片"}</label>
        </div>
        {resource.status === "error" ? (
          <div className="inline-resource-state" role="alert"><CircleAlert size={18} /><div><strong>读取失败</strong><p>{resource.error?.message}</p></div><button type="button" className="secondary-button" onClick={() => setReloadKey((value) => value + 1)}>重试</button></div>
        ) : resource.status === "loading" ? (
          <div className="inline-resource-state"><RefreshCw className="spin" size={18} /><div><strong>正在读取图片</strong><p>同步服务端资源目录。</p></div></div>
        ) : resource.items.length ? (
          <div className="self-image-grid">
            {resource.items.map((item) => (
              <article key={item.name}>
                <img src={item.imageUrl} alt={item.name} loading="lazy" />
                <div><strong title={item.name}>{item.name}</strong><span>{formatBytes(item.size)}</span></div>
                <button type="button" aria-label={`删除 ${item.name}`} title="删除" onClick={() => remove(item)}><Trash2 size={15} /></button>
              </article>
            ))}
          </div>
        ) : <div className="resource-empty"><ImageIcon size={23} /><strong>还没有自身形象</strong><span>上传后即可由 Core 按能力使用。</span></div>}
      </section>
    </div>
  );
}

function WorkspaceSettings() {
  const [resource, setResource] = useState({ status: "loading", data: null, error: null });
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    loadWorkspace({ signal: controller.signal })
      .then((data) => setResource({ status: "ready", data, error: null }))
      .catch((error) => {
        if (error?.name !== "AbortError") setResource({ status: "error", data: null, error });
      });
    return () => controller.abort();
  }, [reloadKey]);

  return (
    <div className="settings-content resource-settings-content">
      <SettingsHeader eyebrow="ISOLATED RUNTIME" title="Linux 工作区" description="查看隔离执行环境、安装位置和资源状态。" badge="只读状态" />
      {resource.status === "loading" ? (
        <div className="inline-resource-state panel-window"><RefreshCw className="spin" size={18} /><div><strong>正在探测工作区</strong><p>读取配置与实际安装状态。</p></div></div>
      ) : resource.status === "error" ? (
        <div className="inline-resource-state panel-window" role="alert"><CircleAlert size={18} /><div><strong>工作区状态不可用</strong><p>{resource.error?.message}</p></div><button type="button" className="secondary-button" onClick={() => setReloadKey((value) => value + 1)}>重试</button></div>
      ) : (
        <>
          <section className="workspace-status-grid">
            <article><Server size={18} /><span>服务状态</span><strong>{resource.data.stateLabel}</strong></article>
            <article><HardDrive size={18} /><span>安装路径</span><strong title={resource.data.config.install_root}>{resource.data.config.install_root || "未配置"}</strong></article>
            <article><RefreshCw size={18} /><span>路径锁定</span><strong>{resource.data.observed.path_locked ? "已锁定" : "可调整"}</strong></article>
          </section>
          <section className="form-section workspace-readonly-panel">
            <div className="form-section-head"><div><h3>资源配置</h3><p>构建、扩容和卸载属于高风险维护操作，将在维护页由服务端确认后执行。</p></div></div>
            <dl>
              <div><dt>启用</dt><dd>{resource.data.config.enabled ? "是" : "否"}</dd></div>
              <div><dt>CPU</dt><dd>{resource.data.config.resources?.cpus ?? "—"}</dd></div>
              <div><dt>内存</dt><dd>{resource.data.config.resources?.memory_gib ? `${resource.data.config.resources.memory_gib} GiB` : "—"}</dd></div>
              <div><dt>磁盘</dt><dd>{resource.data.config.resources?.disk_gib ? `${resource.data.config.resources.disk_gib} GiB` : "—"}</dd></div>
              <div><dt>当前任务</dt><dd>{resource.data.job?.status || "无"}</dd></div>
            </dl>
          </section>
        </>
      )}
    </div>
  );
}

function CacheSettings({ onNavigate }) {
  const [resource, setResource] = useState({ status: "loading", data: null, error: null });
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    loadCacheMaintenance({ signal: controller.signal })
      .then((data) => setResource({ status: "ready", data, error: null }))
      .catch((error) => {
        if (error?.name !== "AbortError") setResource({ status: "error", data: null, error });
      });
    return () => controller.abort();
  }, [reloadKey]);

  const targets = Object.values(resource.data?.overview?.targets || {});

  return (
    <div className="settings-content resource-settings-content">
      <SettingsHeader eyebrow="REGENERABLE DATA" title="缓存" description="查看服务端实际缓存目录占用；清理动作统一在维护页确认执行。" badge="只读状态" />
      {resource.status === "loading" ? (
        <div className="inline-resource-state panel-window"><RefreshCw className="spin" size={18} /><div><strong>正在统计缓存</strong><p>扫描图片、语音和表情派生缓存。</p></div></div>
      ) : resource.status === "error" ? (
        <div className="inline-resource-state panel-window" role="alert"><CircleAlert size={18} /><div><strong>缓存状态不可用</strong><p>{resource.error?.message}</p></div><button type="button" className="secondary-button" onClick={() => setReloadKey((value) => value + 1)}>重试</button></div>
      ) : (
        <>
          <section className="cache-status-grid">
            {targets.map((target) => (
              <article key={target.path}>
                <Archive size={18} />
                <span>{target.label}</span>
                <strong>{formatBytes(target.bytes)}</strong>
                <small>{target.files} 个文件</small>
              </article>
            ))}
          </section>
          <section className="form-section cache-readonly-panel">
            <div className="form-section-head">
              <div><h3>安全清理</h3><p>维护页会显示精确目录、影响范围和服务端确认字符串；本页不会直接删除文件。</p></div>
              <button type="button" className="secondary-button" onClick={() => onNavigate("maintenance")}>
                前往维护页 <ChevronRight size={15} />
              </button>
            </div>
            <dl>
              <div><dt>总占用</dt><dd>{formatBytes(resource.data.overview.total_bytes)}</dd></div>
              <div><dt>文件总数</dt><dd>{resource.data.overview.total_files}</dd></div>
              <div><dt>备份策略</dt><dd>缓存可重新生成，清理前不创建备份</dd></div>
            </dl>
          </section>
        </>
      )}
    </div>
  );
}

export function ResourceSettingsPage({ section, onToast, onNavigate }) {
  if (section === "self-image") return <SelfImageSettings onToast={onToast} />;
  if (section === "workspace") return <WorkspaceSettings />;
  return <CacheSettings onNavigate={onNavigate} />;
}
