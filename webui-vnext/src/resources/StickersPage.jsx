import { useEffect, useMemo, useRef, useState } from "react";
import {
  CircleAlert,
  Image as ImageIcon,
  RefreshCw,
  Save,
  Search,
  Trash2,
  Upload,
} from "lucide-react";
import {
  deleteSticker,
  loadStickers,
  reconcileStickers,
  updateSticker,
  uploadSticker,
} from "../api/resourceApi.js";

function StickerCard({ sticker, busy, onSave, onDelete }) {
  const [description, setDescription] = useState(sticker.description);
  const dirty = description.trim() !== sticker.description;

  return (
    <article className="sticker-card panel-window">
      <div className="sticker-image-wrap">
        <img src={sticker.imageUrl} alt={sticker.description || `表情包 ${sticker.id}`} loading="lazy" />
        <span>#{sticker.id}</span>
      </div>
      <label>
        <span>用途描述</span>
        <textarea value={description} onChange={(event) => setDescription(event.target.value)} maxLength={200} rows={2} placeholder="说明适合在什么语境使用" />
      </label>
      <div className="sticker-card-actions">
        <button type="button" className="secondary-button" disabled={!dirty || busy} onClick={() => onSave(sticker.id, description.trim())}><Save size={14} />保存</button>
        <button type="button" className="danger-text-button" disabled={busy} onClick={() => onDelete(sticker)}><Trash2 size={14} />删除</button>
      </div>
    </article>
  );
}

export function StickersPage({ onToast }) {
  const [resource, setResource] = useState({ status: "loading", items: [], error: null });
  const [query, setQuery] = useState("");
  const [description, setDescription] = useState("");
  const [busyIds, setBusyIds] = useState(new Set());
  const [uploading, setUploading] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  const inputRef = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    loadStickers({ signal: controller.signal })
      .then((items) => setResource({ status: "ready", items, error: null }))
      .catch((error) => {
        if (error?.name !== "AbortError") setResource((current) => ({ ...current, status: "error", error }));
      });
    return () => controller.abort();
  }, [reloadKey]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase("zh-CN");
    if (!needle) return resource.items;
    return resource.items.filter((item) => `${item.id} ${item.description}`.toLocaleLowerCase("zh-CN").includes(needle));
  }, [query, resource.items]);

  const refresh = () => setReloadKey((value) => value + 1);
  const markBusy = (id, busy) => setBusyIds((current) => {
    const next = new Set(current);
    if (busy) next.add(id); else next.delete(id);
    return next;
  });

  const uploadFiles = async (files) => {
    if (!files.length) return;
    setUploading(true);
    try {
      let duplicates = 0;
      for (const file of files) {
        const result = await uploadSticker(file, description.trim());
        if (result?.duplicate) duplicates += 1;
      }
      setDescription("");
      onToast(duplicates ? `上传完成，其中 ${duplicates} 张已存在` : `已上传 ${files.length} 张表情包`);
      refresh();
    } catch (error) {
      onToast(error?.message || "上传失败");
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const saveDescription = async (id, nextDescription) => {
    markBusy(id, true);
    try {
      await updateSticker(id, nextDescription);
      setResource((current) => ({ ...current, items: current.items.map((item) => item.id === id ? { ...item, description: nextDescription } : item) }));
      onToast(`表情包 #${id} 描述已保存`);
    } catch (error) {
      onToast(error?.message || "描述保存失败");
    } finally {
      markBusy(id, false);
    }
  };

  const remove = async (sticker) => {
    if (!window.confirm(`确定删除表情包 #${sticker.id}？此操作不会删除其他资源。`)) return;
    markBusy(sticker.id, true);
    try {
      await deleteSticker(sticker.id);
      setResource((current) => ({ ...current, items: current.items.filter((item) => item.id !== sticker.id) }));
      onToast(`表情包 #${sticker.id} 已删除`);
    } catch (error) {
      onToast(error?.message || "删除失败");
      markBusy(sticker.id, false);
    }
  };

  const reconcile = async () => {
    setUploading(true);
    try {
      await reconcileStickers();
      onToast("表情包索引已核对并修复");
      refresh();
    } catch (error) {
      onToast(error?.message || "索引核对失败");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="stickers-workspace">
      <div className="sticker-sticky-tools">
        <section className="sticker-upload-panel panel-window">
          <div className="panel-header">
            <div><div className="eyebrow">RESOURCE LIBRARY</div><h3>添加表情包</h3></div>
            <span>{resource.items.length} / 30</span>
          </div>
          <div className="sticker-upload-row">
            <span className="upload-icon"><Upload size={20} /></span>
            <label className="upload-description"><span>统一描述（可选）</span><input value={description} onChange={(event) => setDescription(event.target.value)} maxLength={200} placeholder="例如：轻松回应或表示赞同" /></label>
            <label className={`primary-button file-button ${uploading ? "disabled" : ""}`}>
              <input ref={inputRef} type="file" accept="image/*" multiple disabled={uploading} onChange={(event) => uploadFiles([...event.target.files])} />
              {uploading ? <RefreshCw className="spin" size={16} /> : <ImageIcon size={16} />}{uploading ? "处理中" : "选择图片"}
            </label>
          </div>
        </section>

        <section className="sticker-toolbar panel-window">
          <label className="search-box"><Search size={16} /><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索编号或描述" /></label>
          <button type="button" className="secondary-button" disabled={uploading} onClick={reconcile}><RefreshCw size={15} />核对索引</button>
        </section>
      </div>

      {resource.status === "loading" && !resource.items.length ? (
        <div className="inline-resource-state panel-window"><RefreshCw className="spin" size={19} /><div><strong>正在读取表情包</strong><p>同步图片索引与描述。</p></div></div>
      ) : resource.status === "error" ? (
        <div className="inline-resource-state panel-window" role="alert"><CircleAlert size={19} /><div><strong>表情包读取失败</strong><p>{resource.error?.message}</p></div><button type="button" className="secondary-button" onClick={refresh}>重试</button></div>
      ) : filtered.length ? (
        <section className="sticker-grid">
          {filtered.map((sticker) => <StickerCard key={`${sticker.id}-${sticker.description}`} sticker={sticker} busy={busyIds.has(sticker.id)} onSave={saveDescription} onDelete={remove} />)}
        </section>
      ) : (
        <div className="resource-empty panel-window"><ImageIcon size={24} /><strong>{resource.items.length ? "没有匹配项" : "还没有表情包"}</strong><span>{resource.items.length ? "换一个关键词试试。" : "选择图片后，它们会出现在这里。"}</span></div>
      )}
    </div>
  );
}
