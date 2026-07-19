"use client";

import { useEffect, useRef } from "react";
import { BookOpen, ChevronRight, FileText, ShieldCheck, X } from "lucide-react";
import {
  MEMORY_QUERY_EXAMPLES,
  MEMORYQL_CLAUSE_REFERENCE,
  MEMORYQL_GUIDE_SECTIONS,
  MEMORYQL_META,
  MEMORYQL_QUERY_SHAPE,
} from "./memoryQueryGuide.js";

export function MemoryQueryGuide({ open, onClose, onUseExample }) {
  const dialogRef = useRef(null);
  const closeButtonRef = useRef(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (open && !dialog.open) {
      dialog.showModal();
      window.requestAnimationFrame(() => closeButtonRef.current?.focus());
    } else if (!open && dialog.open) {
      dialog.close();
    }
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const closeOnEscape = (event) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      window.removeEventListener("keydown", closeOnEscape);
      document.body.style.overflow = previousOverflow;
    };
  }, [open, onClose]);

  const scrollToSection = (sectionId) => {
    dialogRef.current
      ?.querySelector(`[data-memoryql-section="${sectionId}"]`)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const applyGuideExample = (example) => {
    onUseExample(example);
    onClose();
  };

  return (
    <dialog
      id="memoryql-guide-dialog"
      ref={dialogRef}
      className="memory-doc-dialog"
      aria-labelledby="memory-doc-title"
      aria-describedby="memory-doc-description"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClick={(event) => {
        if (event.target === dialogRef.current) onClose();
      }}
    >
      <div className="memory-doc-shell">
        <header className="memory-doc-header">
          <div className="memory-doc-heading">
            <span className="memory-doc-icon"><BookOpen size={19} /></span>
            <div>
              <div className="eyebrow">MEMORYQL REFERENCE</div>
              <h2 id="memory-doc-title">语法说明书</h2>
            </div>
          </div>
          <div className="memory-doc-header-actions">
            <span>{MEMORYQL_META.version}</span>
            <button ref={closeButtonRef} type="button" aria-label="关闭语法说明书" onClick={onClose}>
              <X size={19} />
            </button>
          </div>
        </header>

        <div className="memory-doc-layout">
          <nav className="memory-doc-nav" aria-label="MemoryQL 文档章节">
            <div className="eyebrow">CONTENTS</div>
            {MEMORYQL_GUIDE_SECTIONS.map((section) => (
              <button key={section.id} type="button" onClick={() => scrollToSection(section.id)}>
                <span>{section.index}</span>
                {section.title}
              </button>
            ))}
          </nav>

          <article className="memory-doc-content">
            <section className="memory-doc-intro">
              <div>
                <span className="memory-doc-kicker"><FileText size={14} /> {MEMORYQL_META.status}</span>
                <h3>先限定问题，再观察局部关系</h3>
                <p id="memory-doc-description">
                  MemoryQL 面向大型记忆图的精确读取。它要求查询显式声明匹配范围、返回形式和结果预算，避免把上千节点重新塞回一张失去作用的全局图。
                </p>
              </div>
              <pre><code>{MEMORYQL_QUERY_SHAPE}</code></pre>
            </section>

            <section className="memory-doc-reference" aria-labelledby="memoryql-quick-reference">
              <div className="memory-doc-section-heading">
                <div><span>QUICK REFERENCE</span><h3 id="memoryql-quick-reference">子句速查</h3></div>
              </div>
              <div className="memory-doc-reference-grid">
                {MEMORYQL_CLAUSE_REFERENCE.map(([clause, purpose, requirement]) => (
                  <div key={clause}>
                    <code>{clause}</code>
                    <span>{purpose}</span>
                    <small>{requirement}</small>
                  </div>
                ))}
              </div>
            </section>

            {MEMORYQL_GUIDE_SECTIONS.map((section) => (
              <section className="memory-doc-section" key={section.id} data-memoryql-section={section.id}>
                <div className="memory-doc-section-heading">
                  <span>{section.index}</span>
                  <div><span>SYNTAX</span><h3>{section.title}</h3></div>
                </div>
                <p>{section.summary}</p>
                {section.code && <pre><code>{section.code}</code></pre>}
                <ul>
                  {section.bullets.map((bullet) => <li key={bullet}>{bullet}</li>)}
                </ul>
              </section>
            ))}

            <section className="memory-doc-examples" aria-labelledby="memoryql-examples-title">
              <div className="memory-doc-section-heading">
                <div><span>READY TO RUN</span><h3 id="memoryql-examples-title">完整示例</h3></div>
              </div>
              <div className="memory-doc-example-list">
                {MEMORY_QUERY_EXAMPLES.map((example) => (
                  <section key={example.label}>
                    <div>
                      <div><strong>{example.label}</strong><p>{example.description}</p></div>
                      <button type="button" onClick={() => applyGuideExample(example)}>载入编辑器 <ChevronRight size={14} /></button>
                    </div>
                    <pre><code>{example.query}</code></pre>
                  </section>
                ))}
              </div>
            </section>

            <aside className="memory-doc-boundary">
              <ShieldCheck size={18} />
              <div><strong>服务边界</strong><p>查询读取真实记忆，但只返回本次请求的有界投影；它不会加载完整实例图，也不提供任何写入或维护命令。</p></div>
            </aside>
          </article>
        </div>
      </div>
    </dialog>
  );
}
