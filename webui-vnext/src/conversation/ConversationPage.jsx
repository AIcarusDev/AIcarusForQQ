import { useEffect, useRef, useState } from "react";
import {
  ChevronDown,
  CircleAlert,
  Clock3,
  MessageCircle,
  RefreshCw,
  Send,
  Users,
} from "lucide-react";
import {
  loadCoreChat,
  loadFocusContext,
  loadFocusOverview,
  sendCoreChat,
} from "../api/conversationApi.js";
import { loadRuntimeOverview } from "../api/runtimeApi.js";
import { AgentPage } from "./AgentPage.jsx";

function formatMoment(value) {
  if (!value) return "—";
  const date = typeof value === "number" ? new Date(value) : new Date(String(value));
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function mergeById(current, incoming) {
  const map = new Map(current.map((item) => [item.id, item]));
  for (const item of incoming) map.set(item.id, item);
  return [...map.values()];
}

function createClientId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function InlineState({ icon: Icon = CircleAlert, title, detail, action, actionLabel = "重试" }) {
  return (
    <div className="inline-resource-state" role={action ? "alert" : "status"}>
      <span><Icon size={19} /></span>
      <div><strong>{title}</strong><p>{detail}</p></div>
      {action && <button type="button" className="secondary-button" onClick={action}><RefreshCw size={15} />{actionLabel}</button>}
    </div>
  );
}

function MessageBody({ message }) {
  if (!message.segments.length) return <p>{message.content || "（空消息）"}</p>;
  return (
    <div className="message-segments">
      {message.segments.map((segment, index) => {
        const key = `${segment.type}-${index}`;
        if ((segment.type === "image" || segment.type === "sticker") && segment.src) {
          return <img key={key} src={segment.src} alt={segment.text} loading="lazy" />;
        }
        if (["text", "mention", "emoji"].includes(segment.type)) {
          return <span key={key} className={`segment-${segment.type}`}>{segment.text}</span>;
        }
        return <span key={key} className="segment-placeholder">[{segment.text}]</span>;
      })}
    </div>
  );
}

function MessageList({ messages, pending = [], personalizeUser = false, scrollKey = "default" }) {
  const listRef = useRef(null);
  const contentRef = useRef(null);
  const pinnedToBottomRef = useRef(true);
  const initialFollowRef = useRef(true);
  const positionedKeyRef = useRef("");
  const count = messages.length + pending.length;

  useEffect(() => {
    const target = listRef.current;
    if (!target) return;
    const contextChanged = positionedKeyRef.current !== scrollKey;
    if (contextChanged) initialFollowRef.current = true;
    if (contextChanged || pinnedToBottomRef.current) {
      target.scrollTop = target.scrollHeight;
      pinnedToBottomRef.current = true;
    }
    positionedKeyRef.current = scrollKey;
  }, [count, scrollKey]);

  useEffect(() => {
    const content = contentRef.current;
    if (!content || typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(() => {
      const target = listRef.current;
      if (target && (initialFollowRef.current || pinnedToBottomRef.current)) {
        target.scrollTop = target.scrollHeight;
      }
    });
    observer.observe(content);
    return () => observer.disconnect();
  }, [scrollKey]);

  if (!count) {
    return <div className="message-list conversation-empty"><MessageCircle size={23} /><strong>还没有消息</strong><span>新消息会在这里按时间顺序出现。</span></div>;
  }

  return (
    <div
      className="message-list"
      ref={listRef}
      tabIndex={0}
      onWheel={() => { initialFollowRef.current = false; }}
      onTouchStart={() => { initialFollowRef.current = false; }}
      onPointerDown={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        if (event.clientX >= rect.right - 20) initialFollowRef.current = false;
      }}
      onKeyDown={(event) => {
        if (["ArrowUp", "PageUp", "Home"].includes(event.key)) initialFollowRef.current = false;
      }}
      onScroll={(event) => {
        const target = event.currentTarget;
        pinnedToBottomRef.current = target.scrollHeight - target.scrollTop - target.clientHeight < 56;
      }}
    >
      <div className="message-list-content" ref={contentRef}>
        {messages.map((message) => {
          const mine = personalizeUser && message.role === "user";
          return (
            <article className={`message ${mine ? "mine" : "core"}`} key={message.id}>
              <header><strong>{mine ? "你" : message.sender}</strong><time>{formatMoment(message.timestamp)}</time></header>
              <MessageBody message={message} />
              {message.deliveryState && <small>{message.deliveryState}</small>}
            </article>
          );
        })}
        {pending.map((item) => (
          <article className={`message mine outbox-${item.status}`} key={item.clientId}>
            <header><strong>你</strong><time>{item.status === "failed" ? "发送失败" : "正在发送"}</time></header>
            <p>{item.content}</p>
            {item.error && <small>{item.error}</small>}
          </article>
        ))}
      </div>
    </div>
  );
}

function ChatPage() {
  const [resource, setResource] = useState({ status: "loading", messages: [], runtime: null, error: null });
  const [draft, setDraft] = useState("");
  const [outbox, setOutbox] = useState([]);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    let firstLoad = true;
    const refresh = async () => {
      try {
        const chat = await loadCoreChat({ signal: controller.signal });
        let runtime = null;
        if (firstLoad) runtime = await loadRuntimeOverview({ signal: controller.signal });
        firstLoad = false;
        setResource((current) => ({
          status: "ready",
          messages: mergeById(current.messages, chat.messages),
          runtime: runtime || current.runtime,
          error: null,
        }));
      } catch (error) {
        if (error?.name !== "AbortError") {
          setResource((current) => ({ ...current, status: current.messages.length ? "ready" : "error", error }));
        }
      }
    };
    refresh();
    const timer = window.setInterval(refresh, 4_000);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, [reloadKey]);

  const transmit = async (item) => {
    setOutbox((items) => items.map((current) => current.clientId === item.clientId
      ? { ...current, status: "pending", error: "" }
      : current));
    try {
      const accepted = await sendCoreChat({ content: item.content, clientId: item.clientId });
      setResource((current) => ({ ...current, messages: mergeById(current.messages, [accepted.message]) }));
      setOutbox((items) => items.filter((current) => current.clientId !== item.clientId));
    } catch (error) {
      setOutbox((items) => items.map((current) => current.clientId === item.clientId
        ? { ...current, status: "failed", error: error?.message || "发送失败" }
        : current));
    }
  };

  const send = (event) => {
    event.preventDefault();
    const content = draft.trim();
    if (!content) return;
    const item = { clientId: createClientId(), content, status: "pending", error: "" };
    setDraft("");
    setOutbox((items) => [...items, item]);
    transmit(item);
  };

  if (resource.status === "loading") {
    return <InlineState icon={RefreshCw} title="正在载入 Core 会话" detail="读取最近消息与真实运行状态。" />;
  }
  if (resource.status === "error") {
    return <InlineState title="无法读取 Core 会话" detail={resource.error?.message} action={() => setReloadKey((value) => value + 1)} />;
  }

  const runtime = resource.runtime;
  const running = runtime?.runtime?.state === "running";
  return (
    <div className="wide-workspace chat-workspace">
      <section className="conversation-panel panel-window">
        <div className="panel-header">
          <div><div className="eyebrow">CORE SESSION</div><h3>与 Core 对话</h3></div>
          <span className={`connection-chip ${running ? "live" : "offline"}`}><span />{running ? "可用" : "未运行"}</span>
        </div>
        <MessageList messages={resource.messages} pending={outbox} personalizeUser />
        {outbox.some((item) => item.status === "failed") && (
          <div className="outbox-retry-row">
            <span>有消息尚未送达；重试会沿用同一请求标识，不会重复写入。</span>
            <button type="button" onClick={() => outbox.filter((item) => item.status === "failed").forEach(transmit)}>重试失败项</button>
          </div>
        )}
        <form className="composer" onSubmit={send}>
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder="输入消息，Enter 发送，Shift + Enter 换行"
            rows={1}
            maxLength={8_000}
          />
          <button type="submit" aria-label="发送消息" disabled={!draft.trim()}><Send size={18} /></button>
        </form>
      </section>
      <aside className="chat-context panel-window">
        <div className="eyebrow">RUNTIME</div><h3>运行状态</h3>
        <dl>
          <div><dt>模式</dt><dd>{runtime?.runtime?.modeLabel || "—"}</dd></div>
          <div><dt>模型</dt><dd>{runtime?.identity?.model || "未配置"}</dd></div>
          <div><dt>当前焦点</dt><dd title={runtime?.activity?.currentFocus || ""}>{runtime?.activity?.currentFocus || "暂无"}</dd></div>
          <div><dt>会话消息</dt><dd>{resource.messages.length} 条</dd></div>
        </dl>
      </aside>
    </div>
  );
}

function FocusPage() {
  const [overview, setOverview] = useState({ status: "loading", data: null, error: null });
  const [context, setContext] = useState({ status: "idle", messages: [], error: null });
  const [selectedKey, setSelectedKey] = useState("");
  const [reloadKey, setReloadKey] = useState(0);
  const [contextReloadKey, setContextReloadKey] = useState(0);
  const [sessionsExpanded, setSessionsExpanded] = useState(true);
  const [turnsExpanded, setTurnsExpanded] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    const refresh = async () => {
      try {
        const data = await loadFocusOverview({ signal: controller.signal });
        setOverview({ status: "ready", data, error: null });
        setSelectedKey((current) => current || data.currentFocus || data.sessions[0]?.key || "");
      } catch (error) {
        if (error?.name !== "AbortError") setOverview({ status: "error", data: null, error });
      }
    };
    refresh();
    const timer = window.setInterval(refresh, 6_000);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [reloadKey]);

  useEffect(() => {
    if (!selectedKey) {
      setContext({ status: "ready", messages: [], error: null });
      return undefined;
    }
    const controller = new AbortController();
    const refresh = async () => {
      try {
        const data = await loadFocusContext(selectedKey, { signal: controller.signal });
        setContext({ status: "ready", messages: data.messages, error: null });
      } catch (error) {
        if (error?.name !== "AbortError") setContext((current) => ({ ...current, status: "error", error }));
      }
    };
    setContext((current) => ({ ...current, status: "loading", error: null }));
    refresh();
    const timer = window.setInterval(refresh, 4_000);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [contextReloadKey, selectedKey]);

  if (overview.status === "loading") return <InlineState icon={RefreshCw} title="正在读取焦点" detail="同步会话、最近轮次与上下文。" />;
  if (overview.status === "error") return <InlineState title="焦点数据不可用" detail={overview.error?.message} action={() => setReloadKey((value) => value + 1)} />;

  const data = overview.data;
  return (
    <div className="focus-workspace">
      <section className="focus-sidebar panel-window">
        <div className="panel-header">
          <div><div className="eyebrow">SESSIONS</div><h3>已知会话</h3></div>
          <span className="focus-section-count">{data.sessions.length} 个</span>
          <button
            type="button"
            className="focus-mobile-section-toggle"
            aria-expanded={sessionsExpanded}
            aria-controls="focus-session-list"
            onClick={() => setSessionsExpanded((expanded) => !expanded)}
          >
            <span>{data.sessions.length} 个</span>
            <ChevronDown size={16} />
          </button>
        </div>
        <div id="focus-session-list" className={`session-list ${sessionsExpanded ? "" : "is-mobile-collapsed"}`}>
          {data.sessions.length ? data.sessions.map((session) => (
            <button key={session.key} type="button" className={selectedKey === session.key ? "active" : ""} onClick={() => setSelectedKey(session.key)}>
              <span><strong>{session.label}</strong><small>{session.platform} · {session.type || "session"}</small></span>
              {data.currentFocus === session.key && <i title="当前焦点" />}
            </button>
          )) : <InlineState icon={Users} title="暂无会话" detail="Core 识别会话后会自动出现在这里。" />}
        </div>
        <div className={`recent-turns ${turnsExpanded ? "" : "is-mobile-collapsed"}`}>
          <div className="subsection-heading">
            <span>最近轮次</span>
            <small>{data.turns.length}</small>
            <button
              type="button"
              className="focus-mobile-section-toggle"
              aria-expanded={turnsExpanded}
              aria-controls="focus-recent-turns"
              onClick={() => setTurnsExpanded((expanded) => !expanded)}
            >
              <span>{turnsExpanded ? "收起" : "展开"}</span>
              <ChevronDown size={16} />
            </button>
          </div>
          <div id="focus-recent-turns" className="recent-turn-list">
          {data.turns.slice(0, 6).map((turn) => (
            <button key={turn.id} type="button" onClick={() => turn.sessionKey && setSelectedKey(turn.sessionKey)}>
              <Clock3 size={14} /><span><strong>{turn.conversation}</strong><small>{formatMoment(turn.createdAt)} · {turn.toolCount} 个工具</small></span>
            </button>
          ))}
          </div>
        </div>
      </section>
      <section className="focus-context panel-window">
        <div className="panel-header">
          <div><div className="eyebrow">CONTEXT</div><h3>{data.sessions.find((session) => session.key === selectedKey)?.label || "会话上下文"}</h3></div>
          {selectedKey && <span className="context-key" title={selectedKey}>{selectedKey}</span>}
        </div>
        {context.status === "loading" && !context.messages.length
          ? <InlineState icon={RefreshCw} title="正在读取消息" detail="从会话存储载入最近上下文。" />
          : context.status === "error"
            ? <InlineState title="上下文读取失败" detail={context.error?.message} action={() => setContextReloadKey((value) => value + 1)} />
            : <MessageList messages={context.messages} scrollKey={selectedKey} />}
      </section>
    </div>
  );
}

export function ConversationPage({ page }) {
  if (page === "chat") return <ChatPage />;
  if (page === "focus") return <FocusPage />;
  return <AgentPage />;
}
