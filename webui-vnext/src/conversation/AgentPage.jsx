import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Activity,
  ArrowDown,
  Bot,
  Braces,
  Check,
  ChevronDown,
  Copy,
  Globe2,
  History,
  ListTree,
  LoaderCircle,
  RefreshCw,
  Target,
  Wrench,
} from "lucide-react";
import {
  loadAgentState,
  loadAgentTurns,
  subscribeAgentEvents,
} from "../api/conversationApi.js";
import { smoothScrollElement } from "../scrolling.js";
import {
  AGENT_EVENT_FILTERS,
  agentEventDetail,
  agentEventLabel,
  agentPayloadText,
  agentRoundStatus,
  projectAgentRounds,
  roundMatchesAgentFilter,
  summarizeAgentPayload,
} from "./agentProjection.js";

const EVENT_LIMIT = 600;
const HISTORY_PAGE_SIZE = 24;
const FOLLOW_THRESHOLD = 64;
const USER_SCROLL_INTENT_MS = 800;

function formatClock(value) {
  if (!value) return "—";
  const date = new Date(Number(value) || value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function formatMoment(value) {
  if (!value) return "—";
  const date = new Date(Number(value) || value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function formatDuration(value) {
  const milliseconds = Number(value);
  if (!Number.isFinite(milliseconds) || milliseconds < 0) return "—";
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  if (milliseconds < 60_000) return `${(milliseconds / 1000).toFixed(milliseconds < 10_000 ? 1 : 0)} s`;
  const minutes = Math.floor(milliseconds / 60_000);
  const seconds = Math.round((milliseconds % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

function eventKey(event, index) {
  const sequence = Number(event?.seq);
  return Number.isFinite(sequence) && sequence > 0
    ? `seq:${sequence}`
    : `${event?.type || "event"}:${event?.created_at || 0}:${event?.round_id || ""}:${index}`;
}

function mergeAgentEvents(current, incoming) {
  const merged = new Map(current.map((event, index) => [eventKey(event, index), event]));
  for (const [index, event] of incoming.entries()) merged.set(eventKey(event, current.length + index), event);
  return [...merged.values()]
    .sort((left, right) => (Number(left.seq) || 0) - (Number(right.seq) || 0))
    .slice(-EVENT_LIMIT);
}

function mergeAgentTurns(current, incoming) {
  const merged = new Map(current.map((turn) => [turn.id, turn]));
  for (const turn of incoming) {
    if (turn?.id) merged.set(turn.id, turn);
  }
  return [...merged.values()].sort((left, right) => right.createdAt - left.createdAt);
}

function AgentResourceState({ title, detail, action }) {
  return (
    <div className="inline-resource-state" role={action ? "alert" : "status"}>
      <span><Activity size={19} /></span>
      <div><strong>{title}</strong><p>{detail}</p></div>
      {action && <button type="button" className="secondary-button" onClick={action}><RefreshCw size={15} />重试</button>}
    </div>
  );
}

function CopyTextButton({ value, label }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(String(value || ""));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 900);
    } catch {
      setCopied(false);
    }
  };

  return (
    <button
      type="button"
      className="agent-icon-button"
      onClick={copy}
      aria-label={copied ? "已复制" : label}
      title={copied ? "已复制" : label}
    >
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </button>
  );
}

function ToolExecution({ tool }) {
  const statusLabel = {
    planned: "待执行",
    running: "执行中",
    done: "已完成",
    blocked: "已阻止",
    skipped: "已跳过",
    error: "失败",
  }[tool.status] || tool.status;
  const preview = tool.argsPreview || summarizeAgentPayload(tool.arguments) || tool.progress;

  return (
    <details className={`agent-tool-execution is-${tool.status}`}>
      <summary>
        <span className="agent-tool-state" aria-label={statusLabel} />
        <strong>{tool.name}</strong>
        {preview && <span className="agent-tool-inline-preview" title={preview}>{preview}</span>}
        <span className="agent-tool-status">{statusLabel}</span>
        <time>{formatDuration(tool.elapsedMs)}</time>
        <ChevronDown size={15} aria-hidden="true" />
      </summary>
      <div className="agent-tool-detail-grid">
        <section>
          <span>参数</span>
          <pre>{agentPayloadText(tool.arguments)}</pre>
        </section>
        <section>
          <span>结果</span>
          <pre>{agentPayloadText(tool.result ?? tool.resultPreview ?? tool.progress)}</pre>
        </section>
      </div>
    </details>
  );
}

function RuntimeDetails({ events }) {
  const visibleEvents = events.filter((event) => event.type !== "cognition_delta");
  if (!visibleEvents.length) return null;
  return (
    <details className="agent-runtime-details">
      <summary><Braces size={15} />运行细节 <span>{visibleEvents.length} 条</span></summary>
      <ol>
        {visibleEvents.map((event, index) => {
          const detail = agentEventDetail(event);
          return (
            <li key={eventKey(event, index)}>
              <time>{formatClock(event.created_at)}</time>
              <strong>{agentEventLabel(event.type)}</strong>
              {detail && <span title={detail}>{detail}</span>}
            </li>
          );
        })}
      </ol>
    </details>
  );
}

function AgentRound({ round, filter, worldOpen, onToggleWorld }) {
  const status = agentRoundStatus(round);
  const showCognition = filter === "all" || filter === "cognition";
  const showTools = filter === "all" || filter === "tools";
  const showRuntime = filter === "all" || filter === "runtime";
  const actionStarted = round.events.some((event) => event.type === "action_start");
  const modelLabel = [round.provider, round.model].filter(Boolean).join(" · ");
  const cognitionPending = round.status === "running" && round.cognitionState !== "discarded";

  return (
    <article className={`agent-round is-${status.id}`} data-agent-round-id={round.id}>
      <header className="agent-round-header">
        <span className={`agent-round-status-dot is-${status.id}`} />
        <div>
          <strong>{status.label}</strong>
          <span title={round.sessionKey}>{round.conversation}</span>
        </div>
        <div className="agent-round-header-meta">
          {round.retries > 0 && <span>重试 {round.retries}</span>}
          {round.tools.length > 0 && <span>{round.tools.length} 工具</span>}
          {(round.promptTokens > 0 || round.outputTokens > 0) && <span>↑{round.promptTokens} ↓{round.outputTokens}</span>}
          {round.elapsedMs !== null && <span>{formatDuration(round.elapsedMs)}</span>}
          <time>{formatMoment(round.createdAt)}</time>
        </div>
        {modelLabel && <small title={modelLabel}>{modelLabel}</small>}
      </header>

      <div className="agent-round-flow">
        {showRuntime && round.worldXml && (
          <section className="agent-phase agent-world-phase">
            <span className="agent-phase-marker"><Globe2 size={15} /></span>
            <div className="agent-phase-surface">
              <header>
                <div><strong>World</strong><span>此轮模型观察到的世界帧 · {round.worldXml.length} chars</span></div>
                <div>
                  <CopyTextButton value={round.worldXml} label="复制 World 原文" />
                  <button
                    type="button"
                    className="agent-world-toggle"
                    onClick={onToggleWorld}
                    aria-expanded={worldOpen}
                  >
                    {worldOpen ? "收起" : "展开"}<ChevronDown size={14} />
                  </button>
                </div>
              </header>
              {worldOpen && <pre className="agent-world-content">{round.worldXml}</pre>}
            </div>
          </section>
        )}

        {showCognition && (round.cognition || cognitionPending || round.cognitionState === "discarded") && (
          <section className="agent-phase agent-cognition-phase">
            <span className="agent-phase-marker"><Bot size={15} /></span>
            <div className="agent-phase-surface">
              <header>
                <div>
                  <strong>认知</strong>
                  <span>{round.cognitionState === "streaming" ? "正在生成" : round.cognitionState === "discarded" ? "本次认知已丢弃" : "连续记录"}</span>
                </div>
                {round.cognition && <CopyTextButton value={round.cognition} label="复制认知原文" />}
              </header>
              {round.cognition
                ? <p className="agent-cognition-text">{round.cognition}</p>
                : round.cognitionState === "discarded"
                  ? <p className="agent-phase-placeholder">已丢弃重复认知，等待重试 · {round.cognitionDiscardReason}</p>
                  : <p className="agent-phase-placeholder"><LoaderCircle size={15} />等待认知输出…</p>}
            </div>
          </section>
        )}

        {showTools && (round.tools.length > 0 || actionStarted) && (
          <section className="agent-phase agent-action-phase">
            <span className="agent-phase-marker"><Wrench size={15} /></span>
            <div className="agent-phase-surface">
              <header>
                <div><strong>动作</strong><span>一次调用链合并显示规划、执行与结果</span></div>
                <span className="agent-phase-count">{round.tools.length} 个工具</span>
              </header>
              {round.tools.length
                ? <div className="agent-tool-list">{round.tools.map((tool) => <ToolExecution key={tool.key} tool={tool} />)}</div>
                : <p className="agent-phase-placeholder"><LoaderCircle size={15} />正在规划工具…</p>}
            </div>
          </section>
        )}

        {showRuntime && <RuntimeDetails events={round.events} />}
      </div>

      {round.status === "error" && <p className="agent-round-error">{round.error || "此轮运行未成功完成"}</p>}
    </article>
  );
}

function AgentRunIndex({
  rounds,
  selectedRoundId,
  expanded,
  loadingOlder,
  hasOlder,
  historyError,
  onToggle,
  onSelect,
  onLoadOlder,
}) {
  const newestFirst = [...rounds].reverse();
  return (
    <aside className={`agent-run-index panel-window ${expanded ? "is-expanded" : "is-collapsed"}`} aria-label="Agent 轮次跳转索引">
      <header>
        <div><ListTree size={16} /><strong>轮次索引</strong><span>{rounds.length}</span></div>
        <button type="button" className="agent-index-toggle" onClick={onToggle} aria-expanded={expanded} aria-controls="agent-run-list">
          {expanded ? "收起" : "展开"}<ChevronDown size={14} />
        </button>
      </header>
      <div className="agent-run-list" id="agent-run-list">
        {newestFirst.map((round) => {
          const status = agentRoundStatus(round);
          return (
            <button
              type="button"
              key={round.id}
              className={selectedRoundId === round.id ? "active" : ""}
              onClick={() => onSelect(round.id)}
              aria-current={selectedRoundId === round.id ? "true" : undefined}
            >
              <span className={`agent-round-status-dot is-${status.id}`} />
              <span>
                <strong title={round.conversation}>{round.conversation}</strong>
                <small>{status.label} · {round.tools.length} 工具 · {formatDuration(round.elapsedMs)}</small>
              </span>
              <time>{formatClock(round.createdAt)}</time>
            </button>
          );
        })}
        {!rounds.length && <p className="agent-index-empty">暂无可跳转轮次</p>}
        {(hasOlder || historyError) && (
          <button type="button" className="agent-load-history" onClick={onLoadOlder} disabled={loadingOlder}>
            {loadingOlder ? <LoaderCircle size={14} /> : <History size={14} />}
            {loadingOlder ? "正在加载…" : historyError ? "加载失败，重试" : "加载更早记录"}
          </button>
        )}
        {historyError && <p className="agent-history-error">{historyError}</p>}
      </div>
    </aside>
  );
}

function AgentStatusCards({ data, connection, stats, eventCursor }) {
  const connectionLabel = connection === "live" ? "已连接" : connection === "reconnecting" ? "正在重连" : "正在连接";
  return (
    <section className="agent-summary-grid" aria-label="Agent 当前状态">
      <article className="panel-window"><Bot size={18} /><span>模型</span><strong>{data.model || "未配置"}</strong><small>{data.provider || "未知供应商"}</small></article>
      <article className="panel-window"><Target size={18} /><span>当前焦点</span><strong title={data.currentFocus}>{data.currentFocus || "暂无"}</strong><small>{data.sessions.length} 个已知会话</small></article>
      <article className="panel-window" role="status"><Activity size={18} /><span>实时连接</span><strong>{connectionLabel}</strong><small>游标 {stats.latest_seq || eventCursor || 0}</small></article>
    </section>
  );
}

export function AgentPage() {
  const [resource, setResource] = useState({ status: "loading", data: null, error: null });
  const [events, setEvents] = useState([]);
  const [turns, setTurns] = useState([]);
  const [stats, setStats] = useState({});
  const [connection, setConnection] = useState("connecting");
  const [filter, setFilter] = useState("all");
  const [reloadKey, setReloadKey] = useState(0);
  const [selectedRoundId, setSelectedRoundId] = useState("");
  const [expandedWorlds, setExpandedWorlds] = useState(() => new Set());
  const [indexExpanded, setIndexExpanded] = useState(false);
  const [following, setFollowing] = useState(true);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [hasOlder, setHasOlder] = useState(true);
  const [historyError, setHistoryError] = useState("");
  const timelineRef = useRef(null);
  const timelineContentRef = useRef(null);
  const pinnedToBottomRef = useRef(true);
  const lastUserScrollIntentRef = useRef(0);
  const initiallyPositionedRef = useRef(false);
  const prependAnchorRef = useRef(null);
  const historyAbortRef = useRef(null);
  const scrollAnimationCancelRef = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    let stream = null;
    let flushTimer = 0;
    let pendingEvents = [];
    let resetPending = false;

    const flushEvents = () => {
      flushTimer = 0;
      const batch = pendingEvents;
      pendingEvents = [];
      const shouldReset = resetPending;
      resetPending = false;
      if (batch.length || shouldReset) {
        setEvents((current) => mergeAgentEvents(shouldReset ? [] : current, batch));
      }
    };

    const queueEvents = (incoming, envelope) => {
      if (envelope?.cursor_reset === true) resetPending = true;
      pendingEvents.push(...incoming);
      if (!flushTimer) flushTimer = window.setTimeout(flushEvents, 50);
    };

    loadAgentState({ signal: controller.signal })
      .then((data) => {
        setResource({ status: "ready", data, error: null });
        setEvents(data.events.slice(-EVENT_LIMIT));
        setTurns(data.turns);
        setHasOlder(data.turns.length >= HISTORY_PAGE_SIZE);
        setStats(data.stats);
        stream = subscribeAgentEvents({
          initialCursor: data.latestSeq,
          initialStreamId: data.streamId,
          signal: controller.signal,
          onEvents: queueEvents,
          onStats: setStats,
          onStatus: setConnection,
          onError: (error) => setResource((current) => ({ ...current, streamError: error })),
        });
      })
      .catch((error) => {
        if (error?.name !== "AbortError") setResource({ status: "error", data: null, error });
      });

    return () => {
      controller.abort();
      stream?.close();
      window.clearTimeout(flushTimer);
    };
  }, [reloadKey]);

  useEffect(() => () => {
    historyAbortRef.current?.abort();
    scrollAnimationCancelRef.current?.();
  }, []);

  const rounds = useMemo(() => projectAgentRounds(events, turns), [events, turns]);
  const visibleRounds = useMemo(
    () => rounds.filter((round) => roundMatchesAgentFilter(round, filter)),
    [rounds, filter],
  );
  const eventCursor = Number(events.at(-1)?.seq) || 0;
  const scrollSignature = `${filter}:${visibleRounds.length}:${eventCursor}:${visibleRounds.at(-1)?.updatedAt || 0}`;
  const effectiveSelectedRoundId = visibleRounds.some((round) => round.id === selectedRoundId)
    ? selectedRoundId
    : visibleRounds.at(-1)?.id || "";

  useLayoutEffect(() => {
    const timeline = timelineRef.current;
    if (!timeline) return;
    if (prependAnchorRef.current) {
      const anchor = prependAnchorRef.current;
      prependAnchorRef.current = null;
      timeline.scrollTop = anchor.top + (timeline.scrollHeight - anchor.height);
      return;
    }
    if (!initiallyPositionedRef.current || pinnedToBottomRef.current) {
      timeline.scrollTop = timeline.scrollHeight;
      initiallyPositionedRef.current = true;
    }
  }, [scrollSignature]);

  useEffect(() => {
    const timeline = timelineRef.current;
    const content = timelineContentRef.current;
    if (!timeline || !content || typeof ResizeObserver === "undefined") return undefined;
    let frame = 0;
    const observer = new ResizeObserver(() => {
      if (!pinnedToBottomRef.current) return;
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => { timeline.scrollTop = timeline.scrollHeight; });
    });
    observer.observe(content);
    return () => {
      observer.disconnect();
      window.cancelAnimationFrame(frame);
    };
  }, [resource.status]);

  const markTimelineUserScrollIntent = useCallback(() => {
    scrollAnimationCancelRef.current?.();
    scrollAnimationCancelRef.current = null;
    lastUserScrollIntentRef.current = performance.now();
  }, []);

  const handleTimelineScroll = useCallback(() => {
    const timeline = timelineRef.current;
    if (!timeline) return;
    const nearBottom = timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight <= FOLLOW_THRESHOLD;
    const hasUserIntent = performance.now() - lastUserScrollIntentRef.current <= USER_SCROLL_INTENT_MS;
    if (hasUserIntent) {
      pinnedToBottomRef.current = nearBottom;
      setFollowing((current) => current === nearBottom ? current : nearBottom);
    } else if (pinnedToBottomRef.current && !nearBottom) {
      window.requestAnimationFrame(() => { timeline.scrollTop = timeline.scrollHeight; });
      return;
    }

    const threshold = timeline.scrollTop + 96;
    const nodes = [...timeline.querySelectorAll("[data-agent-round-id]")];
    let visibleId = nodes[0]?.dataset.agentRoundId || "";
    for (const node of nodes) {
      if (node.offsetTop <= threshold) visibleId = node.dataset.agentRoundId || visibleId;
      else break;
    }
    if (nearBottom && nodes.length) visibleId = nodes.at(-1).dataset.agentRoundId || visibleId;
    if (visibleId) setSelectedRoundId((current) => current === visibleId ? current : visibleId);
  }, []);

  const scrollToLatest = useCallback(() => {
    const timeline = timelineRef.current;
    if (!timeline) return;
    scrollAnimationCancelRef.current?.();
    scrollAnimationCancelRef.current = null;
    pinnedToBottomRef.current = false;
    lastUserScrollIntentRef.current = 0;
    setFollowing(false);
    setSelectedRoundId(visibleRounds.at(-1)?.id || "");
    scrollAnimationCancelRef.current = smoothScrollElement(timeline, timeline.scrollHeight, {
      minimumDuration: 320,
      maximumDuration: 720,
      onComplete: () => {
        pinnedToBottomRef.current = true;
        setFollowing(true);
      },
    });
  }, [visibleRounds]);

  const jumpToRound = useCallback((roundId) => {
    const timeline = timelineRef.current;
    if (!timeline) return;
    const isLatest = visibleRounds.at(-1)?.id === roundId;
    if (isLatest) {
      scrollToLatest();
      return;
    }
    pinnedToBottomRef.current = false;
    lastUserScrollIntentRef.current = 0;
    setFollowing(false);
    setSelectedRoundId(roundId);
    const node = [...timeline.querySelectorAll("[data-agent-round-id]")]
      .find((item) => item.dataset.agentRoundId === roundId);
    if (node) {
      scrollAnimationCancelRef.current?.();
      const timelineRect = timeline.getBoundingClientRect();
      const nodeRect = node.getBoundingClientRect();
      const targetTop = timeline.scrollTop + nodeRect.top - timelineRect.top - 12;
      scrollAnimationCancelRef.current = smoothScrollElement(
        timeline,
        Math.max(0, targetTop),
        { minimumDuration: 320, maximumDuration: 720 },
      );
    }
  }, [scrollToLatest, visibleRounds]);

  const changeFilter = (nextFilter) => {
    scrollAnimationCancelRef.current?.();
    scrollAnimationCancelRef.current = null;
    pinnedToBottomRef.current = true;
    lastUserScrollIntentRef.current = 0;
    setFollowing(true);
    setFilter(nextFilter);
  };

  const toggleWorld = (roundId) => {
    setExpandedWorlds((current) => {
      const next = new Set(current);
      if (next.has(roundId)) next.delete(roundId);
      else next.add(roundId);
      return next;
    });
  };

  const loadOlder = useCallback(async () => {
    if (loadingOlder || !hasOlder) return;
    scrollAnimationCancelRef.current?.();
    scrollAnimationCancelRef.current = null;
    const oldest = turns.reduce((minimum, turn) => {
      if (!turn.createdAt) return minimum;
      return minimum ? Math.min(minimum, turn.createdAt) : turn.createdAt;
    }, 0);
    if (!oldest) {
      setHasOlder(false);
      return;
    }

    historyAbortRef.current?.abort();
    const controller = new AbortController();
    historyAbortRef.current = controller;
    setLoadingOlder(true);
    setHistoryError("");
    try {
      const older = await loadAgentTurns({ before: oldest, limit: HISTORY_PAGE_SIZE, signal: controller.signal });
      const timeline = timelineRef.current;
      if (timeline) prependAnchorRef.current = { top: timeline.scrollTop, height: timeline.scrollHeight };
      setTurns((current) => mergeAgentTurns(current, older));
      setHasOlder(older.length >= HISTORY_PAGE_SIZE);
    } catch (error) {
      if (error?.name !== "AbortError") setHistoryError(error?.message || "无法读取更早记录");
    } finally {
      if (!controller.signal.aborted) setLoadingOlder(false);
    }
  }, [hasOlder, loadingOlder, turns]);

  const retryAgent = () => {
    historyAbortRef.current?.abort();
    scrollAnimationCancelRef.current?.();
    scrollAnimationCancelRef.current = null;
    setResource({ status: "loading", data: null, error: null });
    setEvents([]);
    setTurns([]);
    setStats({});
    setConnection("connecting");
    setSelectedRoundId("");
    setHistoryError("");
    setLoadingOlder(false);
    initiallyPositionedRef.current = false;
    pinnedToBottomRef.current = true;
    setFollowing(true);
    setReloadKey((value) => value + 1);
  };

  if (resource.status === "loading") {
    return <AgentResourceState title="正在连接 Agent 时间线" detail="先读取轮次快照，再接续实时事件。" />;
  }
  if (resource.status === "error") {
    return <AgentResourceState title="Agent 状态不可用" detail={resource.error?.message} action={retryAgent} />;
  }

  const data = resource.data;
  return (
    <div className="agent-workspace">
      <AgentStatusCards data={data} connection={connection} stats={stats} eventCursor={eventCursor} />
      <div className="agent-body-layout">
        <AgentRunIndex
          rounds={visibleRounds}
          selectedRoundId={effectiveSelectedRoundId}
          expanded={indexExpanded}
          loadingOlder={loadingOlder}
          hasOlder={hasOlder}
          historyError={historyError}
          onToggle={() => setIndexExpanded((value) => !value)}
          onSelect={jumpToRound}
          onLoadOlder={loadOlder}
        />
        <section className="agent-timeline panel-window">
          <div className="panel-header agent-timeline-header">
            <div><div className="eyebrow">ROUND STREAM</div><h3>执行时间线</h3></div>
            <div className="agent-timeline-actions">
              <div className="segmented-control" aria-label="轮次内容筛选">
                {AGENT_EVENT_FILTERS.map(([id, label]) => (
                  <button key={id} type="button" className={filter === id ? "active" : ""} onClick={() => changeFilter(id)}>{label}</button>
                ))}
              </div>
              <button
                type="button"
                className={`agent-follow-button ${following ? "active" : ""}`}
                onClick={() => scrollToLatest()}
                aria-pressed={following}
                title={following ? "正在跟随最新轮次" : "回到并跟随最新轮次"}
              >
                <ArrowDown size={15} />{following ? "跟随中" : "回到最新"}
              </button>
            </div>
          </div>
          <div
            className="agent-round-list"
            ref={timelineRef}
            onScroll={handleTimelineScroll}
            onWheel={markTimelineUserScrollIntent}
            onTouchMove={markTimelineUserScrollIntent}
            onPointerDown={(event) => {
              if (event.target === event.currentTarget) markTimelineUserScrollIntent();
            }}
            onKeyDown={(event) => {
              if (event.target === event.currentTarget && ["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(event.key)) {
                markTimelineUserScrollIntent();
              }
            }}
            role="feed"
            aria-label="Agent 执行轮次"
            tabIndex={0}
          >
            <div className="agent-round-list-content" ref={timelineContentRef}>
              {visibleRounds.length
                ? visibleRounds.map((round) => (
                  <AgentRound
                    key={round.id}
                    round={round}
                    filter={filter}
                    worldOpen={expandedWorlds.has(round.id)}
                    onToggleWorld={() => toggleWorld(round.id)}
                  />
                ))
                : <AgentResourceState title="暂无匹配轮次" detail="新轮次开始后，认知与工具执行会按轮次出现在这里。" />}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
