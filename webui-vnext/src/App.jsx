"use client";

import { lazy, Suspense, useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  Activity,
  BarChart3,
  Bot,
  Brain,
  ChevronRight,
  CircleAlert,
  CircleCheck,
  Clock3,
  CloudCog,
  Code2,
  Cpu,
  Database,
  Gauge,
  HardDrive,
  Image,
  Layers3,
  LayoutDashboard,
  LogOut,
  Mail,
  Menu,
  MessageCircle,
  Monitor,
  Moon,
  Network,
  Palette,
  PanelLeftClose,
  PanelLeftOpen,
  PlugZap,
  RefreshCw,
  ScrollText,
  Search,
  Settings2,
  Shield,
  ShieldCheck,
  SlidersHorizontal,
  Smile,
  Sparkles,
  Sun,
  Target,
  Terminal,
  Volume2,
  Wrench,
  X,
} from "lucide-react";
import { loadRuntimeOverview, logoutSession } from "./api/runtimeApi.js";
import { ThemeControl } from "./theme/ThemeControl.jsx";
import { UpdateCenter } from "./updates/UpdateCenter.jsx";
import {
  getThemePalette,
  loadThemePreferences,
  THEME_STORAGE_KEYS,
} from "./theme/themePalettes.js";

const MemoryPage = lazy(() =>
  import("./memory/MemoryPage.jsx").then((module) => ({ default: module.MemoryPage })),
);
const MaintenancePage = lazy(() =>
  import("./maintenance/MaintenancePage.jsx").then((module) => ({ default: module.MaintenancePage })),
);
const ObservabilityPage = lazy(() =>
  import("./observability/ObservabilityPage.jsx").then((module) => ({
    default: module.ObservabilityPage,
  })),
);
const ConversationPage = lazy(() =>
  import("./conversation/ConversationPage.jsx").then((module) => ({
    default: module.ConversationPage,
  })),
);
const StickersPage = lazy(() =>
  import("./resources/StickersPage.jsx").then((module) => ({ default: module.StickersPage })),
);
const ResourceSettingsPage = lazy(() =>
  import("./settings/ResourceSettingsPage.jsx").then((module) => ({
    default: module.ResourceSettingsPage,
  })),
);
const SettingsDomainPage = lazy(() =>
  import("./settings/SettingsDomainPage.jsx").then((module) => ({
    default: module.SettingsDomainPage,
  })),
);

const SOURCE_REPOSITORY_URL = "https://github.com/AIcarusDev/AIcarusForQQ";

const NAV_GROUPS = [
  {
    label: "概览与工作",
    items: [
      ["home", "首页", LayoutDashboard],
      ["chat", "Core 聊天", MessageCircle],
      ["focus", "焦点视图", Target],
      ["agent", "Agent 视图", Bot],
    ],
  },
  {
    label: "记忆与资源",
    items: [
      ["memory", "记忆", Brain],
      ["stickers", "表情包", Smile],
    ],
  },
  {
    label: "可观测性",
    items: [
      ["logs", "运行日志", ScrollText],
      ["tools", "工具统计", BarChart3],
      ["tokens", "Token 用量", Gauge],
    ],
  },
];

const SYSTEM_NAV = [
  ["settings", "设置", Settings2],
  ["maintenance", "维护", Wrench],
];

const SETTINGS_GROUPS = [
  {
    label: "模型与推理",
    items: [
      ["providers", "模型供应商", CloudCog],
      ["main-model", "主模型", Cpu],
      ["specialized-models", "专用模型", Layers3],
    ],
  },
  {
    label: "记忆与身份",
    items: [
      ["memory-system", "记忆系统", Brain],
      ["persona", "角色与身份", Sparkles],
      ["self-image", "自身形象", Image],
    ],
  },
  {
    label: "接入与表达",
    items: [
      ["qq-adapter", "QQ / Adapter", Network],
      ["tts", "TTS", Volume2],
    ],
  },
  {
    label: "工具与通知",
    items: [
      ["services", "外部服务", PlugZap],
      ["alerts", "告警与邮件", Mail],
    ],
  },
  {
    label: "运行与数据",
    items: [
      ["workspace", "Linux 工作区", Terminal],
      ["cache", "缓存", HardDrive],
      ["advanced", "网络与高级", SlidersHorizontal],
    ],
  },
  {
    label: "界面与安全",
    items: [
      ["appearance", "外观", Palette],
      ["security", "面板安全", Shield],
    ],
  },
];

const SETTINGS_BY_ID = Object.fromEntries(
  SETTINGS_GROUPS.flatMap(({ items }) =>
    items.map(([id, label, Icon]) => [id, { label, Icon }]),
  ),
);

const DOMAIN_SETTINGS_SECTIONS = new Set([
  "providers",
  "main-model",
  "specialized-models",
  "persona",
  "qq-adapter",
  "tts",
  "services",
  "alerts",
  "advanced",
  "security",
]);

const PAGE_COPY = {
  home: ["仪表盘", "系统状态、运行健康度与近期活动"],
  chat: ["Core 聊天", "读取真实会话，并可靠地向 Core 发送消息"],
  focus: ["焦点视图", "查看 Core 当前关注的会话与上下文"],
  agent: ["Agent 视图", "观察规划、工具调用与执行时间线"],
  memory: ["记忆", "以 Schema 总览和查询结果探索长期记忆"],
  stickers: ["表情包", "管理可供 Core 使用的媒体资源"],
  logs: ["运行日志", "实时查看模块日志与错误事件"],
  tools: ["工具统计", "观察工具调用频率、延迟和变化"],
  tokens: ["Token 用量", "按模型和能力查看推理用量"],
  settings: ["设置", "配置模型、记忆、接入与运行环境"],
  maintenance: ["维护", "执行安全、可解释的系统维护操作"],
};

function routeFromHash() {
  const route = window.location.hash.replace(/^#\/?/, "");
  if (!route) return { page: "home", section: "main-model" };
  const [page, section] = route.split("/");
  const resolvedPage = PAGE_COPY[page] ? page : "home";
  return {
    page: resolvedPage,
    section: resolvedPage === "settings" && Object.hasOwn(SETTINGS_BY_ID, section)
      ? section
      : "main-model",
  };
}

function formatDate(date) {
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}

function formatTime(date) {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function formatCount(value) {
  return new Intl.NumberFormat("zh-CN").format(Math.max(0, Math.round(Number(value) || 0)));
}

function useMediaQuery(query) {
  const [matches, setMatches] = useState(() =>
    typeof window !== "undefined" && window.matchMedia(query).matches,
  );

  useEffect(() => {
    const media = window.matchMedia(query);
    const onChange = (event) => setMatches(event.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}

export function App() {
  const initialRoute = routeFromHash();
  const [page, setPage] = useState(initialRoute.page);
  const [settingsSection, setSettingsSection] = useState(initialRoute.section);
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem("aicarus-vnext-sidebar") === "collapsed",
  );
  const [mobileOpen, setMobileOpen] = useState(false);
  const [themePreferences, setThemePreferences] = useState(() =>
    loadThemePreferences(localStorage),
  );
  const [systemTheme, setSystemTheme] = useState(() =>
    window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light",
  );
  const [toast, setToast] = useState("");
  const mobileMenuButtonRef = useRef(null);
  const routeViewportRef = useRef(null);
  const isMobileLayout = useMediaQuery("(max-width: 860px)");

  const { mode: themeMode, lightPalette, darkPalette } = themePreferences;
  const effectiveTheme = themeMode === "system" ? systemTheme : themeMode;
  const effectivePalette = effectiveTheme === "dark" ? darkPalette : lightPalette;

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (event) => setSystemTheme(event.matches ? "dark" : "light");
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = effectiveTheme;
    document.documentElement.dataset.palette = effectivePalette;
    document.documentElement.style.colorScheme = effectiveTheme;
    localStorage.setItem(THEME_STORAGE_KEYS.mode, themeMode);
    localStorage.setItem(THEME_STORAGE_KEYS.light, lightPalette);
    localStorage.setItem(THEME_STORAGE_KEYS.dark, darkPalette);
  }, [darkPalette, effectivePalette, effectiveTheme, lightPalette, themeMode]);

  useEffect(() => {
    localStorage.setItem(
      "aicarus-vnext-sidebar",
      collapsed ? "collapsed" : "expanded",
    );
  }, [collapsed]);

  useEffect(() => {
    routeViewportRef.current?.scrollTo({ top: 0, left: 0 });
    window.scrollTo({ top: 0, left: 0 });
  }, [page, settingsSection]);

  useEffect(() => {
    if (!isMobileLayout || !mobileOpen) return undefined;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [isMobileLayout, mobileOpen]);

  useEffect(() => {
    const onHashChange = () => {
      const route = routeFromHash();
      setPage(route.page);
      setSettingsSection(route.section);
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    if (!toast) return undefined;
    const timer = window.setTimeout(() => setToast(""), 2600);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const navigate = (nextPage) => {
    const hash = nextPage === "settings" ? "settings/main-model" : nextPage;
    window.location.hash = hash;
    setPage(nextPage);
    setMobileOpen(false);
  };

  const navigateSettings = (section) => {
    window.location.hash = `settings/${section}`;
    setPage("settings");
    setSettingsSection(section);
  };

  const changeThemeMode = (mode) => {
    setThemePreferences((current) => ({ ...current, mode }));
  };

  const changeThemePalette = (tone, paletteId) => {
    const paletteKey = tone === "light" ? "lightPalette" : "darkPalette";
    setThemePreferences((current) => ({
      ...current,
      mode: tone,
      [paletteKey]: paletteId,
    }));
  };

  const closeMobileNavigation = () => {
    setMobileOpen(false);
    window.requestAnimationFrame(() => mobileMenuButtonRef.current?.focus());
  };

  const logout = async () => {
    try {
      await logoutSession();
      window.location.assign("/login?next=%2Fnew%2F");
    } catch (error) {
      setToast(error?.message || "退出登录失败");
    }
  };

  const mobileNavigationOpen = isMobileLayout && mobileOpen;

  return (
    <div className={`app-root ${collapsed ? "sidebar-collapsed" : ""}`}>
      <div className="app-frame">
        <Sidebar
          page={page}
          collapsed={collapsed}
          mobileOpen={mobileNavigationOpen}
          isMobileLayout={isMobileLayout}
          themeMode={themeMode}
          lightPalette={lightPalette}
          darkPalette={darkPalette}
          onNavigate={navigate}
          onCollapse={() => setCollapsed((value) => !value)}
          onCloseMobile={closeMobileNavigation}
          onThemeMode={changeThemeMode}
          onThemePalette={changeThemePalette}
          onLogout={logout}
        />

        {mobileNavigationOpen && (
          <button
            className="drawer-scrim"
            type="button"
            aria-label="关闭导航遮罩"
            onClick={closeMobileNavigation}
          />
        )}

        <main
          className="main-stage"
          aria-hidden={mobileNavigationOpen || undefined}
          inert={mobileNavigationOpen || undefined}
        >
          <Topbar
            page={page}
            mobileOpen={mobileNavigationOpen}
            mobileMenuButtonRef={mobileMenuButtonRef}
            onOpenMobile={() => setMobileOpen(true)}
            onToast={setToast}
          />
          <div ref={routeViewportRef} className={`route-viewport route-${page}`}>
            <PageContent
              page={page}
              settingsSection={settingsSection}
              themeMode={themeMode}
              effectiveTheme={effectiveTheme}
              effectivePalette={effectivePalette}
              lightPalette={lightPalette}
              darkPalette={darkPalette}
              onThemeMode={changeThemeMode}
              onNavigate={navigate}
              onSettingsSection={navigateSettings}
              onToast={setToast}
            />
          </div>
        </main>
      </div>
      {toast && (
        <div className="toast" role="status">
          <CircleCheck size={17} />
          {toast}
        </div>
      )}
    </div>
  );
}

function Sidebar({
  page,
  collapsed,
  mobileOpen,
  isMobileLayout,
  themeMode,
  lightPalette,
  darkPalette,
  onNavigate,
  onCollapse,
  onCloseMobile,
  onThemeMode,
  onThemePalette,
  onLogout,
}) {
  const sidebarRef = useRef(null);

  useEffect(() => {
    if (!mobileOpen) return undefined;
    const frame = window.requestAnimationFrame(() => {
      sidebarRef.current
        ?.querySelector('.primary-nav .nav-button[aria-current="page"]')
        ?.scrollIntoView({ block: "nearest", inline: "nearest" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [mobileOpen, page]);

  return (
    <aside
      ref={sidebarRef}
      id="global-navigation"
      className={`global-sidebar ${mobileOpen ? "mobile-open" : ""}`}
      aria-label="全局导航"
      aria-hidden={(isMobileLayout && !mobileOpen) || undefined}
      inert={(isMobileLayout && !mobileOpen) || undefined}
    >
      <div className="sidebar-window">
        <div className="window-strip">
          <div className="window-dots" aria-hidden="true">
            <span className="dot rose" />
            <span className="dot amber" />
            <span className="dot mint" />
          </div>
          <a
            className="repo-link"
            href={SOURCE_REPOSITORY_URL}
            target="_blank"
            rel="noreferrer"
            aria-label="打开 AIcarusForQQ 源代码"
            title="打开 AIcarusForQQ 源代码"
          >
            <Code2 size={13} />
            <span>CODE</span>
          </a>
          <button
            className="mobile-close"
            type="button"
            aria-label="关闭导航"
            onClick={onCloseMobile}
          >
            <X size={19} />
          </button>
        </div>

        <div className="sidebar-body">
          <div className="brand-row">
            <button className="brand" type="button" onClick={() => onNavigate("home")}>
              <span className="brand-mark">AI</span>
              <span className="brand-copy">
                <strong>AIcarus</strong>
                <small>WebUI vNext</small>
              </span>
            </button>
            <button
              className="collapse-button"
              type="button"
              onClick={onCollapse}
              aria-label={collapsed ? "展开侧栏" : "收起侧栏"}
              title={collapsed ? "展开侧栏" : "收起侧栏"}
            >
              {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
            </button>
          </div>

          <nav className="primary-nav" aria-label="主导航">
            {NAV_GROUPS.map((group) => (
              <div className="nav-group" key={group.label}>
                <div className="nav-group-label">{group.label}</div>
                {group.items.map(([id, label, Icon]) => (
                  <NavButton
                    key={id}
                    active={page === id}
                    label={label}
                    Icon={Icon}
                    collapsed={collapsed}
                    onClick={() => onNavigate(id)}
                  />
                ))}
              </div>
            ))}
          </nav>

          <div className="sidebar-bottom">
            <div className="nav-group system-group">
              <div className="nav-group-label">系统</div>
              {SYSTEM_NAV.map(([id, label, Icon]) => (
                <NavButton
                  key={id}
                  active={page === id}
                  label={label}
                  Icon={Icon}
                  collapsed={collapsed}
                  onClick={() => onNavigate(id)}
                />
              ))}
            </div>

            <div className="sidebar-divider" />

            <div className="account-card">
              <ShieldCheck size={17} />
              <div className="account-copy">
                <strong>admin</strong>
                <span>WebUI 会话</span>
              </div>
              <button type="button" aria-label="退出登录" title="退出登录" onClick={onLogout}>
                <LogOut size={16} />
              </button>
            </div>

            <ThemeControl
              key={collapsed && !isMobileLayout ? "compact" : "expanded"}
              value={themeMode}
              lightPalette={lightPalette}
              darkPalette={darkPalette}
              onChange={onThemeMode}
              onPaletteChange={onThemePalette}
              compact={collapsed && !isMobileLayout}
            />
          </div>
        </div>
      </div>
    </aside>
  );
}

function NavButton({ active, label, Icon, collapsed, onClick }) {
  return (
    <button
      className={`nav-button ${active ? "active" : ""}`}
      type="button"
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      aria-label={label}
      title={collapsed ? label : undefined}
    >
      <Icon size={18} strokeWidth={1.8} />
      <span>{label}</span>
    </button>
  );
}

function Topbar({ page, mobileOpen, mobileMenuButtonRef, onOpenMobile, onToast }) {
  const [title, description] = PAGE_COPY[page];
  return (
    <header className="topbar">
      <div className="topbar-title-wrap">
        <button
          ref={mobileMenuButtonRef}
          className="mobile-menu"
          type="button"
          aria-label="打开导航"
          aria-controls="global-navigation"
          aria-expanded={mobileOpen}
          onClick={onOpenMobile}
        >
          <Menu size={20} />
        </button>
        <div>
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
      </div>
      <div className="topbar-actions">
        <span className="status-chip">
          <Activity size={14} />
          VNEXT
        </span>
        <UpdateCenter onToast={onToast} />
      </div>
    </header>
  );
}

function PageContent(props) {
  switch (props.page) {
    case "home":
      return <DashboardPage onNavigate={props.onNavigate} />;
    case "settings":
      return <SettingsPage {...props} />;
    case "chat":
    case "focus":
    case "agent":
      return (
        <Suspense fallback={<PageLoadingState label="正在加载会话工作区" />}>
          <ConversationPage page={props.page} />
        </Suspense>
      );
    case "stickers":
      return (
        <Suspense fallback={<PageLoadingState label="正在加载表情包资源" />}>
          <StickersPage onToast={props.onToast} />
        </Suspense>
      );
    case "memory":
      return (
        <Suspense fallback={<PageLoadingState label="正在加载记忆工作台" />}>
          <MemoryPage
            effectiveTheme={props.effectiveTheme}
            onToast={props.onToast}
          />
        </Suspense>
      );
    case "maintenance":
      return (
        <Suspense fallback={<PageLoadingState label="正在读取维护边界" />}>
          <MaintenancePage onToast={props.onToast} />
        </Suspense>
      );
    case "logs":
    case "tools":
    case "tokens":
      return (
        <Suspense fallback={<PageLoadingState label="正在加载诊断图表" />}>
          <ObservabilityPage page={props.page} />
        </Suspense>
      );
    default:
      return <DashboardPage onNavigate={props.onNavigate} />;
  }
}

function PageLoadingState({ label }) {
  return (
    <section className="panel-window route-loading-state" role="status">
      <RefreshCw className="spin" size={20} />
      <span>{label}</span>
    </section>
  );
}

function formatUptime(totalSeconds) {
  const seconds = Math.max(0, Number(totalSeconds) || 0);
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (days) return `${days} 天 ${hours} 小时`;
  if (hours) return `${hours} 小时 ${minutes} 分`;
  return `${minutes} 分钟`;
}

function greetingFor(date) {
  const hour = date.getHours();
  if (hour < 6) return "夜深了";
  if (hour < 11) return "早上好";
  if (hour < 14) return "中午好";
  if (hour < 18) return "下午好";
  return "晚上好";
}

function botGreetingFor(name, date) {
  const speaker = String(name || "AIcarus").trim() || "AIcarus";
  return `${speaker}：“${greetingFor(date)}”`;
}

function DashboardPage({ onNavigate }) {
  const [now, setNow] = useState(new Date());
  const [reloadKey, setReloadKey] = useState(0);
  const [resource, setResource] = useState({
    status: "loading",
    data: null,
    error: null,
  });

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    loadRuntimeOverview({ signal: controller.signal })
      .then((data) => setResource({ status: "ready", data, error: null }))
      .catch((error) => {
        if (error?.name !== "AbortError") {
          setResource({ status: "error", data: null, error });
        }
      });
    return () => controller.abort();
  }, [reloadKey]);

  const refreshRuntime = () => {
    setResource((current) => ({ status: "loading", data: current.data, error: null }));
    setReloadKey((value) => value + 1);
  };

  const overview = resource.data;
  const runtimeState = overview?.runtime.state;
  const runtimeValue = {
    running: "在线",
    stopped: "未启动",
    unavailable: "不可用",
  }[runtimeState] || "读取中";
  const runtimeNote = overview
    ? `${overview.runtime.modeLabel} · 已运行 ${formatUptime(overview.activity.uptimeSeconds)}`
    : "正在读取 Core 状态";

  const checks = overview ? [
    {
      title: "Core 运行时",
      note: runtimeState === "running"
        ? "Core 能力在当前运行模式中可用"
        : runtimeState === "stopped"
          ? "当前仅启动 WebUI，Core 未运行"
          : "当前模式不提供 Core 能力",
      meta: overview.runtime.modeLabel,
      tone: runtimeState === "running" ? "ok" : "info",
    },
    {
      title: "状态契约",
      note: "首页状态与 vNext 能力描述已成功读取",
      meta: overview.identity.model,
      tone: "ok",
    },
    {
      title: "记忆概览",
      note: `已读取 ${formatCount(overview.activity.memoryEvents)} 个记忆事件`,
      meta: `${formatCount(overview.activity.memoryRelations)} 条${overview.activity.memoryRelationLabel}`,
      tone: "ok",
    },
  ] : [];

  return (
    <div className="page-stack dashboard-page">
      <section className="hero-window">
        <div className="hero-kicker">DASHBOARD</div>
        <div className="hero-time-chip">
          <Clock3 size={16} />
          {formatTime(now).slice(0, 5)}
        </div>
        <h2>{botGreetingFor(overview?.identity.name, now)}</h2>
        <p>{formatDate(now)} · 同源 WebUI vNext</p>
        <div className="hero-badges">
          <span>vNext 0.1</span>
          {resource.status === "error" ? (
            <span><CircleAlert size={13} /> 状态同步失败</span>
          ) : (
            <span><CircleCheck size={13} /> {overview?.runtime.modeLabel || "正在同步"}</span>
          )}
        </div>
      </section>

      <div className="metric-row">
        <MetricCard label="Core 状态" value={runtimeValue} note={runtimeNote} Icon={Cpu} />
        <MetricCard
          label="今日消息"
          value={overview ? formatCount(overview.activity.todayMessages) : "—"}
          note={overview?.activity.currentFocus ? `当前焦点：${overview.activity.currentFocus}` : "当前没有活动焦点"}
          Icon={MessageCircle}
        />
        <MetricCard
          label="记忆事件"
          value={overview ? formatCount(overview.activity.memoryEvents) : "—"}
          note={overview ? `${formatCount(overview.activity.memoryRelations)} 条${overview.activity.memoryRelationLabel}` : "正在读取计数"}
          Icon={Database}
        />
      </div>

      <div className="dashboard-grid">
        <section className="panel-window health-panel">
          <div className="panel-header">
            <div>
              <div className="eyebrow">SYSTEM CHECK</div>
              <h3>运行体检</h3>
            </div>
            <button
              className="quiet-button"
              type="button"
              onClick={refreshRuntime}
              disabled={resource.status === "loading"}
            >
              <RefreshCw className={resource.status === "loading" ? "spin" : ""} size={15} />
              {resource.status === "loading" ? "检查中" : "重新检查"}
            </button>
          </div>
          <div className="check-list">
            {resource.status === "error" && (
              <div className="dashboard-resource-state" role="alert">
                <CircleAlert size={18} />
                <div>
                  <strong>无法读取后端状态</strong>
                  <p>{resource.error?.message || "请确认 WebUI 服务仍在运行。"}</p>
                </div>
              </div>
            )}
            {resource.status === "loading" && !overview && (
              <div className="dashboard-resource-state" role="status">
                <RefreshCw className="spin" size={18} />
                <div><strong>正在同步</strong><p>读取运行模式、首页状态与能力边界。</p></div>
              </div>
            )}
            {checks.map((check) => (
              <div className="check-row" key={check.title}>
                <span className={`ok-mark ${check.tone}`}>
                  {check.tone === "ok" ? <CircleCheck size={15} /> : <Activity size={15} />}
                  {check.tone === "ok" ? "OK" : "INFO"}
                </span>
                <div>
                  <strong>{check.title}</strong>
                  <p>{check.note}</p>
                </div>
                <code>{check.meta}</code>
              </div>
            ))}
          </div>
        </section>

        <section className="panel-window activity-panel">
          <div className="panel-header">
            <div>
              <div className="eyebrow">RUNTIME</div>
              <h3>当前摘要</h3>
            </div>
            <span className="count-chip">实时</span>
          </div>
          <div className="activity-list">
            <ActivityItem time="模型" title={overview?.identity.model || "正在读取"} detail="当前主模型标识" />
            <ActivityItem
              time="焦点"
              title={overview?.activity.currentFocus || "暂无活动焦点"}
              detail={overview ? "来自 /api/status" : "等待状态契约"}
            />
            <ActivityItem
              time="模式"
              title={overview?.runtime.modeLabel || "正在读取"}
              detail={overview?.runtime.launcherManaged ? "由 launcher 管理" : "非 launcher 管理"}
            />
          </div>
          <button className="text-link" type="button" onClick={() => onNavigate("tools")}>
            查看真实调用趋势 <ChevronRight size={15} />
          </button>
        </section>
      </div>
    </div>
  );
}

function MetricCard({ label, value, note, Icon }) {
  return (
    <article className="metric-card">
      <div className="metric-label"><span>{label}</span><Icon size={17} /></div>
      <strong>{value}</strong>
      <p>{note}</p>
    </article>
  );
}

function ActivityItem({ time, title, detail }) {
  return (
    <div className="activity-item">
      <time>{time}</time>
      <div><strong>{title}</strong><span>{detail}</span></div>
    </div>
  );
}

function SettingsPage({
  settingsSection,
  themeMode,
  effectiveTheme,
  effectivePalette,
  lightPalette,
  darkPalette,
  onThemeMode,
  onSettingsSection,
  onToast,
  onNavigate,
}) {
  const [query, setQuery] = useState("");
  const [dirtyDomains, setDirtyDomains] = useState({});
  const [directoryOpen, setDirectoryOpen] = useState(() =>
    window.matchMedia("(min-width: 1081px)").matches,
  );
  const searchRef = useRef(null);
  const directoryNavRef = useRef(null);
  const activeDirectoryItemRef = useRef(null);
  const workspaceRef = useRef(null);
  const normalized = query.trim().toLocaleLowerCase("zh-CN");
  const activeSettingLabel = SETTINGS_BY_ID[settingsSection]?.label ?? "设置";
  const ActiveSettingIcon = SETTINGS_BY_ID[settingsSection]?.Icon ?? Settings2;
  const filtered = SETTINGS_GROUPS.map((group) => ({
    ...group,
    items: group.items.filter(([, label]) =>
      `${group.label} ${label}`.toLocaleLowerCase("zh-CN").includes(normalized),
    ),
  })).filter((group) => group.items.length > 0);

  useEffect(() => {
    const wideLayout = window.matchMedia("(min-width: 1081px)");
    const syncDirectory = (event) => setDirectoryOpen(event.matches);
    wideLayout.addEventListener("change", syncDirectory);
    return () => wideLayout.removeEventListener("change", syncDirectory);
  }, []);

  useEffect(() => {
    const focusSearch = (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setDirectoryOpen(true);
        window.requestAnimationFrame(() => searchRef.current?.focus());
      }
    };
    window.addEventListener("keydown", focusSearch);
    return () => window.removeEventListener("keydown", focusSearch);
  }, []);

  useLayoutEffect(() => {
    if (directoryOpen || !window.matchMedia("(min-width: 1081px)").matches) return undefined;
    const navigation = directoryNavRef.current;
    const activeItem = activeDirectoryItemRef.current;
    if (!navigation || !activeItem) return undefined;

    const revealActiveItem = () => {
      const navigationBounds = navigation.getBoundingClientRect();
      const activeBounds = activeItem.getBoundingClientRect();
      const revealMargin = 8;

      if (activeBounds.top < navigationBounds.top + revealMargin) {
        navigation.scrollTop -= navigationBounds.top + revealMargin - activeBounds.top;
      } else if (activeBounds.bottom > navigationBounds.bottom - revealMargin) {
        navigation.scrollTop += activeBounds.bottom - navigationBounds.bottom + revealMargin;
      }
    };

    const layout = navigation.closest(".settings-layout");
    const revealAfterLayoutTransition = (event) => {
      if (event.target === layout) revealActiveItem();
    };

    revealActiveItem();
    layout?.addEventListener("transitionend", revealAfterLayoutTransition);
    return () => layout?.removeEventListener("transitionend", revealAfterLayoutTransition);
  }, [directoryOpen, settingsSection]);

  useLayoutEffect(() => {
    if (!window.matchMedia("(min-width: 1081px)").matches) return;
    if (workspaceRef.current) workspaceRef.current.scrollTop = 0;
  }, [settingsSection]);

  const selectSetting = (id) => {
    onSettingsSection(id);
    if (window.matchMedia("(max-width: 1080px)").matches) {
      setDirectoryOpen(false);
    }
  };

  return (
    <div className={`settings-layout ${directoryOpen ? "directory-open" : "directory-collapsed"}`}>
      <aside
        className={`settings-directory ${directoryOpen ? "is-open" : "is-collapsed"}`}
        aria-label="设置目录"
      >
        <div className="settings-directory-head">
          <div className="settings-directory-title-row">
            <div>
              <div className="eyebrow">SETTINGS</div>
              <h2>配置目录</h2>
            </div>
            <button
              className="settings-directory-toggle"
              type="button"
              aria-expanded={directoryOpen}
              aria-controls="settings-directory-navigation"
              aria-label={directoryOpen ? "收起配置目录" : "展开配置目录"}
              onClick={() => setDirectoryOpen((open) => !open)}
            >
              <span>{directoryOpen ? "收起" : "展开"}</span>
              {directoryOpen ? <PanelLeftClose size={15} /> : <PanelLeftOpen size={15} />}
            </button>
          </div>
          <label className="search-box">
            <Search size={16} />
            <input
              ref={searchRef}
              type="search"
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                if (event.target.value) setDirectoryOpen(true);
              }}
              onFocus={() => setDirectoryOpen(true)}
              placeholder="搜索设置"
              aria-label="搜索设置"
            />
            <kbd>⌘ K</kbd>
          </label>
          <div className="settings-directory-current" aria-live="polite" title={activeSettingLabel}>
            <ActiveSettingIcon size={15} aria-hidden="true" />
            <span>当前</span>
            <strong>{activeSettingLabel}</strong>
          </div>
        </div>
        <nav ref={directoryNavRef} className="settings-nav" id="settings-directory-navigation">
          {filtered.length ? filtered.map((group) => (
            <div className="settings-group" key={group.label}>
              <div className="settings-group-label">{group.label}</div>
              {group.items.map(([id, label, Icon]) => (
                <button
                  key={id}
                  ref={settingsSection === id ? activeDirectoryItemRef : null}
                  className={settingsSection === id ? "active" : ""}
                  type="button"
                  title={label}
                  aria-label={label}
                  aria-current={settingsSection === id ? "page" : undefined}
                  onClick={() => selectSetting(id)}
                >
                  <Icon size={16} />
                  <span>{label}</span>
                  {dirtyDomains[id] && (
                    <i className="dirty-dot" aria-label="有未保存项" />
                  )}
                </button>
              ))}
            </div>
          )) : (
            <div className="empty-search">没有匹配的设置</div>
          )}
        </nav>
      </aside>

      <section ref={workspaceRef} className="settings-workspace">
        {settingsSection === "appearance" ? (
          <AppearanceSettings
            themeMode={themeMode}
            effectiveTheme={effectiveTheme}
            effectivePalette={effectivePalette}
            lightPalette={lightPalette}
            darkPalette={darkPalette}
            onThemeMode={onThemeMode}
          />
        ) : ["self-image", "workspace", "cache"].includes(settingsSection) ? (
          <Suspense fallback={<PageLoadingState label="正在加载资源状态" />}>
            <ResourceSettingsPage section={settingsSection} onToast={onToast} onNavigate={onNavigate} />
          </Suspense>
        ) : DOMAIN_SETTINGS_SECTIONS.has(settingsSection) ? (
          <Suspense fallback={<PageLoadingState label="正在加载领域设置" />}>
            <SettingsDomainPage
              key={settingsSection}
              domain={settingsSection}
              onToast={onToast}
              onDirtyChange={(dirty) => setDirtyDomains((current) => (
                current[settingsSection] === dirty
                  ? current
                  : { ...current, [settingsSection]: dirty }
              ))}
            />
          </Suspense>
        ) : settingsSection === "memory-system" ? (
          <MemorySystemDeferredPage onNavigate={onNavigate} />
        ) : null}
      </section>
    </div>
  );
}

function FormSection({ title, description, children }) {
  return (
    <section className="form-section">
      <div className="form-section-header">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
      <div className="form-rows">{children}</div>
    </section>
  );
}

function AppearanceSettings({
  themeMode,
  effectiveTheme,
  effectivePalette,
  lightPalette,
  darkPalette,
  onThemeMode,
}) {
  const currentPalette = getThemePalette(effectiveTheme, effectivePalette);
  const selectedLightPalette = getThemePalette("light", lightPalette);
  const selectedDarkPalette = getThemePalette("dark", darkPalette);

  return (
    <div className="settings-form-page narrow-settings-page">
      <div className="settings-page-header">
        <div>
          <div className="breadcrumb">界面与安全 <ChevronRight size={14} /> 外观</div>
          <h2>外观</h2>
          <p>默认跟随操作系统，也可以只为当前浏览器选择浅色或深色。</p>
        </div>
      </div>
      <FormSection
        title="主题模式"
        description={`当前实际显示为${effectiveTheme === "dark" ? "深色" : "浅色"} · ${currentPalette.label}。`}
      >
        <div className="theme-choice-grid">
          {[
            ["system", "跟随系统", "响应操作系统的外观变化", Monitor],
            ["light", "浅色", "保持温暖、清晰的纸面工作台", Sun],
            ["dark", "深色", "适合低照度环境下持续使用", Moon],
          ].map(([id, title, note, Icon]) => (
            <button
              key={id}
              className={themeMode === id ? "active" : ""}
              type="button"
              onClick={() => onThemeMode(id)}
            >
              <Icon size={20} />
              <strong>{title}</strong>
              <span>{note}</span>
              {themeMode === id && <CircleCheck size={18} />}
            </button>
          ))}
        </div>
        <div className="theme-palette-summary" aria-label="当前配色偏好">
          {[
            ["浅色", selectedLightPalette],
            ["深色", selectedDarkPalette],
          ].map(([tone, palette]) => (
            <div key={tone}>
              <span className="theme-palette-summary-swatch" aria-hidden="true">
                {palette.preview.map((color) => <i key={color} style={{ background: color }} />)}
              </span>
              <span><small>{tone}配色</small><strong>{palette.label}</strong></span>
            </div>
          ))}
          <p>在左侧栏底部点击“浅色”或“深色”，即可向上展开更多配色。</p>
        </div>
      </FormSection>
    </div>
  );
}

function MemorySystemDeferredPage({ onNavigate }) {
  return (
    <div className="settings-form-page narrow-settings-page">
      <div className="settings-page-header">
        <div>
          <div className="breadcrumb">记忆与身份 <ChevronRight size={14} /> 记忆系统</div>
          <h2>记忆系统</h2>
          <p>配置结构仍在随记忆系统演进；这里不会把未稳定的内部字段固化成新版公共契约。</p>
        </div>
      </div>
      <section className="memory-settings-deferred" aria-labelledby="memory-settings-deferred-title">
        <span className="memory-settings-deferred-icon"><Brain size={22} /></span>
        <div>
          <div className="eyebrow">VERSIONED CONTRACT REQUIRED</div>
          <h3 id="memory-settings-deferred-title">配置契约尚未稳定</h3>
          <p>新版当前只提供稳定的 Schema 与只读 MemoryQL。配置写入继续沿用旧 UI，直到服务端能提供 schema version、revision 和兼容迁移说明。</p>
          <ul>
            <li>不会通过跨领域的完整设置接口猜测字段。</li>
            <li>不会在浏览器中模拟保存或保留无效开关。</li>
            <li>语义结构变化时，记忆查询会明确报告兼容状态。</li>
          </ul>
          <div className="memory-settings-deferred-actions">
            <button className="primary-button" type="button" onClick={() => onNavigate("memory")}>
              打开记忆工作台 <ChevronRight size={16} />
            </button>
            <a className="secondary-button" href="/settings">在旧 UI 配置</a>
          </div>
        </div>
      </section>
    </div>
  );
}
