from __future__ import annotations

from pathlib import Path
import re

from web.settings_domains import SUPPORTED_DOMAINS


REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = REPO_ROOT / "webui-vnext" / "src"
APP_SOURCE = (FRONTEND_ROOT / "App.jsx").read_text(encoding="utf-8")
REALTIME_SOURCE = (FRONTEND_ROOT / "api" / "realtime.js").read_text(encoding="utf-8")
CONVERSATION_SOURCE = (FRONTEND_ROOT / "conversation" / "ConversationPage.jsx").read_text(encoding="utf-8")
AGENT_SOURCE = (FRONTEND_ROOT / "conversation" / "AgentPage.jsx").read_text(encoding="utf-8")
AGENT_PROJECTION_SOURCE = (FRONTEND_ROOT / "conversation" / "agentProjection.js").read_text(encoding="utf-8")
MEMORY_SOURCE = (FRONTEND_ROOT / "memory" / "MemoryPage.jsx").read_text(encoding="utf-8")
OBSERVABILITY_SOURCE = (FRONTEND_ROOT / "observability" / "ObservabilityPage.jsx").read_text(encoding="utf-8")
STICKERS_SOURCE = (FRONTEND_ROOT / "resources" / "StickersPage.jsx").read_text(encoding="utf-8")
STYLES_SOURCE = (FRONTEND_ROOT / "styles.css").read_text(encoding="utf-8")


def _frontend_sources() -> list[Path]:
    return sorted(
        path
        for path in FRONTEND_ROOT.rglob("*")
        if path.suffix in {".js", ".jsx"}
    )


def _quoted_values_from_block(source: str, start: str, end: str) -> set[str]:
    block = source.split(start, 1)[1].split(end, 1)[0]
    return set(re.findall(r'"([a-z][a-z0-9-]*)"', block))


def test_frontend_http_calls_stay_behind_the_shared_transport() -> None:
    offenders: list[str] = []
    for path in _frontend_sources():
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bfetch\s*\(", text):
            offenders.append(path.relative_to(FRONTEND_ROOT).as_posix())

    assert offenders == ["api/http.js"]


def test_browser_storage_is_limited_to_shell_preferences() -> None:
    storage_sources = []
    combined = []
    for path in _frontend_sources():
        text = path.read_text(encoding="utf-8")
        combined.append(text)
        if "localStorage" in text or "sessionStorage" in text:
            storage_sources.append(path.relative_to(FRONTEND_ROOT).as_posix())

    assert storage_sources == ["App.jsx"]
    assert "sessionStorage" not in "\n".join(combined)
    assert 'localStorage.getItem("aicarus-vnext-sidebar")' in APP_SOURCE
    assert "localStorage.setItem(THEME_STORAGE_KEYS.mode" in APP_SOURCE
    assert "localStorage.setItem(THEME_STORAGE_KEYS.light" in APP_SOURCE
    assert "localStorage.setItem(THEME_STORAGE_KEYS.dark" in APP_SOURCE


def test_every_current_main_destination_has_an_explicit_page_branch() -> None:
    page_copy = APP_SOURCE.split("const PAGE_COPY = {", 1)[1].split("};", 1)[0]
    destinations = set(re.findall(r"^\s{2}([a-z]+):", page_copy, re.MULTILINE))

    assert destinations == {
        "home",
        "chat",
        "focus",
        "agent",
        "memory",
        "stickers",
        "logs",
        "tools",
        "tokens",
        "settings",
        "maintenance",
    }
    for destination in destinations:
        assert f'case "{destination}":' in APP_SOURCE


def test_settings_routes_match_supported_domains_and_one_explicit_deferred_domain() -> None:
    frontend_domains = _quoted_values_from_block(
        APP_SOURCE,
        "const DOMAIN_SETTINGS_SECTIONS = new Set([",
        "]);",
    )

    assert frontend_domains == set(SUPPORTED_DOMAINS) | {"security"}
    assert 'settingsSection === "memory-system"' in APP_SOURCE
    assert "MemorySystemDeferredPage" in APP_SOURCE
    assert "配置契约尚未稳定" in APP_SOURCE


def test_production_surfaces_do_not_keep_simulated_controls_or_preview_fallbacks() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in _frontend_sources())
    forbidden = (
        "CLICKABLE PREVIEW",
        "原型模拟",
        "原型不会读写真实配置文件",
        "此分区用于展示未来配置",
        "减少动态效果",
        "紧凑数据显示",
    )

    assert [marker for marker in forbidden if marker in combined] == []


def test_realtime_transport_recovers_stream_restarts_and_expired_sessions() -> None:
    assert 'url.searchParams.set("stream_id", streamId)' in REALTIME_SOURCE
    assert "envelope?.cursor_reset === true" in REALTIME_SOURCE
    assert "envelopeStreamId !== streamId" in REALTIME_SOURCE
    assert "status?.enabled && !status?.authenticated" in REALTIME_SOURCE
    assert "redirectToLogin()" in REALTIME_SOURCE


def test_dense_runtime_surfaces_keep_readability_and_direct_manipulation() -> None:
    assert "projectAgentRounds(events, turns)" in AGENT_SOURCE
    assert "function upsertTool(round, source" in AGENT_PROJECTION_SOURCE
    assert "if (TERMINAL_TOOL_STATES.has(tool.status)) return;" in AGENT_PROJECTION_SOURCE
    assert "round.cognitionDraft += text(event.text)" in AGENT_PROJECTION_SOURCE
    assert "left.createdAt - right.createdAt" in AGENT_PROJECTION_SOURCE
    assert "round_persisted" in AGENT_PROJECTION_SOURCE
    assert "window.setTimeout(flushEvents, 50)" in AGENT_SOURCE
    assert "pinnedToBottomRef" in AGENT_SOURCE
    assert "ResizeObserver" in AGENT_SOURCE
    assert 'data-agent-round-id={round.id}' in AGENT_SOURCE
    assert "loadAgentTurns" in AGENT_SOURCE
    assert 'className="agent-summary-grid"' in AGENT_SOURCE
    assert "smoothScrollElement" in AGENT_SOURCE
    assert "smoothScrollElement(timeline, timeline.scrollHeight" in AGENT_SOURCE
    assert "timeline.scrollTop + nodeRect.top - timelineRect.top - 12" in AGENT_SOURCE
    assert 'className="agent-world-content"' in AGENT_SOURCE
    assert 'className="agent-runtime-details"' in AGENT_SOURCE
    assert "pinnedToBottomRef" in CONVERSATION_SOURCE
    assert "scrollKey={selectedKey}" in CONVERSATION_SOURCE

    assert "useNodesState" in MEMORY_SOURCE
    assert "onNodesChange={onNodesChange}" in MEMORY_SOURCE
    assert "function layoutResultGraph(nodes, edges)" in MEMORY_SOURCE
    assert "function graphTraversalOrder(nodes, edges)" in MEMORY_SOURCE
    assert "GRAPH_NODE_WIDTH" in MEMORY_SOURCE
    assert "GRAPH_COLUMN_GAP" in MEMORY_SOURCE
    assert "seededUnit" not in MEMORY_SOURCE
    assert "graphExpanded" in MEMORY_SOURCE
    assert "nodeStrokeColor" in MEMORY_SOURCE

    assert "selectedSources" in OBSERVABILITY_SOURCE
    assert 'aria-controls="log-source-filter"' in OBSERVABILITY_SOURCE
    assert "const [autoScroll, setAutoScroll]" in OBSERVABILITY_SOURCE
    assert "pinnedToBottomRef" in OBSERVABILITY_SOURCE
    assert "smoothScrollElement" in OBSERVABILITY_SOURCE
    assert 'aria-label="运行日志时间线"' in OBSERVABILITY_SOURCE
    assert "}).slice().reverse();" not in OBSERVABILITY_SOURCE


def test_shell_identity_links_and_scroll_ownership_are_explicit() -> None:
    assert 'const SOURCE_REPOSITORY_URL = "https://github.com/AIcarusDev/AIcarusForQQ"' in APP_SOURCE
    assert "function botGreetingFor(name, date)" in APP_SOURCE
    assert '<h2>{botGreetingFor(overview?.identity.name, now)}</h2>' in APP_SOURCE
    assert "routeViewportRef.current?.scrollTo" in APP_SOURCE
    assert 'className={`route-viewport route-${page}`}' in APP_SOURCE

    desktop_shell = STYLES_SOURCE.split("@media (min-width: 861px) {", 1)[1]
    assert "overflow: hidden" in desktop_shell
    assert ".route-viewport" in desktop_shell
    assert "scrollbar-gutter: stable" in desktop_shell
    assert ".route-focus" in desktop_shell
    assert ".route-agent" in desktop_shell
    assert ".route-logs" in desktop_shell
    assert ".agent-round-list" in desktop_shell
    assert ".route-logs .log-stream" in desktop_shell


def test_mobile_logs_keep_an_internal_scroll_owner() -> None:
    mobile_shell = STYLES_SOURCE.rsplit("@media (max-width: 860px) {", 1)[1].split(
        "@media (max-width: 640px) {", 1
    )[0]
    assert ".log-stream { height: clamp(420px, 64dvh, 560px);" in mobile_shell


def test_theme_and_settings_layout_use_shared_tokens_and_stable_geometry() -> None:
    query_editor_styles = STYLES_SOURCE.split(".query-editor {", 1)[1].split(
        ".memory-doc-dialog",
        1,
    )[0]

    assert "#172123" not in query_editor_styles
    assert "var(--surface-muted)" in query_editor_styles
    assert "scrollbar-gutter: stable" in STYLES_SOURCE
    assert ".resource-settings-content { width: min(100%, 960px); margin: 0 auto;" in STYLES_SOURCE
    assert ".cache-readonly-panel dl { margin: 0; padding: 4px 20px 8px; }" in STYLES_SOURCE


def test_dense_surfaces_scope_scrolling_and_disclose_long_content() -> None:
    assert "sessionsExpanded" in CONVERSATION_SOURCE
    assert 'aria-controls="focus-session-list"' in CONVERSATION_SOURCE
    assert 'className={`recent-turns ${turnsExpanded ? "" : "is-mobile-collapsed"}`}' in CONVERSATION_SOURCE

    assert 'const summary = String(properties.summary || "").trim()' in MEMORY_SOURCE
    assert ".route-memory .memory-inspector" in STYLES_SOURCE
    assert ".route-memory {" in STYLES_SOURCE
    assert "overflow-y: hidden" in STYLES_SOURCE.split(".route-memory {", 1)[1].split("}", 1)[0]

    assert "function isLongLog(message)" in OBSERVABILITY_SOURCE
    assert "expandedRecords" in OBSERVABILITY_SOURCE
    assert 'aria-controls={messageId}' in OBSERVABILITY_SOURCE
    assert ".log-record-message.is-collapsed" in STYLES_SOURCE

    assert 'className="sticker-sticky-tools"' in STICKERS_SOURCE
    assert ".sticker-sticky-tools" in STYLES_SOURCE
    assert "border: 1px solid var(--border-strong)" in STYLES_SOURCE

    assert "activeDirectoryItemRef" in APP_SOURCE
    assert "directoryNavRef" in APP_SOURCE
    assert 'layout?.addEventListener("transitionend", revealAfterLayoutTransition)' in APP_SOURCE
    assert "workspaceRef.current.scrollTop = 0" in APP_SOURCE
    assert 'aria-current={settingsSection === id ? "page" : undefined}' in APP_SOURCE
    assert ".settings-directory.is-collapsed .settings-nav" in STYLES_SOURCE

    desktop_settings = STYLES_SOURCE.split("@media (min-width: 1081px) {", 1)[1].split(
        ".route-memory {",
        1,
    )[0]
    assert ".route-settings {" in desktop_settings
    assert "overflow-y: hidden" in desktop_settings
    assert ".route-settings .settings-workspace" in desktop_settings
    assert "overflow-y: auto" in desktop_settings
