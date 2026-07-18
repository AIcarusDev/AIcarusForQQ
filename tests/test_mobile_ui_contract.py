from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _template(name: str) -> str:
    return (ROOT / "src" / "templates" / name).read_text(encoding="utf-8")


def test_base_shell_owns_mobile_navigation_contract() -> None:
    source = _template("_base.html")

    assert 'id="mobile-nav-toggle"' in source
    assert 'aria-controls="app-sidebar"' in source
    assert 'id="mobile-nav-scrim"' in source
    assert "body.mobile-nav-open .sidebar" in source
    assert "window.matchMedia('(max-width: 760px)')" in source
    assert "sidebar.toggleAttribute('inert', mobile && !open)" in source
    assert "main.toggleAttribute('inert', open)" in source
    assert "const focusTarget = restoreFocusTo" in source
    assert "focusTarget.focus({ preventScroll: true })" in source
    assert "document.activeElement === first" in source
    assert "document.activeElement === last" in source
    assert "height: 100dvh" in source
    assert "env(safe-area-inset-bottom, 0)" in source
    assert 'href="/settings#security"' in source
    assert 'id="sidebar-security-state"' in source


def test_chat_uses_shared_mobile_shell_and_touch_targets() -> None:
    source = _template("chat.html")

    assert 'body[data-page="chat"] .sidebar' not in source
    assert ".composer .icon-btn" in source
    assert "min-width: 44px" in source
    assert "env(safe-area-inset-bottom, 0)" in source
    assert 'els.botMutedBtn.setAttribute("aria-label", statusLabel)' in source


def test_home_and_settings_have_mobile_content_contracts() -> None:
    home = _template("home.html")
    settings = _template("settings.html")

    assert "@media (max-width: 760px)" in home
    assert "grid-template-columns: 1fr" in home
    assert "grid-template-columns: auto minmax(0, 1fr) minmax(0, 1fr)" in settings
    assert "align-items: stretch" in settings
    assert ".mobile-section-select" in settings
    assert 'class="mobile-section-picker"' in settings
    assert "<span>设置分类</span>" in settings
    assert "min-height: 44px" in settings
    assert settings.index('["security", "面板安全"]') < settings.index(
        '["providers", "模型供应商"]'
    )
    assert 'new URLSearchParams(location.search).get("section")' in settings
    assert "togglePasswordVisibility" in settings
    assert 'id="webuiAuthDisableBtn"' in settings
    assert "disableBtn.hidden = !enabled" in settings
    assert "启用后，WebUI 页面、API 与实时连接均需通过登录验证。" in settings
    assert "如果 WebUI 只在本机访问" not in settings


def test_complex_pages_define_mobile_reflow_and_touch_targets() -> None:
    focus = _template("focus.html")
    agent = _template("agent.html")
    tool_stats = _template("tool_stats.html")
    token_stats = _template("token_stats.html")
    log = _template("log.html")
    memory = _template("memory.html")
    stickers = _template("stickers.html")
    maintenance = _template("maintenance.html")

    assert ".focus-layout" in focus and "flex-direction: column" in focus
    assert ".context-col" in focus and "order: 1" in focus
    assert ".stream-actions .icon-btn" in agent
    assert ".tool-chip" in tool_stats and "min-height: 44px" in tool_stats
    assert ".timeline-range" in token_stats and "min-height: 44px" in token_stats
    assert '["ArrowLeft", "ArrowRight", "Home", "End"]' in token_stats
    assert "tokenTimelineScroller.scrollTo" in token_stats
    assert ".mode-btn" in log and "min-height: 44px" in log
    assert ".mg-icon-btn" in memory and "width: 44px" in memory
    assert ".stk-overlay-btn" in stickers and "height: 44px" in stickers
    assert ".stk-card-img-overlay" in stickers and "opacity: 1" in stickers
    assert ".action-run" in maintenance and "min-height: 44px" in maintenance


def test_auth_surfaces_are_opaque_and_offer_password_visibility() -> None:
    base = _template("_base.html")
    login = _template("login.html")

    assert ".webui-auth-screen-overlay" in base
    assert "linear-gradient(135deg" in base
    assert 'background: var(--bg)' in login
    assert 'class="password-toggle"' in login
    assert 'aria-label="显示访问密码"' in login
    assert "togglePasswordVisibility()" in login
    assert "输入访问密码，登录 WebUI。" in login
    assert 'localStorage.getItem("afq_ui_theme")' in login
    for theme in (
        "dark-night",
        "dark-deep",
        "dark-soft",
        "light-blue",
        "light-pink",
    ):
        assert f'html[data-theme="{theme}"]' in login
