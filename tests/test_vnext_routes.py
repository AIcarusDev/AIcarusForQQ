from __future__ import annotations

import asyncio

from quart import Quart

import app_state
from web import (
    auth,
    debug_server,
    routes_agent,
    routes_core,
    routes_dashboard,
    routes_settings,
    routes_ui_v1,
    routes_ui_v1_maintenance,
    routes_ui_v1_settings,
    routes_updates,
    routes_vnext,
    routes_workspace,
)


class FakeToolStats:
    def __init__(self) -> None:
        self.kwargs = None

    async def timeline(self, **kwargs):
        self.kwargs = kwargs
        return {"summary": {"total_calls": 0}, "tools": []}


class FakeTokenStats:
    def __init__(self) -> None:
        self.kwargs = None

    async def timeline(self, **kwargs):
        self.kwargs = kwargs
        return {"group_by": kwargs["group_by"], "summary": {"total_tokens": 0}, "series": []}


def _make_vnext_dist(path) -> None:
    path.mkdir()
    (path / "index.html").write_text(
        "<!doctype html><html><head><title>AIcarus vNext</title></head><body>vNext</body></html>",
        encoding="utf-8",
    )


def test_vnext_shell_redirects_to_canonical_path_and_serves_build(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        dist = tmp_path / "dist"
        _make_vnext_dist(dist)
        monkeypatch.setattr(routes_vnext, "VNEXT_DIST_DIR", dist)
        app = Quart(__name__)
        app.register_blueprint(routes_vnext.vnext_bp)
        client = app.test_client()

        redirect_response = await client.get("/new")
        assert redirect_response.status_code == 308
        assert redirect_response.headers["Location"].endswith("/new/")

        response = await client.get("/new/")
        assert response.status_code == 200
        assert "AIcarus vNext" in (await response.get_data(as_text=True))

    asyncio.run(scenario())


def test_vnext_shell_reports_missing_build_without_affecting_legacy_routes(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(routes_vnext, "VNEXT_DIST_DIR", tmp_path / "missing")
        app = Quart(__name__)

        @app.get("/")
        async def legacy_home():
            return "legacy"

        app.register_blueprint(routes_vnext.vnext_bp)
        client = app.test_client()

        missing = await client.get("/new/")
        assert missing.status_code == 503
        assert (await missing.get_json())["error"] == "vnext_build_missing"
        legacy = await client.get("/")
        assert legacy.status_code == 200
        assert await legacy.get_data(as_text=True) == "legacy"

    asyncio.run(scenario())


def test_vnext_shell_uses_existing_session_auth(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        dist = tmp_path / "dist"
        _make_vnext_dist(dist)
        monkeypatch.setattr(routes_vnext, "VNEXT_DIST_DIR", dist)
        monkeypatch.setattr(
            app_state,
            "config",
            {
                "webui_auth": {
                    "enabled": True,
                    "password_hash": auth._hash_password("secret1"),
                    "skipped_setup": False,
                    "session_days": 7,
                }
            },
        )
        app = Quart(__name__)
        auth.install_auth(app)
        app.register_blueprint(routes_vnext.vnext_bp)
        app.register_blueprint(auth.auth_bp)
        client = app.test_client()

        denied = await client.get("/new/")
        assert denied.status_code == 302
        assert denied.headers["Location"].endswith("/login?next=/new/")

        logged_in = await client.post("/api/auth/login", json={"password": "secret1"})
        assert logged_in.status_code == 200
        allowed = await client.get("/new/")
        assert allowed.status_code == 200

    asyncio.run(scenario())


def test_v1_observability_routes_validate_and_forward_query(monkeypatch) -> None:
    async def scenario() -> None:
        tools = FakeToolStats()
        tokens = FakeTokenStats()
        monkeypatch.setattr(routes_ui_v1, "tool_stats_service", tools)
        monkeypatch.setattr(routes_ui_v1, "token_stats_service", tokens)
        app = Quart(__name__)
        app.register_blueprint(routes_ui_v1.ui_v1_bp)
        client = app.test_client()

        tool_response = await client.get(
            "/api/ui/v1/observability/tools?range=7d&granularity=day"
            "&tools=core.web_search,core.calculator&limit=4&tz_offset_minutes=480"
        )
        assert tool_response.status_code == 200
        tool_payload = await tool_response.get_json()
        assert tool_payload["ok"] is True
        assert tool_payload["api_version"] == "1"
        assert tools.kwargs == {
            "granularity": "day",
            "range_preset": "7d",
            "tool_names": ["core.web_search", "core.calculator"],
            "limit": 4,
            "start_ms": None,
            "end_ms": None,
            "tz_offset_minutes": 480,
        }

        token_response = await client.get(
            "/api/ui/v1/observability/tokens?range=30d&group_by=feature"
        )
        assert token_response.status_code == 200
        token_payload = await token_response.get_json()
        assert token_payload["data"]["group_by"] == "feature"
        assert tokens.kwargs["group_by"] == "feature"

        invalid = await client.get(
            "/api/ui/v1/observability/tokens?group_by=unsupported"
        )
        assert invalid.status_code == 400
        assert (await invalid.get_json())["error"]["code"] == "invalid_group_by"

    asyncio.run(scenario())


def test_v1_capabilities_describe_migration_boundary(monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(app_state, "webui_only", True)
        monkeypatch.setattr(app_state, "webui_standalone", False, raising=False)
        app = Quart(__name__)
        app.register_blueprint(routes_ui_v1.ui_v1_bp)
        client = app.test_client()

        response = await client.get("/api/ui/v1/capabilities")
        payload = await response.get_json()

        assert response.status_code == 200
        assert payload["api_version"] == "1"
        assert payload["runtime"] == {"mode": "webui_only", "core_available": False}
        assert payload["capabilities"]["observability"]["tokens"]["group_by"] == [
            "feature",
            "model",
        ]
        assert payload["migration"]["legacy_path"] == "/"
        assert payload["migration"]["vnext_path"] == "/new/"

    asyncio.run(scenario())


def test_update_manifest_announces_vnext_as_a_gradual_migration(monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            app_state,
            "config",
            {"webui_updates": {"ack_version": "2026.06-webui-auth"}},
        )
        app = Quart(__name__)
        app.register_blueprint(routes_updates.updates_bp)
        client = app.test_client()

        response = await client.get("/api/updates/current")
        payload = await response.get_json()
        latest = payload["items"][0]

        assert response.status_code == 200
        assert payload["current_version"] == "2026.07-webui-vnext"
        assert payload["needs_popup"] is True
        assert latest["level"] == "info"
        assert "/new/" in latest["summary"]
        assert any("原面板仍保留在 /" in change for change in latest["changes"])

    asyncio.run(scenario())


def test_every_frontend_adapter_target_has_a_registered_quart_route() -> None:
    app = Quart(__name__)
    for blueprint in (
        auth.auth_bp,
        debug_server.debug_bp,
        routes_agent.agent_bp,
        routes_core.core_bp,
        routes_dashboard.dashboard_bp,
        routes_settings.settings_bp,
        routes_ui_v1.ui_v1_bp,
        routes_ui_v1_maintenance.ui_v1_maintenance_bp,
        routes_ui_v1_settings.ui_v1_settings_bp,
        routes_updates.updates_bp,
        routes_workspace.workspace_bp,
    ):
        app.register_blueprint(blueprint)

    registered = {
        (rule.rule, method, bool(rule.websocket))
        for rule in app.url_map.iter_rules()
        for method in rule.methods
        if method not in {"HEAD", "OPTIONS"}
    }
    expected = {
        ("/api/auth/logout", "POST", False),
        ("/api/auth/password", "POST", False),
        ("/api/auth/status", "GET", False),
        ("/api/status", "GET", False),
        ("/api/core/status", "GET", False),
        ("/api/core/chat", "GET", False),
        ("/api/core/chat", "POST", False),
        ("/api/focus/state", "GET", False),
        ("/api/focus/context", "GET", False),
        ("/api/agent/state", "GET", False),
        ("/agent/ws/events", "GET", True),
        ("/log/ws/log", "GET", True),
        ("/api/updates/current", "GET", False),
        ("/api/updates/ack", "POST", False),
        ("/api/updates/migrations/napcat-to-qq-adapter", "POST", False),
        ("/api/sticker/<sticker_id>", "GET", False),
        ("/api/stickers/list", "GET", False),
        ("/api/stickers/upload", "POST", False),
        ("/api/stickers/<sticker_id>", "PATCH", False),
        ("/api/stickers/<sticker_id>", "DELETE", False),
        ("/api/stickers/reconcile", "POST", False),
        ("/settings/self_image", "GET", False),
        ("/settings/self_image", "POST", False),
        ("/settings/self_image/<path:filename>", "DELETE", False),
        ("/api/computer", "GET", False),
        ("/api/ui/v1/capabilities", "GET", False),
        ("/api/ui/v1/observability/tools", "GET", False),
        ("/api/ui/v1/observability/tokens", "GET", False),
        ("/api/ui/v1/memory/schema", "GET", False),
        ("/api/ui/v1/memory/query", "POST", False),
        ("/api/ui/v1/settings/<domain>", "GET", False),
        ("/api/ui/v1/settings/<domain>", "PATCH", False),
        ("/api/ui/v1/maintenance", "GET", False),
        ("/api/ui/v1/maintenance/cache", "GET", False),
        ("/api/ui/v1/maintenance/actions/<domain>/<action>", "POST", False),
        (
            "/api/ui/v1/maintenance/workspace/jobs/<job_id>",
            "GET",
            False,
        ),
    }

    assert expected <= registered
