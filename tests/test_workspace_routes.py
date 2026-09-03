from __future__ import annotations

import asyncio
from types import SimpleNamespace

from quart import Quart

import app_state
from consciousness.flow import ConsciousnessFlow
from tools.namespaces import NamespaceRuntimeState, load_namespace_registry
from web import routes_workspace


class FakeControl:
    def __init__(self):
        self.started = None

    def status_payload(self, config):
        return {
            "ok": True,
            "state": "not_built",
            "config": config.to_public_dict(),
            "observed": {"state": "not_built", "path_locked": False},
            "job": None,
        }

    def probe(self, config):
        return SimpleNamespace(path_locked=False, installed_resources=None)

    def start_job(self, action, config, *, confirmation=""):
        self.started = (action, config, confirmation)
        return {"job_id": "job-1", "action": action, "status": "building"}

    def get_job(self, job_id, *, cursor=0):
        return {"job_id": job_id, "status": "ready", "log": "done", "log_cursor": cursor + 4}


def make_app() -> Quart:
    app = Quart(__name__)
    app.register_blueprint(routes_workspace.workspace_bp)
    return app


def test_workspace_config_api_is_independent_from_full_settings(monkeypatch) -> None:
    async def scenario() -> None:
        fake = FakeControl()
        saved = []
        monkeypatch.setattr(routes_workspace, "workspace_control", fake)
        def fake_save(workspace, *, base_config=None):
            merged = dict(base_config or {})
            merged["workspace"] = workspace
            saved.append(merged)
            return merged

        monkeypatch.setattr(routes_workspace, "save_workspace_config", fake_save)
        monkeypatch.setattr(app_state, "config", {
            "model": "unchanged",
            "workspace": {
                "enabled": False,
                "install_root": "E:\\Aic_forQ\\workspace-data",
                "resources": {"cpus": 4, "memory_gib": 8, "disk_gib": 64},
            },
        })
        client = make_app().test_client()
        response = await client.put("/api/computer/config", json={
            "enabled": True,
            "install_root": "E:\\Aic_forQ\\workspace-new",
            "resources": {"cpus": 4, "memory_gib": 8, "disk_gib": 96},
        })
        data = await response.get_json()
        assert response.status_code == 200
        assert data["ok"] is True
        assert saved[0]["model"] == "unchanged"
        assert saved[0]["workspace"]["enabled"] is True
        assert saved[0]["workspace"]["install_root"] == "E:\\Aic_forQ\\workspace-new"
        assert saved[0]["workspace"]["resources"]["disk_gib"] == 96

    asyncio.run(scenario())


def test_disabling_workspace_closes_namespace_and_injects_system_info(monkeypatch) -> None:
    async def scenario() -> None:
        fake = FakeControl()
        state = NamespaceRuntimeState()
        state.open("computer", load_namespace_registry(), 1)
        flow = ConsciousnessFlow()
        saved_snapshots = []

        def fake_save(workspace, *, base_config=None):
            merged = dict(base_config or {})
            merged["workspace"] = workspace
            return merged

        async def fake_save_namespace_state(snapshot):
            saved_snapshots.append(snapshot)

        monkeypatch.setattr(routes_workspace, "workspace_control", fake)
        monkeypatch.setattr(routes_workspace, "save_workspace_config", fake_save)
        monkeypatch.setattr(
            routes_workspace,
            "save_namespace_runtime_state",
            fake_save_namespace_state,
        )
        monkeypatch.setattr(app_state, "namespace_runtime_state", state)
        monkeypatch.setattr(app_state, "consciousness_flow", flow)
        monkeypatch.setattr(app_state, "config", {
            "workspace": {
                "enabled": True,
                "install_root": "E:\\Aic_forQ\\workspace-data",
                "resources": {"cpus": 4, "memory_gib": 8, "disk_gib": 64},
            },
        })

        client = make_app().test_client()
        response = await client.put("/api/computer/config", json={
            "enabled": False,
            "install_root": "E:\\Aic_forQ\\workspace-data",
            "resources": {"cpus": 4, "memory_gib": 8, "disk_gib": 64},
        })

        assert response.status_code == 200
        assert "computer" not in state.open_order
        assert saved_snapshots == [state.to_snapshot()]
        assert flow.to_xml_messages() == [{
            "role": "user",
            "content": (
                "[system info] 命名空间 `computer` 因对应功能当前不可用，"
                "已被系统关闭。"
            ),
        }]

    asyncio.run(scenario())


def test_workspace_job_and_log_apis_delegate_to_persistent_controller(monkeypatch) -> None:
    async def scenario() -> None:
        fake = FakeControl()
        monkeypatch.setattr(routes_workspace, "workspace_control", fake)
        monkeypatch.setattr(app_state, "config", {
            "workspace": {
                "enabled": False,
                "install_root": "E:\\Aic_forQ\\workspace-data",
                "resources": {"cpus": 4, "memory_gib": 8, "disk_gib": 64},
            },
        })
        client = make_app().test_client()
        started = await client.post("/api/computer/jobs", json={"action": "build"})
        assert started.status_code == 202
        assert fake.started[0] == "build"

        read = await client.get("/api/computer/jobs/job-1?cursor=7")
        data = await read.get_json()
        assert read.status_code == 200
        assert data["job"]["log"] == "done"
        assert data["job"]["log_cursor"] == 11

    asyncio.run(scenario())


def test_workspace_directory_selection_api_consumes_native_result(monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            routes_workspace,
            "consume_workspace_directory_selection",
            lambda selection_id: {
                "selection_id": selection_id,
                "status": "selected",
                "path": "E:\\Aic_forQ\\workspace-new",
                "error": "",
            },
        )
        client = make_app().test_client()
        response = await client.get("/api/computer/directory-selections/selection-1")
        data = await response.get_json()

        assert response.status_code == 200
        assert data["selection"]["selection_id"] == "selection-1"
        assert data["selection"]["path"] == "E:\\Aic_forQ\\workspace-new"

    asyncio.run(scenario())
