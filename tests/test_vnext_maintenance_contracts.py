from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from quart import Quart

import app_state
from runtime.cache_maintenance import CacheMaintenanceError, CacheMaintenanceService
from web import routes_ui_v1_maintenance, routes_workspace


def test_cache_actions_expose_exact_scope_and_require_server_confirmation(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    image_cache = cache_root / "image"
    image_cache.mkdir(parents=True)
    (image_cache / "one.png").write_bytes(b"1234")
    (image_cache / "two.png").write_bytes(b"56")
    service = CacheMaintenanceService(cache_root)

    image_action = {item["id"]: item for item in service.describe_actions()}["image"]
    assert image_action["target"] == "图片缓存"
    assert image_action["metrics"]["bytes"] == 6
    assert image_action["metrics"]["files"] == 2
    assert image_action["expected_confirmation"] == "CLEAR IMAGE CACHE"
    assert image_action["backup"]["created"] is False

    with pytest.raises(CacheMaintenanceError):
        service.perform("image", confirmation="CLEAR CACHE")
    assert (image_cache / "one.png").exists()

    result = service.perform("image", confirmation=image_action["expected_confirmation"])
    assert result["ok"] is True
    assert result["deleted_files"] == 2
    assert result["reclaimed_bytes"] == 6
    assert list(image_cache.iterdir()) == []

    empty_action = {item["id"]: item for item in service.describe_actions()}["image"]
    assert empty_action["available"] is False
    with pytest.raises(CacheMaintenanceError) as unavailable:
        service.perform("image", confirmation=empty_action["expected_confirmation"])
    assert unavailable.value.status_code == 409
    assert unavailable.value.details["metrics"] == {"bytes": 0, "files": 0}


class _FakeMaintenanceResult:
    def to_dict(self):
        return {"ok": True, "message": "done", "maintenance_id": "maint-1"}


class _FakeMaintenanceService:
    def __init__(self) -> None:
        self.performed: list[str] = []

    async def overview(self):
        return {"MemoryEvents": 12, "chat_messages": 34}

    def describe_actions(self):
        return [{
            "id": "reset_cognition",
            "label": "重置认知",
            "domain": "cognition",
            "danger": "medium",
            "available": True,
            "disabled_reason": "",
            "expected_confirmation": "RESET TestBot",
            "confirmation_required": True,
            "target": "当前 Core 认知运行时",
            "summary": "reset",
            "effects": ["clear runtime"],
            "preserves": ["memory"],
            "backup": {"created": False, "kind": "none", "description": "none"},
        }]

    def expected_confirmation(self, action):
        if action != "reset_cognition":
            raise AssertionError("unexpected action")
        return "RESET TestBot"

    async def perform(self, action):
        self.performed.append(action)
        return _FakeMaintenanceResult()


class _FakeCacheService:
    def __init__(self) -> None:
        self.performed: list[tuple[str, str]] = []

    def overview(self):
        return {"image": {"label": "图片缓存", "path": "cache/image", "bytes": 8, "files": 1}}

    def describe_actions(self, *, overview=None):
        assert overview is not None
        return [{
            "id": "image",
            "label": "清理图片缓存",
            "domain": "cache",
            "danger": "medium",
            "available": True,
            "disabled_reason": "",
            "expected_confirmation": "CLEAR IMAGE CACHE",
            "confirmation_required": True,
            "target": "图片缓存",
            "summary": "clear",
            "effects": ["delete cache"],
            "preserves": ["source"],
            "backup": {"created": False, "kind": "none", "description": "none"},
            "metrics": overview["image"],
        }]

    def perform(self, action, *, confirmation):
        self.performed.append((action, confirmation))
        return {"ok": True, "message": "cleared", "deleted_files": 1}


class _FakeWorkspaceControl:
    def __init__(self) -> None:
        self.started: list[tuple[str, str]] = []

    def probe(self, _config):
        return SimpleNamespace(state="ready")

    def current_job(self):
        return None

    def status_payload(self, config, *, observed=None, job=None):
        assert observed.state == "ready"
        return {
            "ok": True,
            "state": "ready",
            "config": config.to_public_dict(),
            "observed": {"state": "ready", "path_locked": True},
            "job": job,
        }

    def describe_actions(self, _config, *, observed=None, current_job=None):
        assert observed.state == "ready"
        assert current_job is None
        return [{
            "id": "restart",
            "label": "重启工作区",
            "domain": "workspace",
            "danger": "medium",
            "available": True,
            "disabled_reason": "",
            "expected_confirmation": "RESTART WORKSPACE",
            "confirmation_required": True,
            "target": "AICQ-Workspace",
            "summary": "restart",
            "effects": ["restart"],
            "preserves": ["files"],
            "backup": {"created": False, "kind": "none", "description": "none"},
        }]

    def start_job(self, action, _config, *, confirmation=""):
        self.started.append((action, confirmation))
        return {"job_id": "job-1", "action": action, "status": "restarting"}

    def get_job(self, job_id, *, cursor=0):
        return {"job_id": job_id, "status": "ready", "log": "done", "log_cursor": cursor + 4}


def test_v1_maintenance_contract_describes_domains_and_enforces_confirmation(monkeypatch) -> None:
    async def scenario() -> None:
        data_service = _FakeMaintenanceService()
        cache_service = _FakeCacheService()
        workspace_control = _FakeWorkspaceControl()
        monkeypatch.setattr(routes_ui_v1_maintenance, "maintenance_service", data_service)
        monkeypatch.setattr(routes_ui_v1_maintenance, "cache_maintenance_service", cache_service)
        monkeypatch.setattr(routes_workspace, "workspace_control", workspace_control)
        monkeypatch.setattr(app_state, "config", {
            "workspace": {
                "enabled": True,
                "install_root": "E:\\Aic_forQ\\workspace-data",
                "resources": {"cpus": 4, "memory_gib": 8, "disk_gib": 64},
            },
        })

        app = Quart(__name__)
        app.register_blueprint(routes_ui_v1_maintenance.ui_v1_maintenance_bp)
        client = app.test_client()

        overview = await client.get("/api/ui/v1/maintenance")
        payload = await overview.get_json()
        assert overview.status_code == 200
        assert set(payload["data"]["domains"]) == {"data", "cache", "workspace"}
        assert payload["data"]["domains"]["data"]["overview"]["total_rows"] == 46
        assert payload["data"]["domains"]["cache"]["overview"]["total_bytes"] == 8

        mismatch = await client.post(
            "/api/ui/v1/maintenance/actions/data/reset_cognition",
            json={"confirmation": "RESET"},
        )
        assert mismatch.status_code == 400
        mismatch_payload = await mismatch.get_json()
        assert mismatch_payload["error"]["code"] == "confirmation_mismatch"
        assert mismatch_payload["error"]["details"]["expected_confirmation"] == "RESET TestBot"
        assert data_service.performed == []

        accepted = await client.post(
            "/api/ui/v1/maintenance/actions/data/reset_cognition",
            json={"confirmation": "RESET TestBot"},
        )
        assert accepted.status_code == 200
        assert data_service.performed == ["reset_cognition"]

        workspace = await client.post(
            "/api/ui/v1/maintenance/actions/workspace/restart",
            json={"confirmation": "RESTART WORKSPACE"},
        )
        assert workspace.status_code == 202
        assert workspace_control.started == [("restart", "RESTART WORKSPACE")]

        job = await client.get("/api/ui/v1/maintenance/workspace/jobs/job-1?cursor=7")
        assert job.status_code == 200
        assert (await job.get_json())["data"]["job"]["log_cursor"] == 11

    asyncio.run(scenario())
