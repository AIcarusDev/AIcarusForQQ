from __future__ import annotations

import json
import time
from types import SimpleNamespace

import app_state
from platforms.focus import FocusRef
from runtime.core_restart import (
    _clean_text,
    _atomic_write_json,
    apply_startup_intent,
    build_restart_completed_tool_result,
    consume_pending_intent,
    read_pending_intent,
)
from runtime.maintenance import (
    CLEAR_ALL_DATA,
    DELETE_LONG_TERM_MEMORY,
    RESET_COGNITION,
    EmergencyResetResult,
    MaintenanceActionResult,
    MaintenanceService,
)
from tools.core import restart as restart_tool


SANDBOX_GROUP_ID = "6990" + "19840"
SANDBOX_GROUP_KEY = f"qq:group:{SANDBOX_GROUP_ID}"


def test_core_restart_intent_file_helpers_round_trip_json(tmp_path):
    path = tmp_path / "restart.json"
    payload = {"version": 1, "focus_key": "qq:group:sandbox"}

    _atomic_write_json(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert read_pending_intent(path) == payload
    assert consume_pending_intent(path) == payload
    assert read_pending_intent(path) is None


def test_restart_completed_result_reports_focus_and_elapsed_time():
    result = build_restart_completed_tool_result(
        {"requested_at": time.time() - 2, "focus_key": "qq:group:sandbox"},
        focus_key=None,
    )

    assert result["ok"] is True
    assert result["restarted"] is True
    assert result["focus_key"] == "qq:group:sandbox"
    assert result["offline_seconds"] >= 0


def test_restart_tool_persists_parseable_focus_key(monkeypatch):
    captured: dict[str, str | None] = {}

    def fake_request_restart(*, focus_key, requested_by):
        captured["focus_key"] = focus_key
        captured["requested_by"] = requested_by
        return {"ok": True, "restart_scheduled": True}

    monkeypatch.setattr(app_state, "current_focus", FocusRef("qq", "group", SANDBOX_GROUP_ID, "松窗听雨阁"))
    monkeypatch.setattr(restart_tool.core_restart, "request_restart", fake_request_restart)

    result = restart_tool.make_handler(session=None)()

    assert result["ok"] is True
    assert result["deferred"] is True
    assert captured == {
        "focus_key": SANDBOX_GROUP_KEY,
        "requested_by": "tool:restart",
    }


def test_apply_startup_intent_restores_focus_ref(monkeypatch):
    monkeypatch.setattr(app_state, "current_focus", None)
    app_state.first_input_event.clear()

    restored = apply_startup_intent({"focus_key": SANDBOX_GROUP_KEY})

    assert restored == SANDBOX_GROUP_KEY
    assert app_state.current_focus == FocusRef("qq", "group", SANDBOX_GROUP_ID)
    assert app_state.first_input_event.is_set()


def test_clean_text_trims_defaults_and_truncates_long_values():
    assert _clean_text("  value  ") == "value"
    assert _clean_text("", default="fallback") == "fallback"
    assert _clean_text("x" * 400).endswith("...")


def test_maintenance_confirmation_and_descriptions_separate_dangerous_actions(monkeypatch):
    service = MaintenanceService()
    monkeypatch.setattr(app_state, "SELF_NAME", "SandboxBot")
    monkeypatch.setattr(app_state, "webui_only", True)
    monkeypatch.setattr(app_state, "consciousness_flow", None)

    assert service.expected_confirmation(RESET_COGNITION) == "RESET SandboxBot"
    assert service.expected_confirmation(DELETE_LONG_TERM_MEMORY) == "DELETE MEMORY SandboxBot"
    assert service.expected_confirmation(CLEAR_ALL_DATA) == "CLEAR DB SandboxBot"

    actions = {item["id"]: item for item in service.describe_actions()}
    assert actions[RESET_COGNITION]["available"] is False
    assert actions[DELETE_LONG_TERM_MEMORY]["available"] is True
    assert actions[CLEAR_ALL_DATA]["available"] is True


def test_maintenance_long_term_memory_scope_includes_consolidation_tables():
    service = MaintenanceService()
    delete_order = service._LONG_TERM_DELETE_ORDER
    overview_tables = service._OVERVIEW_TABLES

    consolidation_tables = {
        "MemoryMounts",
        "MemoryThreadStates",
        "MemoryClusterRelations",
        "MemorySummaryInputs",
        "MemorySummaryCache",
        "MemoryCanonicalEntities",
    }

    assert consolidation_tables <= set(delete_order)
    assert consolidation_tables <= set(overview_tables)
    assert delete_order.index("MemoryMounts") < delete_order.index("MemoryEvents")
    assert delete_order.index("MemorySummaryCache") < delete_order.index("MemoryEvents")


def test_maintenance_result_dataclasses_include_ok_and_nested_reset():
    reset = EmergencyResetResult(
        reset_id="reset-1",
        epoch=2,
        previous_focus="qq:group:sandbox",
        cleared_flow_rounds=3,
        cleared_compression_pending_jobs=1,
        cleared_compression_inflight_job=True,
        woken_waits=1,
        woken_sleeps=0,
        main_loop_restarted=False,
    )
    result = MaintenanceActionResult(
        action=RESET_COGNITION,
        maintenance_id="maint-1",
        epoch=2,
        message="done",
        reset=reset,
    ).to_dict()

    assert result["ok"] is True
    assert result["reset"]["reset_id"] == "reset-1"


def test_maintenance_helpers_mark_runtime_epoch_abort(monkeypatch):
    service = MaintenanceService()
    monkeypatch.setattr(app_state, "runtime_reset_epoch", 3)
    result = SimpleNamespace(failed=False)

    assert service.is_runtime_epoch_stale(2) is True
    assert service.make_runtime_epoch_checker(3)() is False

    marked = service.mark_result_aborted_by_reset(result, 3)
    assert marked.failed is True
    assert marked.aborted_by_runtime_reset is True
    assert marked.runtime_reset_epoch == 3
