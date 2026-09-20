from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import app_state
from consciousness import main_loop
from llm.prompt import container


def test_only_successful_completed_main_round_spends_custom_lifetime(monkeypatch):
    async def run_case(failed: bool):
        shutdown = asyncio.Event()
        monkeypatch.setattr(app_state, "shutdown_event", shutdown)
        monkeypatch.setattr(app_state, "current_focus", "test")
        monkeypatch.setattr(main_loop, "current_focus_key", lambda _: "test")
        session = SimpleNamespace(conv_type="group", conv_id="1", conv_name="test")
        monkeypatch.setattr(main_loop, "get_or_create_session", lambda _: session)
        monkeypatch.setattr(main_loop, "_maybe_reset_transient_session_views", lambda *_: None)
        monkeypatch.setattr(main_loop, "emit_agent_event", lambda *_, **__: None)
        monkeypatch.setattr(main_loop.maintenance_service, "is_runtime_epoch_stale", lambda *_: False)
        from browser import image_confirmation
        monkeypatch.setattr(image_confirmation, "current_pending", lambda *_: None)
        monkeypatch.setattr(image_confirmation, "expire_pending_after_round", lambda *_: None)
        result = SimpleNamespace(
            failed=failed, tool_calls_log=[], prompt_tokens=0, output_tokens=0,
            runtime_reset_epoch=0, agent_run_id="test", llm_error=None,
        )
        monkeypatch.setattr(main_loop, "_run_one_round", AsyncMock(return_value=result))
        monkeypatch.setattr(main_loop, "_persist_round", AsyncMock(return_value=True))
        advance = AsyncMock()
        monkeypatch.setattr(container, "advance_completed_round", advance)
        monkeypatch.setattr(main_loop.core_restart, "shutdown_after_round_if_requested",
                            AsyncMock(return_value=True))

        async def stop_after_failure(_seconds):
            shutdown.set()

        monkeypatch.setattr(main_loop.asyncio, "sleep", stop_after_failure)
        await main_loop.consciousness_main_loop()
        assert advance.await_count == (0 if failed else 1)

    asyncio.run(run_case(False))
    asyncio.run(run_case(True))
