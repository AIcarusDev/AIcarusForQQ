import asyncio
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import app_state
from consciousness import main_loop
from llm.core.provider import RoundResult


class _FakeRateLimiter:
    async def acquire(self) -> None:
        return None


class _FakeSession:
    def __init__(self) -> None:
        self.context_messages = [
            {"role": "user", "message_id": "m1", "content": "？"},
        ]
        self.pending_early_trigger = None

    async def prepare_memory_recall(self) -> None:
        return None

    def build_system_prompt(self, activated_names=None, latent_names=None) -> str:
        return "system"


class RetryOnNewMessageTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_one_round_interrupts_and_retries_only_once(self) -> None:
        session = _FakeSession()
        call_checkers = []

        async def fake_run_in_daemon_thread(*args, **kwargs):
            checker = args[6]
            call_checkers.append(checker)
            if len(call_checkers) == 1:
                self.assertIsNotNone(checker)
                session.context_messages.append(
                    {"role": "user", "message_id": "m2", "content": "是你"}
                )
                self.assertTrue(checker())
                return RoundResult(new_message_during_thinking=True)
            self.assertIsNone(checker)
            return RoundResult(had_tool_call=True)

        old_gen = app_state.GEN
        old_rate_limiter = app_state.rate_limiter
        old_llm_lock = app_state.llm_lock
        old_adapter = app_state.adapter
        old_flow = app_state.consciousness_flow
        try:
            app_state.GEN = {"retry_on_new_message": True}
            app_state.rate_limiter = _FakeRateLimiter()
            app_state.llm_lock = asyncio.Lock()
            app_state.adapter = SimpleNamespace(call_one_round=lambda *args, **kwargs: None)
            app_state.consciousness_flow = object()

            with (
                patch.object(main_loop, "build_main_user_prompt", return_value="user"),
                patch.object(main_loop, "_build_tool_collection", return_value=object()),
                patch.object(main_loop, "_restore_latent_tools_from_flow"),
                patch.object(
                    main_loop,
                    "run_in_daemon_thread",
                    side_effect=fake_run_in_daemon_thread,
                ),
                patch(
                    "llm.prompt.quote_prefetch.prefetch_quoted_messages",
                    new=AsyncMock(),
                ),
            ):
                result = await main_loop._run_one_round(session, "group_1")
        finally:
            app_state.GEN = old_gen
            app_state.rate_limiter = old_rate_limiter
            app_state.llm_lock = old_llm_lock
            app_state.adapter = old_adapter
            app_state.consciousness_flow = old_flow

        self.assertTrue(result.had_tool_call)
        self.assertEqual(len(call_checkers), 2)
        self.assertIsNotNone(call_checkers[0])
        self.assertIsNone(call_checkers[1])


if __name__ == "__main__":
    unittest.main()
