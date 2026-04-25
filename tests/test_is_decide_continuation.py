import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import app_state
from llm.IS.core import _call_is_model_sync
from llm.core.internal_tool import InternalToolSpec


class _FakeIsAdapter:
    def __init__(self, result: dict | None) -> None:
        self.result = result
        self.calls: list[dict] = []

    def _call_forced_tool(self, system_prompt, user_content, gen, tool_decl, log_tag="IS"):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_content": user_content,
                "gen": gen,
                "tool_decl": tool_decl,
                "log_tag": log_tag,
            }
        )
        return self.result


class IsDecideContinuationTests(unittest.TestCase):
    def test_call_is_model_sync_uses_internal_tool_spec(self) -> None:
        fake_adapter = _FakeIsAdapter({"continue": False, "reason": " stop now "})
        prev_is_adapter = app_state.is_adapter
        prev_adapter = app_state.adapter
        prev_is_cfg = app_state.is_cfg
        try:
            app_state.is_adapter = fake_adapter
            app_state.adapter = None
            app_state.is_cfg = {}

            should_continue, reason = _call_is_model_sync("system", "user")

            self.assertFalse(should_continue)
            self.assertEqual(reason, " stop now ")
            self.assertIsInstance(fake_adapter.calls[0]["tool_decl"], InternalToolSpec)
            self.assertEqual(fake_adapter.calls[0]["tool_decl"].name, "decide_continuation")
        finally:
            app_state.is_adapter = prev_is_adapter
            app_state.adapter = prev_adapter
            app_state.is_cfg = prev_is_cfg


if __name__ == "__main__":
    unittest.main()