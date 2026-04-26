import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import app_state
from llm.core.internal_tool import InternalToolSpec
from memory.archive_memories import read_result, repair_schema_args
from memory.archiver import archive_turn_memories


class _SessionStub:
    conv_type = "group"
    conv_id = "123"
    conv_name = "测试群"
    context_messages = [
        {
            "role": "user",
            "sender_name": "Alice",
            "content": "我最近在学摄影。",
        }
    ]


class MemoryArchiveToolTests(unittest.TestCase):
    def test_archive_schema_repair_fills_missing_events(self) -> None:
        repaired, changes = repair_schema_args({})

        self.assertEqual(repaired["events"], [])
        self.assertIn("filled missing events with []", changes)

    def test_archive_read_result_returns_events_list(self) -> None:
        events = read_result(
            {
                "events": [
                    {
                        "event_type": "学习",
                        "summary": "Alice 在学摄影",
                        "roles": [
                            {"role": "agent", "entity": "User:qq_1"},
                            {"role": "theme", "value_text": "摄影"},
                        ],
                    }
                ],
            }
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "学习")

    def test_archive_read_result_drops_non_list_events(self) -> None:
        events = read_result({"events": "bad-shape"})
        self.assertEqual(events, [])


class MemoryArchiverFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_archive_turn_memories_passes_internal_tool_spec(self) -> None:
        captured: dict[str, object] = {}
        fake_adapter = SimpleNamespace(_call_forced_tool=lambda *args, **kwargs: None)
        prev_adapter = app_state.adapter
        prev_config = app_state.config

        async def fake_to_thread(func, *args):
            captured["func"] = func
            captured["args"] = args
            return {"events": []}

        try:
            app_state.adapter = fake_adapter
            app_state.config = {
                "memory": {
                    "auto_archive": {
                        "enabled": True,
                        "context_turns": 1,
                        "max_per_turn": 1,
                    }
                }
            }

            with patch("memory.archiver.asyncio.to_thread", new=fake_to_thread):
                await archive_turn_memories(_SessionStub(), "42", [])

            self.assertIn("args", captured)
            self.assertIsInstance(captured["args"][3], InternalToolSpec)
            self.assertEqual(captured["args"][3].name, "archive_memories")
        finally:
            app_state.adapter = prev_adapter
            app_state.config = prev_config


if __name__ == "__main__":
    unittest.main()
