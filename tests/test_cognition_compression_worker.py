import asyncio
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import app_state
from consciousness.flow import ConsciousnessFlow, ToolCall, ToolResponse
from llm.compression import worker


class CognitionCompressionWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_runs_serial_jobs_with_latest_summary(self) -> None:
        old_flow = app_state.consciousness_flow
        old_gen = app_state.GEN
        old_lock = app_state.llm_lock
        old_task = app_state.cognition_compression_task
        old_pending = app_state.cognition_compression_pending_jobs
        old_inflight = app_state.cognition_compression_inflight_job
        flow = ConsciousnessFlow()
        for i in range(1, 4):
            flow.append_round(
                [ToolCall(name="wait", args={"timeout": 600}, call_id=f"call_{i}")],
                [ToolResponse(name="wait", response={"ok": True}, call_id=f"call_{i}")],
                cognition=f"第 {i} 轮。",
            )

        calls: list[str] = []

        async def fake_run_in_daemon_thread(fn, task_xml, **kwargs):
            del fn, kwargs
            calls.append(task_xml)
            self.assertRegex(
                task_xml,
                r"^<current_time>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}</current_time>\n<task>",
            )
            if len(calls) == 1:
                self.assertIn("<last_compression/>", task_xml)
                for i in range(4, 7):
                    flow.append_round(
                        [ToolCall(name="wait", args={"timeout": 600}, call_id=f"call_{i}")],
                        [ToolResponse(name="wait", response={"ok": True}, call_id=f"call_{i}")],
                        cognition=f"第 {i} 轮。",
                    )
                worker.schedule_cognition_compression()
                return "<summary>第一段摘要</summary>"
            self.assertIn("<last_compression>第一段摘要</last_compression>", task_xml)
            return "<summary>第二段摘要</summary>"

        async def fake_save_adapter_contents(*_args, **_kwargs):
            return None

        app_state.consciousness_flow = flow
        app_state.GEN = {
            "llm_contents_max_rounds": 6,
            "cognition_compression_trigger_rounds": 3,
        }
        app_state.llm_lock = asyncio.Lock()
        app_state.cognition_compression_task = None
        app_state.cognition_compression_pending_jobs = []
        app_state.cognition_compression_inflight_job = None
        try:
            with (
                patch.object(worker, "run_in_daemon_thread", fake_run_in_daemon_thread),
                patch.object(worker, "save_adapter_contents", fake_save_adapter_contents),
            ):
                worker.schedule_cognition_compression()
                await app_state.cognition_compression_task
        finally:
            app_state.consciousness_flow = old_flow
            app_state.GEN = old_gen
            app_state.llm_lock = old_lock
            app_state.cognition_compression_task = old_task
            app_state.cognition_compression_pending_jobs = old_pending
            app_state.cognition_compression_inflight_job = old_inflight

        self.assertEqual(len(calls), 2)
        self.assertEqual(flow.active_compression_summary.text, "第二段摘要")
        self.assertEqual(flow.active_compression_summary.coverage_end_seq, 6)

    async def test_schedule_freezes_current_time_before_worker_runs(self) -> None:
        old_flow = app_state.consciousness_flow
        old_gen = app_state.GEN
        old_lock = app_state.llm_lock
        old_task = app_state.cognition_compression_task
        old_pending = app_state.cognition_compression_pending_jobs
        old_inflight = app_state.cognition_compression_inflight_job
        flow = ConsciousnessFlow()
        for i in range(1, 4):
            flow.append_round(
                [ToolCall(name="wait", args={"timeout": 600}, call_id=f"call_{i}")],
                [ToolResponse(name="wait", response={"ok": True}, call_id=f"call_{i}")],
                cognition=f"第 {i} 轮。",
            )

        captured: list[str] = []

        async def fake_run_in_daemon_thread(fn, task_xml, **kwargs):
            del fn, kwargs
            captured.append(task_xml)
            return "<summary>第一段摘要</summary>"

        async def fake_save_adapter_contents(*_args, **_kwargs):
            return None

        app_state.consciousness_flow = flow
        app_state.GEN = {
            "llm_contents_max_rounds": 6,
            "cognition_compression_trigger_rounds": 3,
        }
        app_state.llm_lock = asyncio.Lock()
        app_state.cognition_compression_task = None
        app_state.cognition_compression_pending_jobs = []
        app_state.cognition_compression_inflight_job = None
        try:
            with (
                patch.object(worker, "run_in_daemon_thread", fake_run_in_daemon_thread),
                patch.object(worker, "save_adapter_contents", fake_save_adapter_contents),
                patch(
                    "consciousness.flow._format_os_timestamp",
                    return_value="2026-05-26T12:00:00+08:00",
                ),
            ):
                worker.schedule_cognition_compression()
                await app_state.cognition_compression_task
        finally:
            app_state.consciousness_flow = old_flow
            app_state.GEN = old_gen
            app_state.llm_lock = old_lock
            app_state.cognition_compression_task = old_task
            app_state.cognition_compression_pending_jobs = old_pending
            app_state.cognition_compression_inflight_job = old_inflight

        self.assertEqual(len(captured), 1)
        self.assertTrue(
            captured[0].startswith(
                "<current_time>2026-05-26T12:00:00+08:00</current_time>\n<task>"
            )
        )

    async def test_call_compressor_uses_dedicated_model_binding(self) -> None:
        old_config = app_state.config
        old_gen = app_state.GEN
        old_cfg = app_state.cognition_compression_cfg
        old_adapter = app_state.cognition_compression_adapter

        class FakeAdapter:
            def __init__(self) -> None:
                self.call_args = None

            def call_simple_text(self, *args, **kwargs):
                self.call_args = (args, kwargs)
                return "<summary>ok</summary>"

        fake_adapter = FakeAdapter()
        created_cfgs: list[dict] = []

        def fake_create_adapter(cfg):
            created_cfgs.append(cfg)
            return fake_adapter

        app_state.config = {
            "provider": "main_provider",
            "model": "main-model",
            "model_providers": {
                "main_provider": {"base_url": "http://main"},
                "compress_provider": {"base_url": "http://compress"},
            },
        }
        app_state.GEN = {"llm_contents_max_rounds": 8}
        app_state.cognition_compression_cfg = {
            "provider": "compress_provider",
            "model": "compress-model",
            "generation": {
                "temperature": 0.2,
                "max_output_tokens": 1234,
            },
        }
        app_state.cognition_compression_adapter = None
        try:
            with patch.object(worker, "create_adapter", fake_create_adapter):
                text = worker._call_compressor("<task/>")
        finally:
            app_state.config = old_config
            app_state.GEN = old_gen
            app_state.cognition_compression_cfg = old_cfg
            app_state.cognition_compression_adapter = old_adapter

        self.assertEqual(text, "<summary>ok</summary>")
        self.assertEqual(created_cfgs[0]["provider"], "compress_provider")
        self.assertEqual(created_cfgs[0]["model"], "compress-model")
        call_args, _call_kwargs = fake_adapter.call_args
        self.assertEqual(call_args[2]["temperature"], 0.2)
        self.assertEqual(call_args[2]["max_output_tokens"], 1234)


if __name__ == "__main__":
    unittest.main()
