from __future__ import annotations

import asyncio

import app_state
from consciousness.flow import extract_structured_compression_summary
from llm.compression import worker


def test_structured_compression_summary_requires_analysis_then_summary():
    text = """
<analysis>
这里是不会注入上下文的分析。
</analysis>

<summary>
这是真正的摘要。
</summary>
"""

    assert extract_structured_compression_summary(text) == "这是真正的摘要。"


def test_structured_compression_summary_rejects_leaked_analysis_inside_summary():
    text = """
<summary>
`，确保历史信息不被丢弃。

**重要性评估：**
- 这些本该属于 analysis。

</analysis>

<summary>
这是后面的真正摘要。
</summary>
"""

    assert extract_structured_compression_summary(text) == ""


def test_compression_worker_retries_until_schema_valid(monkeypatch):
    outputs = [
        "<summary>bad</summary>",
        "<analysis>ok</analysis><summary>good summary</summary>",
    ]
    calls = []
    queued = []

    async def fake_run_in_daemon_thread(func, *args, thread_name=None):
        calls.append((func, args, thread_name))
        return outputs.pop(0)

    class FakeFlow:
        def queue_compression_summary(self, summary, coverage_end_seq):
            queued.append((summary, coverage_end_seq))
            return False

    monkeypatch.setattr(worker, "run_in_daemon_thread", fake_run_in_daemon_thread)
    monkeypatch.setattr(worker.maintenance_service, "is_runtime_epoch_stale", lambda epoch: False)
    monkeypatch.setattr(app_state, "runtime_reset_epoch", 0)
    monkeypatch.setattr(app_state, "GEN", {})
    monkeypatch.setattr(app_state, "config", {})
    monkeypatch.setattr(app_state, "cognition_compression_cfg", {"generation": {"schema_retries": 1}})
    monkeypatch.setattr(app_state, "llm_lock", asyncio.Lock())
    monkeypatch.setattr(app_state, "consciousness_flow", FakeFlow())

    ok = asyncio.run(
        worker._run_cognition_compression(
            "<compression_input/>",
            coverage_end_seq=120,
            round_count=5,
            expected_epoch=0,
        )
    )

    assert ok is True
    assert len(calls) == 2
    assert queued == [("good summary", 120)]
