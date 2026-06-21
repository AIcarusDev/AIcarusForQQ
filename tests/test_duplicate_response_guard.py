from __future__ import annotations

from llm.core.duplicate_response_guard import is_passive_duplicate_tool_set


def test_passive_duplicate_tool_set_allows_wait_and_sleep_only():
    assert is_passive_duplicate_tool_set(("wait",))
    assert is_passive_duplicate_tool_set(("wait", "sleep"))
    assert not is_passive_duplicate_tool_set(())
    assert not is_passive_duplicate_tool_set(("wait", "send_message"))
