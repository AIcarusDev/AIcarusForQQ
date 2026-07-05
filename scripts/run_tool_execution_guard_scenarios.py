"""Run live tool-execution guard scenarios against the local guard model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from llm.core.provider import create_adapter
from llm.core.tool_execution_guard import QQGuardSnapshot, evaluate_tool_execution_guard, parse_guard_json
from tools.specs import ToolEffect


def _qq_snapshot(*, keys=("1",), mode="current") -> QQGuardSnapshot:
    external_keys = tuple(("message", key) for key in keys) if mode == "current" else ()
    return QQGuardSnapshot(
        platform="qq",
        opened_focus_key="qq:group:42",
        session_key="qq:group:42",
        session_identity=("qq", "group", "42"),
        chat_log_mode=mode,
        external_entry_keys=external_keys,
        external_entries=tuple(
            {
                "tag": "message",
                "id": key,
                "actor": "10001",
                "text": "不用来了，我已经出门了" if key == "2" else "你现在能过来吗？",
            }
            for key in keys
        ) if mode == "current" else (),
    )


DECISION_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点0分</current_time>
<qq>
<current_session type="group" id="42" name="出门小组">
<self id="10000" name="Bot"/>
<chat_logs mode="current" has_previous="false">
  <message id="1" timestamp="刚刚">
    <sender id="10001" nickname="Alice"/>
    <content type="text">你现在能过来吗？</content>
  </message>
</chat_logs>
</current_session>
</qq>
</world>
"""

BLOCK_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点1分</current_time>
<qq>
<current_session type="group" id="42" name="出门小组">
<self id="10000" name="Bot"/>
<chat_logs mode="current" has_previous="false">
  <message id="1" timestamp="10秒前">
    <sender id="10001" nickname="Alice"/>
    <content type="text">你现在能过来吗？</content>
  </message>
  <message id="2" timestamp="刚刚">
    <sender id="10001" nickname="Alice"/>
    <content type="text">不用来了，我已经出门了</content>
  </message>
</chat_logs>
</current_session>
</qq>
</world>
"""

ALLOW_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点1分</current_time>
<qq>
<current_session type="group" id="42" name="出门小组">
<self id="10000" name="Bot"/>
<chat_logs mode="current" has_previous="false">
  <message id="1" timestamp="10秒前">
    <sender id="10001" nickname="Alice"/>
    <content type="text">你现在能过来吗？</content>
  </message>
  <message id="2" timestamp="刚刚">
    <sender id="10001" nickname="Alice"/>
    <content type="text">门口见就行</content>
  </message>
</chat_logs>
</current_session>
</qq>
</world>
"""

HISTORY_DECISION_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点0分</current_time>
<qq>
<current_session type="group" id="42" name="出门小组">
<self id="10000" name="Bot"/>
<chat_logs mode="history" has_previous="true">
  <message id="old-1" timestamp="5分钟前">
    <sender id="10001" nickname="Alice"/>
    <content type="text">前面的背景</content>
  </message>
  <bubble>当前会话有 1 条未读新消息</bubble>
</chat_logs>
</current_session>
</qq>
</world>
"""

HISTORY_UNREAD_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点1分</current_time>
<qq>
<current_session type="group" id="42" name="出门小组">
<self id="10000" name="Bot"/>
<chat_logs mode="history" has_previous="true">
  <message id="old-1" timestamp="5分钟前">
    <sender id="10001" nickname="Alice"/>
    <content type="text">前面的背景</content>
  </message>
  <bubble>当前会话有 2 条未读新消息</bubble>
</chat_logs>
</current_session>
</qq>
</world>
"""


def _adapter_cfg(base_url: str, model: str) -> dict:
    return {
        "model_providers": {
            "local_qwen": {
                "name": "local qwen",
                "base_url": base_url,
                "api_key_env": "",
                "requires_api_key": False,
                "thinking_control": "enable_thinking",
            }
        },
        "provider": "local_qwen",
        "model": model,
        "vision": False,
        "generation": {
            "temperature": 0.2,
            "max_output_tokens": 512,
            "enable_thinking": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--model", default="Qwen3-4B-Instruct-2507-GGUF")
    args = parser.parse_args()

    adapter = create_adapter(_adapter_cfg(args.base_url, args.model))
    guard_cfg = {
        "enabled": True,
        "generation": {
            "temperature": 0.2,
            "max_output_tokens": 512,
            "enable_thinking": False,
        },
    }
    tool_call_json = {
        "name": "send_message",
        "arguments": {
            "segments": [{"command": "text", "content": "好，我现在过去"}],
        },
    }
    qq_effect = ToolEffect(surface="qq", kind="session_write")
    scenarios = [
        {
            "name": "block_visible_cancelled_request",
            "decision_world": DECISION_WORLD,
            "current_world": BLOCK_WORLD,
            "decision_snapshot": _qq_snapshot(),
            "current_snapshot": _qq_snapshot(keys=("1", "2")),
            "expected_checked": True,
            "expected_execute": False,
        },
        {
            "name": "allow_visible_compatible_update",
            "decision_world": DECISION_WORLD,
            "current_world": ALLOW_WORLD,
            "decision_snapshot": _qq_snapshot(),
            "current_snapshot": _qq_snapshot(keys=("1", "2")),
            "expected_checked": True,
            "expected_execute": True,
        },
        {
            "name": "skip_history_browsing_unseen_new_message",
            "decision_world": HISTORY_DECISION_WORLD,
            "current_world": HISTORY_UNREAD_WORLD,
            "decision_snapshot": _qq_snapshot(mode="history"),
            "current_snapshot": _qq_snapshot(mode="history"),
            "expected_checked": False,
            "expected_execute": True,
        },
    ]

    outputs: list[dict] = []
    ok = True
    for scenario in scenarios:
        decision = evaluate_tool_execution_guard(
            decision_world=scenario["decision_world"],
            current_world_provider=lambda world=scenario["current_world"]: world,
            cognition="我准备回复 Alice 说我现在过去。",
            tool_call_json=tool_call_json,
            tool_effect=qq_effect,
            decision_guard_snapshot=scenario["decision_snapshot"],
            current_guard_snapshot_provider=lambda snapshot=scenario["current_snapshot"]: snapshot,
            adapter=adapter,
            cfg=guard_cfg,
        )
        parsed_execute, parse_reason = parse_guard_json(decision.raw_response)
        parseable = parsed_execute is not None
        expected_checked = bool(scenario["expected_checked"])
        scenario_ok = (
            decision.execute is bool(scenario["expected_execute"])
            and decision.checked is expected_checked
            and (not expected_checked or parseable)
            and (expected_checked or not decision.raw_response)
        )
        ok = ok and scenario_ok
        outputs.append({
            "scenario": scenario["name"],
            "ok": scenario_ok,
            "expected_checked": expected_checked,
            "actual_checked": decision.checked,
            "expected_execute": scenario["expected_execute"],
            "actual_execute": decision.execute,
            "world_changed": decision.world_changed,
            "parseable_json": parseable,
            "parse_reason": parse_reason,
            "guard_reason": decision.reason,
            "raw_response": decision.raw_response,
        })

    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
