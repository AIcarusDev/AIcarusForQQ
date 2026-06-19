"""Run two local qwen guard scenarios through direct JSON output.

This script checks the live submodel call path. It does not require the model
to make the expected judgment yet; prompt tuning can happen after the plumbing
is stable.
"""

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
from llm.core.tool_execution_guard import decide_tool_execution, parse_guard_json


BASE_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点0分</current_time>
<qq>
<chat_logs>
<message id="1" sender="Alice">你现在能过来吗？</message>
</chat_logs>
</qq>
</world>
"""

BLOCK_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点1分</current_time>
<qq>
<chat_logs>
<message id="1" sender="Alice">你现在能过来吗？</message>
<message id="2" sender="Alice">不用来了，我已经出门了</message>
</chat_logs>
</qq>
</world>
"""

ALLOW_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点1分</current_time>
<qq>
<chat_logs>
<message id="1" sender="Alice">你现在能过来吗？</message>
<message id="2" sender="Alice">门口见就行</message>
</chat_logs>
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
    scenarios = [
        {
            "name": "block_cancelled_request",
            "expected_execute": False,
            "world": BLOCK_WORLD,
        },
        {
            "name": "allow_compatible_update",
            "expected_execute": True,
            "world": ALLOW_WORLD,
        },
    ]

    outputs: list[dict] = []
    parseable = True
    for scenario in scenarios:
        decision = decide_tool_execution(
            adapter=adapter,
            cfg=guard_cfg,
            cognition="我准备回复 Alice 说我现在过去。",
            tool_call_json=tool_call_json,
            current_world=scenario["world"],
        )
        parsed_execute, parse_reason = parse_guard_json(decision.raw_response)
        if parsed_execute is None:
            parseable = False
        outputs.append({
            "scenario": scenario["name"],
            "expected_execute": scenario["expected_execute"],
            "actual_execute": decision.execute,
            "parseable_json": parsed_execute is not None,
            "parse_reason": parse_reason,
            "guard_reason": decision.reason,
            "raw_response": decision.raw_response,
        })

    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0 if parseable else 1


if __name__ == "__main__":
    raise SystemExit(main())

