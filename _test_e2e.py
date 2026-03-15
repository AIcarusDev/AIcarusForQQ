"""端到端测试：实际调用 Gemini 原生 API，验证工具调用 + JSON 结构化输出。"""
from dotenv import load_dotenv
load_dotenv()

import json
from provider import create_adapter
from tools import TOOL_DECLARATIONS, TOOL_REGISTRY
from schema import RESPONSE_SCHEMA

cfg = {
    "provider": "gemini",
    "model": "gemini-3.1-flash-lite-preview",
    "thinking": {"level": "high"},
}
adapter = create_adapter(cfg)

gen = {"temperature": 1.0, "max_output_tokens": 8192, "max_tool_rounds": 5}

def system_prompt_builder(tool_budget):
    budget_lines = []
    for name, info in tool_budget.items():
        budget_lines.append(f"- {name}: {info['remaining']}/{info['total']}")
    budget_text = "\n".join(budget_lines) if budget_lines else "无可用工具"
    return f"""你是一个测试机器人。请用中文回复。
可用工具配额：
{budget_text}
"""

user_content = "你好，你能查一下你运行在什么设备上吗？"

print("=" * 60)
print("调用 Gemini 原生 API...")
print(f"模型: {adapter.model}")
print(f"用户消息: {user_content}")
print("=" * 60)

result, grounding, repaired, tool_calls_log = adapter.call(
    system_prompt_builder,
    user_content,
    gen,
    RESPONSE_SCHEMA,
    tool_declarations=TOOL_DECLARATIONS,
    tool_registry=TOOL_REGISTRY,
)

print(f"\n工具调用日志 ({len(tool_calls_log)} 次):")
for log in tool_calls_log:
    print(f"  Round {log['round']}: {log['function']}({log.get('motivation', '')})")
    print(f"    结果: {json.dumps(log['result'], ensure_ascii=False)[:200]}...")

print(f"\nJSON 修复: {repaired}")
print(f"\n模型输出:")
if result:
    print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    print("(None — 调用失败)")
