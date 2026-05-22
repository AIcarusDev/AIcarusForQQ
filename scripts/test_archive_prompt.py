"""test_archive_prompt.py — 测试记忆提取 prompt 的实际效果。

用法（在项目根目录执行）:
    python scripts/test_archive_prompt.py

输出内容：
  - LLM 返回的原始 tool_call 参数 (JSON)
  - 逐条解析后的 events 结构化信息
  - 简单的规则自检报告（event_type 是否合规、roles 是否含 entity 等）
"""

import json
import os
import re
import sys
import textwrap

# ── 路径 ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from dotenv import load_dotenv
load_dotenv()

from config_loader import load_config
from llm.core.provider import build_archiver_adapter_cfg, create_adapter
from llm.core.tool_calling import parse_tool_arguments
from memory.archive_memories import ARCHIVE_GEN, DECLARATION, TOOL as ARCHIVE_TOOL, read_result
from memory.archive_prompt import ARCHIVE_SYSTEM_PROMPT

# ── 测试素材（原始 prompt 中的对话 + 候选记忆）────────────────────────────────
_TEST_DIALOGUE = textwrap.dedent("""\
    [场景: group/1090411227]
    <conversation type="group" id="1090411227" name="AICQ测试群" member_count="6">
    <self id="213628848" name="Icarus" card="Icc"/>
    <chat_logs mode="current" has_previous="false">
      <message id="760017362" timestamp="22小时前">
        <sender id="1321807442" nickname="未來星织" role="admin"/>
        <content type="text">算了不管你好不好玩也要去睡觉了</content>
      </message>
      <message id="1380065337" timestamp="22小时前">
        <sender id="1321807442" nickname="未來星织" role="admin"/>
        <content type="text">再见</content>
      </message>
      <message id="1853463779" timestamp="22小时前">
        <sender id="3533611951" nickname="吹雪" role="member"/>
        <quote ref_id="1015118457">
          <preview>[ERROR: Message_lost]</preview>
        </quote>
        <content type="text"><at uid="1321807442">@未來星织</at> 晚上好呀 ~</content>
      </message>
      <message id="1223766926" timestamp="22小时前">
        <sender id="3533611951" nickname="吹雪" role="member"/>
        <content type="text">都凌晨两点二十五了... 还没睡吗 ?</content>
      </message>
      <message id="710715194" timestamp="22小时前">
        <sender id="2514624910" nickname="智慧米塔" role="owner"/>
        <content type="text"><at uid="self">@Icc</at> 早</content>
      </message>
      <message id="1917380804" timestamp="22小时前">
        <sender id="self"/>
        <content type="text">早</content>
      </message>
      <message id="2073948705" timestamp="18小时前">
        <sender id="2514624910" nickname="智慧米塔" role="owner"/>
        <content type="text"><at uid="self">@Icc</at> 测试测试</content>
      </message>
      <message id="1624764207" timestamp="18小时前">
        <sender id="self"/>
        <content type="text">在的</content>
      </message>
      <message id="700438043" timestamp="18小时前">
        <sender id="2514624910" nickname="智慧米塔" role="owner"/>
        <content type="text">嗷，我调整了 prompt 结构，在测试缓存命中率</content>
      </message>
      <message id="1373503030" timestamp="18小时前">
        <sender id="self"/>
        <content type="text">原来如此</content>
      </message>
    </chat_logs>
    </conversation>

    <existing_candidates>
    #1489  ctx=episodic | 智慧米塔#qq_2514624910 at我并说「测试测试」进行功能测试 | roles: agent=User:qq_2514624910, recipient=self, theme="测试测试"
    #770  ctx=episodic | 智慧米塔#qq_2514624910 说他单独测试gemma 4模型时，同时返回content and tool_calls是稳定的 | roles: agent=User:qq_2514624910, theme="单独测试gemma 4模型时，同时返回content and tool_calls是稳定的"
    #769  ctx=episodic | 智慧米塔#qq_2514624910 问我能否同时返回content and tool_calls | roles: agent=User:qq_2514624910, recipient=Bot:self, theme="能否同时返回content and tool_calls"
    #765  ctx=episodic | 智慧米塔#qq_2514624910 问我能否同时返回 content 和 tool_calls | roles: agent=User:qq_2514624910, recipient=Bot:self, theme="能否同时返回 content 和 tool_calls"
    #766  ctx=episodic | 我解释理论上可以混合返回 content 和 tool_calls，但实际测试时文字内容被系统漏掉 | roles: agent=Bot:self, recipient=User:qq_2514624910, theme="理论上可以混合返回 content 和 tool_calls，但实际测试时文字内容被系统漏掉"
    #767  ctx=episodic | 智慧米塔#qq_2514624910 分享说单独测试 gemma 4 模型时同时返回 content 和 tool_calls 是稳定的 | roles: agent=User:qq_2514624910, recipient=Bot:self, theme="单独测试 gemma 4 模型时同时返回 content 和 tool_calls 是稳定的"
    #1371  ctx=episodic | 智慧米塔#qq_2514624910 告知我为Bot换成了稠密模型 | roles: agent=User:qq_2514624910, recipient=Bot:self, theme="换成了一个稠密模型"
    #1498  ctx=episodic | 智慧米塔#qq_2514624910 在凌晨2点25分左右对Iccc说「早」进行问候 | roles: agent=User:qq_2514624910, recipient=self
    </existing_candidates>
""")

# 合法的 event_type 词表（来自 archive_memories.py 描述）
_VALID_EVENT_TYPES = {
    "say", "share", "complain", "joke", "update",
    "teach", "correct", "ask", "answer",
    "promise", "refuse", "agree",
    "like", "dislike", "feel", "experience",
    "own", "be", "do", "isA",
}

_VALID_CONFIDENCE = {0.95, 0.80, 0.50, 0.30}


def _check_event(idx: int, ev: dict) -> list[str]:
    """对单个 event 执行简单规则自检，返回违规描述列表。"""
    issues: list[str] = []

    et = ev.get("event_type", "")
    if et not in _VALID_EVENT_TYPES:
        issues.append(f"event_type={et!r} 不在词表中（可能用了屈折形式或自创词）")

    conf = ev.get("confidence")
    try:
        conf_f = float(conf)
        if conf_f not in _VALID_CONFIDENCE:
            issues.append(f"confidence={conf_f} 不是四档锚点之一（0.95/0.80/0.50/0.30）")
    except (TypeError, ValueError):
        issues.append(f"confidence={conf!r} 无法转为数字")

    roles = ev.get("roles") or []
    has_entity = any(r.get("entity") for r in roles if isinstance(r, dict))
    if not has_entity:
        issues.append("roles 中没有任何 entity 字段（孤岛节点，违反连接性铁则）")

    summary = ev.get("summary", "")
    if "self" in summary or "User:qq_" in summary:
        issues.append(f"summary 包含图谱 ID 格式字符串（应改为自然语言称谓）: {summary!r}")

    return issues


def _print_event(idx: int, ev: dict) -> None:
    sep = "─" * 60
    print(f"\n  [{idx}] event_type={ev.get('event_type')!r}")
    print(f"      summary    : {ev.get('summary')}")
    print(f"      confidence : {ev.get('confidence')}  recall_scope={ev.get('recall_scope')}")

    roles = ev.get("roles") or []
    for r in roles:
        if isinstance(r, dict):
            entity = r.get("entity") or ""
            vt = r.get("value_text") or ""
            te = r.get("target_event") or ""
            detail = entity or f'value_text={vt!r}' or f'→#{te}'
            print(f"      role/{r.get('role'):12s}: {detail}")

    if ev.get("merge_into") is not None:
        print(f"      merge_into : #{ev['merge_into']}")
    if ev.get("supersedes") is not None:
        print(f"      supersedes : #{ev['supersedes']}")
    if ev.get("reason"):
        print(f"      reason     : {ev['reason']}")

    issues = _check_event(idx, ev)
    if issues:
        print(f"      ⚠  规则违规:")
        for iss in issues:
            print(f"         - {iss}")
    else:
        print(f"      ✓  规则自检通过")


def main() -> None:
    print("=" * 70)
    print(" 记忆提取 Prompt 效果测试")
    print("=" * 70)

    # ── 1. 加载配置 ──
    print("\n[1/3] 加载配置...")
    config, _ = load_config()
    archiver_cfg = config.get("memory", {}).get("auto_archive", {})
    adapter_cfg = build_archiver_adapter_cfg(config, archiver_cfg)
    print(f"      provider = {adapter_cfg.get('provider')}  model = {adapter_cfg.get('model')}")

    gen_cfg = archiver_cfg.get("generation", {})
    gen = {
        "temperature": float(gen_cfg.get("temperature", ARCHIVE_GEN["temperature"])),
        "max_output_tokens": int(gen_cfg.get("max_output_tokens", ARCHIVE_GEN["max_output_tokens"])),
    }
    print(f"      gen      = temperature={gen['temperature']}  max_output_tokens={gen['max_output_tokens']}")

    # ── 2. 创建适配器并调用 LLM ──
    print("\n[2/3] 调用 LLM...")
    adapter = create_adapter(adapter_cfg)

    # 直接调用底层 API，以便捕获思维链
    _tool_decl = {
        "type": "function",
        "function": {
            "name": DECLARATION["name"],
            "description": DECLARATION.get("description", ""),
            "parameters": DECLARATION.get("parameters", {}),
        },
    }
    response = adapter.client.chat.completions.create(
        model=adapter.model,
        messages=[
            {"role": "system", "content": ARCHIVE_SYSTEM_PROMPT},
            {"role": "user", "content": _TEST_DIALOGUE},
        ],
        tools=[_tool_decl],
        temperature=gen["temperature"],
        max_tokens=gen["max_output_tokens"],
    )

    msg = response.choices[0].message if response.choices else None

    # ── 思维链输出 ──
    reasoning = getattr(msg, "reasoning_content", None) if msg else None
    if reasoning:
        print("\n── 思维链 (reasoning_content) " + "─" * 38)
        print(reasoning)
    elif msg and msg.content:
        think_match = re.search(r"<think>(.*?)</think>", msg.content, re.DOTALL)
        if think_match:
            print("\n── 思维链 (<think>) " + "─" * 50)
            print(think_match.group(1).strip())

    # ── 解析工具调用参数 ──
    raw = None
    if msg and msg.tool_calls:
        args_json = msg.tool_calls[0].function.arguments
        raw, ok = parse_tool_arguments(
            args_json,
            DECLARATION["name"],
            "test_archive_prompt",
            DECLARATION,
            ARCHIVE_TOOL.schema_repairer,
            ARCHIVE_TOOL.semantic_sanitizer,
        )
        if not ok:
            raw = None

    print("\n── 原始返回 (JSON) ──────────────────────────────────────────────────")
    if raw is None:
        print("  <None — 模型未返回函数调用>")
    else:
        print(json.dumps(raw, ensure_ascii=False, indent=2))

    # ── 3. 解析 & 自检 ──
    print("\n[3/3] 解析事件 & 规则自检")
    print("── Events ───────────────────────────────────────────────────────────")
    events = read_result(raw)
    if not events:
        print("  (空，未提取到任何事件)")
    else:
        print(f"  共 {len(events)} 个事件:")
        for i, ev in enumerate(events, 1):
            _print_event(i, ev)

    total_issues = sum(len(_check_event(i, ev)) for i, ev in enumerate(events, 1))
    print("\n" + "=" * 70)
    print(f" 汇总: {len(events)} 个事件  /  {total_issues} 处规则违规")
    print("=" * 70)


if __name__ == "__main__":
    main()
