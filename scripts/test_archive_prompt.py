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
import sys
import textwrap

# ── 路径 ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from dotenv import load_dotenv
load_dotenv()

from config_loader import load_config
from llm.core.provider import build_archiver_adapter_cfg, create_adapter
from memory.archive_memories import ARCHIVE_GEN, TOOL as ARCHIVE_TOOL, read_result
from memory.archive_prompt import ARCHIVE_SYSTEM_PROMPT

# ── 测试素材（原始 prompt 中的对话 + 候选记忆）────────────────────────────────
_TEST_DIALOGUE = textwrap.dedent("""\
    [场景: group/1030770193]
    <conversation type="group" id="1030770193" name="deepseek无禁交流群">
    <self id="213628848" name="Iccc"/>
    <chat_logs mode="current" has_previous="false">
      <message id="778814082" timestamp="46分钟前">
        <sender id="self"/>
        <content type="sticker">[动画表情 id="003"]</content>
      </message>
      <message id="28991260" timestamp="46分钟前">
        <sender id="2514624910" nickname="智慧米塔" role="admin"/>
        <content type="sticker">[动画表情 ref="54935ca62ca8"]</content>
        <!-- 1张图片（base64已省略） -->
      </message>
      <message id="698486325" timestamp="40分钟前">
        <sender id="1833114026" nickname="四两七星" role="member"/>
        <content type="sticker">[动画表情 ref="ea9f161efa25"]</content>
        <!-- 1张图片（base64已省略） -->
      </message>
      <message id="1720336362" timestamp="40分钟前">
        <sender id="1833114026" nickname="四两七星" role="member"/>
        <content type="text">进了类脑发现自己太懒狗导致还有好长时间的缓冲期</content>
      </message>
      <message id="1770037022" timestamp="40分钟前">
        <sender id="1833114026" nickname="四两七星" role="member"/>
        <content type="image">[图片 ref="5f1a58148958"]</content>
        <!-- 1张图片（base64已省略） -->
      </message>
      <message id="1380299935" timestamp="39分钟前">
        <sender id="self"/>
        <content type="text">也挺正常的</content>
      </message>
      <message id="523806214" timestamp="17秒前">
        <sender id="2019880148" nickname="古希腊の电流表" role="member"/>
        <content type="text">ai无限流真的不好玩</content>
      </message>
      <message id="913274259" timestamp="11秒前">
        <sender id="2019880148" nickname="古希腊の电流表" role="member"/>
        <content type="text">用户推出来的就是对的</content>
      </message>
      <message id="1376390699" timestamp="刚刚">
        <sender id="1833114026" nickname="四两七星" role="member"/>
        <content type="text"><at uid="2514624910">@智慧米塔</at> 不懂就问，用酒馆是会有很多ai同时回答吗</content>
      </message>
      <message id="967785762" timestamp="刚刚">
        <sender id="2019880148" nickname="古希腊の电流表" role="member"/>
        <quote ref_id="913274259">
          <preview>古希腊の电流表: 用户推出来的就是对的</preview>
        </quote>
        <content type="text">无论怎么样都会往这个方向走</content>
      </message>
    </chat_logs>
    </conversation>

    <existing_candidates>
    #2528  ctx=episodic pol=positive  | 我对四两七星#qq_1833114026 说在类脑缓冲期长也挺正常的 | roles: agent=Bot:self, recipient=User:qq_1833114026, theme="也挺正常的"
    #612  ctx=episodic pol=positive  | 智慧米塔#qq_2514624910 说明 id 字段应为 string 类型 | roles: agent=User:qq_2514624910, recipient=User:qq_1321807442, theme="id 字段的类型定义应为 string"
    #2439  ctx=episodic pol=positive  | 智慧米塔#qq_2514624910 纠正我别说得跟知道一样，说话没头没尾 | roles: agent=User:qq_2514624910, recipient=Bot:self, theme="别说得跟你知道一样，说的话没头没尾的"
    #614  ctx=episodic pol=positive  | 智慧米塔#qq_2514624910 建议根据 ID 格式决定是否让 json repair 修复类型错误 | roles: agent=User:qq_2514624910, recipient=User:qq_1321807442, theme="如果确认 ID 不会是 "qq_xxxxx" 格式，可以让 json repair 负责修复 string 到 number 的转换"
    #613  ctx=episodic pol=positive  | 智慧米塔#qq_2514624910 建议若 ID 无前缀可使用 JSON repair 修复类型错误 | roles: agent=User:qq_2514624910, recipient=User:qq_1321807442, theme="如果确认 ID 不包含 "qq_" 等非数字字符，JSON repair 可以修复 string/number 类型错误"
    #2458  ctx=episodic pol=positive  | 我同意智慧米塔#qq_2514624910 并发语音消息 | roles: agent=Bot:self, recipient=User:qq_2514624910, theme="发送语音消息"
    #2457  ctx=episodic pol=positive  | 智慧米塔#qq_2514624910 让我再发几条语音试试 | roles: agent=User:qq_2514624910, recipient=Bot:self, theme="再发几条语音试试"
    #2453  ctx=episodic pol=positive  | 智慧米塔#qq_2514624910 说要教我然后看记忆压缩和召回是否有效 | roles: agent=User:qq_2514624910, recipient=Bot:self, theme="需要教，然后看看之后相关的记忆压缩和召回是否有效"
    </existing_candidates>
""")

# 合法的 event_type 词表（来自 archive_memories.py 描述）
_VALID_EVENT_TYPES = {
    "say", "share", "complain", "joke", "update",
    "teach", "correct", "ask", "answer",
    "promise", "refuse", "agree",
    "like", "dislike", "feel", "experience",
    "own", "be", "do",
}

_VALID_POLARITIES = {"positive", "negative"}
_VALID_MODALITIES = {"actual", "hypothetical", "possible"}
_VALID_CONTEXT_TYPES = {"meta", "contract", "episodic", "hypothetical"}
_VALID_CONFIDENCE = {0.95, 0.80, 0.50, 0.30}


def _check_event(idx: int, ev: dict) -> list[str]:
    """对单个 event 执行简单规则自检，返回违规描述列表。"""
    issues: list[str] = []

    et = ev.get("event_type", "")
    if et not in _VALID_EVENT_TYPES:
        issues.append(f"event_type={et!r} 不在词表中（可能用了屈折形式或自创词）")

    pol = ev.get("polarity", "")
    if pol not in _VALID_POLARITIES:
        issues.append(f"polarity={pol!r} 非法（应为 positive/negative）")

    mod = ev.get("modality", "")
    if mod not in _VALID_MODALITIES:
        issues.append(f"modality={mod!r} 非法")

    ctx = ev.get("context_type", "")
    if ctx not in _VALID_CONTEXT_TYPES:
        issues.append(f"context_type={ctx!r} 非法")

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
    if "Bot:self" in summary or "User:qq_" in summary:
        issues.append(f"summary 包含图谱 ID 格式字符串（应改为自然语言称谓）: {summary!r}")

    return issues


def _print_event(idx: int, ev: dict) -> None:
    sep = "─" * 60
    print(f"\n  [{idx}] event_type={ev.get('event_type')!r}")
    print(f"      summary    : {ev.get('summary')}")
    print(f"      polarity   : {ev.get('polarity')}  modality={ev.get('modality')}  ctx={ev.get('context_type')}")
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
    raw = adapter._call_forced_tool(
        ARCHIVE_SYSTEM_PROMPT,
        _TEST_DIALOGUE,
        gen,
        ARCHIVE_TOOL,
        "test_archive_prompt",
    )

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
