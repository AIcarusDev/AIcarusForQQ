"""scripts/test_evidence_track.py — 第二轨 (evidence) 证据推断原型测试

从 day_core.db 读取真实群聊片段，用原型 evidence 提取 Prompt 调用 LLM，
对比轨道1 (episodic) 与轨道2 (evidence) 的提取结果。

用法（项目根目录执行）：
    python scripts/test_evidence_track.py --group 643700843 --window 30
    python scripts/test_evidence_track.py --group 883059749 --skip 200 --window 20 --compare
    python scripts/test_evidence_track.py --group 643700843 --date 2025-07-15 --window 25
    python scripts/test_evidence_track.py --db data/AICQ.db ...   # 用生产库

选项：
    --group     群号（必填）
    --window    连续消息数量（默认 20）
    --skip      跳过前 N 条文本消息（默认 0）
    --date      从指定日期（YYYY-MM-DD）开始取消息
    --compare   同时调用轨道1归档做对比
    --db        源数据库路径（默认 day_core.db）
    --output    输出报告文件路径（默认仅终端打印）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── 路径 ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── 颜色输出 ──────────────────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _h1(t: str) -> str:
    bar = "═" * 64
    return _c("1;36", f"\n{bar}\n  {t}\n{bar}")


def _h2(t: str) -> str:
    return _c("1;33", f"\n{'─' * 56}\n  {t}\n{'─' * 56}")


def _ok(t: str) -> str:
    return _c("32", t)


def _warn(t: str) -> str:
    return _c("33", t)


def _err(t: str) -> str:
    return _c("31", t)


def _dim(t: str) -> str:
    return _c("2", t)


_output_lines: list[str] = []


def _print(*args, sep=" ", end="\n") -> None:
    line = sep.join(str(a) for a in args) + end
    sys.stdout.write(line)
    sys.stdout.flush()
    _output_lines.append(line)


def _flush_to_file(path: str) -> None:
    raw = "".join(_output_lines)
    clean = re.sub(r"\033\[[0-9;]*m", "", raw)
    Path(path).write_text(clean, encoding="utf-8")
    print(f"\n报告已保存至: {path}")


# ── 消息文本提取（支持 CQ 码 + segment list） ────────────────────────────────

_CQ_CODE_RE = re.compile(r"\[CQ:[^\]]+\]")
# NOTE: str.maketrans 仅支持单字符键，HTML 反转义通过 str.replace 逐一处理

_PURE_REACTION_WORDS = frozenset({
    "哈", "哈哈", "哈哈哈", "哈哈哈哈", "哈哈哈哈哈",
    "呵呵", "嘿嘿", "嗯", "嗯嗯", "哦", "哦哦",
    "啊", "啊啊", "哇", "ok", "OK",
    "好", "好的", "好好", "行", "可以",
    "6", "66", "666", "6666",
    "hh", "hhh", "hhhh", "hhhhh",
    "nb", "NB", "gg", "GG", "yyds", "awsl",
    "好耶", "坏耶", "草了", "寄了",
})
_EMOJI_NOISE_RE = re.compile(
    r'^[\U0001F000-\U0001FFFF\U00002600-\U000027FF\s'
    r'\[表情\]\[图片\]\[视频\]\[语音\]\[文件\]]+$',
    re.UNICODE,
)
_QQ_EMOJI_SHORTCUT_RE = re.compile(r'^\[[\u4e00-\u9fa5a-zA-Z0-9·]{1,8}\]$')


_URL_ONLY_RE = re.compile(r'^\s*https?://\S+\s*$')


def _extract_text(raw: Any) -> str:
    """从 CQ 码字符串或 segment list 中提取纯文字内容。

    顺序：先剥 CQ 码（此时 &#91;&#93; 还是字面文本，不干扰外层括号匹配），
    再做 HTML 反转义。
    """
    if isinstance(raw, list):
        return " ".join(
            seg.get("data", {}).get("text", "")
            for seg in raw
            if seg.get("type") == "text"
        ).strip()
    if not isinstance(raw, str):
        raw = str(raw)
    # 1. 先剥离 CQ 码（内层的 &#91;&#93; 仍是字面文本，不影响正则匹配外层括号）
    text = _CQ_CODE_RE.sub("", raw).strip()
    # 2. 再做 HTML 反转义
    for src, dst in [("&amp;", "&"), ("&#91;", "["), ("&#93;", "]"),
                     ("&lt;", "<"), ("&gt;", ">")]:
        text = text.replace(src, dst)
    return text.strip()


def _is_trivial(text: str) -> bool:
    s = text.strip()
    if len(s) <= 1:
        return True
    if s in _PURE_REACTION_WORDS:
        return True
    if _EMOJI_NOISE_RE.match(s):
        return True
    if _QQ_EMOJI_SHORTCUT_RE.match(s):
        return True
    # 纯 URL（无附加文字）一般不携带可提取的命题信息
    if _URL_ONLY_RE.match(s):
        return True
    return False


# ── day_core.db 读取 ──────────────────────────────────────────────────────────

def _load_messages(
    db_path: str,
    group_id: str,
    window: int,
    skip: int = 0,
    date_from: str | None = None,
) -> list[dict]:
    """从 event_records 表读取一段群聊消息，过滤纯噪声，返回文本消息列表。

    返回格式: [{"uid": "xxx", "nick": "xxx", "text": "xxx", "time": "HH:MM"}]
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    params: list[Any] = ["message", "group", group_id]
    where_extra = ""
    if date_from:
        where_extra = " AND time >= ?"
        params.append(date_from + " 00:00:00")

    c.execute(
        f"""
        SELECT time, user_id, details
        FROM event_records
        WHERE post_type=? AND event_type=? AND group_id=?{where_extra}
        ORDER BY id
        """,
        params,
    )

    result: list[dict] = []
    skipped = 0
    for row in c:
        ts, uid, details_json = row
        try:
            d = json.loads(details_json)
        except Exception:
            continue

        raw = d.get("raw_message") or d.get("message", "")
        text = _extract_text(raw)
        if not text or _is_trivial(text):
            continue

        sender = d.get("sender", {})
        nick = (sender.get("nickname") or str(uid)) if isinstance(sender, dict) else str(uid)
        time_str = ts[11:16] if isinstance(ts, str) and len(ts) >= 16 else "??"

        if skipped < skip:
            skipped += 1
            continue

        result.append({"uid": str(uid), "nick": nick, "text": text, "time": time_str})
        if len(result) >= window:
            break

    conn.close()
    return result


# ── dialogue 格式化 ──────────────────────────────────────────────────────────

def _format_dialogue(msgs: list[dict], group_id: str, bot_id: str = "") -> str:
    """把消息列表格式化为归档器使用的 plain text dialogue 格式。"""
    lines: list[str] = [f"[场景: group/{group_id}]"]
    for m in msgs:
        uid = m["uid"]
        nick = m["nick"]
        text = m["text"]
        t = m["time"]
        if uid == bot_id:
            lines.append(f"[{t}] 我 (Bot:self): {text}")
        else:
            lines.append(f"[{t}] User:qq_{uid}({nick}): {text}")

    # member_aliases 块
    alias_map: dict[str, str] = {}
    for m in msgs:
        if m["uid"] != bot_id:
            alias_map[m["nick"]] = f"User:qq_{m['uid']}"
    if alias_map:
        lines.append("")
        lines.append("<member_aliases>")
        for nick, entity in alias_map.items():
            lines.append(f'  "{nick}" → {entity}')
        lines.append("</member_aliases>")

    return "\n".join(lines)


# ── EVIDENCE 提取工具定义 ─────────────────────────────────────────────────────

EVIDENCE_SYSTEM_PROMPT = """
你是「证据提取助手」。本任务以函数调用形式工作：你必须且只能调用工具 extract_evidence。

输入是一批从群聊提取的 episodic 事件（记录「谁说了/做了什么」）。
你的任务是识别：这些话语中，是否有人描述了一个第三方实体的状态/行为/属性？
如果有，提取一条 evidence 事件，描述那个第三方实体，并标注说话者为证人。

核心原则：
- 你不判断命题是否为真——只判断「这段话构成某命题成立的证据」
- 必须有第三方实体：说话者说的是关于他人/工具/组织的事才值得提取
- 说话者说关于自己的事（agent = 被描述实体）→ 已在 episodic 中，跳过
- 无可提取内容时仍要调用工具并把 events 填为空数组
"""

EVIDENCE_TOOL_PROMPT = """
从群聊的 episodic 事件列表中，识别哪些事件暗示了第三方实体的状态/行为/属性，提取为 evidence 事件。

=== 核心转换模式 ===

模式1：转述他人行为
  Episodic #1: [say] User:qq_123(小明) 说了关于 "B砸了C家的窗户" 的事
  → B 不是说话者 User:qq_123，且内容描述了 B 的行为
  → Evidence: agent=Person:B, theme="砸了C家的窗户",
              instrument=User:qq_123, source_episodic_idx=1, conf=0.50
  （conf=0.50 因为是二手转述，真实性未知）

模式2：描述第三方工具/产品属性
  Episodic #2: [complain] User:qq_456(老王) 说了关于 "RTX 4090 散热太差" 的事
  → RTX 4090 不是人，是被描述的实体 Tool:RTX4090
  → Evidence: agent=Tool:RTX4090, theme="散热性能差",
              instrument=User:qq_456, source_episodic_idx=2, conf=0.70
  （conf=0.70 因为是亲身使用体验，比纯传闻更可信）

模式3：推测性询问暗示对方状态
  Episodic #3: [ask] User:qq_789(小红) 问 User:qq_123 "你最近是不是在玩原神？"
  → 询问方在推测 User:qq_123 的状态
  → Evidence: agent=User:qq_123, theme="可能在玩原神",
              instrument=User:qq_789, source_episodic_idx=3, conf=0.40
  （conf=0.40 因为是他人推测，不是当事人陈述）

=== 不要提取 ===
- agent 说关于自己的事 → 已在 episodic 中，跳过（避免重复）
  例: User:qq_A 说「我最近压力大」→ 跳过（agent=instrument，无新信息）
- 泛泛感慨，无具体实体（「好难啊」「好累」）→ 跳过
- 无法锁定命题主语的陈述 → 跳过
- Bot 说关于自己角色设定的话 → 跳过

=== 字段说明 ===
source_episodic_idx: 来源 episodic 事件的序号（从 1 开始，对应输入列表）
agent:        被描述的第三方实体（非说话者！）
              格式: User:qq_xxx / Tool:工具名 / Person:人名 / Org:组织名
summary:      命题的简洁摘要，格式「[实体] [命题]」，脱离上下文可读
theme_text:   关于该实体的完整命题文字（必填，不能为空）
instrument:   提供这条证据的说话者（User:qq_xxx 格式）
confidence:   0.30～0.80（证据只是「指向」而非「确认」，不能超过 0.80）
  0.80 = 说话者亲身使用/接触后描述的第三方事物（如「我用的4090散热很差」）
  0.60 = 说话者明确陈述他人/第三方情况（如「我朋友说...」但比较明确）
  0.50 = 二手转述，来源不确定（如「A说B砸了C的窗户」）
  0.40 = 推测性询问或间接暗示
  0.30 = 极弱暗示，仅提供参考
raw_quote:    触发这条证据的原始文字片段（用于溯源）
"""

EVIDENCE_TOOL_DECL: dict[str, Any] = {
    "name": "extract_evidence",
    "description": EVIDENCE_TOOL_PROMPT,
    "parameters": {
        "type": "object",
        "required": ["events"],
        "properties": {
            "events": {
                "type": "array",
                "description": "证据事件列表（空数组表示无可提取的证据）",
                "items": {
                    "type": "object",
                    "required": ["source_episodic_idx", "agent", "summary",
                                 "theme_text", "instrument", "confidence"],
                    "properties": {
                        "source_episodic_idx": {
                            "type": "integer",
                            "description": "来源 episodic 事件的序号（从 1 开始）",
                        },
                        "agent": {
                            "type": "string",
                            "description": "被描述的第三方实体（非说话者），格式: User:qq_xxx / Tool:xxx / Person:xxx / Org:xxx",
                        },
                        "summary": {
                            "type": "string",
                            "description": "命题的简洁摘要，格式'[实体] [命题]'，脱离上下文可独立阅读",
                        },
                        "theme_text": {
                            "type": "string",
                            "description": "关于该实体的完整命题文字（必填）",
                        },
                        "instrument": {
                            "type": "string",
                            "description": "提供这条证据的说话者（User:qq_xxx 格式）",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "证据强度 0.30~0.80",
                        },
                        "raw_quote": {
                            "type": "string",
                            "description": "触发证据的原始文字（简短，用于溯源）",
                        },
                    },
                },
            }
        },
    },
}

EVIDENCE_GEN = {"temperature": 0.3, "max_output_tokens": 6000}


def _read_evidence_result(raw: dict | None) -> list[dict]:
    if not isinstance(raw, dict):
        return []
    events = raw.get("events") or []
    if not isinstance(events, list):
        return []
    return events


def _format_episodic_for_evidence(episodic_events: list[dict]) -> str:
    """将 episodic 事件列表格式化为 evidence 提取器的输入文本。

    每条事件标注序号（1-based），供 LLM 在输出的 source_episodic_idx 中引用。
    """
    if not episodic_events:
        return "（本段对话无 episodic 事件）"

    lines: list[str] = ["=== 待处理的 Episodic 事件列表 ===", ""]
    for i, ev in enumerate(episodic_events, 1):
        etype  = ev.get("event_type", "?")
        conf   = ev.get("confidence", 0)
        ctx    = ev.get("context_type", "")
        mod    = ev.get("modality", "actual")
        summary = ev.get("summary", "")
        roles  = ev.get("roles") or []

        lines.append(f"#{i}  [{etype}/{mod}]  conf={conf}  ctx={ctx}")
        lines.append(f"    摘要: {summary}")

        role_parts: list[str] = []
        for r in roles:
            rname = r.get("role", "?")
            entity = r.get("entity", "")
            vtext  = r.get("value_text", "")
            if entity:
                role_parts.append(f"{rname}={entity}")
            elif vtext:
                role_parts.append(f'{rname}="{vtext[:50]}"')
        if role_parts:
            lines.append("    roles: " + ", ".join(role_parts))
        lines.append("")

    lines.append("请对每条 episodic 事件，判断是否暗示了第三方实体的命题，提取为 evidence 事件。")
    return "\n".join(lines)


# ── 最小化 app_state 初始化 ──────────────────────────────────────────────────

def _init_minimal(config: dict, bot_name: str = "Bot", bot_id: str = "0") -> None:
    """初始化测试脚本所需的最小 app_state（仅 archiver_adapter）。"""
    import app_state
    from llm.core.provider import create_adapter, build_archiver_adapter_cfg
    from llm.core.profiles import normalize_profile_config_inplace

    normalize_profile_config_inplace(config)
    app_state.config = config
    app_state.BOT_NAME = bot_name
    app_state.napcat_client = None
    app_state.archive_tasks = set()

    # 主模型适配器（仅 --compare 用）
    try:
        app_state.adapter = create_adapter(config)
        _print(_ok(f"  [✓] 主模型: {app_state.adapter.model} ({app_state.adapter.provider})"))
    except Exception as e:
        _print(_warn(f"  [!] 主模型未配置或初始化失败: {e}"))
        app_state.adapter = None

    # archiver 适配器
    archiver_cfg = config.get("memory", {}).get("auto_archive", {})
    if archiver_cfg.get("provider") and archiver_cfg.get("model"):
        try:
            app_state.archiver_adapter = create_adapter(
                build_archiver_adapter_cfg(config, archiver_cfg)
            )
            _print(_ok(
                f"  [✓] 归档适配器: {app_state.archiver_adapter.model}"
                f" ({app_state.archiver_adapter.provider})"
            ))
        except Exception as e:
            _print(_err(f"  [✗] 归档适配器初始化失败: {e}"))
            app_state.archiver_adapter = None
    else:
        _print(_warn("  [!] 未配置归档适配器 (memory.auto_archive.provider/model)"))
        app_state.archiver_adapter = None


# ── 调用 LLM（同步 via asyncio.to_thread） ───────────────────────────────────

async def _call_tool(adapter, system: str, dialogue: str, gen: dict, tool_decl: dict, tag: str) -> dict | None:
    """在线程中调用 adapter._call_forced_tool，避免阻塞事件循环。"""
    return await asyncio.to_thread(
        adapter._call_forced_tool,
        system,
        dialogue,
        gen,
        tool_decl,
        tag,
    )


# ── 格式化打印 ────────────────────────────────────────────────────────────────

def _print_messages(msgs: list[dict]) -> None:
    for m in msgs:
        _print(_dim(f"  [{m['time']}] {m['nick']}({m['uid']}): {m['text'][:100]}"))


def _print_evidence_events(
    events: list[dict],
    episodic_events: list[dict] | None = None,
) -> None:
    """打印 evidence 事件，若提供 episodic_events 则显示溯源箭头。"""
    if not events:
        _print(_warn("  (无证据事件)"))
        return
    for i, ev in enumerate(events, 1):
        src_idx  = ev.get("source_episodic_idx")  # 1-based
        agent    = ev.get("agent", "?")
        conf     = ev.get("confidence", 0)
        summary  = ev.get("summary", "")
        theme    = ev.get("theme_text", "")
        instr    = ev.get("instrument", "")
        quote    = ev.get("raw_quote", "")

        conf_color = "32" if conf >= 0.6 else ("33" if conf >= 0.4 else "2")
        _print(_c("1", f"  #{i}") + f"  agent={_c('1;35', agent)}  "
               + f"conf={_c(conf_color, str(conf))}"
               + (f"  {_dim(f'← episodic#{src_idx}')}" if src_idx else ""))

        # 显示来源 episodic 事件摘要（方便人工核查）
        if src_idx and episodic_events and 1 <= src_idx <= len(episodic_events):
            src_ev = episodic_events[src_idx - 1]
            src_summary = src_ev.get("summary", "")
            _print(_dim(f"      来源:  [{src_ev.get('event_type','?')}] {src_summary[:80]}"))

        _print(f"      摘要:  {summary}")
        _print(f"      命题:  {_c('36', theme)}")
        _print(f"      证人:  {instr}")
        if quote:
            _print(_dim(f"      原文:  「{quote[:80]}」"))


def _print_episodic_events(events: list[dict]) -> None:
    if not events:
        _print(_warn("  (无轨道1事件)"))
        return
    for i, ev in enumerate(events, 1):
        etype = ev.get("event_type", "?")
        summary = ev.get("summary", "")
        conf = ev.get("confidence", 0)
        ctx = ev.get("context_type", "")
        mod = ev.get("modality", "")
        conf_color = "32" if conf >= 0.8 else ("33" if conf >= 0.5 else "2")
        _print(_c("1", f"  #{i}") + f"  [{etype}/{mod}]  ctx={ctx}  "
               + f"conf={_c(conf_color, str(conf))}")
        _print(f"      {summary}")
        roles = ev.get("roles", [])
        if roles:
            role_strs = []
            for r in roles:
                rv = r.get("entity") or (f'"{r["value_text"]}"' if r.get("value_text") else None)
                if rv:
                    role_strs.append(f"{r.get('role','?')}={rv}")
            if role_strs:
                _print(_dim(f"      roles: " + ", ".join(role_strs)))


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

async def _main(args: argparse.Namespace) -> None:
    # ── 加载配置 ─────────────────────────────────────────────────────────────
    import yaml

    config_path = _ROOT / "config" / "config_user.yaml"
    if not config_path.exists():
        _print(_err(f"[✗] 配置文件不存在: {config_path}"))
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    bot_id = str(config.get("bot_id") or config.get("self_id") or "")
    bot_name = str(config.get("bot_name") or "Bot")

    _print(_h1(f"第二轨 (evidence) 测试  —  群 {args.group}"))
    _print(f"\n  数据库: {args.db}")
    _print(f"  窗口:   {args.window} 条文本消息（跳过前 {args.skip} 条）")
    if args.date:
        _print(f"  日期:   >= {args.date}")
    _print(f"  详细输入: {'是（--verbose-input）' if args.verbose_input else '否'}\n")

    # ── 初始化适配器 ──────────────────────────────────────────────────────────
    _init_minimal(config, bot_name=bot_name, bot_id=bot_id)

    import app_state
    adapter = app_state.archiver_adapter or app_state.adapter
    if adapter is None:
        _print(_err("[✗] 无可用 LLM 适配器，退出"))
        sys.exit(1)

    # ── 读取真实消息 ──────────────────────────────────────────────────────────
    _print(_h2("原始对话"))
    t0 = time.perf_counter()
    db_path = str(_ROOT / args.db) if not Path(args.db).is_absolute() else args.db
    msgs = _load_messages(
        db_path=db_path,
        group_id=str(args.group),
        window=args.window,
        skip=args.skip,
        date_from=args.date,
    )
    if not msgs:
        _print(_err(f"[✗] 群 {args.group} 中未找到文本消息（db={db_path}，skip={args.skip}，date={args.date}）"))
        sys.exit(1)

    _print(f"  取到 {len(msgs)} 条文本消息：\n")
    _print_messages(msgs)

    dialogue = _format_dialogue(msgs, group_id=str(args.group), bot_id=bot_id)

    # ── no-cot：为 gen 注入 extra_body ────────────────────────────────────────
    no_cot_extra: dict = {"extra_body": {"enable_thinking": False}} if args.no_cot else {}

    # ── 轨道 1：episodic（永远先跑，为轨道 2 提供结构化输入）────────────────
    from memory.archive_memories import ARCHIVE_GEN, TOOL as ARCHIVE_TOOL, read_result
    from memory.archive_prompt import ARCHIVE_SYSTEM_PROMPT as ARC_SYS

    arc_gen = {**ARCHIVE_GEN, **no_cot_extra}

    _print(_h2("轨道 1 — episodic（从对话提取结构化事件）"
               + (" [no-CoT]" if args.no_cot else "")))
    _print(_dim("  调用中…"))
    t1 = time.perf_counter()
    raw1 = await _call_tool(
        adapter,
        ARC_SYS,
        dialogue,
        arc_gen,
        ARCHIVE_TOOL,
        "archive-track1",
    )
    t2 = time.perf_counter()
    ev1 = read_result(raw1)
    _print(f"  耗时: {t2 - t1:.1f}s  提取到 {len(ev1)} 条事件\n")
    _print_episodic_events(ev1)

    if not ev1:
        _print(_warn("\n  轨道1无输出，轨道2跳过（无 episodic 事件可处理）"))
        _print(f"\n  总耗时: {time.perf_counter() - t0:.1f}s")
        if args.output:
            _flush_to_file(args.output)
        return

    # ── 轨道 2：evidence（以 episodic 事件为输入，推断第三方命题）─────────────
    _print(_h2("轨道 2 — evidence（从 episodic 事件推断第三方证据）"
               + (" [no-CoT]" if args.no_cot else "")))
    _print(_dim("  构建输入（episodic 事件 → evidence 格式化文本）…"))

    evidence_input = _format_episodic_for_evidence(ev1)
    if args.verbose_input:
        _print(_dim("  --- 发给 LLM 的输入 ---"))
        for line in evidence_input.splitlines():
            _print(_dim("  " + line))
        _print(_dim("  --- 结束 ---\n"))

    _print(_dim("  调用中…"))
    t3 = time.perf_counter()
    ev_gen = {**EVIDENCE_GEN, **no_cot_extra}
    raw2 = await _call_tool(
        adapter,
        EVIDENCE_SYSTEM_PROMPT,
        evidence_input,
        ev_gen,
        EVIDENCE_TOOL_DECL,
        "evidence-track2",
    )
    t4 = time.perf_counter()
    ev2 = _read_evidence_result(raw2)
    _print(f"  耗时: {t4 - t3:.1f}s  提取到 {len(ev2)} 条证据\n")
    _print_evidence_events(ev2, episodic_events=ev1)

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    _print(_h2("汇总"))
    _print(f"  原始消息:  {len(msgs)} 条")
    _print(f"  轨道1事件: {len(ev1)} 条")
    _print(f"  轨道2证据: {len(ev2)} 条", end="")
    if ev2:
        dist = ", ".join(
            f"{c:.2f}×{sum(1 for e in ev2 if abs(e.get('confidence', 0) - c) < 0.05)}"
            for c in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
            if any(abs(e.get('confidence', 0) - c) < 0.05 for e in ev2)
        )
        _print(f"  (conf 分布: {dist})")
    else:
        _print()

    # 溯源覆盖率：有多少 episodic 事件至少产生了一条 evidence
    sourced_idxs = {e.get("source_episodic_idx") for e in ev2 if e.get("source_episodic_idx")}
    _print(f"  溯源覆盖:  {len(sourced_idxs)}/{len(ev1)} 条 episodic 事件产生了证据")
    not_sourced = [i for i in range(1, len(ev1) + 1) if i not in sourced_idxs]
    if not_sourced:
        _print(_dim(f"  未产生证据的 episodic 事件: {not_sourced}"))
    _print(f"  总耗时: {time.perf_counter() - t0:.1f}s")

    if args.output:
        _flush_to_file(args.output)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="第二轨 (evidence) 证据推断原型测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--group", required=True, help="群号")
    p.add_argument("--window", type=int, default=20, help="取多少条文本消息（默认 20）")
    p.add_argument("--skip", type=int, default=0, help="跳过前 N 条文本消息（默认 0）")
    p.add_argument("--date", default=None, help="从指定日期开始（YYYY-MM-DD）")
    p.add_argument("--verbose-input", action="store_true",
                   help="打印发给 LLM 的 episodic 格式化文本（默认隐藏）")
    p.add_argument("--no-cot", action="store_true",
                   help="禁用 CoT（Qwen3等支持 enable_thinking 的模型），通过 extra_body 传入")
    p.add_argument("--db", default="day_core.db", help="源数据库路径（默认 day_core.db）")
    p.add_argument("--output", default=None, help="输出报告文件路径")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(_main(args))
