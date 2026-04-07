"""test_memory_io.py — 记忆系统输入输出交互式检查脚本

用法：
    # 完整自动流程（写入示例数据 → 搜索 → 渲染 XML）
    python test_memory_io.py

    # 交互模式：手动输入文本逐步测试
    python test_memory_io.py --interactive

    # 只测试某一环节
    python test_memory_io.py --step tokenize
    python test_memory_io.py --step write
    python test_memory_io.py --step search
    python test_memory_io.py --step xml
"""

import asyncio
import os
import sys
import argparse
import textwrap
from datetime import datetime, timezone

# ── 路径初始化 ───────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))

# 屏蔽 jieba 初始化时的 stderr 输出，避免 PowerShell 误报退出码 1
import logging
logging.getLogger("jieba").setLevel(logging.ERROR)

# 使用临时 DB，不污染生产数据
import tempfile
_TMPDIR = tempfile.mkdtemp(prefix="memory_io_test_")
import database as _database
_database.DB_PATH = os.path.join(_TMPDIR, "test_memory_io.db")

# ── 颜色输出 ─────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def hdr(text: str)  -> str: return _c(f"\n{'─'*60}\n  {text}\n{'─'*60}", "1;36")
def ok(text: str)   -> str: return _c(f"  ✓  {text}", "32")
def inp(text: str)  -> str: return _c(f"  ▶  {text}", "33")
def out(text: str)  -> str: return _c(f"  ◀  {text}", "34")
def warn(text: str) -> str: return _c(f"  !  {text}", "31")
def dim(text: str)  -> str: return _c(text, "2")


# ════════════════════════════════════════════════════════
#  STEP 1 — 分词
# ════════════════════════════════════════════════════════

def step_tokenize(text: str | None = None) -> tuple[str, str]:
    """输入：原始文本  →  输出：(分词串, FTS查询串)"""
    from llm.memory_tokenizer import tokenize, build_fts_query
    if text is None:
        text = "用户非常喜欢苹果手机和乒乓球"
    tok = tokenize(text)
    fts = build_fts_query(text)
    return tok, fts


def demo_tokenize():
    print(hdr("STEP 1 · 分词 (tokenize & build_fts_query)"))
    samples = [
        "用户非常喜欢苹果手机和乒乓球",
        "bot的名字叫凤凰，性格温柔",
        "今天天气很好",
        "的了在是也都就",   # 全停用词，触发回退
    ]
    for s in samples:
        tok, fts = step_tokenize(s)
        print(inp(f"原文  : {s}"))
        print(out(f"分词串: {tok}"))
        print(out(f"FTS串 : {fts}"))
        print()


def interactive_tokenize():
    print(hdr("交互模式 · 分词"))
    print(dim("  输入任意文本，查看 jieba 分词结果和 FTS5 查询串。输入 q 退出。\n"))
    while True:
        try:
            raw = input("  输入文本> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if raw.lower() in ("q", "quit", "exit", ""):
            break
        tok, fts = step_tokenize(raw)
        print(out(f"分词串 : {tok}"))
        print(out(f"FTS串  : {fts}"))
        print()


# ════════════════════════════════════════════════════════
#  STEP 2 — 写入记忆
# ════════════════════════════════════════════════════════

DEMO_MEMORIES = [
    dict(subject="User:qq_12345", predicate="[喜好]",
         object_text="用户非常喜欢苹果手机",
         source="和小明聊天时", reason="用户主动提及"),
    dict(subject="User:qq_12345", predicate="[爱好]",
         object_text="用户喜欢乒乓球运动",
         source="和小明聊天时", reason="用户说自己经常去打球"),
    dict(subject="User:qq_12345", predicate="[饮食]",
         object_text="用户不喜欢吃香菜",
         source="饭局闲聊", reason="用户说香菜有异味"),
    dict(subject="Self", predicate="[自我]",
         object_text="bot的名字叫凤凰，性格温柔体贴",
         source="系统初始化", reason="persona 设定"),
    dict(subject="User:qq_99999", predicate="[职业]",
         object_text="另一个用户是程序员，平时写Python",
         source="技术讨论", reason="用户自我介绍"),
]


async def step_write_demo() -> list[int]:
    """写入演示数据，返回写入的 id 列表"""
    await _database.init_db()
    from llm.memory_tokenizer import tokenize, register_word
    from database import write_triple
    ids = []
    for m in DEMO_MEMORIES:
        tok = tokenize(m["object_text"])
        register_word(m["object_text"])
        tid = await write_triple(
            subject=m["subject"],
            predicate=m["predicate"],
            object_text=m["object_text"],
            object_text_tok=tok,
            source=m.get("source", ""),
            reason=m.get("reason", ""),
        )
        ids.append(tid)
    return ids


def demo_write():
    print(hdr("STEP 2 · 写入记忆 (write_triple)"))
    for m in DEMO_MEMORIES:
        print(inp(f"写入: [{m['subject']}] {m['predicate']} → {m['object_text']}"))

    ids = asyncio.run(step_write_demo())

    print()
    for i, (m, tid) in enumerate(zip(DEMO_MEMORIES, ids)):
        print(ok(f"id={tid}  {m['object_text'][:30]}"))
    print()
    print(dim(f"  DB 路径: {_database.DB_PATH}"))


async def _write_one(subject: str, predicate: str, text: str,
                     source: str = "手动输入", reason: str = "") -> int:
    from llm.memory_tokenizer import tokenize, register_word
    from database import write_triple
    tok = tokenize(text)
    register_word(text)
    return await write_triple(
        subject=subject, predicate=predicate,
        object_text=text, object_text_tok=tok,
        source=source, reason=reason,
    )


def interactive_write():
    print(hdr("交互模式 · 写入记忆"))
    print(dim("  格式: subject | predicate | 记忆内容  （subject 例: User:qq_123 / Self）"))
    print(dim("  输入 q 退出\n"))
    asyncio.run(_database.init_db())
    while True:
        try:
            raw = input("  写入> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if raw.lower() in ("q", "quit", "exit", ""):
            break
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) < 3:
            print(warn("格式不对，需要: subject | predicate | 记忆内容"))
            continue
        subject, predicate, text = parts[0], parts[1], parts[2]
        tid = asyncio.run(_write_one(subject, predicate, text))
        print(ok(f"写入成功，id={tid}  分词: {__import__('llm.memory_tokenizer', fromlist=['tokenize']).tokenize(text)}"))
        print()


# ════════════════════════════════════════════════════════
#  STEP 3 — 搜索召回
# ════════════════════════════════════════════════════════

async def step_search(message: str, sender_id: str = "12345") -> list[dict]:
    from llm.prompt.memory import recall_memories
    return await recall_memories(message_text=message, sender_id=sender_id)


def demo_search():
    print(hdr("STEP 3 · FTS5 搜索召回 (recall_memories)"))
    queries = [
        ("我最近想买新手机", "12345"),
        ("你喜欢运动吗", "12345"),
        ("你知道我的职业吗", "99999"),
        ("今天吃什么好", "12345"),
        ("完全不相关的话题比如量子力学", "12345"),
    ]
    for msg, sid in queries:
        results = asyncio.run(step_search(msg, sid))
        print(inp(f"消息: 「{msg}」  sender_id={sid}"))
        if results:
            for r in results:
                score = r.get("rank", "?")
                print(out(f"  [{r['id']}] {r['subject']} / {r['object_text'][:35]}  rank={score}"))
        else:
            print(out("  （无匹配记忆）"))
        print()


def interactive_search():
    print(hdr("交互模式 · 搜索召回"))
    print(dim("  格式: 消息文本  （可选在末尾加 | sender_id，默认 12345）"))
    print(dim("  输入 q 退出\n"))

    # 确保有演示数据
    rows = asyncio.run(_database.load_all_triples())
    if not rows:
        print(dim("  DB 为空，先写入演示数据..."))
        asyncio.run(step_write_demo())

    while True:
        try:
            raw = input("  搜索> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if raw.lower() in ("q", "quit", "exit", ""):
            break
        parts = raw.split("|")
        msg = parts[0].strip()
        sid = parts[1].strip() if len(parts) > 1 else "12345"
        results = asyncio.run(step_search(msg, sid))
        if results:
            for r in results:
                print(out(f"  [{r['id']}] {r['subject']}  {r['object_text'][:40]}  rank={r.get('rank','?')}"))
        else:
            print(out("  （无匹配记忆）"))
        print()


# ════════════════════════════════════════════════════════
#  STEP 4 — XML 渲染
# ════════════════════════════════════════════════════════

def demo_xml():
    print(hdr("STEP 4 · XML 渲染 (build_active_memory_xml)"))
    from llm.prompt.memory import build_active_memory_xml, restore
    from llm.memory_tokenizer import tokenize

    # 直接用内存中的演示条目构造 XML（不依赖 DB 状态）
    fake_rows = []
    import time
    for i, m in enumerate(DEMO_MEMORIES):
        fake_rows.append({
            "id": i + 1,
            "subject": m["subject"],
            "predicate": m["predicate"],
            "object_text": m["object_text"],
            "confidence": 0.8,
            "context": "",
            "created_at": int((time.time() - 3600 * (i + 1)) * 1000),
            "last_accessed": int(time.time() * 1000),
            "source": m.get("source", ""),
            "reason": m.get("reason", ""),
            "conv_type": "group",
            "conv_id": "987654",
            "conv_name": "测试群",
        })

    restore(fake_rows)
    now = datetime.now(timezone.utc)

    print(inp("场景 A：全量缓存（recalled=None，无召回时的回退模式）"))
    xml_all = build_active_memory_xml(now=now, recalled=None)
    print(dim(textwrap.indent(xml_all, "    ")))

    recalled_subset = fake_rows[:2]
    print(inp(f"\n场景 B：FTS5 召回 {len(recalled_subset)} 条（正常运行时注入 system prompt 的样子）"))
    xml_partial = build_active_memory_xml(now=now, recalled=recalled_subset)
    print(dim(textwrap.indent(xml_partial, "    ")))

    print(inp("\n场景 C：召回为空（本轮对话无相关记忆，recalled=[]）"))
    xml_empty = build_active_memory_xml(now=now, recalled=[])
    print(dim(textwrap.indent(xml_empty, "    ")))


# ════════════════════════════════════════════════════════
#  STEP 5 — 端到端：消息 → 召回 → XML
# ════════════════════════════════════════════════════════

async def _e2e(message: str, sender_id: str):
    from llm.memory_tokenizer import build_fts_query
    from llm.prompt.memory import build_active_memory_xml, restore, recall_memories
    from database import load_all_triples

    rows = await load_all_triples()
    restore(rows)

    fts_q = build_fts_query(message)
    recalled = await recall_memories(message_text=message, sender_id=sender_id)
    xml = build_active_memory_xml(recalled=recalled)
    return fts_q, recalled, xml


def demo_e2e():
    print(hdr("STEP 5 · 端到端：消息 → 召回 → XML（system prompt 最终形态）"))

    # 确保有数据
    rows = asyncio.run(_database.load_all_triples())
    if not rows:
        asyncio.run(step_write_demo())

    cases = [
        ("我最近想买 iPhone，你觉得苹果怎么样", "12345"),
        ("你知道我不喜欢什么食物", "12345"),
    ]
    for msg, sid in cases:
        fts_q, recalled, xml = asyncio.run(_e2e(msg, sid))
        print(inp(f"消息   : 「{msg}」  sender={sid}"))
        print(out(f"FTS串  : {fts_q}"))
        print(out(f"召回数 : {len(recalled)} 条"))
        print(out("XML    :"))
        print(dim(textwrap.indent(xml, "      ")))
        print()


def interactive_e2e():
    print(hdr("交互模式 · 端到端"))
    print(dim("  格式: 消息文本 | sender_id（默认 12345）"))
    print(dim("  输入 q 退出\n"))

    rows = asyncio.run(_database.load_all_triples())
    if not rows:
        print(dim("  DB 为空，先写入演示数据..."))
        asyncio.run(step_write_demo())

    while True:
        try:
            raw = input("  消息> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if raw.lower() in ("q", "quit", "exit", ""):
            break
        parts = raw.split("|")
        msg = parts[0].strip()
        sid = parts[1].strip() if len(parts) > 1 else "12345"
        fts_q, recalled, xml = asyncio.run(_e2e(msg, sid))
        print(out(f"FTS串  : {fts_q}"))
        print(out(f"召回数 : {len(recalled)} 条"))
        if recalled:
            for r in recalled:
                print(dim(f"    [{r['id']}] {r['object_text'][:40]}"))
        print(out("XML    :"))
        print(dim(textwrap.indent(xml, "    ")))
        print()


# ════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════

STEPS = {
    "tokenize": (demo_tokenize, interactive_tokenize),
    "write":    (demo_write,    interactive_write),
    "search":   (demo_search,   interactive_search),
    "xml":      (demo_xml,      None),
    "e2e":      (demo_e2e,      interactive_e2e),
}


def main():
    parser = argparse.ArgumentParser(
        description="记忆系统 IO 检查脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="进入交互模式（可手动输入文本测试）",
    )
    parser.add_argument(
        "--step", "-s", choices=list(STEPS.keys()),
        help="只运行某一环节",
    )
    args = parser.parse_args()

    # 所有步骤都需要 DB 初始化
    asyncio.run(_database.init_db())

    if args.interactive:
        # 交互模式
        if args.step:
            _, ifn = STEPS[args.step]
            if ifn:
                ifn()
            else:
                print(warn(f"{args.step} 暂无交互模式，改为运行自动演示"))
                STEPS[args.step][0]()
        else:
            # 交互菜单
            menu = {str(i+1): k for i, k in enumerate(STEPS)}
            print(hdr("记忆系统 IO 交互检查"))
            for k, v in menu.items():
                print(f"  {k}. {v}")
            print()
            while True:
                try:
                    choice = input("  选择环节 (1-5, q 退出)> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if choice.lower() in ("q", "quit", "exit", ""):
                    break
                if choice not in menu:
                    print(warn("无效选项"))
                    continue
                step_name = menu[choice]
                _, ifn = STEPS[step_name]
                if ifn:
                    ifn()
                else:
                    STEPS[step_name][0]()
    else:
        # 自动演示模式：依次运行所有步骤
        if args.step:
            STEPS[args.step][0]()
        else:
            demo_tokenize()
            demo_write()
            demo_search()
            demo_xml()
            demo_e2e()

    print(hdr("完成"))
    print(dim(f"  临时 DB: {_database.DB_PATH}"))
    print(dim(f"  （脚本退出后临时文件会保留在 {_TMPDIR}\n   下次运行会重新创建新的临时 DB）"))
    print()


if __name__ == "__main__":
    main()
