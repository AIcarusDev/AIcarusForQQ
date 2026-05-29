"""scripts/smoke_archiver.py — 归档流水线冒烟测试（Track1 + Track2 全流程）

直接调用生产代码 memory.archiver._run_archive_job，写入真实 data/AICQ.db，
然后查询新写入的 episodic / evidence 事件并打印。

用法：
    python scripts/smoke_archiver.py --group 643700843 --skip 500 --window 25
    python scripts/smoke_archiver.py --group 883059749 --skip 2000 --window 20
    python scripts/smoke_archiver.py --group 643700843 --window 30 --no-write
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# ── 路径 ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_SRC  = _ROOT / "src"
sys.path.insert(0, str(_SRC))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── 消息提取（与 test_evidence_track 保持一致） ────────────────────────────────
_CQ_RE = re.compile(r"\[CQ:[^\]]+\]")
_NOISE_RE = re.compile(
    r'^[\U0001F000-\U0001FFFF\U00002600-\U000027FF\s\[表情\]\[图片\]\[视频\]\[语音\]\[文件\]]+$',
    re.UNICODE,
)
_REACTION_WORDS = frozenset({
    "哈", "哈哈", "哈哈哈", "嗯", "嗯嗯", "哦", "啊", "好", "好的",
    "6", "66", "666", "hh", "hhh", "ok", "OK", "好耶", "坏耶",
})
_URL_RE = re.compile(r'^\s*https?://\S+\s*$')
_QQ_EMOJI_RE = re.compile(r'^\[[\u4e00-\u9fa5a-zA-Z0-9·]{1,8}\]$')


def _extract_text(raw: Any) -> str:
    if isinstance(raw, list):
        return " ".join(
            s.get("data", {}).get("text", "") for s in raw if s.get("type") == "text"
        ).strip()
    if not isinstance(raw, str):
        raw = str(raw)
    text = _CQ_RE.sub("", raw).strip()
    for src, dst in [("&amp;", "&"), ("&#91;", "["), ("&#93;", "]"), ("&lt;", "<"), ("&gt;", ">")]:
        text = text.replace(src, dst)
    return text.strip()


def _is_trivial(text: str) -> bool:
    s = text.strip()
    return (
        len(s) <= 1
        or s in _REACTION_WORDS
        or bool(_NOISE_RE.match(s))
        or bool(_QQ_EMOJI_RE.match(s))
        or bool(_URL_RE.match(s))
    )


def _load_messages(
    db_path: str,
    group_id: str,
    window: int,
    skip: int = 0,
) -> list[dict]:
    import sqlite3
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        "SELECT time, user_id, details FROM event_records "
        "WHERE post_type='message' AND event_type='group' AND group_id=? ORDER BY id",
        (group_id,),
    )
    result: list[dict] = []
    skipped = 0
    for row in c:
        ts, uid, details_json = row
        try:
            d = json.loads(details_json)
        except Exception:
            continue
        raw  = d.get("raw_message") or d.get("message", "")
        text = _extract_text(raw)
        if not text or _is_trivial(text):
            continue
        if skipped < skip:
            skipped += 1
            continue
        sender = d.get("sender", {})
        nick = (sender.get("nickname") or str(uid)) if isinstance(sender, dict) else str(uid)
        time_str = ts[11:16] if isinstance(ts, str) and len(ts) >= 16 else "??"
        result.append({"uid": str(uid), "nick": nick, "text": text, "time": time_str})
        if len(result) >= window:
            break
    conn.close()
    return result


def _format_dialogue(msgs: list[dict], group_id: str, bot_id: str) -> str:
    """构造与生产代码一致的 dialogue 文本。"""
    lines: list[str] = [f"[场景: group/{group_id}]"]
    for m in msgs:
        if m["uid"] == bot_id:
            lines.append(f"[{m['time']}] 我 (Bot:self): {m['text']}")
        else:
            lines.append(f"[{m['time']}] User:qq_{m['uid']}({m['nick']}): {m['text']}")
    alias_map = {m["nick"]: f"User:qq_{m['uid']}" for m in msgs if m["uid"] != bot_id}
    if alias_map:
        lines += ["", "<member_aliases>"]
        for nick, entity in alias_map.items():
            lines.append(f'  "{nick}" → {entity}')
        lines.append("</member_aliases>")
    return "\n".join(lines)


# ── 查询新写入事件 ─────────────────────────────────────────────────────────────
async def _query_new_events(min_event_id: int) -> list[dict]:
    """查询 event_id >= min_event_id 的 episodic/evidence 事件。"""
    from database import _connect
    async with _connect() as db:
        cursor = await db.execute(
            """
            SELECT e.event_id, e.event_type, e.context_type, e.modality,
                   e.confidence, e.summary,
                   GROUP_CONCAT(
                       r.role || '=' || COALESCE(r.entity,'') || COALESCE(r.value_text,''),
                       ' | '
                   ) AS roles_str
            FROM MemoryEvents e
            LEFT JOIN MemoryRoles r ON r.event_id = e.event_id
            WHERE e.event_id >= ? AND e.context_type IN ('episodic','evidence')
              AND e.is_deleted = 0
            GROUP BY e.event_id
            ORDER BY e.event_id ASC
            """,
            (min_event_id,),
        )
        rows = await cursor.fetchall()
    return [
        {
            "event_id":    r[0],
            "event_type":  r[1],
            "context_type":r[2],
            "modality":    r[3],
            "confidence":  r[4],
            "summary":     r[5],
            "roles_str":   r[6] or "",
        }
        for r in rows
    ]


async def _next_event_id() -> int:
    """返回下一条 event_id（当前最大值 + 1），用于划定写入前的基线。"""
    from database import _connect
    async with _connect() as db:
        cur = await db.execute("SELECT COALESCE(MAX(event_id),0)+1 FROM MemoryEvents")
        row = await cur.fetchone()
    return row[0] if row else 1


# ── 主流程 ────────────────────────────────────────────────────────────────────
async def _main(args: argparse.Namespace) -> None:
    import yaml
    import app_state
    from llm.core.provider import create_adapter, build_archiver_adapter_cfg
    from llm.core.profiles import normalize_profile_config_inplace
    from database import init_db

    # ── 配置 ──────────────────────────────────────────────────────────────────
    config_path = _ROOT / "config" / "config_user.yaml"
    if not config_path.exists():
        print(f"[✗] 配置文件不存在: {config_path}")
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    normalize_profile_config_inplace(config)
    bot_id   = str(config.get("bot_id") or config.get("self_id") or "")
    bot_name = str(config.get("bot_name") or "Bot")

    app_state.config   = config
    app_state.BOT_NAME = bot_name
    app_state.qq_adapter_client = None
    app_state.archive_tasks = set()

    archiver_cfg = config.get("memory", {}).get("auto_archive", {})
    app_state.archiver_adapter = create_adapter(
        build_archiver_adapter_cfg(config, archiver_cfg)
    ) if archiver_cfg.get("provider") else None
    app_state.adapter = create_adapter(config)

    adapter = app_state.archiver_adapter or app_state.adapter
    if adapter is None:
        print("[✗] 无可用适配器")
        sys.exit(1)
    print(f"[✓] 适配器: {adapter.model} ({adapter.provider})")

    # ── 数据库 ─────────────────────────────────────────────────────────────────
    await init_db()
    print(f"[✓] 数据库已初始化: {_ROOT / 'data' / 'AICQ.db'}")
    min_eid = await _next_event_id()

    # ── 读取消息 ───────────────────────────────────────────────────────────────
    day_db = str(_ROOT / "day_core.db")
    msgs = _load_messages(
        db_path=day_db,
        group_id=str(args.group),
        window=args.window,
        skip=args.skip,
    )
    if not msgs:
        print(f"[✗] 群 {args.group} 未找到文本消息（skip={args.skip}）")
        sys.exit(1)

    print(f"\n── 原始对话（{len(msgs)} 条）────────────────────────────")
    for m in msgs:
        print(f"  [{m['time']}] {m['nick']}({m['uid']}): {m['text'][:80]}")

    dialogue = _format_dialogue(msgs, group_id=str(args.group), bot_id=bot_id)

    if args.no_write:
        print("\n[--no-write] 跳过实际写入，仅打印 dialogue 输入：\n")
        print(dialogue)
        return

    # ── 构造 payload，调用生产 _run_archive_job ────────────────────────────────
    from memory.archiver import _run_archive_job

    payload: dict[str, Any] = {
        "job_id":              -1,          # 不存在的行，delete 会静默无操作
        "conv_type":           "group",
        "conv_id":             str(args.group),
        "conv_name":           f"群{args.group}(smoke)",
        "sender_id":           msgs[-1]["uid"] if msgs else "0",
        "dialogue":            dialogue,
        "signature":           f"smoke-{int(time.time())}",
        "prev_signature":      "",
        "valid_candidate_ids": [],
    }

    print(f"\n── 调用 _run_archive_job（Track1 + Track2）────────────────")
    t0 = time.perf_counter()
    try:
        await _run_archive_job(payload)
    except Exception as e:
        print(f"[✗] _run_archive_job 异常: {e}")
        raise
    elapsed = time.perf_counter() - t0
    print(f"   完成，耗时 {elapsed:.1f}s")

    # ── 查询并打印新写入的事件 ─────────────────────────────────────────────────
    new_events = await _query_new_events(min_eid)
    episodic = [e for e in new_events if e["context_type"] == "episodic"]
    evidence = [e for e in new_events if e["context_type"] == "evidence"]

    print(f"\n── 新写入 episodic 事件（{len(episodic)} 条）────────────────")
    for ev in reversed(episodic):
        print(f"  #{ev['event_id']}  [{ev['event_type']}/{ev['modality']}]  conf={ev['confidence']}")
        print(f"    {ev['summary']}")
        if ev["roles_str"]:
            print(f"    roles: {ev['roles_str'][:120]}")

    print(f"\n── 新写入 evidence 事件（{len(evidence)} 条）────────────────")
    for ev in reversed(evidence):
        print(f"  #{ev['event_id']}  [{ev['event_type']}]  conf={ev['confidence']}")
        print(f"    {ev['summary']}")
        if ev["roles_str"]:
            print(f"    roles: {ev['roles_str'][:120]}")

    print(f"\n── 汇总 ────────────────────────────────────────────────────")
    print(f"  消息: {len(msgs)} 条  episodic: {len(episodic)} 条  evidence: {len(evidence)} 条")
    print(f"  耗时: {elapsed:.1f}s")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="归档流水线冒烟测试（Track1 + Track2）")
    p.add_argument("--group",  required=True, type=str, help="群号")
    p.add_argument("--window", type=int, default=20,   help="取多少条消息（默认 20）")
    p.add_argument("--skip",   type=int, default=0,    help="跳过前 N 条（默认 0）")
    p.add_argument("--no-write", action="store_true",   help="仅打印 dialogue，不调用 LLM")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(_main(_parse_args()))
