"""sample_check_memory.py — 记忆系统随机抽样质量检查

用法:
    python scripts/sample_check_memory.py [--sample N] [--seed S]

选项:
    --sample N   每个检查类别抽取的样本数（默认 20）
    --seed   S   随机种子，0 表示不固定（默认 42）
    --verbose    打印每条样本的详细信息（默认仅打印问题条目）
    --table  T   只检查指定表（MemoryEvents / MemoryRoles / entities / entity_profiles）
"""

import argparse
import json
import os
import random
import sqlite3
import sys
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

# ── 路径 ───────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
DB_PATH = os.path.join(_ROOT_DIR, "data", "AICQ.db")

# ── 合法枚举值（与 database.py / events.py 保持一致）─────────────────────────
VALID_ROLES         = frozenset({"agent", "patient", "theme", "recipient",
                                  "instrument", "location", "time", "attribute"})
VALID_CONTEXT_TYPES = frozenset({"episodic", "hypothetical"})
VALID_POLARITY      = frozenset({"positive", "negative"})
VALID_MODALITY      = frozenset({"actual", "hypothetical", "possible"})
VALID_EVENT_TYPES   = frozenset({
    "teach", "correct", "ask", "answer", "promise", "refuse", "agree",
    "like", "dislike", "feel", "experience", "share", "complain", "joke",
    "update", "say", "tell", "do", "be", "own", "understand",
    # 进行时形式（未规范化的历史数据可能存在）
    "teaching", "correcting", "asking", "answering", "promising",
    "refusing", "agreeing", "liking", "disliking", "feeling",
    "experiencing", "sharing", "complaining", "joking", "updating",
    "saying", "telling", "doing", "being", "owning", "understanding",
})

# ── 颜色辅助 ──────────────────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def red(t):    return _c(t, "31")
def yellow(t): return _c(t, "33")
def green(t):  return _c(t, "32")
def cyan(t):   return _c(t, "36")
def bold(t):   return _c(t, "1")


# ── 数据库连接 ─────────────────────────────────────────────────────────────────
def _connect(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        print(red(f"[ERROR] 数据库文件不存在: {db_path}"))
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _fetch_sample(conn: sqlite3.Connection, table: str, where: str,
                  n: int, rng: random.Random) -> list[sqlite3.Row]:
    """从 `table` 中随机抽取最多 n 行（ORDER BY RANDOM() 对大表性能够用）。"""
    sql = f"SELECT * FROM {table}"
    if where:
        sql += f" WHERE {where}"
    # SQLite RANDOM() 不可控种子，先取所有符合条件的 rowid，再 Python 侧抽样
    cur = conn.execute(f"SELECT rowid FROM {table}" + (f" WHERE {where}" if where else ""))
    rowids = [r[0] for r in cur.fetchall()]
    if not rowids:
        return []
    sampled = rng.sample(rowids, min(n, len(rowids)))
    placeholders = ",".join("?" * len(sampled))
    rows = conn.execute(
        f"SELECT * FROM {table} WHERE rowid IN ({placeholders})",
        sampled,
    ).fetchall()
    return rows


# ── 问题收集器 ────────────────────────────────────────────────────────────────
class Report:
    def __init__(self) -> None:
        self.issues: list[dict[str, Any]] = []
        self.counts: dict[str, int] = defaultdict(int)

    def ok(self, section: str) -> None:
        self.counts[section] += 1

    def warn(self, section: str, row_id: Any, field: str, msg: str,
             value: Any = None) -> None:
        self.counts[section] += 1
        self.issues.append({
            "level": "WARN",
            "section": section,
            "id": row_id,
            "field": field,
            "msg": msg,
            "value": value,
        })

    def error(self, section: str, row_id: Any, field: str, msg: str,
              value: Any = None) -> None:
        self.counts[section] += 1
        self.issues.append({
            "level": "ERROR",
            "section": section,
            "id": row_id,
            "field": field,
            "msg": msg,
            "value": value,
        })

    def print_issues(self, verbose: bool = False) -> None:
        if not self.issues:
            print(green("  ✓ 未发现问题"))
            return
        for issue in sorted(self.issues, key=lambda x: (x["section"], str(x["id"]))):
            lvl = red("ERROR") if issue["level"] == "ERROR" else yellow("WARN ")
            val_str = ""
            if issue["value"] is not None:
                val_repr = repr(issue["value"])
                if len(val_repr) > 80:
                    val_repr = val_repr[:77] + "..."
                val_str = f"  值={val_repr}"
            print(f"  [{lvl}] {issue['section']}#{issue['id']}  "
                  f"{cyan(issue['field'])}: {issue['msg']}{val_str}")

    def summary(self) -> str:
        total_issues = len(self.issues)
        errors = sum(1 for i in self.issues if i["level"] == "ERROR")
        warns  = total_issues - errors
        if total_issues == 0:
            return green(f"全部通过，共检查 {sum(self.counts.values())} 项")
        return (f"{red(f'ERROR×{errors}')}  {yellow(f'WARN×{warns}')}  "
                f"共检查 {sum(self.counts.values())} 项")


# ── 各表检查函数 ───────────────────────────────────────────────────────────────

def check_memory_events(conn: sqlite3.Connection, report: Report,
                        n: int, rng: random.Random, verbose: bool) -> None:
    section = "MemoryEvents"
    rows = _fetch_sample(conn, section, "is_deleted=0", n, rng)
    total = conn.execute("SELECT COUNT(*) FROM MemoryEvents WHERE is_deleted=0").fetchone()[0]
    deleted = conn.execute("SELECT COUNT(*) FROM MemoryEvents WHERE is_deleted=1").fetchone()[0]
    print(bold(f"\n[{section}]  总计 {total} 条有效 / {deleted} 条已删除  →  抽样 {len(rows)} 条"))

    # 预取所有有效 event_id 集合（用于引用合法性检查）
    valid_ids = {r[0] for r in conn.execute(
        "SELECT event_id FROM MemoryEvents WHERE is_deleted=0"
    )}

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    for row in rows:
        eid = row["event_id"]
        if verbose:
            print(f"  → event_id={eid}  type={row['event_type']!r}  "
                  f"ctx={row['context_type']}  conf={row['confidence']:.2f}  "
                  f"summary={row['summary'][:60]!r}")

        # 1. event_type
        if not row["event_type"]:
            report.error(section, eid, "event_type", "为空")
        elif row["event_type"] not in VALID_EVENT_TYPES:
            report.warn(section, eid, "event_type", "不在已知词表中",
                        row["event_type"])
        else:
            report.ok(section)

        # 2. summary
        if not row["summary"] or not row["summary"].strip():
            report.error(section, eid, "summary", "为空")
        else:
            report.ok(section)

        # 3. summary_tok
        if not row["summary_tok"] or not row["summary_tok"].strip():
            report.warn(section, eid, "summary_tok", "为空（FTS 索引失效）")
        else:
            report.ok(section)

        # 4. polarity
        if row["polarity"] not in VALID_POLARITY:
            report.error(section, eid, "polarity", "非法值", row["polarity"])
        else:
            report.ok(section)

        # 5. modality
        if row["modality"] not in VALID_MODALITY:
            report.error(section, eid, "modality", "非法值", row["modality"])
        else:
            report.ok(section)

        # 6. context_type
        if row["context_type"] not in VALID_CONTEXT_TYPES:
            report.error(section, eid, "context_type", "非法值", row["context_type"])
        else:
            report.ok(section)

        # 7. confidence
        conf = float(row["confidence"])
        if not (0.0 <= conf <= 1.0):
            report.error(section, eid, "confidence", "超出 [0,1] 范围", conf)
        elif conf < 0.1:
            report.warn(section, eid, "confidence", "极低置信度（<0.1），是否误写？", conf)
        else:
            report.ok(section)

        # 8. occurred_at 合理性
        if row["occurred_at"] <= 0:
            report.warn(section, eid, "occurred_at", "时间戳为 0 或负数")
        elif row["occurred_at"] > now_ms + 86400_000:
            report.warn(section, eid, "occurred_at", "时间戳在未来（>当前+1天）",
                        row["occurred_at"])
        else:
            report.ok(section)

        # 9. merge_into 引用合法性
        if row["merge_into"] is not None:
            if row["merge_into"] not in valid_ids:
                report.warn(section, eid, "merge_into",
                            "引用了不存在或已删除的事件", row["merge_into"])
            elif row["merge_into"] == eid:
                report.error(section, eid, "merge_into", "自引用（循环合并）")
            else:
                report.ok(section)

        # 10. supersedes 引用合法性
        if row["supersedes"] is not None:
            if row["supersedes"] not in valid_ids:
                # supersedes 的旧事件通常已被 soft-delete，这是正常情况
                pass
            elif row["supersedes"] == eid:
                report.error(section, eid, "supersedes", "自引用（循环替换）")
            else:
                report.ok(section)


def check_memory_roles(conn: sqlite3.Connection, report: Report,
                       n: int, rng: random.Random, verbose: bool) -> None:
    section = "MemoryRoles"
    rows = _fetch_sample(conn, section, "", n, rng)
    total = conn.execute("SELECT COUNT(*) FROM MemoryRoles").fetchone()[0]
    print(bold(f"\n[{section}]  总计 {total} 条  →  抽样 {len(rows)} 条"))

    valid_event_ids = {r[0] for r in conn.execute(
        "SELECT event_id FROM MemoryEvents"  # 包含已删除，保持 FK 完整性即可
    )}

    for row in rows:
        rid = row["id"]
        if verbose:
            print(f"  → id={rid}  event_id={row['event_id']}  role={row['role']!r}  "
                  f"entity={row['entity']!r}  value_text={str(row['value_text'])[:40]!r}")

        # 1. role 合法性
        if row["role"] not in VALID_ROLES:
            report.error(section, rid, "role", "非法角色名", row["role"])
        else:
            report.ok(section)

        # 2. event_id 外键
        if row["event_id"] not in valid_event_ids:
            report.error(section, rid, "event_id", "指向不存在的 MemoryEvents 行",
                         row["event_id"])
        else:
            report.ok(section)

        # 3. 至少有一个有效载荷
        has_payload = (
            (row["entity"] is not None and str(row["entity"]).strip()) or
            (row["value_text"] is not None and str(row["value_text"]).strip()) or
            (row["target_event"] is not None)
        )
        if not has_payload:
            report.error(section, rid, "payload",
                         "entity / value_text / target_event 全为空")
        else:
            report.ok(section)

        # 4. entity 格式简检
        entity = row["entity"]
        if entity:
            if not (entity.startswith("User:") or entity.startswith("Bot:") or
                    entity.startswith("Group:") or ":" in entity):
                report.warn(section, rid, "entity",
                            "格式不符合 'Namespace:id' 规范", entity)
            else:
                report.ok(section)

        # 5. value_tok 与 value_text 同步性
        if row["value_text"] and not row["value_tok"]:
            report.warn(section, rid, "value_tok",
                        "value_text 非空但 value_tok 为空（FTS 索引可能失效）")
        else:
            report.ok(section)


def check_entities(conn: sqlite3.Connection, report: Report,
                   n: int, rng: random.Random, verbose: bool) -> None:
    section = "entities"
    rows = _fetch_sample(conn, section, "", n, rng)
    total = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    print(bold(f"\n[{section}]  总计 {total} 条  →  抽样 {len(rows)} 条"))

    valid_profile_ids = {r[0] for r in conn.execute(
        "SELECT profile_id FROM entity_profiles"
    )}

    for row in rows:
        uid = row["account_uid"]
        if verbose:
            print(f"  → account_uid={uid}  platform={row['platform']}  "
                  f"platform_id={row['platform_id']}  nickname={row['nickname']!r}")

        # 1. platform 非空
        if not row["platform"] or not str(row["platform"]).strip():
            report.error(section, uid, "platform", "为空")
        else:
            report.ok(section)

        # 2. platform_id 非空
        if not row["platform_id"] or not str(row["platform_id"]).strip():
            report.error(section, uid, "platform_id", "为空")
        else:
            report.ok(section)

        # 3. profile_id FK
        if row["profile_id"] not in valid_profile_ids:
            report.error(section, uid, "profile_id",
                         "引用了不存在的 entity_profiles 行", row["profile_id"])
        else:
            report.ok(section)

        # 4. nickname 建议有值
        if not row["nickname"] or not str(row["nickname"]).strip():
            report.warn(section, uid, "nickname", "昵称为空（可能未同步）")
        else:
            report.ok(section)

        # 5. extra JSON 合法性
        if row["extra"]:
            try:
                json.loads(row["extra"])
                report.ok(section)
            except json.JSONDecodeError as exc:
                report.warn(section, uid, "extra",
                            f"extra 字段不是合法 JSON: {exc}")


def check_entity_profiles(conn: sqlite3.Connection, report: Report,
                           n: int, rng: random.Random, verbose: bool) -> None:
    section = "entity_profiles"
    rows = _fetch_sample(conn, section, "", n, rng)
    total = conn.execute("SELECT COUNT(*) FROM entity_profiles").fetchone()[0]
    print(bold(f"\n[{section}]  总计 {total} 条  →  抽样 {len(rows)} 条"))

    for row in rows:
        pid = row["profile_id"]
        if verbose:
            notes_preview = str(row["notes"] or "")[:60]
            print(f"  → profile_id={pid}  sex={row['sex']!r}  age={row['age']}  "
                  f"notes={notes_preview!r}")

        # 1. notes 建议有值
        if not row["notes"] or not str(row["notes"]).strip():
            report.warn(section, pid, "notes", "AI 侧写备注为空")
        else:
            report.ok(section)

        # 2. sex 合法性（允许 None）
        VALID_SEX = {None, "", "male", "female", "unknown", "other",
                     "男", "女", "未知"}
        if row["sex"] not in VALID_SEX:
            report.warn(section, pid, "sex", "性别值不在预期集合内", row["sex"])
        else:
            report.ok(section)

        # 3. age 合理范围（允许 None）
        if row["age"] is not None:
            age = int(row["age"])
            if not (0 <= age <= 150):
                report.warn(section, pid, "age", "年龄超出合理范围 [0,150]", age)
            else:
                report.ok(section)

        # 4. extra JSON 合法性
        if row["extra"]:
            try:
                json.loads(row["extra"])
                report.ok(section)
            except json.JSONDecodeError as exc:
                report.warn(section, pid, "extra",
                            f"extra 字段不是合法 JSON: {exc}")


def check_cross_table_integrity(conn: sqlite3.Connection,
                                report: Report, n: int,
                                rng: random.Random) -> None:
    """跨表抽样完整性检查：随机取 n 条 MemoryEvents，验证其 roles 是否可正常关联。"""
    section = "跨表完整性"
    rows = _fetch_sample(conn, "MemoryEvents", "is_deleted=0", n, rng)
    print(bold(f"\n[{section}]  抽样 {len(rows)} 条 MemoryEvents，核查关联 Roles"))

    for row in rows:
        eid = row["event_id"]
        roles_rows = conn.execute(
            "SELECT * FROM MemoryRoles WHERE event_id=?", (eid,)
        ).fetchall()

        if not roles_rows:
            report.warn(section, eid, "MemoryRoles",
                        "该事件无任何角色边（孤立事件）")
        else:
            for rr in roles_rows:
                role = rr["role"]
                if role not in VALID_ROLES:
                    report.error(section, eid, "role",
                                 f"关联 MemoryRoles.id={rr['id']} 角色非法", role)
                else:
                    report.ok(section)

    # 检查是否存在 MemoryRoles 指向不存在 event_id 的孤立行（全量，轻量级）
    orphans = conn.execute(
        """SELECT COUNT(*) FROM MemoryRoles mr
           WHERE NOT EXISTS (
               SELECT 1 FROM MemoryEvents me WHERE me.event_id = mr.event_id
           )"""
    ).fetchone()[0]
    if orphans > 0:
        report.error(section, "MemoryRoles", "event_id",
                     f"存在 {orphans} 条孤立 MemoryRoles（event_id 无对应 MemoryEvents）")
    else:
        report.ok(section)


# ── 统计摘要 ──────────────────────────────────────────────────────────────────

def print_stats(conn: sqlite3.Connection) -> None:
    """打印数据库基本统计信息。"""
    tables = [
        ("MemoryEvents",    "is_deleted=0"),
        ("MemoryEvents",    "is_deleted=1"),
        ("MemoryRoles",     ""),
        ("entity_profiles", ""),
        ("entities",        ""),
        ("chat_sessions",   ""),
        ("chat_messages",   ""),
        ("bot_turns",       ""),
        ("bot_goals",       "is_deleted=0 AND status='active'"),
    ]
    print(bold("\n[数据库统计]"))
    for table, where in tables:
        sql = f"SELECT COUNT(*) FROM {table}"
        if where:
            sql += f" WHERE {where}"
        try:
            cnt = conn.execute(sql).fetchone()[0]
            label = f"{table}" + (f"[{where}]" if where else "")
            print(f"  {label:<55} {cnt:>8} 条")
        except sqlite3.OperationalError:
            pass  # 表不存在时跳过

    # MemoryEvents 各 context_type 分布
    print()
    for ctx in sorted(VALID_CONTEXT_TYPES):
        cnt = conn.execute(
            "SELECT COUNT(*) FROM MemoryEvents WHERE is_deleted=0 AND context_type=?",
            (ctx,)
        ).fetchone()[0]
        print(f"  MemoryEvents[context_type={ctx}]"
              f"{'':>{max(0, 30 - len(ctx))}} {cnt:>8} 条")

    # 置信度分布
    print()
    bins = [(0.9, 1.01, "高 [0.9,1.0]"), (0.7, 0.9, "中 [0.7,0.9)"),
            (0.4, 0.7, "低 [0.4,0.7)"), (0.0, 0.4, "极低 [0,0.4)")]
    for lo, hi, label in bins:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM MemoryEvents "
            "WHERE is_deleted=0 AND confidence>=? AND confidence<?",
            (lo, hi)
        ).fetchone()[0]
        print(f"  MemoryEvents[confidence {label}]{'':>10} {cnt:>8} 条")


# ── 主程序 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="记忆系统随机抽样质量检查",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    parser.add_argument("--sample",  type=int,  default=20,
                        metavar="N", help="每类抽样数量（默认 20）")
    parser.add_argument("--seed",    type=int,  default=42,
                        metavar="S", help="随机种子，0=不固定（默认 42）")
    parser.add_argument("--verbose", action="store_true",
                        help="打印每条样本的详情")
    parser.add_argument("--table",   type=str,  default=None,
                        metavar="T",
                        help="只检查指定表（MemoryEvents/MemoryRoles/entities/entity_profiles）")
    parser.add_argument("--db",      type=str,  default=DB_PATH,
                        metavar="PATH", help=f"数据库路径（默认 {DB_PATH}）")
    args = parser.parse_args()

    rng = random.Random(args.seed if args.seed else None)
    conn = _connect(args.db)
    report = Report()

    print(bold(f"数据库: {args.db}"))
    print(bold(f"抽样数: {args.sample}  随机种子: {args.seed if args.seed else '随机'}"))

    print_stats(conn)

    target = (args.table or "").strip().lower()

    ALL_CHECKS = {
        "memoryevents":    lambda: check_memory_events(conn, report, args.sample, rng, args.verbose),
        "memoryroles":     lambda: check_memory_roles(conn, report, args.sample, rng, args.verbose),
        "entities":        lambda: check_entities(conn, report, args.sample, rng, args.verbose),
        "entity_profiles": lambda: check_entity_profiles(conn, report, args.sample, rng, args.verbose),
        "_cross":          lambda: check_cross_table_integrity(conn, report, args.sample, rng),
    }

    if target:
        matched = {k: v for k, v in ALL_CHECKS.items() if k.startswith(target)}
        if not matched:
            print(red(f"[ERROR] 未找到表 '{args.table}'，"
                      f"可选: {', '.join(k for k in ALL_CHECKS if not k.startswith('_'))}"))
            sys.exit(1)
        for fn in matched.values():
            fn()
    else:
        for fn in ALL_CHECKS.values():
            fn()

    print(bold("\n── 问题汇总 " + "─" * 50))
    report.print_issues(verbose=args.verbose)
    print(bold("\n── 总评 " + "─" * 55))
    print(" ", report.summary())

    conn.close()

    # 有 ERROR 时返回非零退出码，方便 CI 集成
    has_errors = any(i["level"] == "ERROR" for i in report.issues)
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()
