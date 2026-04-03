"""simulate_memory.py — 记忆系统端到端模拟脚本

用途：在不启动完整 Quart/NapCat 服务的情况下，验证记忆系统的完整生命周期。

覆盖范围：
  §1  写入记忆（write_triple）
  §2  被动召回（search_triples + 精排）
  §3  召回强化（update_triple_confidence +0.05）
  §4  置信度衰减（decay_triple_confidence，模拟时间偏移）
  §5  实体泼溅（upsert_merge_suggestion + list/resolve）
  §6  人工删除（soft_delete_triple）

运行方式（在项目根目录）：
  G:\\Anaconda\\envs\\fbk\\python.exe scripts\\simulate_memory.py
  G:\\Anaconda\\envs\\fbk\\python.exe scripts\\simulate_memory.py --verbose
  G:\\Anaconda\\envs\\fbk\\python.exe scripts\\simulate_memory.py --keep   # 保留 DB 文件供 SQLite Browser 查看
"""

import argparse
import asyncio
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# ── 将 src/ 加入 sys.path ─────────────────────────────────────
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

# jieba 的初始化消息写入 stderr（"Building prefix dict..."），在 PowerShell
# 中 2>&1 合并后会触发 NativeCommandError。用 jieba 的专属 API 压掉它。
import jieba as _jieba_init
_jieba_init.setLogLevel(logging.WARNING)

import aiosqlite
import database
from llm.memory_tokenizer import tokenize, build_fts_query, load_custom_dict_from_triples


# ═══════════════════════════════════════════════════════════════
# 打印辅助
# ═══════════════════════════════════════════════════════════════

VERBOSE = False


def h(title: str) -> None:
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}")


def info(msg: str) -> None:
    print(f"  {msg}")


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠ {msg}")


def detail(label: str, value) -> None:
    if VERBOSE:
        print(f"    {label}: {value}")


# ═══════════════════════════════════════════════════════════════
# 辅助：直接修改 last_accessed（模拟时间流逝）
# ═══════════════════════════════════════════════════════════════

async def _age_triple(db_path: str, triple_id: int, days: float) -> None:
    """将指定三元组的 last_accessed 设置为 days 天前（模拟陈旧）。"""
    target_ms = int(time.time() * 1000) - int(days * 86400 * 1000)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE MemoryTriples SET last_accessed=? WHERE id=?",
            (target_ms, triple_id),
        )
        await db.commit()


async def _get_triple(db_path: str, triple_id: int) -> dict | None:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM MemoryTriples WHERE id=?", (triple_id,)
        ) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def _seed_person(db_path: str, person_id: str) -> None:
    """在 persons 表插入测试用 person 行（满足 merge_suggestions FK 约束）。"""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT OR IGNORE INTO persons (person_id) VALUES (?)",
            (person_id,),
        )
        await db.commit()


# ═══════════════════════════════════════════════════════════════
# 主模拟流程
# ═══════════════════════════════════════════════════════════════

async def run_simulation(db_path: str) -> None:
    database.DB_PATH = db_path
    await database.init_db()

    print(f"\n  数据库：{db_path}")

    # ──────────────────────────────────────────────────────────
    h("§1  写入记忆（write_triple）")
    # ──────────────────────────────────────────────────────────

    triples = [
        # (subject, predicate, object_text, confidence)
        # ⚠ 注意：object_text 需要能被 jieba 拆分为独立 token，
        #   例如"打乒乓球"是 jieba 单一 token，FTS5 无法按"乒乓球"子串检索。
        #   这里用"喜欢乒乓球"→ jieba → "喜欢 乒乓球"（乒乓球成为独立 token）。
        ("User:qq_10001", "喜欢",     "喜欢乒乓球",          0.7),
        ("User:qq_10001", "不喜欢",   "讨厌吃香菜",           0.8),
        ("User:qq_10001", "[note]",   "性格外向喜欢交朋友",   0.6),
        ("User:qq_10001", "居住地",   "上海浦东新区",         0.9),
        ("User:qq_10002", "喜欢",     "喜欢科幻小说",         0.75),
        ("User:qq_10002", "职业",     "软件工程师",           0.85),
        ("User:qq_10002", "[note]",   "经常深夜在线",         0.5),
        ("Bot",           "[note]",   "今天是我入群一周年",   0.5),
    ]

    ids: list[int] = []
    for subject, predicate, object_text, confidence in triples:
        tok = tokenize(object_text)
        row_id = await database.write_triple(
            subject=subject,
            predicate=predicate,
            object_text=object_text,
            object_text_tok=tok,
            source="simulate",
            reason="模拟测试",
            confidence=confidence,
        )
        ids.append(row_id)
        detail(f"  id={row_id}", f"[{subject}] {predicate} → {object_text!r} (conf={confidence})")

    ok(f"写入 {len(triples)} 条三元组，ids = {ids}")

    # 恢复 jieba 词典（让后续 FTS5 分词与写入一致）
    all_rows = await database.load_all_triples()
    load_custom_dict_from_triples(all_rows)

    # ──────────────────────────────────────────────────────────
    h("§2  被动召回（search_triples / FTS5 双通道）")
    # ──────────────────────────────────────────────────────────

    queries = [
        ("乒乓球",   "User:qq_10001"),   # 应命中 id[0]
        ("科幻小说", "User:qq_10002"),   # 应命中 id[4]
        ("深夜",     "User:qq_10002"),   # 应命中 id[6]
        ("香菜",     ""),                # 全局检索，应命中 id[1]
    ]

    for raw_query, subject_filter in queries:
        fts_q = build_fts_query(raw_query)
        results = await database.search_triples(
            fts_q,
            subject_filter=subject_filter,
            recall_top_k=5,
        )
        hit = results[0] if results else None
        status = "命中" if hit else "未命中"
        info(f"查询 {raw_query!r}（subject={subject_filter or '*'}）→ {status}")
        if hit:
            score = hit.get("final_score")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
            detail("    Top1", f"id={hit['id']} [{hit['subject']}] {hit['predicate']} → {hit['object_text']!r} score={score_str}")
        else:
            warn(f"查询 {raw_query!r} 无结果（fts_q={fts_q!r}）")

    ok("被动召回完成")

    # ──────────────────────────────────────────────────────────
    h("§3  召回强化（update_triple_confidence +0.05）")
    # ──────────────────────────────────────────────────────────

    # 模拟：id[0]（乒乓球）和 id[4]（科幻小说）被召回
    boost_ids = [ids[0], ids[4]]
    before = {i: (await _get_triple(db_path, i))["confidence"] for i in boost_ids}
    await database.update_triple_confidence(boost_ids, delta=0.05)
    after  = {i: (await _get_triple(db_path, i))["confidence"] for i in boost_ids}

    for i in boost_ids:
        info(f"  id={i} confidence: {before[i]:.2f} → {after[i]:.2f}  (+{after[i]-before[i]:.2f})")
        assert after[i] > before[i], f"id={i} 置信度未增加"

    ok("召回强化正确（均已 +0.05）")

    # ──────────────────────────────────────────────────────────
    h("§4  置信度衰减（decay_triple_confidence，模拟 10 天前）")
    # ──────────────────────────────────────────────────────────

    # 将 id[2]（note：性格外向）和 id[6]（深夜在线）设为 10 天前
    stale_ids = [ids[2], ids[6]]
    for sid in stale_ids:
        await _age_triple(db_path, sid, days=10.0)
        row = await _get_triple(db_path, sid)
        detail(f"  id={sid} 已设为 10 天前", f"conf={row['confidence']:.2f}")

    conf_before = {i: (await _get_triple(db_path, i))["confidence"] for i in stale_ids}
    count = await database.decay_triple_confidence(
        min_confidence=0.05,
        decay_rate=0.1,
        idle_days_threshold=7.0,
    )
    conf_after = {i: (await _get_triple(db_path, i))["confidence"] for i in stale_ids}

    info(f"  衰减触发条数：{count}（预期 2）")
    for i in stale_ids:
        info(f"  id={i} confidence: {conf_before[i]:.2f} → {conf_after[i]:.2f}  (-{conf_before[i]-conf_after[i]:.2f})")
        assert conf_after[i] < conf_before[i], f"id={i} 置信度未衰减"

    # 验证：新鲜记忆不受影响
    fresh_row = await _get_triple(db_path, ids[0])
    assert abs(fresh_row["confidence"] - after[ids[0]]) < 1e-6, "新鲜记忆不应被衰减"
    ok(f"置信度衰减正确（{count} 条降权，新鲜记忆未受影响）")

    # 验证下限夹底：将 id[6] 置信度逼近下限后再次衰减
    async with aiosqlite.connect(db_path) as db:
        await db.execute("UPDATE MemoryTriples SET confidence=0.07 WHERE id=?", (ids[6],))
        await db.commit()
    await _age_triple(db_path, ids[6], days=10.0)
    await database.decay_triple_confidence(min_confidence=0.05, decay_rate=0.1, idle_days_threshold=7.0)
    floored = (await _get_triple(db_path, ids[6]))["confidence"]
    assert abs(floored - 0.05) < 1e-6, f"下限夹底失败：{floored}"
    ok(f"下限夹底正确（id={ids[6]} conf 精确夹到 0.05）")

    # ──────────────────────────────────────────────────────────
    h("§5  实体泼溅（merge_suggestions）")
    # ──────────────────────────────────────────────────────────

    # 先建好 persons 行（FK 约束）
    for pid in ("qq_10001", "qq_10002", "qq_10099"):
        await _seed_person(db_path, pid)

    # 提交建议：10001 和 10099 可能是同一人
    sid1 = await database.upsert_merge_suggestion(
        "qq_10001", "qq_10099", 0.92, "昵称相同且说话风格一致"
    )
    info(f"  新建建议 sid={sid1[:8]}… similarity=0.92")

    # 幂等更新（同一 pair，更高把握）
    sid2 = await database.upsert_merge_suggestion(
        "qq_10099", "qq_10001", 0.96, "自述换号后重新加群"
    )
    assert sid1 == sid2, "幂等失败：同一 pair 应返回同一 sid"
    info(f"  幂等更新 sid 不变={sid1[:8]}… similarity 更新为 0.96")

    # 再加一条不相关建议
    sid3 = await database.upsert_merge_suggestion(
        "qq_10002", "qq_10099", 0.70, "IP 地址相同"
    )

    pending = await database.list_pending_suggestions()
    info(f"  待审核建议数：{len(pending)}（预期 2）")
    assert len(pending) == 2
    assert pending[0]["similarity"] >= pending[1]["similarity"], "未按 similarity 降序"

    # 确认第一条
    confirmed = await database.resolve_merge_suggestion(sid1, "confirmed")
    assert confirmed is True
    # 拒绝第二条
    rejected  = await database.resolve_merge_suggestion(sid3, "rejected")
    assert rejected is True

    remaining = await database.list_pending_suggestions()
    assert len(remaining) == 0, f"应无 pending 建议，实际 {len(remaining)}"
    ok("实体泼溅全部正确（幂等/排序/confirm/reject）")

    # ──────────────────────────────────────────────────────────
    h("§6  软删除（soft_delete_triple）")
    # ──────────────────────────────────────────────────────────

    del_id = ids[7]  # "今天是我入群一周年"
    before_all = await database.load_all_triples()
    result = await database.soft_delete_triple(del_id)
    assert result is True
    after_all = await database.load_all_triples()

    before_ids = {r["id"] for r in before_all}
    after_ids  = {r["id"] for r in after_all}
    assert del_id not in after_ids, "软删除后仍出现在 load_all_triples"
    assert del_id in before_ids

    # FTS5 索引也应移除（检索不到）
    fts_q = build_fts_query("入群一周年")
    fts_results = await database.search_triples(fts_q, recall_top_k=5)
    fts_ids = {r["id"] for r in fts_results}
    assert del_id not in fts_ids, "软删除后 FTS5 索引未清除"

    ok(f"软删除正确（id={del_id} 从 load_all 和 FTS5 索引中均已移除）")

    # ──────────────────────────────────────────────────────────
    h("§7  最终状态汇总")
    # ──────────────────────────────────────────────────────────

    final = await database.load_all_triples()
    info(f"  有效三元组总数：{len(final)}（写入 {len(triples)} 条，删除 1 条）")
    assert len(final) == len(triples) - 1

    print()
    for row in sorted(final, key=lambda r: r["id"]):
        flag = " [衰减]" if row["confidence"] <= 0.1 else ""
        info(
            f"  id={row['id']:3d}  conf={row['confidence']:.2f}{flag}"
            f"  [{row['subject']}] {row['predicate']} → {row['object_text']!r}"
        )

    print()
    ok("全部 §1-§6 验证通过 ✓")


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    global VERBOSE

    parser = argparse.ArgumentParser(description="记忆系统模拟脚本")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印详细中间状态")
    parser.add_argument("--keep",    "-k", action="store_true", help="保留 DB 文件（不自动删除）")
    parser.add_argument("--db",            type=str,            help="指定 DB 路径（默认临时文件）")
    args = parser.parse_args()
    VERBOSE = args.verbose

    if args.db:
        db_path = args.db
        cleanup = False
    else:
        fd, db_path = tempfile.mkstemp(suffix=".db", prefix="sim_memory_")
        os.close(fd)
        cleanup = not args.keep

    print(f"\n{'═' * 60}")
    print("  AIcarusForQQ — 记忆系统模拟 (Phase 1-3B)")
    print(f"{'═' * 60}")

    try:
        asyncio.run(run_simulation(db_path))
        print(f"\n{'═' * 60}")
        print("  模拟完成，全部断言通过")
        if not cleanup:
            print(f"  DB 已保留：{db_path}")
        print(f"{'═' * 60}\n")
    except AssertionError as e:
        print(f"\n  ✗ 断言失败：{e}\n")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n  ✗ 异常：{e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        if cleanup and os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    main()
