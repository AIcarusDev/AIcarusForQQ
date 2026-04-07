"""routes_memory.py — 记忆图谱相关路由

Blueprint: memory_bp
  GET /memory              — 记忆图谱页面
  GET /memory/graph        — vis.js 节点/边 JSON
  GET /memory/triples      — 指定 subject 的 MemoryTriples 列表
                             ?subject=<str>（默认"Self"） &limit=<int>（默认100）
"""

import logging

import aiosqlite
from quart import Blueprint, render_template, jsonify, request

from database import DB_PATH

logger = logging.getLogger("AICQ.app")

memory_bp = Blueprint("memory", __name__)


# ── helpers ───────────────────────────────────────────────────────────────────

async def _triple_counts_by_subject(db) -> dict:
    """返回 {subject: count} 字典，只计未删除条目。
    若 MemoryTriples 表不存在（旧数据库）则返回空 dict。
    """
    counts: dict = {}
    try:
        async with db.execute(
            "SELECT subject, COUNT(*) AS n FROM MemoryTriples "
            "WHERE is_deleted=0 GROUP BY subject"
        ) as cur:
            async for row in cur:
                counts[row["subject"]] = row["n"]
    except Exception:
        pass
    return counts


# ── routes ────────────────────────────────────────────────────────────────────

@memory_bp.route("/memory")
async def memory_page():
    return await render_template("memory.html", active_page="memory")


@memory_bp.route("/memory/graph")
async def memory_graph():
    """查询数据库，返回 { nodes: [...], edges: [...] } 供前端 vis.js 使用。"""
    nodes: list = []
    edges: list = []

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row

            # 预查 MemoryTriples 各 subject 的三元组计数
            triple_counts = await _triple_counts_by_subject(db)

            # ── Self 节点（Bot 自身记忆）─────────────────────
            self_count = triple_counts.get("Self", 0)
            if self_count > 0:
                nodes.append({
                    "id":    "self",
                    "label": f"Self\n({self_count} 条)",
                    "group": "self",
                    "title": f"Bot 自身记忆：{self_count} 条",
                    "extra": {"subject": "Self", "triple_count": self_count},
                })

            # ── Persons ───────────────────────────────────────
            # columns: person_id, sex, age, area, notes, ...
            async with db.execute(
                "SELECT person_id, notes FROM persons LIMIT 500"
            ) as cur:
                async for row in cur:
                    pid = "p-" + row["person_id"]
                    label = (row["notes"] or row["person_id"])[:20]
                    nodes.append({
                        "id":    pid,
                        "label": label,
                        "group": "person",
                        "title": f"自然人: {row['person_id']}",
                        "extra": {
                            "person_id": row["person_id"],
                            "notes":     row["notes"] or "",
                        },
                    })

            # ── Accounts ──────────────────────────────────────
            # columns: account_uid, person_id, platform, platform_id, nickname, ...
            async with db.execute(
                "SELECT account_uid, person_id, platform, platform_id, nickname "
                "FROM accounts LIMIT 1000"
            ) as cur:
                async for row in cur:
                    aid     = "a-" + row["account_uid"]
                    nick    = row["nickname"] or f"{row['platform']}:{row['platform_id']}"
                    subject = f"User:qq_{row['platform_id']}"
                    tc      = triple_counts.get(subject, 0)
                    label   = nick if not tc else f"{nick}\n({tc} 条)"
                    nodes.append({
                        "id":    aid,
                        "label": label,
                        "group": "account",
                        "title": f"账号: {nick} ({row['platform']})"
                                 + (f" — {tc} 条记忆" if tc else ""),
                        "extra": {
                            "platform":     row["platform"],
                            "platform_id":  row["platform_id"],
                            "nickname":     row["nickname"] or "",
                            "subject":      subject,
                            "triple_count": tc,
                        },
                    })
                    if row["person_id"]:
                        edges.append({
                            "from":  "p-" + row["person_id"],
                            "to":    aid,
                            "label": "has_account",
                        })

            # ── Groups ────────────────────────────────────────
            # columns: group_uid, platform, group_id, group_name, ...
            async with db.execute(
                "SELECT group_uid, platform, group_id, group_name FROM groups LIMIT 500"
            ) as cur:
                async for row in cur:
                    gid   = "g-" + row["group_uid"]
                    label = row["group_name"] or f"{row['platform']}:{row['group_id']}"
                    nodes.append({
                        "id":    gid,
                        "label": label,
                        "group": "group",
                        "title": f"群组: {label} ({row['platform']})",
                        "extra": {
                            "platform": row["platform"],
                            "group_id": row["group_id"],
                        },
                    })

            # ── Memberships ───────────────────────────────────
            # columns: membership_id, account_uid, group_uid, ...
            async with db.execute(
                "SELECT account_uid, group_uid FROM memberships LIMIT 2000"
            ) as cur:
                async for row in cur:
                    edges.append({
                        "from":  "a-" + row["account_uid"],
                        "to":    "g-" + row["group_uid"],
                        "label": "in_group",
                    })

            # ── Chat sessions ─────────────────────────────────
            # columns: session_key, conv_type, conv_id, conv_name, last_active_at
            async with db.execute(
                "SELECT session_key, conv_type, conv_id, conv_name "
                "FROM chat_sessions LIMIT 500"
            ) as cur:
                async for row in cur:
                    sid   = "s-" + row["session_key"]
                    label = row["conv_name"] or row["conv_id"] or row["session_key"]
                    nodes.append({
                        "id":    sid,
                        "label": label,
                        "group": "session",
                        "title": f"会话: {label} ({row['conv_type']})",
                        "extra": {
                            "type":    row["conv_type"],
                            "conv_id": row["conv_id"] or "",
                        },
                    })
                    if row["conv_type"] == "group" and row["conv_id"]:
                        async with db.execute(
                            "SELECT group_uid FROM groups WHERE group_id=? LIMIT 1",
                            (row["conv_id"],),
                        ) as cur2:
                            grow = await cur2.fetchone()
                            if grow:
                                edges.append({
                                    "from":  "g-" + grow["group_uid"],
                                    "to":    sid,
                                    "label": "has_session",
                                })

    except Exception as e:
        logger.warning("memory_graph query failed: %s", e)
        return jsonify({"nodes": [], "edges": [], "error": str(e)})

    return jsonify({"nodes": nodes, "edges": edges})


@memory_bp.route("/memory/triples")
async def memory_triples():
    """返回指定 subject 的 MemoryTriples 列表（未删除，按 created_at DESC）。

    Query params:
      subject — 如 "Self" 或 "User:qq_123456"（默认 "Self"）
      limit   — 最多条数，默认 100，上限 500
    """
    subject = request.args.get("subject", "Self")
    limit   = min(int(request.args.get("limit", 100)), 500)

    rows: list = []
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT id, predicate, object_text, confidence,
                          context, source, conv_name, created_at
                   FROM MemoryTriples
                   WHERE subject=? AND is_deleted=0
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (subject, limit),
            ) as cur:
                async for row in cur:
                    rows.append({
                        "id":          row["id"],
                        "predicate":   row["predicate"],
                        "object_text": row["object_text"],
                        "confidence":  round(row["confidence"], 2),
                        "context":     row["context"],
                        "source":      row["source"] or "",
                        "conv_name":   row["conv_name"] or "",
                        "created_at":  row["created_at"],
                    })
    except Exception as e:
        logger.warning("memory_triples query failed: %s", e)
        return jsonify({"subject": subject, "triples": [], "error": str(e)})

    return jsonify({"subject": subject, "triples": rows})
