"""routes_memory.py — 记忆图谱相关路由

Blueprint: memory_bp
  GET /memory        — 记忆图谱页面
  GET /memory/graph  — 返回 vis.js 可用的节点/边 JSON
"""

import logging

import aiosqlite
from quart import Blueprint, render_template, jsonify

from database import DB_PATH

logger = logging.getLogger("AICQ.app")

memory_bp = Blueprint("memory", __name__)


@memory_bp.route("/memory")
async def memory_page():
    return await render_template("memory.html", active_page="memory")


@memory_bp.route("/memory/graph")
async def memory_graph():
    """查询数据库，返回 { nodes: [...], edges: [...] } 供前端 vis.js 使用。"""
    nodes = []
    edges = []

    # Subject → node-id 映射（供 MemoryTriples 连边使用）
    acct_lookup: dict[str, str] = {}   # platform_id -> "a-{uid}"
    group_lookup: dict[str, str] = {}  # group_id    -> "g-{uid}"
    person_ids: set[str] = set()       # person_id 集合

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row

            # ── Persons ──────────────────────────────────────
            # columns: person_id, sex, age, area, notes, last_seen_at, created_at, updated_at, extra
            async with db.execute(
                "SELECT person_id, notes FROM persons LIMIT 500"
            ) as cur:
                async for row in cur:
                    pid = "p-" + row["person_id"]
                    label = row["notes"] or row["person_id"]
                    nodes.append({
                        "id":    pid,
                        "label": label[:20] if label else row["person_id"],
                        "group": "person",
                        "title": f"自然人: {row['person_id']}",
                        "extra": {"person_id": row["person_id"], "notes": row["notes"] or ""},
                    })
                    person_ids.add(row["person_id"])

            # ── Accounts ─────────────────────────────────────
            # columns: account_uid, person_id, platform, platform_id, nickname, avatar, is_bot, ...
            async with db.execute(
                "SELECT account_uid, person_id, platform, platform_id, nickname FROM accounts LIMIT 1000"
            ) as cur:
                async for row in cur:
                    aid = "a-" + row["account_uid"]
                    label = row["nickname"] or f"{row['platform']}:{row['platform_id']}"
                    nodes.append({
                        "id":    aid,
                        "label": label,
                        "group": "account",
                        "title": f"账号: {label} ({row['platform']})",
                        "extra": {
                            "platform":    row["platform"],
                            "platform_id": row["platform_id"],
                            "nickname":    row["nickname"] or "",
                        },
                    })
                    # Link person → account
                    if row["person_id"]:
                        edges.append({
                            "from":  "p-" + row["person_id"],
                            "to":    aid,
                            "label": "has_account",
                        })
                    # 记录 platform_id → node-id 映射（供 MemoryTriples 连边）
                    if row["platform_id"]:
                        acct_lookup[str(row["platform_id"])] = aid

            # ── Groups ───────────────────────────────────────
            # columns: group_uid, platform, group_id, group_name, bot_card, member_count, updated_at
            async with db.execute(
                "SELECT group_uid, platform, group_id, group_name FROM groups LIMIT 500"
            ) as cur:
                async for row in cur:
                    gid = "g-" + row["group_uid"]
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
                    # 记录 group_id → node-id 映射
                    if row["group_id"]:
                        group_lookup[str(row["group_id"])] = gid

            # ── Memberships (account ↔ group) ─────────────────
            # columns: membership_id, account_uid, group_uid, cardname, ...
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
                "SELECT session_key, conv_type, conv_id, conv_name FROM chat_sessions LIMIT 500"
            ) as cur:
                async for row in cur:
                    sid = "s-" + row["session_key"]
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
                    # Link group session → group node via conv_id matching group_id (platform group id)
                    if row["conv_type"] == "group" and row["conv_id"]:
                        async with db.execute(
                            "SELECT group_uid FROM groups WHERE group_id = ? LIMIT 1",
                            (row["conv_id"],),
                        ) as cur2:
                            grow = await cur2.fetchone()
                            if grow:
                                edges.append({
                                    "from":  "g-" + grow["group_uid"],
                                    "to":    sid,
                                    "label": "has_session",
                                })

            # ── MemoryTriples ─────────────────────────────────
            # 展示记忆三元组节点，并将其连接到对应的实体节点
            async with db.execute(
                """SELECT id, subject, predicate, object_text, context, confidence, source
                   FROM MemoryTriples
                   WHERE is_deleted=0
                   ORDER BY confidence DESC, last_accessed DESC
                   LIMIT 300"""
            ) as cur:
                async for row in cur:
                    mid = f"mt-{row['id']}"
                    pred = row["predicate"] or "[note]"
                    obj = row["object_text"] or ""
                    # label: "谓语: 宾语(截断)"
                    obj_short = obj[:22] + "…" if len(obj) > 22 else obj
                    label = f"{pred}: {obj_short}" if pred != "[note]" else obj_short
                    conf = row["confidence"] if row["confidence"] is not None else 0.6
                    nodes.append({
                        "id":    mid,
                        "label": label,
                        "group": "memory",
                        "title": f"{row['subject']} —[{pred}]→ {obj}",
                        "extra": {
                            "主语":  row["subject"],
                            "谓语":  pred,
                            "宾语":  obj,
                            "语境":  row["context"] or "truth",
                            "置信度": round(conf, 2),
                            "来源":  row["source"] or "",
                        },
                    })

                    # 解析 subject 并连接到对应实体节点
                    subject = row["subject"] or ""
                    from_node_id = None
                    if subject == "Self":
                        pass  # Self 节点暂无对应图节点，孤立显示
                    elif subject.startswith("User:qq_"):
                        plat_id = subject[len("User:qq_"):]
                        from_node_id = acct_lookup.get(plat_id)
                    elif subject.startswith("Person:"):
                        person_id = subject[len("Person:"):]
                        if person_id in person_ids:
                            from_node_id = "p-" + person_id
                    elif subject.startswith("Group:qq_"):
                        grp_id = subject[len("Group:qq_"):]
                        from_node_id = group_lookup.get(grp_id)

                    if from_node_id:
                        edges.append({
                            "from":  from_node_id,
                            "to":    mid,
                            "label": pred,
                        })

    except Exception as e:
        logger.warning("memory_graph query failed: %s", e)
        return jsonify({"nodes": [], "edges": [], "error": str(e)})

    return jsonify({"nodes": nodes, "edges": edges})
