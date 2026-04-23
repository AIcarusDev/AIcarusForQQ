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

    # Subject → node-id 映射（供 MemoryTriples 连边使用）
    acct_lookup: dict[str, str] = {}   # platform_id -> "a-{uid}"
    group_lookup: dict[str, str] = {}  # group_id    -> "g-{uid}"
    profile_ids: set[str] = set()      # entity_profiles.profile_id 集合

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row

            # 预查 MemoryTriples 各 subject 的三元组计数
            triple_counts = await _triple_counts_by_subject(db)

            # ── Self 节点（Bot 自身记忆）─────────────────────
            # 兼容历史：旧数据可能仍有 "Self"，新数据统一为 "Bot:self"
            self_count = (
                triple_counts.get("Bot:self", 0)
                + triple_counts.get("Self", 0)
            )
            # Self 节点恒显示：即使 0 条 triple，也用于承接事件层的 agent=Bot:self 边
            nodes.append({
                "id":    "self",
                "label": f"Self\n({self_count} 条)" if self_count else "Self",
                "group": "self",
                "title": f"Bot 自身记忆：{self_count} 条" if self_count
                          else "Bot 自身（暂无三元组记忆）",
                "extra": {"subject": "Bot:self", "triple_count": self_count},
            })

            # ── EntityProfiles ───────────────────────────────────────────────
            # AI 对意识个体的主观认知侧写（EntityProfile）
            # 标签优先级：notes > 关联 entity 的 nickname > 截断 UUID
            async with db.execute(
                """SELECT ep.profile_id, ep.notes, e.nickname
                   FROM entity_profiles ep
                   LEFT JOIN entities e ON e.profile_id = ep.profile_id
                   LIMIT 500"""
            ) as cur:
                async for row in cur:
                    pid = "p-" + row["profile_id"]
                    label = (row["notes"] or row["nickname"] or row["profile_id"])[:20]
                    nodes.append({
                        "id":    pid,
                        "label": label,
                        "group": "person",
                        "title": f"实体侧写: {row['profile_id']}",
                        "extra": {
                            "profile_id": row["profile_id"],
                            "notes":      row["notes"] or "",
                        },
                    })
                    profile_ids.add(row["profile_id"])

            # ── Entities ──────────────────────────────────────
            # 客观可观测的实体（Entity），通过 profile_id 关联到 entity_profiles
            # columns: account_uid, profile_id, platform, platform_id, nickname, is_bot, ...
            bot_account_uid: str | None = None   # 用于连接 Self 节点
            async with db.execute(
                "SELECT account_uid, profile_id, platform, platform_id, nickname, is_bot "
                "FROM entities LIMIT 1000"
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
                        "title": f"实体: {nick} ({row['platform']})"
                                 + (f" — {tc} 条记忆" if tc else ""),
                        "extra": {
                            "platform":     row["platform"],
                            "platform_id":  row["platform_id"],
                            "nickname":     row["nickname"] or "",
                            "subject":      subject,
                            "triple_count": tc,
                        },
                    })
                    if row["profile_id"]:
                        edges.append({
                            "from":  "p-" + row["profile_id"],
                            "to":    aid,
                            "label": "represents",  # EntityProfile → represents → Entity
                        })
                    # 记录 bot 自身的 account_uid，用于将 Self 节点连至 bot 实体
                    if row["is_bot"]:
                        bot_account_uid = row["account_uid"]
                    # 记录 platform_id → node-id 映射（供 MemoryTriples 连边）
                    if row["platform_id"]:
                        acct_lookup[str(row["platform_id"])] = aid

            # Self 节点连接到 bot 的 entities 节点（Bot 记忆命名空间 → 客观账号实体）
            if bot_account_uid:
                edges.append({
                    "from":  "self",
                    "to":    "a-" + bot_account_uid,
                    "label": "is_bot",
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
                    # 记录 group_id → node-id 映射
                    if row["group_id"]:
                        group_lookup[str(row["group_id"])] = gid

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
                    if subject in ("Self", "Bot:self"):
                        from_node_id = "self"
                    elif subject.startswith("User:qq_"):
                        plat_id = subject[len("User:qq_"):]
                        from_node_id = acct_lookup.get(plat_id)
                    elif subject.startswith("Person:"):
                        profile_id = subject[len("Person:"):]
                        if profile_id in profile_ids:
                            from_node_id = "p-" + profile_id
                    elif subject.startswith("Group:qq_"):
                        grp_id = subject[len("Group:qq_"):]
                        from_node_id = group_lookup.get(grp_id)

                    if from_node_id:
                        edges.append({
                            "from":  from_node_id,
                            "to":    mid,
                            "label": pred,
                        })

            # ── MemoryEvents（Neo-Davidsonian 事件层）─────────
            # 事件作为菱形/六边形节点；角色边把 entity → event 连起来
            # 仅展示未删除事件，按 occurred_at DESC 限量，避免图被淹没
            try:
                async with db.execute(
                    """SELECT event_id, event_type, summary, polarity, modality,
                              confidence, context_type, recall_scope, occurred_at,
                              source, conv_name
                       FROM MemoryEvents
                       WHERE is_deleted=0
                       ORDER BY
                           CASE context_type
                               WHEN 'meta'     THEN 0
                               WHEN 'contract' THEN 1
                               ELSE 2
                           END,
                           occurred_at DESC
                       LIMIT 200"""
                ) as cur:
                    event_rows = [dict(r) for r in await cur.fetchall()]
            except Exception:
                # 旧数据库尚无事件表
                event_rows = []

            event_ids = [e["event_id"] for e in event_rows]
            roles_by_event: dict[int, list[dict]] = {}
            if event_ids:
                ph = ",".join("?" * len(event_ids))
                async with db.execute(
                    f"SELECT event_id, role, entity, value_text, target_event "
                    f"FROM MemoryRoles WHERE event_id IN ({ph})",
                    event_ids,
                ) as cur:
                    async for r in cur:
                        roles_by_event.setdefault(r["event_id"], []).append(dict(r))

            for ev in event_rows:
                eid_str = f"ev-{ev['event_id']}"
                summary = ev["summary"] or "(无摘要)"
                summary_short = summary[:24] + "…" if len(summary) > 24 else summary
                etype = ev["event_type"] or "event"
                ctx = ev["context_type"] or "episodic"
                pol = ev["polarity"] or "positive"
                mod = ev["modality"] or "actual"
                conf = float(ev["confidence"] or 0.6)
                # 否定/假设的事件用前缀标记，肉眼立刻识别
                prefix = ""
                if pol == "negative":
                    prefix = "¬ "
                if mod in ("hypothetical", "possible"):
                    prefix += "? "
                label = f"{prefix}{etype}\n{summary_short}"
                ev_roles = roles_by_event.get(ev["event_id"], [])
                role_brief = " / ".join(
                    f"{r['role']}={r['entity'] or r['value_text']}"
                    for r in ev_roles
                )
                nodes.append({
                    "id":    eid_str,
                    "label": label,
                    "group": "event",
                    "title": f"[{ctx}] {summary}\n{role_brief}",
                    "extra": {
                        "事件ID":    ev["event_id"],
                        "类型":      etype,
                        "摘要":      summary,
                        "context":   ctx,
                        "polarity":  pol,
                        "modality":  mod,
                        "置信度":    round(conf, 2),
                        "scope":     ev["recall_scope"] or "global",
                        "来源":      ev["source"] or "",
                        "会话":      ev["conv_name"] or "",
                        "roles":     ev_roles,
                    },
                })

                # 角色边：entity → event
                for r in ev_roles:
                    role_name = r["role"]
                    entity = r["entity"]
                    target_event = r["target_event"]

                    # 嵌套事件（如反驳 e7）：event → event
                    if target_event:
                        edges.append({
                            "from":  eid_str,
                            "to":    f"ev-{target_event}",
                            "label": role_name,
                        })
                        continue
                    if not entity:
                        continue

                    target_node_id = None
                    if entity in ("Bot:self", "Self"):
                        target_node_id = "self"
                    elif entity.startswith("User:qq_"):
                        target_node_id = acct_lookup.get(entity[len("User:qq_"):])
                    elif entity.startswith("Person:"):
                        profile_id = entity[len("Person:"):]
                        if profile_id in profile_ids:
                            target_node_id = "p-" + profile_id
                    elif entity.startswith("Group:qq_"):
                        target_node_id = group_lookup.get(entity[len("Group:qq_"):])

                    if target_node_id:
                        edges.append({
                            "from":  target_node_id,
                            "to":    eid_str,
                            "label": role_name,
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
