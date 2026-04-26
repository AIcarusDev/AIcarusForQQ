"""routes_memory.py — 记忆图谱相关路由（事件层版）

Blueprint: memory_bp
  GET /memory       — 记忆图谱页面
  GET /memory/graph — vis.js 节点/边 JSON（实体 + MemoryEvents/MemoryRoles）
"""

import logging

import aiosqlite
from quart import Blueprint, jsonify, render_template

from database import DB_PATH

logger = logging.getLogger("AICQ.app")

memory_bp = Blueprint("memory", __name__)


@memory_bp.route("/memory")
async def memory_page():
    return await render_template("memory.html", active_page="memory")


@memory_bp.route("/memory/graph")
async def memory_graph():
    """查询数据库，返回 { nodes: [...], edges: [...] } 供前端 vis.js 使用。"""
    nodes: list = []
    edges: list = []

    acct_lookup: dict[str, str] = {}   # platform_id -> "a-{uid}"
    group_lookup: dict[str, str] = {}  # group_id    -> "g-{uid}"
    profile_ids: set[str] = set()

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row

            # ── Self 节点（Bot 自身命名空间）─────────────────
            nodes.append({
                "id":    "self",
                "label": "Self",
                "group": "self",
                "title": "Bot 自身（Bot:self 命名空间）",
                "extra": {"entity": "Bot:self"},
            })

            # ── EntityProfiles ───────────────────────────────
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

            # ── Entities ─────────────────────────────────────
            bot_account_uid: str | None = None
            async with db.execute(
                "SELECT account_uid, profile_id, platform, platform_id, nickname, is_bot "
                "FROM entities LIMIT 1000"
            ) as cur:
                async for row in cur:
                    aid = "a-" + row["account_uid"]
                    nick = row["nickname"] or f"{row['platform']}:{row['platform_id']}"
                    nodes.append({
                        "id":    aid,
                        "label": nick,
                        "group": "account",
                        "title": f"实体: {nick} ({row['platform']})",
                        "extra": {
                            "platform":    row["platform"],
                            "platform_id": row["platform_id"],
                            "nickname":    row["nickname"] or "",
                            "entity":      f"User:qq_{row['platform_id']}",
                        },
                    })
                    if row["profile_id"]:
                        edges.append({
                            "from":  "p-" + row["profile_id"],
                            "to":    aid,
                            "label": "represents",
                        })
                    if row["is_bot"]:
                        bot_account_uid = row["account_uid"]
                    if row["platform_id"]:
                        acct_lookup[str(row["platform_id"])] = aid

            if bot_account_uid:
                edges.append({
                    "from":  "self",
                    "to":    "a-" + bot_account_uid,
                    "label": "is_bot",
                })

            # ── Groups ───────────────────────────────────────
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
                    if row["group_id"]:
                        group_lookup[str(row["group_id"])] = gid

            # ── Memberships ──────────────────────────────────
            async with db.execute(
                "SELECT account_uid, group_uid FROM memberships LIMIT 2000"
            ) as cur:
                async for row in cur:
                    edges.append({
                        "from":  "a-" + row["account_uid"],
                        "to":    "g-" + row["group_uid"],
                        "label": "in_group",
                    })

            # ── Chat sessions ────────────────────────────────
            async with db.execute(
                "SELECT session_key, conv_type, conv_id, conv_name "
                "FROM chat_sessions LIMIT 500"
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

            # ── MemoryEvents（Neo-Davidsonian 事件层）─────────
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

                for r in ev_roles:
                    role_name = r["role"]
                    entity = r["entity"]
                    target_event = r["target_event"]

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
                    if entity == "Bot:self":
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
