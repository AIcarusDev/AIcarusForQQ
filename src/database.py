"""database.py — SQLite 持久化层

表结构：
  persons     — 自然人表，bot 认知层面的人物画像（跨平台、跨账号）
  accounts    — 平台账号表，关联到 persons
  groups      — 群组表（支持多平台）
  memberships — 群成员关系表（账号×群，保存群名片/头衔/权限）
  chat_sessions  — 会话注册表（记住历史会话的 key → meta）
  chat_messages  — 聊天记录（按 session_key 隔离，可按需恢复上下文）
  bot_turns      — bot 意识流日志（全局唯一，每轮 LLM 输出 + 工具调用记录）

旧表 profiles / group_cards 保留用于数据迁移，迁移后不再写入。
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import aiosqlite

# 数据库路径 (data/AICQ.db)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
os.makedirs(_DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(_DATA_DIR, "AICQ.db")

logger = logging.getLogger("AICQ.db")


def _ms() -> int:
    """返回当前 UTC 时间戳（毫秒）。"""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


@asynccontextmanager
async def _connect():
    """打开数据库连接并启用外键约束。

    PRAGMA foreign_keys=ON 是 SQLite 的连接级设置，不会持久化到文件。
    每条连接都必须单独设置，否则 REFERENCES 约束实际不生效。
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys=ON")
        yield db


# ── 初始化 ────────────────────────────────────────────────

async def init_db() -> None:
    """创建数据库表（如不存在），并执行旧数据迁移。"""
    async with _connect() as db:
        await db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            -- 会话注册表：记住历史会话的 key → meta，重启后可按 key 恢复
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_key   TEXT    PRIMARY KEY,
                conv_type     TEXT    NOT NULL DEFAULT '',
                conv_id       TEXT    NOT NULL DEFAULT '',
                conv_name     TEXT    NOT NULL DEFAULT '',
                last_active_at INTEGER NOT NULL DEFAULT 0
            );

            -- 聊天记录表：每条消息一行，按 session_key 隔离
            CREATE TABLE IF NOT EXISTS chat_messages (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key      TEXT    NOT NULL,
                role             TEXT    NOT NULL,
                message_id       TEXT    NOT NULL DEFAULT '',
                sender_id        TEXT    NOT NULL DEFAULT '',
                sender_name      TEXT    NOT NULL DEFAULT '',
                sender_role      TEXT    NOT NULL DEFAULT '',
                timestamp        TEXT    NOT NULL DEFAULT '',
                content          TEXT    NOT NULL DEFAULT '',
                content_type     TEXT    NOT NULL DEFAULT 'text',
                content_segments TEXT    NOT NULL DEFAULT '[]',
                images           TEXT    NOT NULL DEFAULT '[]',
                created_at       INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                ON chat_messages(session_key, id);

            -- bot 意识流日志：全局唯一，保存每轮 LLM 输出及工具调用，供重启后恢复
            CREATE TABLE IF NOT EXISTS bot_turns (
                turn_id      TEXT    PRIMARY KEY,
                created_at   INTEGER NOT NULL DEFAULT 0,
                conv_type    TEXT    NOT NULL DEFAULT '',
                conv_id      TEXT    NOT NULL DEFAULT '',
                result_json  TEXT    NOT NULL DEFAULT '{}',
                tool_calls   TEXT    NOT NULL DEFAULT '[]'
            );

            -- 意识流活动日志：记录 chat/watcher/hibernate 切换历史，供 LLM prompt 注入
            CREATE TABLE IF NOT EXISTS activity_log (
                entry_id           TEXT    PRIMARY KEY,
                created_at         INTEGER NOT NULL DEFAULT 0,
                ended_at           INTEGER,
                entry_type         TEXT    NOT NULL DEFAULT '',
                conv_type          TEXT    NOT NULL DEFAULT '',
                conv_id            TEXT    NOT NULL DEFAULT '',
                conv_name          TEXT    NOT NULL DEFAULT '',
                enter_attitude     TEXT    NOT NULL DEFAULT '',
                enter_motivation   TEXT    NOT NULL DEFAULT '',
                enter_remark       TEXT    NOT NULL DEFAULT '',
                enter_from         TEXT    NOT NULL DEFAULT '',
                hibernate_minutes  INTEGER NOT NULL DEFAULT 0,
                end_attitude       TEXT    NOT NULL DEFAULT '',
                end_action         TEXT    NOT NULL DEFAULT '',
                end_motivation     TEXT    NOT NULL DEFAULT '',
                end_remark         TEXT    NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_activity_log_created
                ON activity_log(created_at);

            -- watcher 窥屏意识循环日志：每轮窥屏的内心状态与决策
            CREATE TABLE IF NOT EXISTS watcher_cycles (
                cycle_id     TEXT    PRIMARY KEY,
                created_at   INTEGER NOT NULL DEFAULT 0,
                conv_type    TEXT    NOT NULL DEFAULT '',
                conv_id      TEXT    NOT NULL DEFAULT '',
                result_json  TEXT    NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_watcher_cycles_conv
                ON watcher_cycles(conv_type, conv_id, created_at);

            -- 自然人表：bot 认知层面的人物画像
            CREATE TABLE IF NOT EXISTS persons (
                person_id    TEXT    PRIMARY KEY,
                sex          TEXT,
                age          INTEGER,
                area         TEXT,
                notes        TEXT,
                last_seen_at INTEGER,
                created_at   INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                updated_at   INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                extra        TEXT
            );

            -- 平台账号表
            CREATE TABLE IF NOT EXISTS accounts (
                account_uid  TEXT    PRIMARY KEY,
                person_id    TEXT    NOT NULL REFERENCES persons(person_id),
                platform     TEXT    NOT NULL,
                platform_id  TEXT    NOT NULL,
                nickname     TEXT,
                avatar       TEXT,
                is_bot       INTEGER NOT NULL DEFAULT 0,
                last_seen_at INTEGER,
                created_at   INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                updated_at   INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                extra        TEXT,
                UNIQUE(platform, platform_id)
            );

            -- 群组表
            CREATE TABLE IF NOT EXISTS groups (
                group_uid    TEXT    PRIMARY KEY,
                platform     TEXT    NOT NULL,
                group_id     TEXT    NOT NULL,
                group_name   TEXT,
                bot_card     TEXT,
                member_count INTEGER NOT NULL DEFAULT 0,
                updated_at   INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                UNIQUE(platform, group_id)
            );

            -- 群成员关系表
            CREATE TABLE IF NOT EXISTS memberships (
                membership_id    TEXT    PRIMARY KEY,
                account_uid      TEXT    NOT NULL REFERENCES accounts(account_uid),
                group_uid        TEXT    NOT NULL REFERENCES groups(group_uid),
                cardname         TEXT,
                title            TEXT,
                permission_level TEXT    NOT NULL DEFAULT 'member',
                joined_at        INTEGER,
                updated_at       INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                UNIQUE(account_uid, group_uid)
            );

            -- 模型长期记忆表（旧版，保留供迁移脚本读取，新代码不再写入）
            CREATE TABLE IF NOT EXISTS bot_memories (
                memory_id    TEXT    PRIMARY KEY,
                created_at   INTEGER NOT NULL DEFAULT 0,
                content      TEXT    NOT NULL DEFAULT '',
                source       TEXT    NOT NULL DEFAULT '',
                reason       TEXT    NOT NULL DEFAULT '',
                conv_type    TEXT    NOT NULL DEFAULT '',
                conv_id      TEXT    NOT NULL DEFAULT '',
                conv_name    TEXT    NOT NULL DEFAULT '',
                is_deleted   INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_bot_memories_created
                ON bot_memories(created_at) WHERE is_deleted=0;

            -- ── 记忆聚类表（Phase 2：将语义相近的三元组聚合为一组）──────────────────
            CREATE TABLE IF NOT EXISTS MemoryClusters (
                cluster_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                label         TEXT    NOT NULL DEFAULT '',
                confidence    REAL    NOT NULL DEFAULT 0.6,
                created_at    INTEGER NOT NULL DEFAULT 0,
                last_accessed INTEGER NOT NULL DEFAULT 0,
                member_count  INTEGER NOT NULL DEFAULT 0
            );

            -- ── 结构化记忆三元组表（Phase 1+2：subject/predicate/object_text）──────────
            -- object_text     原始文本，供 LLM 阅读
            -- object_text_tok jieba 分词后的空格分隔 token 串，供 FTS5 索引
            -- recall_scope    召回场景隔离：global | group:qq_{id} | private:qq_{id}
            -- cluster_id      所属聚类（NULL 表示孤立条目）
            CREATE TABLE IF NOT EXISTS MemoryTriples (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                subject         TEXT    NOT NULL DEFAULT 'Self',
                predicate       TEXT    NOT NULL DEFAULT '[note]',
                object_text     TEXT    NOT NULL DEFAULT '',
                object_text_tok TEXT    NOT NULL DEFAULT '',
                context         TEXT    NOT NULL DEFAULT 'truth',
                confidence      REAL    NOT NULL DEFAULT 0.6,
                created_at      INTEGER NOT NULL DEFAULT 0,
                last_accessed   INTEGER NOT NULL DEFAULT 0,
                source          TEXT    NOT NULL DEFAULT '',
                reason          TEXT    NOT NULL DEFAULT '',
                conv_type       TEXT    NOT NULL DEFAULT '',
                conv_id         TEXT    NOT NULL DEFAULT '',
                conv_name       TEXT    NOT NULL DEFAULT '',
                origin          TEXT    NOT NULL DEFAULT 'passive',
                recall_scope    TEXT    NOT NULL DEFAULT 'global',
                cluster_id      INTEGER REFERENCES MemoryClusters(cluster_id),
                is_deleted      INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_mt_subject
                ON MemoryTriples(subject) WHERE is_deleted=0;
            CREATE INDEX IF NOT EXISTS idx_mt_context
                ON MemoryTriples(context, confidence) WHERE is_deleted=0;
            CREATE INDEX IF NOT EXISTS idx_mt_created
                ON MemoryTriples(created_at) WHERE is_deleted=0;
            CREATE INDEX IF NOT EXISTS idx_mt_recall_scope
                ON MemoryTriples(recall_scope) WHERE is_deleted=0;
            CREATE INDEX IF NOT EXISTS idx_mt_cluster
                ON MemoryTriples(cluster_id) WHERE is_deleted=0 AND cluster_id IS NOT NULL;

            -- ── 实体泼溅合并建议表（Phase 3B）──────────────────────────────
            -- 绝不自动合并；建议仅供模型/人工二次确认后推进
            CREATE TABLE IF NOT EXISTS merge_suggestions (
                suggestion_id TEXT    PRIMARY KEY,
                person_id_a   TEXT    NOT NULL REFERENCES persons(person_id),
                person_id_b   TEXT    NOT NULL REFERENCES persons(person_id),
                similarity    REAL    NOT NULL DEFAULT 0.0,
                reason        TEXT    NOT NULL DEFAULT '',
                status        TEXT    NOT NULL DEFAULT 'pending',
                created_at    INTEGER NOT NULL DEFAULT 0,
                resolved_at   INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_ms_status
                ON merge_suggestions(status, created_at);
        """)
        await db.commit()

        # FTS5 虚拟表和触发器须单独执行（部分 SQLite 版本在 executescript 中处理虚拟表有兼容问题）
        await db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS MemorySearch USING fts5(
                subject,
                predicate,
                object_text_tok,
                tokenize="unicode61"
            )
        """)
        await db.commit()

        # FTS5 同步触发器：只监听文本字段，不监听 confidence/last_accessed
        # 以避免高频置信度更新触发倒排索引重建（I/O 爆炸）
        for _trigger_sql in [
            # INSERT 新行时同步到 FTS5
            """CREATE TRIGGER IF NOT EXISTS fts_mt_insert
               AFTER INSERT ON MemoryTriples
               WHEN NEW.is_deleted = 0
               BEGIN
                   INSERT INTO MemorySearch(rowid, subject, predicate, object_text_tok)
                   VALUES (NEW.id, NEW.subject, NEW.predicate, NEW.object_text_tok);
               END""",
            # 软删除时从 FTS5 移除
            """CREATE TRIGGER IF NOT EXISTS fts_mt_soft_delete
               AFTER UPDATE OF is_deleted ON MemoryTriples
               WHEN NEW.is_deleted = 1 AND OLD.is_deleted = 0
               BEGIN
                   DELETE FROM MemorySearch WHERE rowid = OLD.id;
               END""",
            # 文本内容变更时重建 FTS5 索引（只监听三个文本字段）
            """CREATE TRIGGER IF NOT EXISTS fts_mt_update_text
               AFTER UPDATE OF subject, predicate, object_text_tok ON MemoryTriples
               WHEN NEW.is_deleted = 0
               BEGIN
                   DELETE FROM MemorySearch WHERE rowid = OLD.id;
                   INSERT INTO MemorySearch(rowid, subject, predicate, object_text_tok)
                   VALUES (NEW.id, NEW.subject, NEW.predicate, NEW.object_text_tok);
               END""",
        ]:
            await db.execute(_trigger_sql)
        await db.commit()

        await _migrate_schema(db)
        await _migrate_legacy(db)
    logger.info("数据库初始化完成: %s", DB_PATH)


async def _migrate_schema(db) -> None:
    """为已有表补充新增列（ALTER TABLE），保证旧数据库可以正常使用。"""
    # activity_log 新增 hibernate_minutes 列
    try:
        await db.execute(
            "ALTER TABLE activity_log ADD COLUMN hibernate_minutes INTEGER NOT NULL DEFAULT 0"
        )
        await db.commit()
        logger.info("[schema] activity_log 已添加 hibernate_minutes 列")
    except Exception:
        pass  # 列已存在则跳过

    # MemoryTriples 新增 origin 列（区分主动/被动记忆）
    try:
        await db.execute(
            "ALTER TABLE MemoryTriples ADD COLUMN origin TEXT NOT NULL DEFAULT 'passive'"
        )
        await db.commit()
        logger.info("[schema] MemoryTriples 已添加 origin 列")
    except Exception:
        pass  # 列已存在则跳过

    # MemoryTriples 新增 recall_scope 列（Phase 2：召回场景隔离）
    try:
        await db.execute(
            "ALTER TABLE MemoryTriples ADD COLUMN recall_scope TEXT NOT NULL DEFAULT 'global'"
        )
        await db.commit()
        logger.info("[schema] MemoryTriples 已添加 recall_scope 列")
    except Exception:
        pass

    # MemoryTriples 新增 cluster_id 列（Phase 2：记忆聚类）
    try:
        await db.execute(
            "ALTER TABLE MemoryTriples ADD COLUMN cluster_id INTEGER"
        )
        await db.commit()
        logger.info("[schema] MemoryTriples 已添加 cluster_id 列")
    except Exception:
        pass


async def _migrate_legacy(db) -> None:
    """将旧 profiles / group_cards 表数据迁移到新表，旧表保留不删除。"""
    now = _ms()

    # 迁移 profiles（bot 自身信息）
    try:
        async with db.execute("SELECT qq_id, nickname FROM profiles WHERE id=0") as cur:
            row = await cur.fetchone()
        if row:
            qq_id, nickname = str(row[0]), str(row[1])
            async with db.execute(
                "SELECT account_uid FROM accounts WHERE platform='qq' AND platform_id=? AND is_bot=1",
                (qq_id,),
            ) as cur2:
                existing = await cur2.fetchone()
            if not existing:
                person_id = str(uuid.uuid4())
                account_uid = str(uuid.uuid4())
                await db.execute(
                    "INSERT OR IGNORE INTO persons (person_id, created_at, updated_at) VALUES (?,?,?)",
                    (person_id, now, now),
                )
                await db.execute(
                    """INSERT OR IGNORE INTO accounts
                       (account_uid, person_id, platform, platform_id, nickname, is_bot, created_at, updated_at)
                       VALUES (?,?,?,?,?,1,?,?)""",
                    (account_uid, person_id, "qq", qq_id, nickname, now, now),
                )
                await db.commit()
                logger.info("旧 profiles 数据迁移完成: qq_id=%s", qq_id)
    except Exception:
        pass  # 旧表不存在则跳过

    # 迁移 group_cards
    try:
        async with db.execute(
            "SELECT group_id, group_name, bot_card, member_count FROM group_cards"
        ) as cur:
            rows = await cur.fetchall()
        migrated = 0
        for row in rows:
            group_id = str(row[0])
            group_uid = f"grp_qq_{group_id}"
            async with db.execute(
                "SELECT group_uid FROM groups WHERE platform='qq' AND group_id=?",
                (group_id,),
            ) as cur2:
                existing = await cur2.fetchone()
            if not existing:
                await db.execute(
                    """INSERT OR IGNORE INTO groups
                       (group_uid, platform, group_id, group_name, bot_card, member_count, updated_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (group_uid, "qq", group_id, str(row[1]), str(row[2]), int(row[3]), now),
                )
                migrated += 1
        if migrated:
            await db.commit()
            logger.info("旧 group_cards 数据迁移完成: %d 条", migrated)
    except Exception:
        pass  # 旧表不存在则跳过


# ── 会话持久化 ───────────────────────────────────────────

async def upsert_chat_session(
    session_key: str,
    conv_type: str,
    conv_id: str,
    conv_name: str = "",
) -> None:
    """写入/更新会话元信息，同时更新 last_active_at。"""
    now = _ms()
    async with _connect() as db:
        await db.execute(
            """INSERT INTO chat_sessions (session_key, conv_type, conv_id, conv_name, last_active_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(session_key) DO UPDATE SET
                   conv_name=excluded.conv_name,
                   last_active_at=excluded.last_active_at""",
            (session_key, conv_type, conv_id, conv_name, now),
        )
        await db.commit()


async def load_chat_sessions() -> list[dict]:
    """返回所有已注册的会话元信息，按 last_active_at 倒序。"""
    async with _connect() as db:
        async with db.execute(
            "SELECT session_key, conv_type, conv_id, conv_name FROM chat_sessions"
            " ORDER BY last_active_at DESC"
        ) as cur:
            rows = await cur.fetchall()
    return [
        {"session_key": r[0], "conv_type": r[1], "conv_id": r[2], "conv_name": r[3]}
        for r in rows
    ]


async def save_chat_message(session_key: str, entry: dict) -> None:
    """将一条上下文条目写入 chat_messages 表。"""
    import json as _json
    now = _ms()
    async with _connect() as db:
        await db.execute(
            """INSERT INTO chat_messages
               (session_key, role, message_id, sender_id, sender_name, sender_role,
                timestamp, content, content_type, content_segments, images, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                session_key,
                entry.get("role", ""),
                entry.get("message_id", ""),
                entry.get("sender_id", ""),
                entry.get("sender_name", ""),
                entry.get("sender_role", ""),
                entry.get("timestamp", ""),
                entry.get("content", ""),
                entry.get("content_type", "text"),
                _json.dumps(entry.get("content_segments", []), ensure_ascii=False),
                _json.dumps(entry.get("images", []), ensure_ascii=False),
                now,
            ),
        )
        await db.commit()


async def update_chat_message_id(session_key: str, old_message_id: str, new_message_id: str) -> None:
    """回填真实 QQ message_id（发送后 NapCat 返回真实 ID 时调用）。"""
    async with _connect() as db:
        await db.execute(
            "UPDATE chat_messages SET message_id=? WHERE session_key=? AND message_id=?",
            (new_message_id, session_key, old_message_id),
        )
        await db.commit()


async def update_chat_message_recalled(message_id: str, operator_name: str, timestamp: str) -> bool:
    """将数据库中的消息更新为撤回状态，与内存中 mark_message_recalled 保持同步。

    返回 True 表示找到并更新了至少一条记录。
    """
    import json as _json
    async with _connect() as db:
        cursor = await db.execute(
            """UPDATE chat_messages
               SET role='note',
                   content=?,
                   content_type='recall',
                   content_segments=?,
                   sender_id='',
                   sender_name='',
                   sender_role=''
               WHERE message_id=?""",
            (f"{operator_name}撤回了一条消息", _json.dumps([], ensure_ascii=False), message_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def get_chat_message_by_id(message_id: str) -> dict | None:
    """按 message_id 在全局范围内查找一条聊天记录（跨所有 session_key）。

    用于引用消息预取：当被引用消息不在当前上下文窗口时，从 DB 恢复基本信息。
    只返回文本相关字段，不含图片 base64。
    """
    import json as _json
    async with _connect() as db:
        async with db.execute(
            """SELECT role, message_id, sender_id, sender_name, sender_role,
                      timestamp, content, content_type, content_segments
               FROM chat_messages
               WHERE message_id=?
               LIMIT 1""",
            (message_id,),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return None
    return {
        "role": row[0],
        "message_id": row[1],
        "sender_id": row[2],
        "sender_name": row[3],
        "sender_role": row[4],
        "timestamp": row[5],
        "content": row[6],
        "content_type": row[7],
        "content_segments": _json.loads(row[8] or "[]"),
    }


async def load_chat_messages(session_key: str, limit: int = 50) -> list[dict]:
    """加载指定会话最近 limit 条聊天记录，按时间正序返回。"""
    import json as _json
    async with _connect() as db:
        async with db.execute(
            """SELECT role, message_id, sender_id, sender_name, sender_role,
                      timestamp, content, content_type, content_segments, images
               FROM (
                   SELECT * FROM chat_messages
                   WHERE session_key=?
                   ORDER BY id DESC
                   LIMIT ?
               ) sub
               ORDER BY id ASC""",
            (session_key, limit),
        ) as cur:
            rows = await cur.fetchall()
    result = []
    for r in rows:
        entry: dict = {
            "role": r[0],
            "message_id": r[1],
            "sender_id": r[2],
            "sender_name": r[3],
            "sender_role": r[4],
            "timestamp": r[5],
            "content": r[6],
            "content_type": r[7],
            "content_segments": _json.loads(r[8] or "[]"),
        }
        images = _json.loads(r[9] or "[]")
        if images:
            entry["images"] = images
        result.append(entry)
    return result


# ── watcher 窥屏意识流 ───────────────────────────────────

async def save_watcher_cycle(
    cycle_id: str,
    conv_type: str,
    conv_id: str,
    result: dict,
) -> None:
    """持久化一轮 watcher 窥屏结果。"""
    import json as _json
    now = _ms()
    async with _connect() as db:
        await db.execute(
            """INSERT INTO watcher_cycles (cycle_id, created_at, conv_type, conv_id, result_json)
               VALUES (?,?,?,?,?)""",
            (cycle_id, now, conv_type, conv_id, _json.dumps(result, ensure_ascii=False)),
        )
        await db.commit()
    logger.debug("已保存 watcher_cycle: cycle_id=%s conv=%s/%s", cycle_id, conv_type, conv_id)


async def load_last_watcher_cycle(
    conv_type: str,
    conv_id: str,
) -> tuple[dict | None, str | None]:
    """加载指定会话最近一轮 watcher 结果，返回 (result, created_at_iso)。"""
    import json as _json
    async with _connect() as db:
        async with db.execute(
            """SELECT result_json, created_at FROM watcher_cycles
               WHERE conv_type=? AND conv_id=?
               ORDER BY created_at DESC LIMIT 1""",
            (conv_type, conv_id),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return None, None
    try:
        result = _json.loads(row[0])
    except Exception:
        result = None
    created_at_iso = (
        datetime.fromtimestamp(row[1] / 1000, tz=timezone.utc).isoformat()
        if row[1]
        else None
    )
    return result, created_at_iso


# ── bot 意识流 ────────────────────────────────────────────

async def save_bot_turn(
    turn_id: str,
    conv_type: str,
    conv_id: str,
    result: dict,
    tool_calls_log: list,
) -> None:
    """持久化一轮 LLM 输出及工具调用日志。"""
    import json as _json
    now = _ms()
    async with _connect() as db:
        await db.execute(
            """INSERT INTO bot_turns (turn_id, created_at, conv_type, conv_id, result_json, tool_calls)
               VALUES (?,?,?,?,?,?)""",
            (
                turn_id,
                now,
                conv_type,
                conv_id,
                _json.dumps(result, ensure_ascii=False),
                _json.dumps(tool_calls_log, ensure_ascii=False),
            ),
        )
        await db.commit()
    logger.debug("已保存 bot_turn: turn_id=%s conv=%s/%s", turn_id, conv_type, conv_id)


async def get_last_tool_call_motivation(function_name: str) -> tuple[str, int] | None:
    """从 bot_turns 日志中找出最近一次指定工具调用的 motivation 参数及时间戳。

    利用 SQLite json_each() 展开 tool_calls 数组，按 bot_turn 创建时间倒序
    返回 (motivation, created_at_ms)，找不到则返回 None。
    """
    async with _connect() as db:
        async with db.execute(
            """
            SELECT json_extract(tc.value, '$.arguments.motivation'), bt.created_at
            FROM bot_turns bt, json_each(bt.tool_calls) tc
            WHERE json_extract(tc.value, '$.function') = ?
            ORDER BY bt.created_at DESC
            LIMIT 1
            """,
            (function_name,),
        ) as cur:
            row = await cur.fetchone()
    if row and row[0]:
        return str(row[0]), int(row[1])
    return None


# ── 活动日志 ─────────────────────────────────────────────

async def save_activity_entry(entry) -> None:
    """写入一条活动日志记录（INSERT OR REPLACE）。"""
    async with _connect() as db:
        await db.execute(
            """INSERT OR REPLACE INTO activity_log
               (entry_id, entry_type, created_at, ended_at,
                conv_type, conv_id, conv_name,
                enter_attitude, enter_motivation, enter_remark, enter_from,
                hibernate_minutes,
                end_attitude, end_action, end_motivation, end_remark)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                entry.entry_id,
                entry.entry_type,
                int(entry.created_at * 1000),
                int(entry.ended_at * 1000) if entry.ended_at else None,
                entry.conv_type,
                entry.conv_id,
                entry.conv_name,
                entry.enter_attitude,
                entry.enter_motivation,
                entry.enter_remark,
                entry.enter_from,
                entry.hibernate_minutes,
                entry.end_attitude,
                entry.end_action,
                entry.end_motivation,
                entry.end_remark,
            ),
        )
        await db.commit()


async def update_activity_entry(entry) -> None:
    """更新已有活动日志记录的 end 相关字段。"""
    async with _connect() as db:
        await db.execute(
            """UPDATE activity_log SET
               ended_at=?, end_attitude=?, end_action=?,
               end_motivation=?, end_remark=?
               WHERE entry_id=?""",
            (
                int(entry.ended_at * 1000) if entry.ended_at else None,
                entry.end_attitude,
                entry.end_action,
                entry.end_motivation,
                entry.end_remark,
                entry.entry_id,
            ),
        )
        await db.commit()


async def load_activity_log(limit: int = 10) -> list[dict]:
    """加载最近 limit 条活动日志，按时间正序（最旧在前）。"""
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT entry_id, entry_type, created_at, ended_at,
                      conv_type, conv_id, conv_name,
                      enter_attitude, enter_motivation, enter_remark, enter_from,
                      hibernate_minutes,
                      end_attitude, end_action, end_motivation, end_remark
               FROM (
                   SELECT * FROM activity_log ORDER BY created_at DESC LIMIT ?
               ) sub ORDER BY created_at ASC""",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    result = []
    for r in rows:
        result.append({
            "entry_id": r["entry_id"],
            "entry_type": r["entry_type"],
            "created_at": r["created_at"] / 1000.0,
            "ended_at": r["ended_at"] / 1000.0 if r["ended_at"] is not None else None,
            "conv_type": r["conv_type"] or "",
            "conv_id": r["conv_id"] or "",
            "conv_name": r["conv_name"] or "",
            "enter_attitude": r["enter_attitude"] or "",
            "enter_motivation": r["enter_motivation"] or "",
            "enter_remark": r["enter_remark"] or "",
            "enter_from": r["enter_from"] or "",
            "hibernate_minutes": int(r["hibernate_minutes"]) if r["hibernate_minutes"] else 0,
            "end_attitude": r["end_attitude"] or "",
            "end_action": r["end_action"] or "",
            "end_motivation": r["end_motivation"] or "",
            "end_remark": r["end_remark"] or "",
        })
    return result


async def load_last_bot_turn() -> tuple[dict | None, list | None, str | None]:
    """加载最新一轮 bot 输出（用于重启后恢复 previous_cycle_json 和 tool_calls）。

    返回 (result, tool_calls, created_at_iso)，created_at_iso 为 UTC ISO 格式时间戳。
    """
    import json as _json
    async with _connect() as db:
        async with db.execute(
            "SELECT result_json, tool_calls, created_at FROM bot_turns ORDER BY created_at DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return None, None, None
    try:
        result = _json.loads(row[0])
    except Exception:
        result = None
    try:
        tool_calls = _json.loads(row[1]) if row[1] else None
    except Exception:
        tool_calls = None
    created_at_iso = (
        datetime.fromtimestamp(row[2] / 1000, tz=timezone.utc).isoformat()
        if row[2]
        else None
    )
    return result, tool_calls, created_at_iso


# ── Bot 自身 ─────────────────────────────────────────────

async def get_bot_self() -> tuple[str, str]:
    """读取机器人自身基本信息，返回 (qq_id, nickname)；不存在则返回 ('', '')。"""
    async with _connect() as db:
        async with db.execute(
            "SELECT platform_id, nickname FROM accounts WHERE platform='qq' AND is_bot=1 LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
    if row:
        return str(row[0]), str(row[1] or "")
    return "", ""


async def upsert_bot_self(qq_id: str, nickname: str) -> None:
    """写入/覆盖机器人自身基本信息。"""
    now = _ms()
    async with _connect() as db:
        async with db.execute(
            "SELECT account_uid FROM accounts WHERE platform='qq' AND platform_id=? AND is_bot=1",
            (qq_id,),
        ) as cur:
            row = await cur.fetchone()
        if row:
            await db.execute(
                "UPDATE accounts SET nickname=?, updated_at=? WHERE account_uid=?",
                (nickname, now, row[0]),
            )
        else:
            person_id = str(uuid.uuid4())
            account_uid = str(uuid.uuid4())
            await db.execute(
                "INSERT OR IGNORE INTO persons (person_id, created_at, updated_at) VALUES (?,?,?)",
                (person_id, now, now),
            )
            await db.execute(
                """INSERT INTO accounts
                   (account_uid, person_id, platform, platform_id, nickname, is_bot, created_at, updated_at)
                   VALUES (?,?,?,?,?,1,?,?)
                   ON CONFLICT(platform, platform_id) DO UPDATE SET
                       nickname=excluded.nickname, updated_at=excluded.updated_at""",
                (account_uid, person_id, "qq", qq_id, nickname, now, now),
            )
        await db.commit()
    logger.info("已同步机器人基本信息: qq_id=%s nickname=%s", qq_id, nickname)


# ── 群组 ─────────────────────────────────────────────────

async def get_group_info(group_id: str, platform: str = "qq") -> tuple[str, int, str]:
    """根据群号查询群名称、人数和机器人群名片，返回 (group_name, member_count, bot_card)；不存在则返回 ('', 0, '')。"""
    async with _connect() as db:
        async with db.execute(
            "SELECT group_name, member_count, bot_card FROM groups WHERE platform=? AND group_id=?",
            (platform, group_id),
        ) as cursor:
            row = await cursor.fetchone()
    return (str(row[0] or ""), int(row[1]), str(row[2] or "")) if row else ("", 0, "")


async def get_group_name(group_id: str, platform: str = "qq") -> str:
    """根据群号查询群名称，不存在则返回空字符串。"""
    name, _, _ = await get_group_info(group_id, platform)
    return name


async def upsert_group(
    group_id: str,
    group_name: str,
    bot_card: str = "",
    member_count: int = 0,
    platform: str = "qq",
) -> str:
    """写入/更新群组信息，返回 group_uid。"""
    now = _ms()
    group_uid = f"grp_{platform}_{group_id}"
    async with _connect() as db:
        await db.execute(
            """INSERT INTO groups
               (group_uid, platform, group_id, group_name, bot_card, member_count, updated_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(platform, group_id) DO UPDATE SET
                   group_name=excluded.group_name,
                   bot_card=excluded.bot_card,
                   member_count=excluded.member_count,
                   updated_at=excluded.updated_at""",
            (group_uid, platform, group_id, group_name, bot_card, member_count, now),
        )
        await db.commit()
    logger.debug(
        "已同步群组: group_id=%s group_name=%s member_count=%d",
        group_id, group_name, member_count,
    )
    return group_uid


async def upsert_group_card(group_id: str, group_name: str, bot_card: str, member_count: int = 0) -> None:
    """兼容旧调用，内部转发到 upsert_group。"""
    await upsert_group(group_id, group_name, bot_card, member_count)


# ── 用户账号 ─────────────────────────────────────────────

async def upsert_account(
    platform: str,
    platform_id: str,
    nickname: str = "",
    avatar: str = "",
    extra: str | None = None,
) -> str:
    """写入/更新用户账号，不存在则自动创建对应的 persons 行，返回 account_uid。"""
    now = _ms()
    async with _connect() as db:
        async with db.execute(
            "SELECT account_uid FROM accounts WHERE platform=? AND platform_id=?",
            (platform, platform_id),
        ) as cur:
            row = await cur.fetchone()
        if row:
            account_uid = str(row[0])
            await db.execute(
                """UPDATE accounts SET nickname=?, avatar=?, last_seen_at=?, updated_at=?
                   WHERE account_uid=?""",
                (nickname or None, avatar or None, now, now, account_uid),
            )
        else:
            person_id = str(uuid.uuid4())
            account_uid = str(uuid.uuid4())
            await db.execute(
                "INSERT INTO persons (person_id, last_seen_at, created_at, updated_at) VALUES (?,?,?,?)",
                (person_id, now, now, now),
            )
            await db.execute(
                """INSERT INTO accounts
                   (account_uid, person_id, platform, platform_id, nickname, avatar,
                    last_seen_at, created_at, updated_at, extra)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (account_uid, person_id, platform, platform_id,
                 nickname or None, avatar or None, now, now, now, extra),
            )
        await db.commit()
    return account_uid


# ── 群成员关系 ────────────────────────────────────────────

async def upsert_membership(
    platform: str,
    platform_id: str,
    group_id: str,
    cardname: str = "",
    title: str = "",
    permission_level: str = "member",
    joined_at: int | None = None,
) -> None:
    """写入/更新群成员关系。账号或群组不存在时会自动创建占位记录。"""
    now = _ms()
    account_uid = await upsert_account(platform, platform_id)
    group_uid = f"grp_{platform}_{group_id}"
    async with _connect() as db:
        # 确保 group 占位行存在
        await db.execute(
            """INSERT OR IGNORE INTO groups
               (group_uid, platform, group_id, updated_at) VALUES (?,?,?,?)""",
            (group_uid, platform, group_id, now),
        )
        membership_id = str(uuid.uuid4())
        await db.execute(
            """INSERT INTO memberships
               (membership_id, account_uid, group_uid, cardname, title,
                permission_level, joined_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(account_uid, group_uid) DO UPDATE SET
                   cardname=excluded.cardname,
                   title=excluded.title,
                   permission_level=excluded.permission_level,
                   updated_at=excluded.updated_at""",
            (membership_id, account_uid, group_uid,
             cardname or None, title or None, permission_level, joined_at, now),
        )
        await db.commit()


# ── 人物侧写更新 ──────────────────────────────────────────

async def update_person_profile(
    platform_id: str,
    platform: str = "qq",
    sex: str | None = None,
    age: int | None = None,
    area: str | None = None,
    notes: str | None = None,
) -> bool:
    """更新 persons 表的侧写字段，通过 platform_id 定位对应 person_id。

    只更新非 None 的字段，返回是否找到了对应账号。
    """
    now = _ms()
    async with _connect() as db:
        # 通过 platform + platform_id 找到 person_id
        async with db.execute(
            "SELECT person_id FROM accounts WHERE platform=? AND platform_id=?",
            (platform, platform_id),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return False
        person_id = row[0]

        # 只更新调用方传入的字段
        updates: list[tuple[str, object]] = []
        if sex is not None:
            updates.append(("sex", sex))
        if age is not None:
            updates.append(("age", age))
        if area is not None:
            updates.append(("area", area))
        if notes is not None:
            updates.append(("notes", notes))

        if not updates:
            return True  # 没有要更新的字段，也算成功

        set_clause = ", ".join(f"{col}=?" for col, _ in updates)
        values = [v for _, v in updates] + [now, person_id]
        await db.execute(
            f"UPDATE persons SET {set_clause}, updated_at=? WHERE person_id=?",
            values,
        )
        await db.commit()
    return True


# ── 实体泼溅合并建议 ─────────────────────────────────────

async def upsert_merge_suggestion(
    person_id_a: str,
    person_id_b: str,
    similarity: float,
    reason: str,
) -> str:
    """写入合并建议（幂等：相同 pair 的 pending 建议重复写入时更新 similarity/reason）。
    自动规范化 person_id 顺序（小值在前），避免 (A,B)/(B,A) 重复建议。
    返回 suggestion_id。
    """
    import uuid
    a, b = (person_id_a, person_id_b) if person_id_a < person_id_b else (person_id_b, person_id_a)
    now = _ms()
    async with _connect() as db:
        async with db.execute(
            "SELECT suggestion_id FROM merge_suggestions WHERE person_id_a=? AND person_id_b=? AND status='pending'",
            (a, b),
        ) as cur:
            row = await cur.fetchone()
        if row:
            sid = row[0]
            await db.execute(
                "UPDATE merge_suggestions SET similarity=?, reason=? WHERE suggestion_id=?",
                (similarity, reason, sid),
            )
        else:
            sid = str(uuid.uuid4())
            await db.execute(
                "INSERT INTO merge_suggestions (suggestion_id, person_id_a, person_id_b, similarity, reason, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (sid, a, b, similarity, reason, now),
            )
        await db.commit()
    return sid


async def list_pending_suggestions(limit: int = 10) -> list[dict]:
    """返回待处理的合并建议，按 similarity 降序。"""
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM merge_suggestions WHERE status='pending' ORDER BY similarity DESC LIMIT ?",
            (limit,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def resolve_merge_suggestion(suggestion_id: str, status: str) -> bool:
    """将建议标记为 confirmed 或 rejected，返回是否找到并更新。
    status 必须为 'confirmed' 或 'rejected'。
    """
    if status not in ("confirmed", "rejected"):
        raise ValueError(f"status 必须为 'confirmed' 或 'rejected'，收到：{status!r}")
    now = _ms()
    async with _connect() as db:
        cur = await db.execute(
            "UPDATE merge_suggestions SET status=?, resolved_at=? WHERE suggestion_id=? AND status='pending'",
            (status, now, suggestion_id),
        )
        await db.commit()
    return cur.rowcount > 0


# ── 显示名查询 ────────────────────────────────────────────

async def get_display_name(platform: str, platform_id: str, group_id: str | None = None) -> str:
    """获取用户显示名：优先群名片，其次全局 nickname，再其次返回 platform_id。"""
    async with _connect() as db:
        if group_id:
            group_uid = f"grp_{platform}_{group_id}"
            async with db.execute(
                """SELECT m.cardname, a.nickname
                   FROM memberships m
                   JOIN accounts a ON a.account_uid = m.account_uid
                   WHERE a.platform=? AND a.platform_id=? AND m.group_uid=?""",
                (platform, platform_id, group_uid),
            ) as cur:
                row = await cur.fetchone()
            if row:
                return str(row[0] or row[1] or platform_id)
        async with db.execute(
            "SELECT nickname FROM accounts WHERE platform=? AND platform_id=?",
            (platform, platform_id),
        ) as cur:
            row = await cur.fetchone()
    return str(row[0] if row and row[0] else platform_id)


# ── 长期记忆 ──────────────────────────────────────────────

async def write_memory(
    memory_id: str,
    content: str,
    source: str,
    reason: str,
    conv_type: str = "",
    conv_id: str = "",
    conv_name: str = "",
) -> None:
    """写入一条新记忆。"""
    now = _ms()
    async with _connect() as db:
        await db.execute(
            """INSERT INTO bot_memories
               (memory_id, created_at, content, source, reason, conv_type, conv_id, conv_name, is_deleted)
               VALUES (?,?,?,?,?,?,?,?,0)""",
            (memory_id, now, content, source, reason, conv_type, conv_id, conv_name),
        )
        await db.commit()
    logger.debug("已写入记忆: memory_id=%s", memory_id)


async def soft_delete_memory(memory_id: str) -> bool:
    """软删除一条记忆，返回是否找到并删除。"""
    async with _connect() as db:
        cur = await db.execute(
            "UPDATE bot_memories SET is_deleted=1 WHERE memory_id=? AND is_deleted=0",
            (memory_id,),
        )
        await db.commit()
    return cur.rowcount > 0


async def load_memories(limit: int = 15) -> list[dict]:
    """加载最近 limit 条未删除的记忆，按 created_at 正序（最旧在前）。"""
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT memory_id, created_at, content, source, reason, conv_type, conv_id, conv_name
               FROM (
                   SELECT * FROM bot_memories
                   WHERE is_deleted=0
                   ORDER BY created_at DESC
                   LIMIT ?
               ) sub ORDER BY created_at ASC""",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


# ── MemoryTriples（Phase 1 结构化记忆）──────────────────

async def write_triple(
    subject: str,
    predicate: str,
    object_text: str,
    object_text_tok: str,
    source: str = "",
    reason: str = "",
    conv_type: str = "",
    conv_id: str = "",
    conv_name: str = "",
    confidence: float = 0.6,
    context: str = "truth",
    origin: str = "passive",
    recall_scope: str = "global",
    cluster_id: int | None = None,
) -> int:
    """写入一条记忆三元组到 MemoryTriples，返回新行的整数 id。

    origin:       'active'（模型工具主动写入）或 'passive'（系统自动归档）。
    recall_scope: 召回场景 —— 'global' | 'group:qq_{id}' | 'private:qq_{id}'。
    FTS5 同步触发器会自动将 object_text_tok 写入 MemorySearch 索引。
    """
    now = _ms()
    async with _connect() as db:
        cur = await db.execute(
            """INSERT INTO MemoryTriples
               (subject, predicate, object_text, object_text_tok,
                context, confidence, created_at, last_accessed,
                source, reason, conv_type, conv_id, conv_name, origin,
                recall_scope, cluster_id, is_deleted)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
            (subject, predicate, object_text, object_text_tok,
             context, confidence, now, now,
             source, reason, conv_type, conv_id, conv_name, origin,
             recall_scope, cluster_id),
        )
        await db.commit()
    logger.debug(
        "已写入 MemoryTriple id=%d subject=%s origin=%s scope=%s",
        cur.lastrowid, subject, origin, recall_scope,
    )
    return cur.lastrowid


async def soft_delete_triple(triple_id: int) -> bool:
    """软删除一条三元组（设 is_deleted=1），返回是否找到并删除。

    FTS5 软删除触发器会自动从 MemorySearch 移除对应索引行。
    """
    async with _connect() as db:
        cur = await db.execute(
            "UPDATE MemoryTriples SET is_deleted=1 WHERE id=? AND is_deleted=0",
            (triple_id,),
        )
        await db.commit()
    return cur.rowcount > 0


async def update_triple_confidence(
    triple_ids: list[int],
    delta: float,
    cap: float = 1.0,
) -> None:
    """批量调整置信度并刷新 last_accessed（艾宾浩斯强化 / 降权均使用此函数）。"""
    now = _ms()
    async with _connect() as db:
        for tid in triple_ids:
            await db.execute(
                """UPDATE MemoryTriples
                   SET confidence = MIN(?, confidence + ?),
                       last_accessed = ?
                   WHERE id = ? AND is_deleted = 0""",
                (cap, delta, now, tid),
            )
        await db.commit()


async def decay_triple_confidence(
    min_confidence: float = 0.05,
    decay_rate: float = 0.01,
    idle_days_threshold: float = 7.0,
) -> int:
    """对长期未访问记忆执行置信度衰减（艾宾浩斯遗忘曲线）。

    超过 idle_days_threshold 天未被访问的记忆，每次调用降低 decay_rate 置信度。
    置信度低于 min_confidence 时停止降权（不软删除，模型仍可主动 recall）。
    返回实际降权的条数。

    设计上不更新 last_accessed（防止衰减调度器刷新访问时间掩盖真实使用频率）。
    """
    now = _ms()
    threshold_ms = idle_days_threshold * 86_400_000
    async with _connect() as db:
        cur = await db.execute(
            """UPDATE MemoryTriples
               SET confidence = MAX(?, confidence - ?)
               WHERE is_deleted = 0
                 AND confidence > ?
                 AND (? - last_accessed) > ?""",
            (min_confidence, decay_rate, min_confidence, now, threshold_ms),
        )
        await db.commit()
    count = cur.rowcount
    logger.debug("置信度衰减：降权 %d 条（阈值 %.1f 天，rate=%.3f）", count, idle_days_threshold, decay_rate)
    return count


async def load_all_triples() -> list[dict]:
    """加载所有未删除的三元组，按 created_at 升序（最旧在前）。

    用于：启动时初始化 jieba 自定义词典 + 填充内存缓存。
    """
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT id, subject, predicate, object_text, object_text_tok,
                      confidence, context, created_at, last_accessed,
                      source, reason, conv_type, conv_id, conv_name, origin,
                      recall_scope, cluster_id
               FROM MemoryTriples
               WHERE is_deleted = 0
               ORDER BY created_at ASC"""
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def search_triples(
    fts_query: str,
    subject_filter: str = "",
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    recall_top_k: int = 20,
    context_scope: str = "",
) -> list[dict]:
    """Stage 1 + Stage 2：双通道 FTS5 粗排召回 + BM25 复合精排。

    fts_query:      由 build_fts_query() 生成的 FTS5 查询串
    subject_filter: 通道 A 的 subject 锁定值（如 'User:qq_123456'）
    context_scope:  当前会话的 recall_scope 值（如 'group:qq_12345'）；
                    空字符串表示不限制，召回所有作用域的记忆。
    返回按 final_score 降序的 top recall_top_k 条结果（含聚类激活传播）。
    """
    import time as _time

    if not fts_query:
        return await _load_recent_triples(recall_top_k, context_scope=context_scope)

    results_a: list[dict] = []
    results_b: list[dict] = []

    # scope 过滤子句：只召回 global 记忆 + 当前场景记忆
    # context_scope 为空时不过滤（允许全域召回，例如 recall_memory 工具）
    scope_clause = (
        "AND (t.recall_scope = 'global' OR t.recall_scope = ?)"
        if context_scope else ""
    )

    _COLS = """t.id, t.subject, t.predicate, t.object_text,
               t.confidence, t.context, t.created_at, t.last_accessed,
               t.source, t.reason, t.conv_type, t.conv_id, t.conv_name, t.origin,
               t.recall_scope, t.cluster_id,
               COALESCE(c.confidence, t.confidence) AS effective_confidence,
               fts.rank AS rank"""

    async with _connect() as db:
        db.row_factory = aiosqlite.Row

        # 通道 A：subject 锁定 + FTS5（当前对话者的记忆优先）
        if subject_filter:
            try:
                params_a: list = [fts_query, subject_filter]
                if context_scope:
                    params_a.append(context_scope)
                async with db.execute(
                    f"""SELECT {_COLS}
                        FROM MemorySearch fts
                        JOIN MemoryTriples t ON fts.rowid = t.id
                        LEFT JOIN MemoryClusters c ON t.cluster_id = c.cluster_id
                        WHERE MemorySearch MATCH ?
                          AND t.is_deleted = 0
                          AND t.subject = ?
                          {scope_clause}
                        ORDER BY fts.rank ASC
                        LIMIT 50""",
                    params_a,
                ) as cur:
                    results_a = [dict(r) for r in await cur.fetchall()]
            except Exception as exc:
                logger.debug("FTS5 通道 A 查询失败（忽略）: %s", exc)

        # 通道 B：纯全文检索（不限 subject，覆盖话题相关记忆）
        try:
            params_b: list = [fts_query]
            if context_scope:
                params_b.append(context_scope)
            async with db.execute(
                f"""SELECT {_COLS}
                    FROM MemorySearch fts
                    JOIN MemoryTriples t ON fts.rowid = t.id
                    LEFT JOIN MemoryClusters c ON t.cluster_id = c.cluster_id
                    WHERE MemorySearch MATCH ?
                      AND t.is_deleted = 0
                      {scope_clause}
                    ORDER BY fts.rank ASC
                    LIMIT 50""",
                params_b,
            ) as cur:
                results_b = [dict(r) for r in await cur.fetchall()]
        except Exception as exc:
            logger.debug("FTS5 通道 B 查询失败（忽略）: %s", exc)

    # 合并去重：A 通道优先，B 通道补充不重叠部分
    seen: set[int] = set()
    merged: list[dict] = []
    for r in results_a:
        if r["id"] not in seen:
            seen.add(r["id"])
            merged.append(r)
    for r in results_b:
        if r["id"] not in seen:
            seen.add(r["id"])
            merged.append(r)

    if not merged:
        return []

    # Stage 2：BM25 归一化（极性反转）+ 置信度（优先用聚类置信度）+ 时间衰减
    now_ms = int(_time.time() * 1000)
    ranks = [r["rank"] for r in merged]
    max_r = max(ranks)
    min_r = min(ranks)
    _eps = 1e-5

    def _bm25(r: float) -> float:
        if abs(max_r - min_r) < _eps:
            return 1.0
        return (max_r - r) / (max_r - min_r)

    scored: list[tuple[float, dict]] = []
    for row in merged:
        delta_days = (now_ms - row["last_accessed"]) / (86_400_000)
        eff_conf = row.get("effective_confidence") or row["confidence"]
        score = (
            alpha * _bm25(row["rank"])
            + beta * eff_conf
            - gamma * delta_days
        )
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_results = [item for _, item in scored[:recall_top_k]]

    # 聚类激活传播：命中某条有 cluster_id 的三元组时，同簇其余成员以折扣分数补充
    cluster_ids_found: set[int] = {
        r["cluster_id"] for r in top_results if r.get("cluster_id") is not None
    }
    if cluster_ids_found:
        already_ids: set[int] = {r["id"] for r in top_results}
        placeholders = ",".join("?" * len(cluster_ids_found))
        scope_extra = "AND (t.recall_scope = 'global' OR t.recall_scope = ?)" if context_scope else ""
        params_extra: list = list(cluster_ids_found) + ([context_scope] if context_scope else [])
        async with _connect() as db:
            db.row_factory = aiosqlite.Row
            try:
                async with db.execute(
                    f"""SELECT t.id, t.subject, t.predicate, t.object_text,
                               t.confidence, t.context, t.created_at, t.last_accessed,
                               t.source, t.reason, t.conv_type, t.conv_id, t.conv_name,
                               t.origin, t.recall_scope, t.cluster_id,
                               COALESCE(c.confidence, t.confidence) AS effective_confidence,
                               0.0 AS rank
                        FROM MemoryTriples t
                        LEFT JOIN MemoryClusters c ON t.cluster_id = c.cluster_id
                        WHERE t.cluster_id IN ({placeholders})
                          AND t.is_deleted = 0
                          {scope_extra}
                        LIMIT 30""",
                    params_extra,
                ) as cur:
                    cluster_members = [dict(r) for r in await cur.fetchall()]
                for member in cluster_members:
                    if member["id"] not in already_ids:
                        top_results.append(member)
                        already_ids.add(member["id"])
            except Exception as exc:
                logger.debug("聚类激活传播查询失败（忽略）: %s", exc)

    return top_results


async def _load_recent_triples(limit: int, context_scope: str = "") -> list[dict]:
    """无关键词时的回退：加载最近 limit 条未删除三元组（按创建时间倒序）。"""
    if context_scope:
        sql = """SELECT id, subject, predicate, object_text,
                      confidence, context, created_at, last_accessed,
                      source, reason, conv_type, conv_id, conv_name, origin,
                      recall_scope, cluster_id, confidence AS effective_confidence,
                      0.0 AS rank
               FROM MemoryTriples
               WHERE is_deleted = 0
                 AND (recall_scope = 'global' OR recall_scope = ?)
               ORDER BY created_at DESC
               LIMIT ?"""
        params: tuple = (context_scope, limit)
    else:
        sql = """SELECT id, subject, predicate, object_text,
                      confidence, context, created_at, last_accessed,
                      source, reason, conv_type, conv_id, conv_name, origin,
                      recall_scope, cluster_id, confidence AS effective_confidence,
                      0.0 AS rank
               FROM MemoryTriples
               WHERE is_deleted = 0
               ORDER BY created_at DESC
               LIMIT ?"""
        params = (limit,)
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def migrate_bot_memories_to_triples(tokenize_fn=None) -> int:
    """将 bot_memories 数据一次性迁移到 MemoryTriples（幂等，已迁移则跳过）。

    tokenize_fn: 分词函数（lifecycle.py 在 jieba 初始化后传入）。
                 若为 None，将原始文本直接存入 object_text_tok（FTS5 仍可检索，精度略低）。
    返回实际迁移的条数。
    """
    if tokenize_fn is None:
        tokenize_fn = lambda x: x  # noqa: E731

    async with _connect() as db:
        # 幂等检查：已有数据则跳过
        async with db.execute("SELECT COUNT(*) FROM MemoryTriples WHERE is_deleted=0") as cur:
            count = (await cur.fetchone())[0]
        if count > 0:
            return 0

        # 读 bot_memories（旧表可能不存在）
        try:
            async with db.execute(
                """SELECT created_at, content, source, reason, conv_type, conv_id, conv_name
                   FROM bot_memories
                   WHERE is_deleted=0
                   ORDER BY created_at ASC"""
            ) as cur:
                rows = await cur.fetchall()
        except Exception:
            return 0

        if not rows:
            return 0

        for created_at, content, source, reason, conv_type, conv_id, conv_name in rows:
            tok = tokenize_fn(content)
            await db.execute(
                """INSERT INTO MemoryTriples
                   (subject, predicate, object_text, object_text_tok,
                    context, confidence, created_at, last_accessed,
                    source, reason, conv_type, conv_id, conv_name, is_deleted)
                   VALUES ('Self', '[note]', ?, ?, 'truth', 0.6, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (content, tok, created_at, created_at, source, reason, conv_type, conv_id, conv_name),
            )
        await db.commit()

    logger.info("[migrate] bot_memories → MemoryTriples: %d 条", len(rows))
    return len(rows)


# ── MemoryClusters（Phase 2 记忆聚类）────────────────────

async def create_cluster(label: str, confidence: float = 0.6) -> int:
    """创建一个新的记忆聚类，返回 cluster_id。"""
    now = _ms()
    async with _connect() as db:
        cur = await db.execute(
            """INSERT INTO MemoryClusters (label, confidence, created_at, last_accessed, member_count)
               VALUES (?,?,?,?,0)""",
            (label, confidence, now, now),
        )
        await db.commit()
    logger.debug("已创建 MemoryCluster id=%d label=%s", cur.lastrowid, label)
    return cur.lastrowid


async def assign_cluster(triple_ids: list[int], cluster_id: int) -> int:
    """将一批三元组分配到指定聚类，同步更新 member_count。返回实际更新条数。"""
    if not triple_ids:
        return 0
    now = _ms()
    async with _connect() as db:
        placeholders = ",".join("?" * len(triple_ids))
        cur = await db.execute(
            f"UPDATE MemoryTriples SET cluster_id=? WHERE id IN ({placeholders}) AND is_deleted=0",
            [cluster_id, *triple_ids],
        )
        updated = cur.rowcount
        async with db.execute(
            "SELECT COUNT(*) FROM MemoryTriples WHERE cluster_id=? AND is_deleted=0",
            (cluster_id,),
        ) as count_cur:
            row = await count_cur.fetchone()
        member_count = row[0] if row else 0
        await db.execute(
            "UPDATE MemoryClusters SET member_count=?, last_accessed=? WHERE cluster_id=?",
            (member_count, now, cluster_id),
        )
        await db.commit()
    return updated


async def get_cluster_members(cluster_id: int) -> list[dict]:
    """获取指定聚类的所有未删除三元组，按 created_at 升序。"""
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT id, subject, predicate, object_text, confidence, context,
                      created_at, last_accessed, source, recall_scope, origin
               FROM MemoryTriples
               WHERE cluster_id=? AND is_deleted=0
               ORDER BY created_at ASC""",
            (cluster_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def list_clusters(limit: int = 30) -> list[dict]:
    """列出所有聚类，按 last_accessed 降序。"""
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT cluster_id, label, confidence, created_at, last_accessed, member_count
               FROM MemoryClusters
               ORDER BY last_accessed DESC
               LIMIT ?""",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]

