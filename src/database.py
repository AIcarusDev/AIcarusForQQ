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


# ── 初始化 ────────────────────────────────────────────────

async def init_db() -> None:
    """创建数据库表（如不存在），并执行旧数据迁移。"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            PRAGMA journal_mode=WAL;

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

            -- 意识流活动日志：记录 chat/watcher 切换历史，供 LLM prompt 注入
            CREATE TABLE IF NOT EXISTS activity_log (
                entry_id         TEXT    PRIMARY KEY,
                created_at       INTEGER NOT NULL DEFAULT 0,
                ended_at         INTEGER,
                entry_type       TEXT    NOT NULL DEFAULT '',
                conv_type        TEXT    NOT NULL DEFAULT '',
                conv_id          TEXT    NOT NULL DEFAULT '',
                conv_name        TEXT    NOT NULL DEFAULT '',
                enter_attitude   TEXT    NOT NULL DEFAULT '',
                enter_motivation TEXT    NOT NULL DEFAULT '',
                enter_remark     TEXT    NOT NULL DEFAULT '',
                enter_from       TEXT    NOT NULL DEFAULT '',
                end_attitude     TEXT    NOT NULL DEFAULT '',
                end_action       TEXT    NOT NULL DEFAULT '',
                end_motivation   TEXT    NOT NULL DEFAULT '',
                end_remark       TEXT    NOT NULL DEFAULT ''
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
        """)
        await db.commit()
        await _migrate_legacy(db)
    logger.info("数据库初始化完成: %s", DB_PATH)


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
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE chat_messages SET message_id=? WHERE session_key=? AND message_id=?",
            (new_message_id, session_key, old_message_id),
        )
        await db.commit()


async def load_chat_messages(session_key: str, limit: int = 50) -> list[dict]:
    """加载指定会话最近 limit 条聊天记录，按时间正序返回。"""
    import json as _json
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
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


# ── 活动日志 ─────────────────────────────────────────────

async def save_activity_entry(entry) -> None:
    """写入一条活动日志记录（INSERT OR REPLACE）。"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR REPLACE INTO activity_log
               (entry_id, entry_type, created_at, ended_at,
                conv_type, conv_id, conv_name,
                enter_attitude, enter_motivation, enter_remark, enter_from,
                end_attitude, end_action, end_motivation, end_remark)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                entry.end_attitude,
                entry.end_action,
                entry.end_motivation,
                entry.end_remark,
            ),
        )
        await db.commit()


async def update_activity_entry(entry) -> None:
    """更新已有活动日志记录的 end 相关字段。"""
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT entry_id, entry_type, created_at, ended_at,
                      conv_type, conv_id, conv_name,
                      enter_attitude, enter_motivation, enter_remark, enter_from,
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
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
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


# ── 显示名查询 ────────────────────────────────────────────

async def get_display_name(platform: str, platform_id: str, group_id: str | None = None) -> str:
    """获取用户显示名：优先群名片，其次全局 nickname，再其次返回 platform_id。"""
    async with aiosqlite.connect(DB_PATH) as db:
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
