"""database.py — SQLite 持久化层

表结构：
  persons     — 自然人表，bot 认知层面的人物画像（跨平台、跨账号）
  accounts    — 平台账号表，关联到 persons
  groups      — 群组表（支持多平台）
  memberships — 群成员关系表（账号×群，保存群名片/头衔/权限）

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
