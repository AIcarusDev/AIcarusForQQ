"""database.py — SQLite 持久化层

表结构：
  profiles   — 名片表，id=0 固定为机器人自身
  group_cards — 机器人在各群的 card（群名片）信息
"""

import logging
from datetime import datetime, timezone

import aiosqlite

import os

# 数据库路径 (data/AICQ.db)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
if not os.path.exists(_DATA_DIR):
    os.makedirs(_DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(_DATA_DIR, "AICQ.db")

logger = logging.getLogger("AICQ.db")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── 初始化 ────────────────────────────────────────────────

async def init_db() -> None:
    """创建数据库表（如不存在）。"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS profiles (
                id         INTEGER PRIMARY KEY,
                qq_id      TEXT    NOT NULL,
                nickname   TEXT    NOT NULL DEFAULT '',
                updated_at TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS group_cards (
                group_id     TEXT PRIMARY KEY,
                group_name   TEXT    NOT NULL DEFAULT '',
                bot_card     TEXT    NOT NULL DEFAULT '',
                member_count INTEGER NOT NULL DEFAULT 0,
                updated_at   TEXT    NOT NULL
            );
        """)
        # 迁移：为旧数据库补加 member_count 列
        try:
            await db.execute("ALTER TABLE group_cards ADD COLUMN member_count INTEGER NOT NULL DEFAULT 0")
            await db.commit()
        except Exception:
            pass  # 列已存在，忽略
        await db.commit()
    logger.info("数据库初始化完成: %s", DB_PATH)


# ── 机器人自身 ────────────────────────────────────────────

async def get_bot_self() -> tuple[str, str]:
    """读取机器人自身基本信息，返回 (qq_id, nickname)；不存在则返回 ('', '')。"""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT qq_id, nickname FROM profiles WHERE id = 0"
        ) as cursor:
            row = await cursor.fetchone()
    if row:
        return str(row[0]), str(row[1])
    return "", ""


async def upsert_bot_self(qq_id: str, nickname: str) -> None:
    """写入/覆盖机器人自身基本信息（profiles.id = 0）。"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO profiles (id, qq_id, nickname, updated_at)
            VALUES (0, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                qq_id      = excluded.qq_id,
                nickname   = excluded.nickname,
                updated_at = excluded.updated_at
            """,
            (qq_id, nickname, _now()),
        )
        await db.commit()
    logger.info("已同步机器人基本信息: qq_id=%s nickname=%s", qq_id, nickname)


# ── 群名片 ────────────────────────────────────────────────

async def get_group_name(group_id: str) -> str:
    """根据群号查询群名称，不存在则返回空字符串。"""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT group_name FROM group_cards WHERE group_id = ?",
            (group_id,),
        ) as cursor:
            row = await cursor.fetchone()
    return str(row[0]) if row else ""


async def get_group_info(group_id: str) -> tuple[str, int]:
    """根据群号查询群名称和人数，返回 (group_name, member_count)；不存在则返回 ('', 0)。"""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT group_name, member_count FROM group_cards WHERE group_id = ?",
            (group_id,),
        ) as cursor:
            row = await cursor.fetchone()
    return (str(row[0]), int(row[1])) if row else ("", 0)


async def upsert_group_card(group_id: str, group_name: str, bot_card: str, member_count: int = 0) -> None:
    """写入/覆盖机器人在某个群的名片信息。"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO group_cards (group_id, group_name, bot_card, member_count, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(group_id) DO UPDATE SET
                group_name   = excluded.group_name,
                bot_card     = excluded.bot_card,
                member_count = excluded.member_count,
                updated_at   = excluded.updated_at
            """,
            (group_id, group_name, bot_card, member_count, _now()),
        )
        await db.commit()
    logger.debug("已同步群名片: group_id=%s group_name=%s card=%s member_count=%d", group_id, group_name, bot_card, member_count)
