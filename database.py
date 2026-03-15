"""database.py — SQLite 持久化层

表结构：
  profiles   — 名片表，id=0 固定为机器人自身
  group_cards — 机器人在各群的 card（群名片）信息
"""

import logging
from datetime import datetime, timezone

import aiosqlite

DB_PATH = "mita.db"

logger = logging.getLogger("mita.db")


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
                group_id   TEXT PRIMARY KEY,
                group_name TEXT    NOT NULL DEFAULT '',
                bot_card   TEXT    NOT NULL DEFAULT '',
                updated_at TEXT    NOT NULL
            );
        """)
        await db.commit()
    logger.info("数据库初始化完成: %s", DB_PATH)


# ── 机器人自身 ────────────────────────────────────────────

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

async def upsert_group_card(group_id: str, group_name: str, bot_card: str) -> None:
    """写入/覆盖机器人在某个群的名片信息。"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO group_cards (group_id, group_name, bot_card, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(group_id) DO UPDATE SET
                group_name = excluded.group_name,
                bot_card   = excluded.bot_card,
                updated_at = excluded.updated_at
            """,
            (group_id, group_name, bot_card, _now()),
        )
        await db.commit()
    logger.debug("已同步群名片: group_id=%s group_name=%s card=%s", group_id, group_name, bot_card)
