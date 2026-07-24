"""database.py — SQLite 持久化层

表结构（核心本体论双层模型，参见《通用实体认知与泼溅系统 V0.1》）：

  ┌─ 客观实体层（Objective / Entity）────────────────────────────────────┐
  │  entities         — 客观可观测的实体（QQ 账号、未来可扩展至任意平台标识符）│
  │                     每行对应一个唯一的 (platform, platform_id) 组合。      │
  │                     存的是事实（fact），不含 AI 推断。                      │
  └──────────────────────────────────────────────────────────────────────┘
  ┌─ 主观侧写层（Subjective / EntityProfile）─────────────────────────────┐
  │  entity_profiles  — AI 对客观实体的主观认知侧写（跨平台、跨账号）。       │
  │                     每行对应一个「意识个体」，存的是推断（inference）。     │
  │                     与 entities 通过 profile_id FK 关联，                  │
  │                     等价于设计文档中的 represents 边：                      │
  │                       EntityProfile ──represents──▶ Entity                 │
  └──────────────────────────────────────────────────────────────────────┘

  groups      — 群组表（支持多平台）
  memberships — 群成员关系表（entities × groups，保存群名片/头衔/权限）
  chat_sessions  — 会话注册表（记住历史会话的 key → meta）
  chat_messages  — 聊天记录（按 session_key 隔离，可按需恢复上下文）
  bot_turns      — bot 意识流日志（全局唯一，每轮 LLM 输出 + 工具调用记录）

旧表 profiles / group_cards 保留用于数据迁移，迁移后不再写入。
"""

import logging
import os
import json
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import aiosqlite

from cognition_sources_schema import COGNITION_SOURCES_SCHEMA_SQL
from platforms.focus import FocusRef, focus_from_session_key, session_key_for_focus

# 数据库路径 (data/AICQ.db)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
os.makedirs(_DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(_DATA_DIR, "AICQ.db")

logger = logging.getLogger("AICQ.db")

# Chat chronology must be based on the message occurrence timestamp, not the
# SQLite insertion id. Recovery/backfill can insert old messages after newer
# live rows, so id is only a deterministic tie-breaker for identical timestamps.
CHAT_MESSAGE_SORT_KEY_SQL = (
    "COALESCE("
    "julianday(NULLIF(timestamp, '')), "
    "CASE WHEN created_at > 0 THEN created_at / 86400000.0 + 2440587.5 END, "
    "0"
    ")"
)
CHAT_MESSAGE_ORDER_ASC_SQL = f"{CHAT_MESSAGE_SORT_KEY_SQL} ASC, id ASC"
CHAT_MESSAGE_ORDER_DESC_SQL = f"{CHAT_MESSAGE_SORT_KEY_SQL} DESC, id DESC"


def _focus_from_legacy(
    *,
    session_key: str = "",
    conv_type: str = "",
    conv_id: str = "",
    conv_name: str = "",
    platform: str = "qq",
) -> FocusRef | None:
    focus = focus_from_session_key(session_key, default_platform=platform)
    if focus is None and conv_type and conv_id:
        focus = FocusRef(platform, str(conv_type), str(conv_id), str(conv_name or ""))
    elif focus is not None and conv_name and not focus.target_name:
        focus = focus.with_name(str(conv_name or ""))
    return focus


def _focus_ref_json(focus: FocusRef | None) -> str:
    if focus is None:
        return "{}"
    return json.dumps(focus.as_dict(), ensure_ascii=False, sort_keys=True)


def _focus_tuple_from_legacy(
    *,
    session_key: str = "",
    conv_type: str = "",
    conv_id: str = "",
    conv_name: str = "",
) -> tuple[str, str, str, str, str, str]:
    focus = _focus_from_legacy(
        session_key=session_key,
        conv_type=conv_type,
        conv_id=conv_id,
        conv_name=conv_name,
    )
    if focus is None:
        return "", "", "", str(conv_name or ""), "", "{}"
    return (
        focus.platform,
        focus.target_type,
        focus.target_id,
        focus.target_name,
        session_key_for_focus(focus),
        _focus_ref_json(focus),
    )


_LLM_USAGE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS llm_usage_events (
    event_id                TEXT    PRIMARY KEY,
    created_at              INTEGER NOT NULL DEFAULT 0,
    provider                TEXT    NOT NULL DEFAULT '',
    model                   TEXT    NOT NULL DEFAULT '',
    feature                 TEXT    NOT NULL DEFAULT '',
    subfeature              TEXT    NOT NULL DEFAULT '',
    input_tokens            INTEGER NOT NULL DEFAULT 0,
    output_tokens           INTEGER NOT NULL DEFAULT 0,
    total_tokens            INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens     INTEGER NOT NULL DEFAULT 0,
    reasoning_output_tokens INTEGER NOT NULL DEFAULT 0,
    usage_available         INTEGER NOT NULL DEFAULT 0,
    status                  TEXT    NOT NULL DEFAULT '',
    raw_usage_json          TEXT    NOT NULL DEFAULT '{}',
    legacy_turn_id          TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_llm_usage_events_created
    ON llm_usage_events(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_events_model
    ON llm_usage_events(provider, model, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_events_feature
    ON llm_usage_events(feature, subfeature, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_usage_events_legacy_turn
    ON llm_usage_events(legacy_turn_id)
    WHERE legacy_turn_id <> '';
"""


def _ms() -> int:
    """返回当前 UTC 时间戳（毫秒）。"""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


@asynccontextmanager
async def _connect():
    """打开数据库连接并启用外键约束。

    PRAGMA foreign_keys=ON 是 SQLite 的连接级设置，不会持久化到文件。
    每条连接都必须单独设置，否则 REFERENCES 约束实际不生效。
    """
    async with aiosqlite.connect(DB_PATH, timeout=30.0) as db:
        await db.execute("PRAGMA foreign_keys=ON")
        await db.execute("PRAGMA busy_timeout=30000")
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
                focus_platform TEXT   NOT NULL DEFAULT '',
                focus_type    TEXT    NOT NULL DEFAULT '',
                focus_id      TEXT    NOT NULL DEFAULT '',
                focus_name    TEXT    NOT NULL DEFAULT '',
                focus_ref_json TEXT   NOT NULL DEFAULT '{}',
                conv_type     TEXT    NOT NULL DEFAULT '',
                conv_id       TEXT    NOT NULL DEFAULT '',
                conv_name     TEXT    NOT NULL DEFAULT '',
                temp_source_group_id   TEXT NOT NULL DEFAULT '',
                temp_source_group_name TEXT NOT NULL DEFAULT '',
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
                sender_card      TEXT    NOT NULL DEFAULT '',
                sender_nickname  TEXT    NOT NULL DEFAULT '',
                sender_role      TEXT    NOT NULL DEFAULT '',
                sender_title     TEXT    NOT NULL DEFAULT '',
                sender_level     TEXT    NOT NULL DEFAULT '',
                timestamp        TEXT    NOT NULL DEFAULT '',
                reply_to         TEXT    NOT NULL DEFAULT '',
                content          TEXT    NOT NULL DEFAULT '',
                content_type     TEXT    NOT NULL DEFAULT 'text',
                content_segments TEXT    NOT NULL DEFAULT '[]',
                images           TEXT    NOT NULL DEFAULT '[]',
                delivery_state   TEXT    NOT NULL DEFAULT '',
                delivery_error   TEXT    NOT NULL DEFAULT '',
                created_at       INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                ON chat_messages(session_key, id);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_timestamp
                ON chat_messages(session_key, timestamp, id);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_chronology
                ON chat_messages(
                    session_key,
                    COALESCE(
                        julianday(NULLIF(timestamp, '')),
                        CASE WHEN created_at > 0 THEN created_at / 86400000.0 + 2440587.5 END,
                        0
                    ),
                    id
                );

            -- bot 意识流日志：全局唯一，保存每轮 LLM 输出及工具调用，供重启后恢复
            CREATE TABLE IF NOT EXISTS bot_turns (
                turn_id      TEXT    PRIMARY KEY,
                created_at   INTEGER NOT NULL DEFAULT 0,
                focus_platform TEXT   NOT NULL DEFAULT '',
                focus_type   TEXT    NOT NULL DEFAULT '',
                focus_id     TEXT    NOT NULL DEFAULT '',
                focus_ref_json TEXT   NOT NULL DEFAULT '{}',
                conv_type    TEXT    NOT NULL DEFAULT '',
                conv_id      TEXT    NOT NULL DEFAULT '',
                result_json  TEXT    NOT NULL DEFAULT '{}',
                tool_calls   TEXT    NOT NULL DEFAULT '[]',
                world_xml    TEXT    NOT NULL DEFAULT ''
            );

            -- LLM token 用量事件：每次模型 API 调用一行。usage 缺失时
            -- usage_available=0，token 字段保持 0，但只作为未知请求统计。
            CREATE TABLE IF NOT EXISTS llm_usage_events (
                event_id                TEXT    PRIMARY KEY,
                created_at              INTEGER NOT NULL DEFAULT 0,
                provider                TEXT    NOT NULL DEFAULT '',
                model                   TEXT    NOT NULL DEFAULT '',
                feature                 TEXT    NOT NULL DEFAULT '',
                subfeature              TEXT    NOT NULL DEFAULT '',
                input_tokens            INTEGER NOT NULL DEFAULT 0,
                output_tokens           INTEGER NOT NULL DEFAULT 0,
                total_tokens            INTEGER NOT NULL DEFAULT 0,
                cached_input_tokens     INTEGER NOT NULL DEFAULT 0,
                reasoning_output_tokens INTEGER NOT NULL DEFAULT 0,
                usage_available         INTEGER NOT NULL DEFAULT 0,
                status                  TEXT    NOT NULL DEFAULT '',
                raw_usage_json          TEXT    NOT NULL DEFAULT '{}',
                legacy_turn_id          TEXT    NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_llm_usage_events_created
                ON llm_usage_events(created_at);
            CREATE INDEX IF NOT EXISTS idx_llm_usage_events_model
                ON llm_usage_events(provider, model, created_at);
            CREATE INDEX IF NOT EXISTS idx_llm_usage_events_feature
                ON llm_usage_events(feature, subfeature, created_at);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_usage_events_legacy_turn
                ON llm_usage_events(legacy_turn_id)
                WHERE legacy_turn_id <> '';

            -- 临时附件任务：只记录动作和结果，不保存附件内容。
            CREATE TABLE IF NOT EXISTS attachment_tasks (
                task_id          TEXT PRIMARY KEY,
                attachment_id    TEXT NOT NULL,
                source_type      TEXT NOT NULL,
                source           TEXT NOT NULL DEFAULT '',
                status           TEXT NOT NULL,
                path             TEXT,
                filename         TEXT,
                mime             TEXT,
                image_ref        TEXT,
                bytes_downloaded INTEGER NOT NULL DEFAULT 0,
                bytes_total      INTEGER,
                sha256           TEXT,
                error            TEXT,
                started_at       TEXT NOT NULL DEFAULT '',
                finished_at      TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_attachment_tasks_status
                ON attachment_tasks(status, started_at);
            CREATE INDEX IF NOT EXISTS idx_attachment_tasks_attachment
                ON attachment_tasks(attachment_id);

            -- watcher 窥屏意识循环日志：每轮窥屏的内心状态与决策
            CREATE TABLE IF NOT EXISTS watcher_cycles (
                cycle_id     TEXT    PRIMARY KEY,
                created_at   INTEGER NOT NULL DEFAULT 0,
                focus_platform TEXT   NOT NULL DEFAULT '',
                focus_type   TEXT    NOT NULL DEFAULT '',
                focus_id     TEXT    NOT NULL DEFAULT '',
                focus_ref_json TEXT   NOT NULL DEFAULT '{}',
                conv_type    TEXT    NOT NULL DEFAULT '',
                conv_id      TEXT    NOT NULL DEFAULT '',
                result_json  TEXT    NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_watcher_cycles_conv
                ON watcher_cycles(conv_type, conv_id, created_at);

            -- ── 主观侧写层：AI 对实体的认知画像 (EntityProfile) ────────────────
            -- 每行代表 AI 认知中的一个「意识个体」，与一或多个客观实体 (entities)
            -- 通过 entities.profile_id FK 关联，对应设计文档的 represents 边。
            -- 只存 AI 的推断/观点（sex/age/area/notes），不存平台事实。
            CREATE TABLE IF NOT EXISTS entity_profiles (
                profile_id   TEXT    PRIMARY KEY,  -- AI 内部生成的唯一 UUID
                sex          TEXT,                 -- 推断性别（AI 主观，非事实）
                age          INTEGER,              -- 推断年龄段
                area         TEXT,                 -- 推断地区
                notes        TEXT,                 -- AI 对该意识个体的综合备注
                last_seen_at INTEGER,
                created_at   INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                updated_at   INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                extra        TEXT
            );

            -- ── 客观实体层：可被直接观测和交互的存在 (Entity) ───────────────────
            -- 每行对应一个唯一的 (platform, platform_id) 组合，存的是客观事实。
            -- profile_id FK → entity_profiles，即 represents 边的关系型表达：
            --   Entity ←represents── EntityProfile
            -- 未来扩展非人类实体（群组概念、物品等）时只需新增行，表结构不变。
            CREATE TABLE IF NOT EXISTS entities (
                account_uid  TEXT    PRIMARY KEY,  -- 内部唯一 UUID（历史遗留名，勿改列名以免迁移）
                profile_id   TEXT    NOT NULL REFERENCES entity_profiles(profile_id),  -- represents 边
                platform     TEXT    NOT NULL,     -- 平台标识，如 'qq'
                platform_id  TEXT    NOT NULL,     -- 平台内唯一 ID，如 QQ 号
                nickname     TEXT,                 -- 客观昵称（来自平台事实，非 AI 推断）
                avatar       TEXT,
                is_bot       INTEGER NOT NULL DEFAULT 0,  -- 1 = 本 bot 自身
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
                account_uid      TEXT    NOT NULL REFERENCES entities(account_uid),
                group_uid        TEXT    NOT NULL REFERENCES groups(group_uid),
                cardname         TEXT,
                title            TEXT,
                title_expire_time INTEGER NOT NULL DEFAULT 0,
                level            TEXT    NOT NULL DEFAULT '',
                permission_level TEXT    NOT NULL DEFAULT 'member',
                joined_at        INTEGER,
                updated_at       INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                UNIQUE(account_uid, group_uid)
            );

            -- 模型活跃目标表：由模型通过工具主动维护
            CREATE TABLE IF NOT EXISTS bot_goals (
                goal_id      TEXT    PRIMARY KEY,
                created_at   INTEGER NOT NULL DEFAULT 0,
                updated_at   INTEGER NOT NULL DEFAULT 0,
                title        TEXT    NOT NULL DEFAULT '',
                content      TEXT    NOT NULL DEFAULT '',
                reason       TEXT    NOT NULL DEFAULT '',
                focus_platform TEXT   NOT NULL DEFAULT '',
                focus_type   TEXT    NOT NULL DEFAULT '',
                focus_id     TEXT    NOT NULL DEFAULT '',
                focus_name   TEXT    NOT NULL DEFAULT '',
                focus_ref_json TEXT   NOT NULL DEFAULT '{}',
                conv_type    TEXT    NOT NULL DEFAULT '',
                conv_id      TEXT    NOT NULL DEFAULT '',
                conv_name    TEXT    NOT NULL DEFAULT '',
                status       TEXT    NOT NULL DEFAULT 'active',
                resolution   TEXT    NOT NULL DEFAULT '',
                is_deleted   INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_bot_goals_active
                ON bot_goals(created_at) WHERE is_deleted=0 AND status='active';

            -- adapter 意识流持久化：跨重启保留函数调用历史
            CREATE TABLE IF NOT EXISTS adapter_state (
                key          TEXT    PRIMARY KEY,
                updated_at   INTEGER NOT NULL DEFAULT 0,
                adapter_type TEXT    NOT NULL DEFAULT '',
                contents     TEXT    NOT NULL DEFAULT '[]',
                timestamps   TEXT    NOT NULL DEFAULT '[]'
            );

            -- 认知块来源身份：独立于长期记忆，用作记忆和未来 world 切片的来源锚点
        """)
        await db.executescript(COGNITION_SOURCES_SCHEMA_SQL)
        await db.executescript("""
            -- ── 实体泼溅合并建议表（Phase 3B）──────────────────────────────
            -- 绝不自动合并；建议仅供模型/人工二次确认后推进
            -- profile_id_a/b 指向 entity_profiles(profile_id)，
            --   即"有多大把把这两个主观侧写当成同一个意识个体"的建议
            CREATE TABLE IF NOT EXISTS merge_suggestions (
                suggestion_id TEXT    PRIMARY KEY,
                profile_id_a  TEXT    NOT NULL REFERENCES entity_profiles(profile_id),
                profile_id_b  TEXT    NOT NULL REFERENCES entity_profiles(profile_id),
                similarity    REAL    NOT NULL DEFAULT 0.0,
                reason        TEXT    NOT NULL DEFAULT '',
                status        TEXT    NOT NULL DEFAULT 'pending',
                created_at    INTEGER NOT NULL DEFAULT 0,
                resolved_at   INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_ms_status
                ON merge_suggestions(status, created_at);

            -- 归档窗口指纹：防止进程重启后对已提取过的不变窗口重复归档
            CREATE TABLE IF NOT EXISTS archive_signatures (
                conv_key   TEXT    PRIMARY KEY,   -- "conv_type/conv_id"
                signature  TEXT    NOT NULL DEFAULT ''
            );

            -- 待归档任务队列：snapshot 已构建好的对话(含 candidates 内联),
            -- 进程退出时不删除,启动时由事件提取流程续跑,避免 Ctrl+C 卡在 LLM 调用上。
            CREATE TABLE IF NOT EXISTS pending_archive_jobs (
                job_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                focus_platform  TEXT    NOT NULL DEFAULT '',
                focus_type      TEXT    NOT NULL DEFAULT '',
                focus_id        TEXT    NOT NULL DEFAULT '',
                focus_name      TEXT    NOT NULL DEFAULT '',
                focus_ref_json  TEXT    NOT NULL DEFAULT '{}',
                conv_type       TEXT    NOT NULL DEFAULT '',
                conv_id         TEXT    NOT NULL DEFAULT '',
                conv_name       TEXT    NOT NULL DEFAULT '',
                sender_id       TEXT    NOT NULL DEFAULT '',
                dialogue        TEXT    NOT NULL DEFAULT '',  -- 已含 <existing_candidates>
                signature       TEXT    NOT NULL DEFAULT '',
                prev_signature  TEXT    NOT NULL DEFAULT '',
                valid_candidate_ids TEXT NOT NULL DEFAULT '[]',  -- JSON array
                enqueued_at     INTEGER NOT NULL DEFAULT 0
            );

            -- 一次性迁移标记表：防止破坏性 DDL/DML 每次启动重跑
            CREATE TABLE IF NOT EXISTS _migrations (
                name       TEXT    PRIMARY KEY,
                applied_at INTEGER NOT NULL DEFAULT 0
            );
        """)
        await db.commit()

        # Core 重启后后台传输进程不存在；遗留 running 不能继续冒充活跃任务。
        await db.execute(
            """UPDATE attachment_tasks
               SET status='interrupted', error='core restarted during download',
                   finished_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
               WHERE status='running'"""
        )
        await db.commit()

        await _migrate_memory_schema_to_primary(db)
        await _migrate_schema(db)
        await _migrate_legacy(db)
        await _migrate_rename_tables(db)
        await _backfill_llm_usage_from_bot_turns(db)
    try:
        from memory.repo.events import ensure_schema as _ensure_memory_schema

        await _ensure_memory_schema()
    except Exception:
        logger.exception("[schema] Memory schema initialization failed")

    logger.info("数据库初始化完成: %s", DB_PATH)


_MEMORY_TABLE_RENAMES: tuple[tuple[str, str], ...] = (
    ("MemoryV2Events", "MemoryEvents"),
    ("MemoryV2Participants", "MemoryParticipants"),
    ("MemoryV2Predicates", "MemoryPredicates"),
    ("MemoryV2Relations", "MemoryRelations"),
    ("MemoryV2EventSources", "MemoryEventSources"),
    ("MemoryV2Vectors", "MemoryVectors"),
    ("MemoryV2EmbeddingJobs", "MemoryEmbeddingJobs"),
    ("MemoryV2PreprocessRuns", "MemoryPreprocessRuns"),
    ("MemoryV2CanonicalEntities", "MemoryCanonicalEntities"),
    ("MemoryV2EntityAliases", "MemoryEntityAliases"),
    ("MemoryV2EntityMentions", "MemoryEntityMentions"),
    ("MemoryV2EntityMergeSuspicions", "MemoryEntityMergeSuspicions"),
    ("MemoryV2EventRelationRuns", "MemoryEventRelationRuns"),
    ("MemoryV2EventRelations", "MemoryEventRelations"),
    ("MemoryV2Episodes", "MemoryEpisodes"),
    ("MemoryV2EpisodeMembers", "MemoryEpisodeMembers"),
    ("MemoryV2RelationRevisions", "MemoryRelationRevisions"),
    ("MemoryV2ClusterRuns", "MemoryStorylineRuns"),
    ("MemoryV2Clusters", "MemoryStorylines"),
    ("MemoryV2ClusterMembers", "MemoryStorylineMembers"),
    ("MemoryV2ClusterMemberRevisions", "MemoryStorylineMemberRevisions"),
    ("MemoryV2ThreadStates", "MemoryThreadStates"),
    ("MemoryV2ThreadStateRevisions", "MemoryThreadStateRevisions"),
    ("MemoryV2ClusterRelations", "MemoryStorylineRelations"),
    ("MemoryV2ClusterRevisions", "MemoryStorylineRevisions"),
    ("MemoryV2SummaryCache", "MemorySummaryCache"),
)

_STALE_MEMORY_INDEXES: tuple[str, ...] = (
    "idx_mv2_embed_jobs",
    "idx_mv2_events_conv",
    "idx_mv2_events_dedupe",
    "idx_mv2_events_pred",
    "idx_mv2_events_time",
    "idx_mv2_part_entity",
    "idx_mv2_part_event",
    "idx_mv2_rel_dst",
    "idx_mv2_rel_src",
    "idx_mv2_sources_event",
    "idx_mv2_sources_source",
    "idx_mv2_sources_uid",
    "idx_mv2_vec_owner",
)


async def _table_exists(db: aiosqlite.Connection, table: str) -> bool:
    async with db.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (table,),
    ) as cur:
        return await cur.fetchone() is not None


async def _table_columns(db: aiosqlite.Connection, table: str) -> set[str]:
    if not await _table_exists(db, table):
        return set()
    async with db.execute(f"PRAGMA table_info({table})") as cur:
        return {str(row[1]) for row in await cur.fetchall()}


async def _drop_legacy_memory_event_tables(db: aiosqlite.Connection) -> None:
    for stmt in (
        "DROP TRIGGER IF EXISTS me_fts_insert",
        "DROP TRIGGER IF EXISTS me_fts_delete",
        "DROP TRIGGER IF EXISTS me_fts_update",
        "DROP TRIGGER IF EXISTS mv2_fts_insert",
        "DROP TRIGGER IF EXISTS mv2_fts_delete",
        "DROP TRIGGER IF EXISTS mv2_fts_update",
        "DROP TRIGGER IF EXISTS memory_fts_insert",
        "DROP TRIGGER IF EXISTS memory_fts_delete",
        "DROP TRIGGER IF EXISTS memory_fts_update",
        "DROP TABLE IF EXISTS MemorySearch",
        "DROP TABLE IF EXISTS MemoryRoles",
        "DROP TABLE IF EXISTS MemoryEvents",
    ):
        await db.execute(stmt)


async def _migrate_memory_schema_to_primary(db: aiosqlite.Connection) -> None:
    """Rename formerly versioned memory tables into the unversioned primary schema."""
    await db.execute(
        "CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY, applied_at INTEGER NOT NULL DEFAULT 0)"
    )
    has_old_primary = await _table_exists(db, "MemoryEvents")
    primary_columns = await _table_columns(db, "MemoryEvents") if has_old_primary else set()
    has_versioned_memory_tables = await _table_exists(db, "MemoryV2Events")

    if has_versioned_memory_tables:
        if has_old_primary and "event_type_norm" not in primary_columns:
            logger.info("[schema] 删除旧 MemoryEvents/MemoryRoles，准备迁移当前记忆主表")
            await _drop_legacy_memory_event_tables(db)
        for stmt in (
            "DROP TRIGGER IF EXISTS mv2_fts_insert",
            "DROP TRIGGER IF EXISTS mv2_fts_delete",
            "DROP TRIGGER IF EXISTS mv2_fts_update",
            "DROP TABLE IF EXISTS MemoryV2Search",
        ):
            await db.execute(stmt)
        await db.execute("PRAGMA foreign_keys=OFF")
        for old_name, new_name in _MEMORY_TABLE_RENAMES:
            if await _table_exists(db, old_name):
                if await _table_exists(db, new_name):
                    logger.warning("[schema] 跳过 %s -> %s：目标表已存在", old_name, new_name)
                    continue
                await db.execute(f"ALTER TABLE {old_name} RENAME TO {new_name}")
                logger.info("[schema] 已迁移记忆表 %s -> %s", old_name, new_name)
        await db.execute("PRAGMA foreign_keys=ON")
        await db.execute(
            "INSERT OR REPLACE INTO _migrations(name, applied_at) VALUES (?, ?)",
            ("memory_primary_table_names", _ms()),
        )
        await db.commit()
    elif has_old_primary and "event_type_norm" not in primary_columns:
        logger.info("[schema] 删除旧 MemoryEvents/MemoryRoles，当前记忆主表将由新 schema 创建")
        await _drop_legacy_memory_event_tables(db)
        await db.execute(
            "INSERT OR REPLACE INTO _migrations(name, applied_at) VALUES (?, ?)",
            ("drop_legacy_memory_events", _ms()),
        )
        await db.commit()
    for index_name in _STALE_MEMORY_INDEXES:
        await db.execute(f"DROP INDEX IF EXISTS {index_name}")
    await db.commit()


async def _migrate_schema(db) -> None:
    """为已有表补充新增列（ALTER TABLE），保证旧数据库可以正常使用。"""
    try:
        await db.executescript(_LLM_USAGE_SCHEMA_SQL)
        await db.commit()
    except Exception:
        logger.exception("[schema] llm_usage_events 表初始化失败")
        raise

    # attachment_tasks 初版未包含展示类型与复用图片引用，补齐旧数据库。
    try:
        await _ensure_columns(db, "attachment_tasks", (
            ("mime", "mime TEXT"),
            ("image_ref", "image_ref TEXT"),
        ))
        await db.commit()
    except Exception:
        logger.exception("[schema] attachment_tasks 迁移失败")
        raise

    # chat_sessions 新增列：临时会话来源群只作为发送/打开入口元数据，不参与会话 key。
    try:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chat_sessions'"
        ) as cur:
            chat_sessions_exists = await cur.fetchone() is not None
        if chat_sessions_exists:
            async with db.execute("PRAGMA table_info(chat_sessions)") as cur:
                session_columns = {str(row[1]) for row in await cur.fetchall()}
            for col, ddl in (
                ("temp_source_group_id", "ALTER TABLE chat_sessions ADD COLUMN temp_source_group_id TEXT NOT NULL DEFAULT ''"),
                ("temp_source_group_name", "ALTER TABLE chat_sessions ADD COLUMN temp_source_group_name TEXT NOT NULL DEFAULT ''"),
            ):
                if col not in session_columns:
                    await db.execute(ddl)
                    logger.info("[schema] chat_sessions 已添加 %s 列", col)
            await db.commit()
    except Exception:
        logger.exception("[schema] chat_sessions 迁移失败")
        raise

    # chat_messages 新增列：持久化 QQ 引用/回复关系与群成员状态快照。
    try:
        async with db.execute("PRAGMA table_info(chat_messages)") as cur:
            chat_columns = {str(row[1]) for row in await cur.fetchall()}
        for col, ddl in (
            ("reply_to", "ALTER TABLE chat_messages ADD COLUMN reply_to TEXT NOT NULL DEFAULT ''"),
            ("sender_card", "ALTER TABLE chat_messages ADD COLUMN sender_card TEXT NOT NULL DEFAULT ''"),
            ("sender_nickname", "ALTER TABLE chat_messages ADD COLUMN sender_nickname TEXT NOT NULL DEFAULT ''"),
            ("sender_title", "ALTER TABLE chat_messages ADD COLUMN sender_title TEXT NOT NULL DEFAULT ''"),
            ("sender_level", "ALTER TABLE chat_messages ADD COLUMN sender_level TEXT NOT NULL DEFAULT ''"),
            ("delivery_state", "ALTER TABLE chat_messages ADD COLUMN delivery_state TEXT NOT NULL DEFAULT ''"),
            ("delivery_error", "ALTER TABLE chat_messages ADD COLUMN delivery_error TEXT NOT NULL DEFAULT ''"),
        ):
            if col not in chat_columns:
                await db.execute(ddl)
                logger.info("[schema] chat_messages 已添加 %s 列", col)
        await db.commit()
    except Exception:
        logger.exception("[schema] chat_messages 迁移失败")
        raise

    # bot_turns 新增列：持久化本轮模型决策前看到的 <world> 文本，供 Agent 视图重启恢复。
    try:
        async with db.execute("PRAGMA table_info(bot_turns)") as cur:
            bot_turn_columns = {str(row[1]) for row in await cur.fetchall()}
        if "world_xml" not in bot_turn_columns:
            await db.execute("ALTER TABLE bot_turns ADD COLUMN world_xml TEXT NOT NULL DEFAULT ''")
            logger.info("[schema] bot_turns 已添加 world_xml 列")
        await db.commit()
    except Exception:
        logger.exception("[schema] bot_turns 迁移失败")
        raise

    # memberships 新增群成员高频状态列：专属头衔、等级。
    try:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memberships'"
        ) as cur:
            memberships_exists = await cur.fetchone() is not None
        if not memberships_exists:
            membership_columns = set()
        else:
            async with db.execute("PRAGMA table_info(memberships)") as cur:
                membership_columns = {str(row[1]) for row in await cur.fetchall()}
        if memberships_exists:
            for col, ddl in (
                ("title_expire_time", "ALTER TABLE memberships ADD COLUMN title_expire_time INTEGER NOT NULL DEFAULT 0"),
                ("level", "ALTER TABLE memberships ADD COLUMN level TEXT NOT NULL DEFAULT ''"),
            ):
                if col not in membership_columns:
                    await db.execute(ddl)
                    logger.info("[schema] memberships 已添加 %s 列", col)
            await db.commit()
    except Exception:
        logger.exception("[schema] memberships 迁移失败")
        raise

    # bot_goals 新增 resolution 列
    try:
        await db.execute(
            "ALTER TABLE bot_goals ADD COLUMN resolution TEXT NOT NULL DEFAULT ''"
        )
        await db.commit()
        logger.info("[schema] bot_goals 已添加 resolution 列")
    except Exception:
        pass  # 列已存在则跳过

    # 兼容旧版：此前 complete_goal 会把 status 直接写成 completed
    try:
        await db.execute(
            "UPDATE bot_goals SET status='resolved', resolution='completed' "
            "WHERE status='completed' AND is_deleted=0 AND (resolution='' OR resolution IS NULL)"
        )
        await db.commit()
    except Exception:
        pass

    await _migrate_focus_refs(db)

    # chat_messages 去重索引：避免 live / recovery 并发或重复补拉导致同一消息多次入库。
    try:
        cursor = await db.execute(
            """DELETE FROM chat_messages
               WHERE message_id<>'' AND role<>'note'
                 AND id NOT IN (
                     SELECT MIN(id)
                     FROM chat_messages
                     WHERE message_id<>'' AND role<>'note'
                     GROUP BY session_key, message_id
                 )"""
        )
        await db.commit()
        if cursor.rowcount > 0:
            logger.info("[schema] chat_messages 已清理重复消息: %d 条", cursor.rowcount)
    except Exception:
        logger.exception("[schema] chat_messages 重复消息清理失败")

    try:
        await db.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_messages_session_message_id
               ON chat_messages(session_key, message_id)
               WHERE message_id<>'' AND role<>'note'"""
        )
        await db.commit()
    except Exception:
        logger.exception("[schema] chat_messages 唯一索引创建失败")


async def _table_exists(db, table: str) -> bool:
    async with db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ) as cur:
        return await cur.fetchone() is not None


async def _ensure_columns(db, table: str, columns: tuple[tuple[str, str], ...]) -> set[str]:
    if not await _table_exists(db, table):
        return set()
    async with db.execute(f"PRAGMA table_info({table})") as cur:
        existing = {str(row[1]) for row in await cur.fetchall()}
    for name, ddl in columns:
        if name not in existing:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
            existing.add(name)
            logger.info("[schema] %s 已添加 %s 列", table, name)
    await db.commit()
    return existing


async def _merge_or_rename_keyed_row(
    db,
    *,
    table: str,
    key_column: str,
    old_key: str,
    new_key: str,
) -> None:
    if old_key == new_key:
        return
    async with db.execute(
        f"SELECT 1 FROM {table} WHERE {key_column}=? LIMIT 1",
        (new_key,),
    ) as cur:
        target_exists = await cur.fetchone() is not None
    if target_exists:
        await db.execute(f"DELETE FROM {table} WHERE {key_column}=?", (old_key,))
    else:
        await db.execute(
            f"UPDATE {table} SET {key_column}=? WHERE {key_column}=?",
            (new_key, old_key),
        )


async def _migrate_focus_refs(db) -> None:
    """一次性把历史 QQ 会话 key 与会话字段迁入平台中性 focus 字段。"""
    try:
        await _ensure_columns(db, "chat_sessions", (
            ("focus_platform", "focus_platform TEXT NOT NULL DEFAULT ''"),
            ("focus_type", "focus_type TEXT NOT NULL DEFAULT ''"),
            ("focus_id", "focus_id TEXT NOT NULL DEFAULT ''"),
            ("focus_name", "focus_name TEXT NOT NULL DEFAULT ''"),
            ("focus_ref_json", "focus_ref_json TEXT NOT NULL DEFAULT '{}'"),
        ))
        await _ensure_columns(db, "bot_turns", (
            ("focus_platform", "focus_platform TEXT NOT NULL DEFAULT ''"),
            ("focus_type", "focus_type TEXT NOT NULL DEFAULT ''"),
            ("focus_id", "focus_id TEXT NOT NULL DEFAULT ''"),
            ("focus_ref_json", "focus_ref_json TEXT NOT NULL DEFAULT '{}'"),
        ))
        await _ensure_columns(db, "watcher_cycles", (
            ("focus_platform", "focus_platform TEXT NOT NULL DEFAULT ''"),
            ("focus_type", "focus_type TEXT NOT NULL DEFAULT ''"),
            ("focus_id", "focus_id TEXT NOT NULL DEFAULT ''"),
            ("focus_ref_json", "focus_ref_json TEXT NOT NULL DEFAULT '{}'"),
        ))
        await _ensure_columns(db, "bot_goals", (
            ("focus_platform", "focus_platform TEXT NOT NULL DEFAULT ''"),
            ("focus_type", "focus_type TEXT NOT NULL DEFAULT ''"),
            ("focus_id", "focus_id TEXT NOT NULL DEFAULT ''"),
            ("focus_name", "focus_name TEXT NOT NULL DEFAULT ''"),
            ("focus_ref_json", "focus_ref_json TEXT NOT NULL DEFAULT '{}'"),
        ))
        await _ensure_columns(db, "pending_archive_jobs", (
            ("focus_platform", "focus_platform TEXT NOT NULL DEFAULT ''"),
            ("focus_type", "focus_type TEXT NOT NULL DEFAULT ''"),
            ("focus_id", "focus_id TEXT NOT NULL DEFAULT ''"),
            ("focus_name", "focus_name TEXT NOT NULL DEFAULT ''"),
            ("focus_ref_json", "focus_ref_json TEXT NOT NULL DEFAULT '{}'"),
        ))
        await _ensure_columns(db, "archive_signatures", (
            ("focus_ref_json", "focus_ref_json TEXT NOT NULL DEFAULT '{}'"),
        ))

        if await _table_exists(db, "chat_sessions"):
            async with db.execute(
                "SELECT session_key, conv_type, conv_id, conv_name FROM chat_sessions"
            ) as cur:
                session_rows = await cur.fetchall()
            for session_key, conv_type, conv_id, conv_name in session_rows:
                platform, focus_type, focus_id, focus_name, new_key, focus_json = _focus_tuple_from_legacy(
                    session_key=str(session_key or ""),
                    conv_type=str(conv_type or ""),
                    conv_id=str(conv_id or ""),
                    conv_name=str(conv_name or ""),
                )
                if not new_key:
                    continue
                old_key = str(session_key or "")
                if old_key != new_key:
                    await db.execute(
                        "UPDATE chat_messages SET session_key=? WHERE session_key=?",
                        (new_key, old_key),
                    )
                    await _merge_or_rename_keyed_row(
                        db,
                        table="chat_sessions",
                        key_column="session_key",
                        old_key=old_key,
                        new_key=new_key,
                    )
                await db.execute(
                    """UPDATE chat_sessions
                       SET focus_platform=?, focus_type=?, focus_id=?, focus_name=?,
                           focus_ref_json=?, conv_type=?, conv_id=?, conv_name=?
                       WHERE session_key=?""",
                    (
                        platform,
                        focus_type,
                        focus_id,
                        focus_name,
                        focus_json,
                        focus_type,
                        focus_id,
                        focus_name,
                        new_key,
                    ),
                )

        if await _table_exists(db, "chat_messages"):
            async with db.execute("SELECT DISTINCT session_key FROM chat_messages") as cur:
                message_keys = [str(row[0] or "") for row in await cur.fetchall()]
            for old_key in message_keys:
                focus = focus_from_session_key(old_key)
                if focus is None:
                    continue
                new_key = session_key_for_focus(focus)
                if old_key != new_key:
                    await db.execute(
                        "UPDATE chat_messages SET session_key=? WHERE session_key=?",
                        (new_key, old_key),
                    )

        for table, pk, select_sql, update_sql in (
            (
                "bot_turns",
                "turn_id",
                "SELECT turn_id, conv_type, conv_id FROM bot_turns",
                "UPDATE bot_turns SET focus_platform=?, focus_type=?, focus_id=?, focus_ref_json=?, conv_type=?, conv_id=? WHERE turn_id=?",
            ),
            (
                "watcher_cycles",
                "cycle_id",
                "SELECT cycle_id, conv_type, conv_id FROM watcher_cycles",
                "UPDATE watcher_cycles SET focus_platform=?, focus_type=?, focus_id=?, focus_ref_json=?, conv_type=?, conv_id=? WHERE cycle_id=?",
            ),
        ):
            if not await _table_exists(db, table):
                continue
            async with db.execute(select_sql) as cur:
                rows = await cur.fetchall()
            for row in rows:
                row_id = str(row[0] or "")
                platform, focus_type, focus_id, _focus_name, _key, focus_json = _focus_tuple_from_legacy(
                    conv_type=str(row[1] or ""),
                    conv_id=str(row[2] or ""),
                )
                if platform and focus_type and focus_id:
                    await db.execute(
                        update_sql,
                        (platform, focus_type, focus_id, focus_json, focus_type, focus_id, row_id),
                    )

        if await _table_exists(db, "bot_goals"):
            async with db.execute("SELECT goal_id, conv_type, conv_id, conv_name FROM bot_goals") as cur:
                rows = await cur.fetchall()
            for goal_id, conv_type, conv_id, conv_name in rows:
                platform, focus_type, focus_id, focus_name, _key, focus_json = _focus_tuple_from_legacy(
                    conv_type=str(conv_type or ""),
                    conv_id=str(conv_id or ""),
                    conv_name=str(conv_name or ""),
                )
                if platform and focus_type and focus_id:
                    await db.execute(
                        """UPDATE bot_goals
                           SET focus_platform=?, focus_type=?, focus_id=?, focus_name=?,
                               focus_ref_json=?, conv_type=?, conv_id=?, conv_name=?
                           WHERE goal_id=?""",
                        (platform, focus_type, focus_id, focus_name, focus_json, focus_type, focus_id, focus_name, goal_id),
                    )

        if await _table_exists(db, "pending_archive_jobs"):
            async with db.execute(
                "SELECT job_id, conv_type, conv_id, conv_name FROM pending_archive_jobs"
            ) as cur:
                rows = await cur.fetchall()
            for job_id, conv_type, conv_id, conv_name in rows:
                platform, focus_type, focus_id, focus_name, _key, focus_json = _focus_tuple_from_legacy(
                    conv_type=str(conv_type or ""),
                    conv_id=str(conv_id or ""),
                    conv_name=str(conv_name or ""),
                )
                if platform and focus_type and focus_id:
                    await db.execute(
                        """UPDATE pending_archive_jobs
                           SET focus_platform=?, focus_type=?, focus_id=?, focus_name=?,
                               focus_ref_json=?, conv_type=?, conv_id=?, conv_name=?
                           WHERE job_id=?""",
                        (platform, focus_type, focus_id, focus_name, focus_json, focus_type, focus_id, focus_name, job_id),
                    )

        if await _table_exists(db, "archive_signatures"):
            async with db.execute("SELECT conv_key, signature FROM archive_signatures") as cur:
                rows = await cur.fetchall()
            for conv_key, _signature in rows:
                raw_key = str(conv_key or "")
                focus = focus_from_session_key(raw_key)
                if focus is None and "/" in raw_key:
                    conv_type, conv_id = raw_key.split("/", 1)
                    if conv_type and conv_id:
                        focus = FocusRef("qq", conv_type, conv_id)
                if focus is None:
                    continue
                new_key = session_key_for_focus(focus)
                focus_json = _focus_ref_json(focus)
                if raw_key != new_key:
                    async with db.execute(
                        "SELECT 1 FROM archive_signatures WHERE conv_key=? LIMIT 1",
                        (new_key,),
                    ) as cur:
                        target_exists = await cur.fetchone() is not None
                    if target_exists:
                        await db.execute("DELETE FROM archive_signatures WHERE conv_key=?", (raw_key,))
                    else:
                        await db.execute(
                            "UPDATE archive_signatures SET conv_key=?, focus_ref_json=? WHERE conv_key=?",
                            (new_key, focus_json, raw_key),
                        )
                else:
                    await db.execute(
                        "UPDATE archive_signatures SET focus_ref_json=? WHERE conv_key=?",
                        (focus_json, raw_key),
                    )

        await db.commit()
    except Exception:
        logger.exception("[schema] focus 引用迁移失败")
        raise


async def _backfill_llm_usage_from_bot_turns(db) -> None:
    """把旧 bot_turns.result_json.tokens 回填为 legacy usage 事件。"""
    import json as _json

    try:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='bot_turns'"
        ) as cur:
            bot_turns_exists = await cur.fetchone() is not None
        if not bot_turns_exists:
            return

        inserted = 0
        async with db.execute(
            "SELECT turn_id, created_at, result_json FROM bot_turns ORDER BY created_at ASC"
        ) as cur:
            async for row in cur:
                turn_id = str(row[0] or "")
                if not turn_id:
                    continue

                created_at = int(row[1] or 0)
                try:
                    result = _json.loads(row[2] or "{}")
                except Exception:
                    result = {}

                tokens = result.get("tokens") if isinstance(result, dict) else None
                input_tokens = 0
                output_tokens = 0
                if isinstance(tokens, dict):
                    try:
                        input_tokens = max(0, int(tokens.get("in") or 0))
                    except Exception:
                        input_tokens = 0
                    try:
                        output_tokens = max(0, int(tokens.get("out") or 0))
                    except Exception:
                        output_tokens = 0

                usage_available = 1 if (input_tokens or output_tokens) else 0
                total_tokens = input_tokens + output_tokens
                raw_usage_json = _json.dumps(
                    {"source": "bot_turns.result_json.tokens", "tokens": tokens or {}},
                    ensure_ascii=False,
                )
                cursor = await db.execute(
                    """INSERT OR IGNORE INTO llm_usage_events (
                           event_id, created_at, provider, model, feature, subfeature,
                           input_tokens, output_tokens, total_tokens,
                           cached_input_tokens, reasoning_output_tokens,
                           usage_available, status, raw_usage_json, legacy_turn_id
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?)""",
                    (
                        f"legacy_{turn_id}",
                        created_at,
                        "legacy",
                        "unknown",
                        "legacy",
                        "bot_turn",
                        input_tokens,
                        output_tokens,
                        total_tokens,
                        usage_available,
                        "legacy" if usage_available else "legacy_unknown",
                        raw_usage_json,
                        turn_id,
                    ),
                )
                if cursor.rowcount:
                    inserted += 1

        await db.commit()
        if inserted:
            logger.info("[schema] llm_usage_events 已回填 bot_turns: %d 条", inserted)
    except Exception:
        logger.exception("[schema] llm_usage_events 历史回填失败")
        raise


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
                "SELECT account_uid FROM entities WHERE platform='qq' AND platform_id=? AND is_bot=1",
                (qq_id,),
            ) as cur2:
                existing = await cur2.fetchone()
            if not existing:
                profile_id = str(uuid.uuid4())
                account_uid = str(uuid.uuid4())
                await db.execute(
                    "INSERT OR IGNORE INTO entity_profiles (profile_id, created_at, updated_at) VALUES (?,?,?)",
                    (profile_id, now, now),
                )
                await db.execute(
                    """INSERT OR IGNORE INTO entities
                       (account_uid, profile_id, platform, platform_id, nickname, is_bot, created_at, updated_at)
                       VALUES (?,?,?,?,?,1,?,?)""",
                    (account_uid, profile_id, "qq", qq_id, nickname, now, now),
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


async def _migrate_rename_tables(db) -> None:
    """平滑迁移：将旧表名 persons/accounts 和旧列名 person_id 重命名为新名称。

    设计说明
    --------
    本次改名仅是"准确化命名"，不改变任何数据或表结构：
      persons  → entity_profiles  （EntityProfile：AI 的主观认知侧写）
      accounts → entities         （Entity：客观可观测的平台实体）
      persons.person_id  → entity_profiles.profile_id
      accounts.person_id → entities.profile_id

    使用 _migrations 表作幂等哨兵，防止每次启动重跑。
    要求 SQLite >= 3.25（RENAME COLUMN，2018 年 9 月发布）。
    """
    MIGRATION_KEY = "rename_persons_accounts_v1"
    async with db.execute(
        "SELECT name FROM _migrations WHERE name=?", (MIGRATION_KEY,)
    ) as cur:
        if await cur.fetchone():
            return  # 已执行过，跳过

    try:
        # 0. 兼容场景：函数调用模式分支的 DB 已存在 `accounts`/`persons` 旧表，
        #    而本次 init_db 又通过 `CREATE TABLE IF NOT EXISTS` 创建了空的
        #    `entities`/`entity_profiles`。直接 ALTER ... RENAME TO 会因目标已
        #    存在而失败。
        #    由于本函数到这里说明迁移哨兵尚未写入，新表里的内容只可能是
        #    上次失败迁移残留的脏数据（本次 startup 的 upsert_* 调用），可以
        #    安全丢弃，再让 RENAME 把旧表搬到新名下。
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='persons'"
        ) as cur:
            _has_persons = await cur.fetchone() is not None
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='accounts'"
        ) as cur:
            _has_accounts = await cur.fetchone() is not None

        if _has_persons or _has_accounts:
            # DROP 顺序：先 entities 后 entity_profiles（前者外键引用后者）。
            # 同时临时关掉 FK 检查，避免对 memberships 等已有外键造成阻塞。
            await db.commit()
            await db.execute("PRAGMA foreign_keys=OFF")
            try:
                if _has_accounts:
                    async with db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='entities'"
                    ) as cur:
                        if await cur.fetchone():
                            await db.execute("DROP TABLE entities")
                            logger.info("[migrate] 丢弃同名脏新表以便迁移: entities")
                if _has_persons:
                    async with db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='entity_profiles'"
                    ) as cur:
                        if await cur.fetchone():
                            await db.execute("DROP TABLE entity_profiles")
                            logger.info("[migrate] 丢弃同名脏新表以便迁移: entity_profiles")
                await db.commit()
            finally:
                await db.execute("PRAGMA foreign_keys=ON")

        # 1. 重命名表 persons → entity_profiles
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='persons'"
        ) as cur:
            if await cur.fetchone():
                await db.execute("ALTER TABLE persons RENAME TO entity_profiles")
                logger.info("[migrate] persons → entity_profiles")

        # 2. 重命名表 accounts → entities
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='accounts'"
        ) as cur:
            if await cur.fetchone():
                await db.execute("ALTER TABLE accounts RENAME TO entities")
                logger.info("[migrate] accounts → entities")

        # 3. 重命名列 entity_profiles.person_id → profile_id（SQLite 3.25+）
        async with db.execute("PRAGMA table_info(entity_profiles)") as cur:
            cols = [row[1] for row in await cur.fetchall()]
        if "person_id" in cols and "profile_id" not in cols:
            await db.execute(
                "ALTER TABLE entity_profiles RENAME COLUMN person_id TO profile_id"
            )
            logger.info("[migrate] entity_profiles.person_id → profile_id")

        # 4. 重命名列 entities.person_id → profile_id
        async with db.execute("PRAGMA table_info(entities)") as cur:
            cols = [row[1] for row in await cur.fetchall()]
        if "person_id" in cols and "profile_id" not in cols:
            await db.execute(
                "ALTER TABLE entities RENAME COLUMN person_id TO profile_id"
            )
            logger.info("[migrate] entities.person_id → profile_id")

        # 5. 重命名 merge_suggestions.person_id_a/b → profile_id_a/b
        async with db.execute("PRAGMA table_info(merge_suggestions)") as cur:
            cols = [row[1] for row in await cur.fetchall()]
        if "person_id_a" in cols:
            await db.execute(
                "ALTER TABLE merge_suggestions RENAME COLUMN person_id_a TO profile_id_a"
            )
            logger.info("[migrate] merge_suggestions.person_id_a → profile_id_a")
        if "person_id_b" in cols:
            await db.execute(
                "ALTER TABLE merge_suggestions RENAME COLUMN person_id_b TO profile_id_b"
            )
            logger.info("[migrate] merge_suggestions.person_id_b → profile_id_b")

        await db.execute(
            "INSERT INTO _migrations (name, applied_at) VALUES (?,?)",
            (MIGRATION_KEY, _ms()),
        )
        await db.commit()
        logger.info("[migrate] 表/列重命名迁移完成 (%s)", MIGRATION_KEY)

    except Exception:
        logger.exception("[migrate] 表/列重命名迁移失败，已有数据保持原状")



async def upsert_chat_session(
    session_key: str,
    conv_type: str,
    conv_id: str,
    conv_name: str = "",
    temp_source_group_id: str = "",
    temp_source_group_name: str = "",
) -> None:
    """写入/更新会话元信息，同时更新 last_active_at。"""
    now = _ms()
    platform, focus_type, focus_id, focus_name, focus_key, focus_json = _focus_tuple_from_legacy(
        session_key=session_key,
        conv_type=conv_type,
        conv_id=conv_id,
        conv_name=conv_name,
    )
    if focus_key:
        session_key = focus_key
        conv_type = focus_type
        conv_id = focus_id
        conv_name = focus_name
    async with _connect() as db:
        await db.execute(
            """INSERT INTO chat_sessions (
                   session_key, focus_platform, focus_type, focus_id, focus_name, focus_ref_json,
                   conv_type, conv_id, conv_name,
                   temp_source_group_id, temp_source_group_name, last_active_at
               )
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(session_key) DO UPDATE SET
                   focus_platform=excluded.focus_platform,
                   focus_type=excluded.focus_type,
                   focus_id=excluded.focus_id,
                   focus_name=excluded.focus_name,
                   focus_ref_json=excluded.focus_ref_json,
                   conv_type=excluded.conv_type,
                   conv_id=excluded.conv_id,
                   conv_name=excluded.conv_name,
                   temp_source_group_id=excluded.temp_source_group_id,
                   temp_source_group_name=excluded.temp_source_group_name,
                   last_active_at=excluded.last_active_at""",
            (
                session_key,
                platform,
                focus_type,
                focus_id,
                focus_name,
                focus_json,
                conv_type,
                conv_id,
                conv_name,
                temp_source_group_id,
                temp_source_group_name,
                now,
            ),
        )
        await db.commit()


async def load_chat_sessions() -> list[dict]:
    """返回所有已注册的会话元信息，按 last_active_at 倒序。"""
    async with _connect() as db:
        async with db.execute(
            "SELECT session_key, focus_platform, focus_type, focus_id, focus_name, focus_ref_json,"
            " conv_type, conv_id, conv_name, temp_source_group_id, temp_source_group_name FROM chat_sessions"
            " ORDER BY last_active_at DESC"
        ) as cur:
            rows = await cur.fetchall()
    out: list[dict] = []
    for r in rows:
        platform = str(r[1] or "")
        focus_type = str(r[2] or "")
        focus_id = str(r[3] or "")
        focus_name = str(r[4] or "")
        focus_key = str(r[0] or "")
        if not (platform and focus_type and focus_id):
            platform, focus_type, focus_id, focus_name, focus_key, _focus_json = _focus_tuple_from_legacy(
                session_key=str(r[0] or ""),
                conv_type=str(r[6] or ""),
                conv_id=str(r[7] or ""),
                conv_name=str(r[8] or ""),
            )
        out.append({
            "session_key": focus_key or str(r[0] or ""),
            "focus_platform": platform,
            "focus_type": focus_type,
            "focus_id": focus_id,
            "focus_name": focus_name,
            "focus_ref_json": r[5] or "{}",
            "conv_type": focus_type,
            "conv_id": focus_id,
            "conv_name": focus_name,
            "temp_source_group_id": r[9],
            "temp_source_group_name": r[10],
        })
    return out


async def get_chat_message_edge(session_key: str, *, newest: bool = True) -> dict | None:
    """返回会话最早或最新的一条真实聊天消息（跳过 note / 空/内部 message_id）。"""
    order = CHAT_MESSAGE_ORDER_DESC_SQL if newest else CHAT_MESSAGE_ORDER_ASC_SQL
    async with _connect() as db:
        async with db.execute(
            f"""SELECT id, message_id, timestamp
                   FROM chat_messages
                   WHERE session_key=?
                     AND message_id<>''
                     AND role<>'note'
                     AND COALESCE(delivery_state, '')=''
                     AND content_type<>'send_failed'
                     AND message_id NOT LIKE 'pending_%'
                     AND message_id NOT LIKE 'failed_%'
                     AND message_id NOT LIKE 'offline_%'
                   ORDER BY {order}
                   LIMIT 1""",
            (session_key,),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return None
    return {"id": int(row[0]), "message_id": str(row[1]), "timestamp": str(row[2] or "")}


async def get_existing_chat_message_ids(session_key: str, message_ids: list[str]) -> set[str]:
    """返回指定会话中已存在的 message_id 集合。"""
    normalized = [str(mid).strip() for mid in message_ids if str(mid).strip()]
    if not normalized:
        return set()

    placeholders = ",".join("?" for _ in normalized)
    async with _connect() as db:
        async with db.execute(
            f"""SELECT message_id
                   FROM chat_messages
                   WHERE session_key=? AND message_id IN ({placeholders})""",
            [session_key, *normalized],
        ) as cur:
            rows = await cur.fetchall()
    return {str(row[0]) for row in rows if row and str(row[0]).strip()}


async def save_chat_message(session_key: str, entry: dict) -> None:
    """将一条上下文条目写入 chat_messages 表。"""
    import json as _json
    now = _ms()
    reply_to = str(entry.get("reply_to", "") or "")
    async with _connect() as db:
        await db.execute(
            """INSERT OR IGNORE INTO chat_messages
               (session_key, role, message_id, sender_id, sender_name,
                sender_card, sender_nickname, sender_role,
                sender_title, sender_level, timestamp, reply_to,
                content, content_type, content_segments, images,
                delivery_state, delivery_error, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                session_key,
                entry.get("role", ""),
                entry.get("message_id", ""),
                entry.get("sender_id", ""),
                entry.get("sender_name", ""),
                entry.get("sender_card", ""),
                entry.get("sender_nickname", ""),
                entry.get("sender_role", ""),
                entry.get("sender_title", ""),
                entry.get("sender_level", ""),
                entry.get("timestamp", ""),
                reply_to,
                entry.get("content", ""),
                entry.get("content_type", "text"),
                _json.dumps(entry.get("content_segments", []), ensure_ascii=False),
                _json.dumps(entry.get("images", []), ensure_ascii=False),
                entry.get("delivery_state", ""),
                entry.get("delivery_error", ""),
                now,
            ),
        )
        if reply_to and entry.get("message_id"):
            await db.execute(
                """UPDATE chat_messages
                   SET reply_to=?
                   WHERE session_key=? AND message_id=? AND (reply_to='' OR reply_to IS NULL)""",
                (reply_to, session_key, entry.get("message_id", "")),
            )
        await db.commit()


async def update_chat_message_id(session_key: str, old_message_id: str, new_message_id: str) -> None:
    """回填真实 QQ message_id（发送后 QQ adapter 返回真实 ID 时调用）。"""
    async with _connect() as db:
        await db.execute(
            """UPDATE chat_messages
               SET message_id=?, delivery_state='', delivery_error=''
               WHERE session_key=? AND message_id=?""",
            (new_message_id, session_key, old_message_id),
        )
        await db.commit()


async def update_chat_message_delivery_state(
    session_key: str,
    message_id: str,
    delivery_state: str,
    delivery_error: str = "",
) -> None:
    """更新本地发送消息的投递状态，不改变内部占位 message_id。"""
    async with _connect() as db:
        await db.execute(
            """UPDATE chat_messages
               SET delivery_state=?, delivery_error=?
               WHERE session_key=? AND message_id=?""",
            (delivery_state, delivery_error, session_key, message_id),
        )
        await db.commit()


async def update_chat_message_recalled(
    message_id: str,
    content: str,
    timestamp: str,
    content_segments: list[dict] | None = None,
    session_key: str = "",
) -> bool:
    """将数据库中的消息更新为撤回状态，与内存中 mark_message_recalled 保持同步。

    返回 True 表示找到并更新了至少一条记录。
    """
    import json as _json
    mid = str(message_id or "").strip()
    if not mid:
        return False
    segments_json = _json.dumps(content_segments or [], ensure_ascii=False)
    async with _connect() as db:
        where = "message_id=?"
        params: list[object] = [
            timestamp,
            content,
            segments_json,
            mid,
        ]
        if session_key:
            where += " AND session_key=?"
            params.append(session_key)
        cursor = await db.execute(
            f"""UPDATE chat_messages
               SET role='note',
                   timestamp=?,
                   content=?,
                   content_type='recall',
                   content_segments=?,
                   images='[]',
                   reply_to='',
                   sender_id='',
                   sender_name='',
                   sender_card='',
                   sender_nickname='',
                   sender_role='',
                   sender_title='',
                   sender_level=''
               WHERE {where}""",
            params,
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
            """SELECT role, message_id, sender_id, sender_name,
                      sender_card, sender_nickname, sender_role,
                      sender_title, sender_level, timestamp, reply_to,
                      content, content_type, content_segments,
                      delivery_state, delivery_error
               FROM chat_messages
               WHERE message_id=?
               LIMIT 1""",
            (message_id,),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return None
    result: dict = {
        "role": row[0],
        "message_id": row[1],
        "sender_id": row[2],
        "sender_name": row[3],
        "sender_card": row[4],
        "sender_nickname": row[5],
        "sender_role": row[6],
        "sender_title": row[7],
        "sender_level": row[8],
        "timestamp": row[9],
        "content": row[11],
        "content_type": row[12],
        "content_segments": _json.loads(row[13] or "[]"),
    }
    if reply_to := str(row[10] or ""):
        result["reply_to"] = reply_to
    if delivery_state := str(row[14] or ""):
        result["delivery_state"] = delivery_state
    if delivery_error := str(row[15] or ""):
        result["delivery_error"] = delivery_error
    return result


async def is_bot_chat_message(session_key: str, message_id: str) -> bool:
    """Return whether ``message_id`` belongs to a bot message in this session."""
    mid = str(message_id or "").strip()
    if not mid:
        return False
    async with _connect() as db:
        async with db.execute(
            """SELECT 1
               FROM chat_messages
               WHERE session_key=? AND message_id=? AND role='bot'
               LIMIT 1""",
            (session_key, mid),
        ) as cur:
            row = await cur.fetchone()
    return row is not None


async def load_chat_messages(session_key: str, limit: int = 50) -> list[dict]:
    """加载指定会话最近 limit 条聊天记录，按时间正序返回。"""
    import json as _json
    async with _connect() as db:
        async with db.execute(
            f"""SELECT role, message_id, sender_id, sender_name,
                      sender_card, sender_nickname, sender_role,
                      sender_title, sender_level, timestamp, reply_to,
                      content, content_type, content_segments, images,
                      delivery_state, delivery_error
               FROM (
                   SELECT * FROM chat_messages
                   WHERE session_key=?
                   ORDER BY {CHAT_MESSAGE_ORDER_DESC_SQL}
                   LIMIT ?
               ) sub
               ORDER BY {CHAT_MESSAGE_ORDER_ASC_SQL}""",
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
            "sender_card": r[4],
            "sender_nickname": r[5],
            "sender_role": r[6],
            "sender_title": r[7],
            "sender_level": r[8],
            "timestamp": r[9],
            "content": r[11],
            "content_type": r[12],
            "content_segments": _json.loads(r[13] or "[]"),
        }
        if reply_to := str(r[10] or ""):
            entry["reply_to"] = reply_to
        images = _json.loads(r[14] or "[]")
        if images:
            entry["images"] = images
        if delivery_state := str(r[15] or ""):
            entry["delivery_state"] = delivery_state
        if delivery_error := str(r[16] or ""):
            entry["delivery_error"] = delivery_error
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
    platform, focus_type, focus_id, _focus_name, _focus_key, focus_json = _focus_tuple_from_legacy(
        conv_type=conv_type,
        conv_id=conv_id,
    )
    async with _connect() as db:
        await db.execute(
            """INSERT INTO watcher_cycles (
                   cycle_id, created_at, focus_platform, focus_type, focus_id, focus_ref_json,
                   conv_type, conv_id, result_json
               )
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                cycle_id,
                now,
                platform,
                focus_type,
                focus_id,
                focus_json,
                focus_type or conv_type,
                focus_id or conv_id,
                _json.dumps(result, ensure_ascii=False),
            ),
        )
        await db.commit()
    logger.debug("已保存 watcher_cycle: cycle_id=%s focus=%s/%s", cycle_id, focus_type or conv_type, focus_id or conv_id)


async def load_last_watcher_cycle(
    conv_type: str,
    conv_id: str,
) -> tuple[dict | None, str | None]:
    """加载指定会话最近一轮 watcher 结果，返回 (result, created_at_iso)。"""
    import json as _json
    platform, focus_type, focus_id, _focus_name, _focus_key, _focus_json = _focus_tuple_from_legacy(
        conv_type=conv_type,
        conv_id=conv_id,
    )
    async with _connect() as db:
        async with db.execute(
            """SELECT result_json, created_at FROM watcher_cycles
               WHERE (focus_platform=? AND focus_type=? AND focus_id=?)
                  OR (conv_type=? AND conv_id=?)
               ORDER BY created_at DESC LIMIT 1""",
            (platform, focus_type, focus_id, conv_type, conv_id),
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

async def load_recent_bot_turns(limit: int = 20, *, before: int | None = None) -> list[dict]:
    """加载最近 limit 轮 bot 意识日志（全局，倒序），供焦点/Agent 视图消费。"""
    import json as _json
    limit = max(1, min(int(limit or 20), 100))
    where = ""
    params: tuple = (limit,)
    if before is not None and int(before) > 0:
        where = "WHERE b.created_at < ?"
        params = (int(before), limit)
    async with _connect() as db:
        async with db.execute(
            f"""SELECT
                   b.turn_id,
                   b.created_at,
                   COALESCE(NULLIF(b.focus_type, ''), b.conv_type) AS focus_type,
                   COALESCE(NULLIF(b.focus_id, ''), b.conv_id) AS focus_id,
                   b.result_json,
                   b.tool_calls,
                   b.world_xml,
                   COALESCE(s.session_key, CASE
                       WHEN COALESCE(NULLIF(b.focus_type, ''), b.conv_type)<>'' AND COALESCE(NULLIF(b.focus_id, ''), b.conv_id)<>''
                       THEN 'qq:' || COALESCE(NULLIF(b.focus_type, ''), b.conv_type) || ':' || COALESCE(NULLIF(b.focus_id, ''), b.conv_id)
                       ELSE ''
                   END) AS session_key,
                   COALESCE(NULLIF(s.focus_name, ''), s.conv_name, '') AS conv_name
               FROM bot_turns AS b
               LEFT JOIN chat_sessions AS s
                 ON s.focus_platform = COALESCE(NULLIF(b.focus_platform, ''), 'qq')
                AND s.focus_type = COALESCE(NULLIF(b.focus_type, ''), b.conv_type)
                AND s.focus_id = COALESCE(NULLIF(b.focus_id, ''), b.conv_id)
               {where}
               ORDER BY b.created_at DESC
               LIMIT ?""",
            params,
        ) as cur:
            rows = await cur.fetchall()
    result = []
    for r in rows:
        try:
            res_json = _json.loads(r[4]) if r[4] else {}
        except Exception:
            res_json = {}
        try:
            tool_calls = _json.loads(r[5]) if r[5] else []
        except Exception:
            tool_calls = []
        result.append({
            "turn_id": r[0],
            "created_at": int(r[1]),
            "conv_type": r[2],
            "conv_id": r[3],
            "result": res_json,
            "tool_calls": tool_calls,
            "world_xml": r[6] or "",
            "session_key": r[7] or (f"qq:{r[2]}:{r[3]}" if r[2] and r[3] else ""),
            "conv_name": r[8] or "",
        })
    return result


async def save_bot_turn(
    turn_id: str,
    conv_type: str,
    conv_id: str,
    result: dict,
    tool_calls_log: list,
    world_xml: str = "",
) -> None:
    """持久化一轮 LLM 输出及工具调用日志。"""
    import json as _json
    now = _ms()
    platform, focus_type, focus_id, _focus_name, _focus_key, focus_json = _focus_tuple_from_legacy(
        conv_type=conv_type,
        conv_id=conv_id,
    )
    async with _connect() as db:
        await db.execute(
            """INSERT INTO bot_turns (
                   turn_id, created_at, focus_platform, focus_type, focus_id, focus_ref_json,
                   conv_type, conv_id, result_json, tool_calls, world_xml
               )
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                turn_id,
                now,
                platform,
                focus_type,
                focus_id,
                focus_json,
                focus_type or conv_type,
                focus_id or conv_id,
                _json.dumps(result, ensure_ascii=False),
                _json.dumps(tool_calls_log, ensure_ascii=False),
                str(world_xml or ""),
            ),
        )
        await db.commit()
    logger.debug("已保存 bot_turn: turn_id=%s focus=%s/%s", turn_id, focus_type or conv_type, focus_id or conv_id)


def save_llm_usage_event_sync(
    *,
    event_id: str | None = None,
    created_at: int | None = None,
    provider: str = "",
    model: str = "",
    feature: str = "",
    subfeature: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    cached_input_tokens: int = 0,
    reasoning_output_tokens: int = 0,
    usage_available: bool = False,
    status: str = "",
    raw_usage_json: str = "{}",
    legacy_turn_id: str = "",
) -> bool:
    """同步记录一条 LLM usage 事件，供 provider 的同步线程调用。"""
    event_id = event_id or uuid.uuid4().hex
    created_at = int(created_at if created_at is not None else _ms())
    if total_tokens <= 0 and (input_tokens or output_tokens):
        total_tokens = max(0, int(input_tokens)) + max(0, int(output_tokens))

    try:
        with sqlite3.connect(DB_PATH, timeout=8) as db:
            db.executescript(_LLM_USAGE_SCHEMA_SQL)
            db.execute(
                """INSERT OR IGNORE INTO llm_usage_events (
                       event_id, created_at, provider, model, feature, subfeature,
                       input_tokens, output_tokens, total_tokens,
                       cached_input_tokens, reasoning_output_tokens,
                       usage_available, status, raw_usage_json, legacy_turn_id
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    created_at,
                    str(provider or ""),
                    str(model or ""),
                    str(feature or ""),
                    str(subfeature or ""),
                    max(0, int(input_tokens or 0)),
                    max(0, int(output_tokens or 0)),
                    max(0, int(total_tokens or 0)),
                    max(0, int(cached_input_tokens or 0)),
                    max(0, int(reasoning_output_tokens or 0)),
                    1 if usage_available else 0,
                    str(status or ""),
                    str(raw_usage_json or "{}"),
                    str(legacy_turn_id or ""),
                ),
            )
            db.commit()
        return True
    except Exception:
        logger.warning("[usage] 保存 LLM token 用量失败", exc_info=True)
        return False


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
            "SELECT platform_id, nickname FROM entities WHERE platform='qq' AND is_bot=1 LIMIT 1"
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
            "SELECT account_uid FROM entities WHERE platform='qq' AND platform_id=? AND is_bot=1",
            (qq_id,),
        ) as cur:
            row = await cur.fetchone()
        if row:
            await db.execute(
                "UPDATE entities SET nickname=?, updated_at=? WHERE account_uid=?",
                (nickname, now, row[0]),
            )
        else:
            profile_id = str(uuid.uuid4())
            account_uid = str(uuid.uuid4())
            await db.execute(
                "INSERT OR IGNORE INTO entity_profiles (profile_id, created_at, updated_at) VALUES (?,?,?)",
                (profile_id, now, now),
            )
            await db.execute(
                """INSERT INTO entities
                   (account_uid, profile_id, platform, platform_id, nickname, is_bot, created_at, updated_at)
                   VALUES (?,?,?,?,?,1,?,?)
                   ON CONFLICT(platform, platform_id) DO UPDATE SET
                       nickname=excluded.nickname, updated_at=excluded.updated_at""",
                (account_uid, profile_id, "qq", qq_id, nickname, now, now),
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
    """写入/更新客观实体（entities），不存在则自动创建对应的 entity_profiles 行，返回 account_uid。

    entity_profiles 行代表 AI 对该实体的主观认知侧写（represents 关系）；
    entities 行代表该平台账号的客观事实。
    """
    now = _ms()
    async with _connect() as db:
        async with db.execute(
            "SELECT account_uid FROM entities WHERE platform=? AND platform_id=?",
            (platform, platform_id),
        ) as cur:
            row = await cur.fetchone()
        if row:
            account_uid = str(row[0])
            # nickname/avatar 只在调用方明确传值时才覆写，空字符串不覆盖已有昵称
            if nickname or avatar:
                await db.execute(
                    """UPDATE entities SET nickname=COALESCE(?,nickname),
                           avatar=COALESCE(?,avatar), last_seen_at=?, updated_at=?
                       WHERE account_uid=?""",
                    (nickname or None, avatar or None, now, now, account_uid),
                )
            else:
                await db.execute(
                    "UPDATE entities SET last_seen_at=?, updated_at=? WHERE account_uid=?",
                    (now, now, account_uid),
                )
        else:
            profile_id = str(uuid.uuid4())
            account_uid = str(uuid.uuid4())
            await db.execute(
                "INSERT INTO entity_profiles (profile_id, last_seen_at, created_at, updated_at) VALUES (?,?,?,?)",
                (profile_id, now, now, now),
            )
            await db.execute(
                """INSERT INTO entities
                   (account_uid, profile_id, platform, platform_id, nickname, avatar,
                    last_seen_at, created_at, updated_at, extra)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (account_uid, profile_id, platform, platform_id,
                 nickname or None, avatar or None, now, now, now, extra),
            )
        await db.commit()
    return account_uid


# ── 群成员关系 ────────────────────────────────────────────

async def upsert_membership(
    platform: str,
    platform_id: str,
    group_id: str,
    nickname: str = "",
    cardname: str = "",
    title: str | None = None,
    title_expire_time: int | None = None,
    level: str | None = None,
    permission_level: str = "member",
    joined_at: int | None = None,
) -> None:
    """写入/更新群成员关系。账号或群组不存在时会自动创建占位记录。

    nickname 会透传给 upsert_account，确保群聊路径也能写入昵称，
    避免用空值覆盖已有 nickname（upsert_account 内部做了 or None 保护）。
    """
    now = _ms()
    account_uid = await upsert_account(platform, platform_id, nickname=nickname)
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
                title_expire_time, level, permission_level, joined_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(account_uid, group_uid) DO UPDATE SET
                   cardname=excluded.cardname,
                   title=CASE WHEN ? THEN excluded.title ELSE memberships.title END,
                   title_expire_time=CASE WHEN ? THEN excluded.title_expire_time ELSE memberships.title_expire_time END,
                   level=CASE WHEN ? THEN excluded.level ELSE memberships.level END,
                   permission_level=excluded.permission_level,
                   updated_at=excluded.updated_at""",
            (membership_id, account_uid, group_uid,
             cardname or None,
             "" if title is None else str(title),
             int(title_expire_time or 0),
             "" if level is None else str(level),
             permission_level, joined_at, now,
             title is not None,
             title_expire_time is not None,
             level is not None),
        )
        await db.commit()


# ── 实体侧写更新 ──────────────────────────────────────────

async def update_person_profile(
    platform_id: str,
    platform: str = "qq",
    sex: str | None = None,
    age: int | None = None,
    area: str | None = None,
    notes: str | None = None,
) -> bool:
    """更新 entity_profiles 表的主观侧写字段，通过 platform_id 定位对应 profile_id。

    只更新非 None 的字段，返回是否找到了对应实体。
    """
    now = _ms()
    async with _connect() as db:
        # 通过 platform + platform_id 在 entities 表找到 profile_id
        async with db.execute(
            "SELECT profile_id FROM entities WHERE platform=? AND platform_id=?",
            (platform, platform_id),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return False
        profile_id = row[0]

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
        values = [v for _, v in updates] + [now, profile_id]
        await db.execute(
            f"UPDATE entity_profiles SET {set_clause}, updated_at=? WHERE profile_id=?",
            values,
        )
        await db.commit()
    return True


# ── 实体泼溅合并建议 ─────────────────────────────────────

async def upsert_merge_suggestion(
    profile_id_a: str,
    profile_id_b: str,
    similarity: float,
    reason: str,
) -> str:
    """写入合并建议（幂等：相同 pair 的 pending 建议重复写入时更新 similarity/reason）。
    自动规范化 profile_id 顺序（小值在前），避免 (A,B)/(B,A) 重复建议。
    返回 suggestion_id。
    """
    import uuid
    a, b = (
        (profile_id_a, profile_id_b)
        if profile_id_a < profile_id_b
        else (profile_id_b, profile_id_a)
    )
    now = _ms()
    async with _connect() as db:
        async with db.execute(
            "SELECT suggestion_id FROM merge_suggestions WHERE profile_id_a=? AND profile_id_b=? AND status='pending'",
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
                "INSERT INTO merge_suggestions (suggestion_id, profile_id_a, profile_id_b, similarity, reason, created_at)"
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
                   JOIN entities a ON a.account_uid = m.account_uid
                   WHERE a.platform=? AND a.platform_id=? AND m.group_uid=?""",
                (platform, platform_id, group_uid),
            ) as cur:
                row = await cur.fetchone()
            if row:
                return str(row[0] or row[1] or platform_id)
        async with db.execute(
            "SELECT nickname FROM entities WHERE platform=? AND platform_id=?",
            (platform, platform_id),
        ) as cur:
            row = await cur.fetchone()
    return str(row[0] if row and row[0] else platform_id)


async def get_group_member_display_info(
    platform: str,
    platform_id: str,
    group_id: str,
) -> dict[str, object]:
    """返回群成员显示信息，包含 card / nickname，并提供 display 回退。"""
    platform_id = str(platform_id or "")
    if not platform_id:
        return {"id": "", "card": "", "nickname": "", "permission_level": "", "title": "", "level": "", "display": ""}
    card = ""
    nickname = ""
    permission_level = ""
    title = ""
    level = ""
    async with _connect() as db:
        group_uid = f"grp_{platform}_{group_id}"
        async with db.execute(
            """SELECT m.cardname, a.nickname, m.permission_level, m.title, m.level
               FROM memberships m
               JOIN entities a ON a.account_uid = m.account_uid
               WHERE a.platform=? AND a.platform_id=? AND m.group_uid=?""",
            (platform, platform_id, group_uid),
        ) as cur:
            row = await cur.fetchone()
        if row:
            card = str(row[0] or "")
            nickname = str(row[1] or "")
            permission_level = str(row[2] or "")
            title = str(row[3] or "")
            level = str(row[4] or "")
        if not nickname:
            async with db.execute(
                "SELECT nickname FROM entities WHERE platform=? AND platform_id=?",
                (platform, platform_id),
            ) as cur:
                row = await cur.fetchone()
            if row:
                nickname = str(row[0] or "")
    return {
        "id": platform_id,
        "card": card,
        "nickname": nickname,
        "permission_level": permission_level,
        "title": title,
        "level": level,
        "display": card or nickname or platform_id,
    }


async def get_nicknames_by_qq_ids(qq_ids: list[str]) -> dict[str, str]:
    """批量查询 platform_id → nickname。空字符串与不存在统一回退为空。"""
    qq_ids = [str(x) for x in qq_ids if x]
    if not qq_ids:
        return {}
    async with _connect() as db:
        ph = ",".join("?" * len(qq_ids))
        async with db.execute(
            f"SELECT platform_id, nickname FROM entities "
            f"WHERE platform='qq' AND platform_id IN ({ph})",
            qq_ids,
        ) as cur:
            return {str(r[0]): (r[1] or "") for r in await cur.fetchall()}


# ── MemoryEvents（Neo-Davidsonian 事件层）──────────────────────────────

# 8 个通用主题角色（对照 entity system / Davidsonian 通用集）
VALID_ROLES: frozenset[str] = frozenset({
    "agent", "patient", "theme", "recipient",
    "instrument", "location", "time", "attribute",
})

VALID_CONTEXT_TYPES: frozenset[str] = frozenset({
    "episodic", "hypothetical",
})

VALID_MODALITY: frozenset[str] = frozenset({"actual", "hypothetical", "possible"})


async def write_event(
    event_type: str,
    summary: str,
    summary_tok: str = "",
    modality: str = "actual",
    confidence: float = 0.6,
    context_type: str = "episodic",
    recall_scope: str = "global",
    source: str = "",
    reason: str = "",
    conv_type: str = "",
    conv_id: str = "",
    conv_name: str = "",
    roles: list[dict] | None = None,
    supersedes: int | None = None,
) -> int:
    """写入事件 + 角色边到事件图，返回新事件 id。"""
    from memory.repo.events import write_event as _impl

    return await _impl(
        event_type=event_type,
        summary=summary,
        summary_tok=summary_tok,
        modality=modality,
        confidence=confidence,
        context_type=context_type,
        recall_scope=recall_scope,
        source=source,
        reason=reason,
        conv_type=conv_type,
        conv_id=conv_id,
        conv_name=conv_name,
        roles=roles,
        supersedes=supersedes,
    )


async def merge_event_occurrence(event_id: int) -> bool:
    """同一事实的再次观测：occurrences+1, 置信度小幅上涨。"""
    from memory.repo.events import merge_event_occurrence as _impl
    return await _impl(event_id)


async def load_events_for_recall(
    sender_entity: str = "",
    context_scope: str = "",
    limit: int = 6,
    query: str = "",
) -> list[dict]:
    """加载与本轮场景相关的事件，附带其所有角色边。"""
    from memory.repo.events import load_events_for_recall as _impl

    return await _impl(
        sender_entity=sender_entity,
        context_scope=context_scope,
        limit=limit,
        query=query,
    )


async def soft_delete_event(event_id: int) -> bool:
    """软删除一个事件（角色边保留，便于审计）。"""
    async with _connect() as db:
        cur = await db.execute(
            "UPDATE MemoryEvents SET is_deleted=1 WHERE event_id=? AND is_deleted=0",
            (event_id,),
        )
        await db.commit()
        return cur.rowcount > 0


# ── 活跃目标 ──────────────────────────────────────────────

async def write_goal(
    goal_id: str,
    title: str,
    content: str,
    reason: str,
    conv_type: str = "",
    conv_id: str = "",
    conv_name: str = "",
    status: str = "active",
    resolution: str = "",
) -> None:
    """写入一条新目标。"""
    now = _ms()
    platform, focus_type, focus_id, focus_name, _focus_key, focus_json = _focus_tuple_from_legacy(
        conv_type=conv_type,
        conv_id=conv_id,
        conv_name=conv_name,
    )
    async with _connect() as db:
        await db.execute(
            """INSERT INTO bot_goals
               (goal_id, created_at, updated_at, title, content, reason,
                focus_platform, focus_type, focus_id, focus_name, focus_ref_json,
                conv_type, conv_id, conv_name, status, resolution, is_deleted)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
            (
                goal_id,
                now,
                now,
                title,
                content,
                reason,
                platform,
                focus_type,
                focus_id,
                focus_name,
                focus_json,
                focus_type or conv_type,
                focus_id or conv_id,
                focus_name or conv_name,
                status,
                resolution,
            ),
        )
        await db.commit()
    logger.debug("已写入目标: goal_id=%s", goal_id)


async def soft_delete_goal(goal_id: str) -> bool:
    """软删除一条目标，返回是否找到并删除。"""
    async with _connect() as db:
        cur = await db.execute(
            "UPDATE bot_goals SET is_deleted=1, updated_at=? WHERE goal_id=? AND is_deleted=0",
            (_ms(), goal_id),
        )
        await db.commit()
    return cur.rowcount > 0


async def resolve_goal(goal_id: str, resolution: str) -> bool:
    """将目标标记为 resolved，并记录 resolution。"""
    async with _connect() as db:
        cur = await db.execute(
            "UPDATE bot_goals SET status='resolved', resolution=?, updated_at=? "
            "WHERE goal_id=? AND is_deleted=0 AND status='active'",
            (resolution, _ms(), goal_id),
        )
        await db.commit()
    return cur.rowcount > 0


async def load_goals(limit: int = 10) -> list[dict]:
    """加载最近 limit 条未删除的活跃目标，按 created_at 正序（最旧在前）。"""
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT goal_id, created_at, updated_at, title, content, reason,
                      COALESCE(NULLIF(focus_platform, ''), 'qq') AS focus_platform,
                      COALESCE(NULLIF(focus_type, ''), conv_type) AS focus_type,
                      COALESCE(NULLIF(focus_id, ''), conv_id) AS focus_id,
                      COALESCE(NULLIF(focus_name, ''), conv_name) AS focus_name,
                      focus_ref_json,
                      status, resolution
               FROM (
                   SELECT * FROM bot_goals
                   WHERE is_deleted=0 AND status='active'
                   ORDER BY created_at DESC
                   LIMIT ?
               ) sub ORDER BY created_at ASC""",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    out: list[dict] = []
    for r in rows:
        item = dict(r)
        item["conv_type"] = item.get("focus_type", "")
        item["conv_id"] = item.get("focus_id", "")
        item["conv_name"] = item.get("focus_name", "")
        out.append(item)
    return out


# ── adapter 意识流持久化 ──────────────────────────────────

async def save_adapter_contents(adapter_type: str, contents: list, timestamps: list) -> None:
    """持久化 adapter 意识流（_contents history + timestamps）。"""
    import json as _json
    async with _connect() as db:
        await db.execute(
            """INSERT OR REPLACE INTO adapter_state (key, updated_at, adapter_type, contents, timestamps)
               VALUES ('main', ?, ?, ?, ?)""",
            (
                _ms(),
                adapter_type,
                _json.dumps(contents, ensure_ascii=False),
                _json.dumps(timestamps, ensure_ascii=False),
            ),
        )
        await db.commit()
    logger.debug("已保存 adapter_contents: type=%s entries=%d", adapter_type, len(contents))


async def load_adapter_contents() -> "tuple[str, list, list] | None":
    """加载 adapter 意识流，返回 (adapter_type, contents, timestamps)；不存在则返回 None。"""
    import json as _json
    async with _connect() as db:
        async with db.execute(
            "SELECT adapter_type, contents, timestamps FROM adapter_state WHERE key = 'main'"
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return None
    try:
        contents = _json.loads(row[1])
        timestamps = _json.loads(row[2])
        return str(row[0]), contents, timestamps
    except Exception:
        return None


# ── 归档窗口指纹持久化 ─────────────────────────────────────────


async def load_archive_signatures() -> dict[tuple[str, str], str]:
    """启动时从数据库加载所有归档签名，返回 {(focus_type, focus_id): signature}。"""
    result: dict[tuple[str, str], str] = {}
    async with _connect() as db:
        async with db.execute("SELECT conv_key, signature FROM archive_signatures") as cur:
            rows = await cur.fetchall()
    for row in rows:
        key_str = str(row[0])
        focus = focus_from_session_key(key_str)
        if focus is None and "/" in key_str:
            parts = key_str.split("/", 1)
            if len(parts) == 2:
                focus = FocusRef("qq", parts[0], parts[1])
        if focus is not None:
            result[(focus.target_type, focus.target_id)] = str(row[1])
    return result


async def save_archive_signature(conv_type: str, conv_id: str, signature: str) -> None:
    """写入/更新单条归档签名。"""
    focus = _focus_from_legacy(conv_type=conv_type, conv_id=conv_id)
    conv_key = session_key_for_focus(focus) if focus is not None else f"{conv_type}/{conv_id}"
    focus_json = _focus_ref_json(focus)
    async with _connect() as db:
        await db.execute(
            """INSERT INTO archive_signatures (conv_key, signature, focus_ref_json)
               VALUES (?, ?, ?)
               ON CONFLICT(conv_key) DO UPDATE SET
                   signature=excluded.signature,
                   focus_ref_json=excluded.focus_ref_json""",
            (conv_key, signature, focus_json),
        )
        await db.commit()


# ── 待归档任务队列持久化 ───────────────────────────────────────


async def enqueue_archive_job(
    *,
    conv_type: str,
    conv_id: str,
    conv_name: str,
    sender_id: str,
    dialogue: str,
    signature: str,
    prev_signature: str,
    valid_candidate_ids: list[int],
) -> int:
    """持久化一条待归档任务，返回 job_id。"""
    import json as _json
    platform, focus_type, focus_id, focus_name, _focus_key, focus_json = _focus_tuple_from_legacy(
        conv_type=conv_type,
        conv_id=conv_id,
        conv_name=conv_name,
    )
    async with _connect() as db:
        cur = await db.execute(
            """INSERT INTO pending_archive_jobs
               (focus_platform, focus_type, focus_id, focus_name, focus_ref_json,
                conv_type, conv_id, conv_name, sender_id, dialogue,
                signature, prev_signature, valid_candidate_ids, enqueued_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                platform,
                focus_type,
                focus_id,
                focus_name,
                focus_json,
                focus_type or conv_type,
                focus_id or conv_id,
                focus_name or conv_name,
                sender_id,
                dialogue,
                signature, prev_signature,
                _json.dumps(list(valid_candidate_ids)),
                _ms(),
            ),
        )
        await db.commit()
        return int(cur.lastrowid or 0)


async def delete_archive_job(job_id: int) -> None:
    """删除一条已处理（成功或永久失败）的归档任务。"""
    async with _connect() as db:
        await db.execute(
            "DELETE FROM pending_archive_jobs WHERE job_id = ?",
            (int(job_id),),
        )
        await db.commit()


async def load_pending_archive_jobs() -> list[dict]:
    """启动时加载所有未完成的归档任务，按入队顺序返回。"""
    import json as _json
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT job_id,
                      COALESCE(NULLIF(focus_type, ''), conv_type) AS focus_type,
                      COALESCE(NULLIF(focus_id, ''), conv_id) AS focus_id,
                      COALESCE(NULLIF(focus_name, ''), conv_name) AS focus_name,
                      focus_ref_json,
                      sender_id,
                      dialogue, signature, prev_signature,
                      valid_candidate_ids, enqueued_at
               FROM pending_archive_jobs
               ORDER BY job_id ASC"""
        ) as cur:
            rows = await cur.fetchall()
    out: list[dict] = []
    for row in rows:
        try:
            cand_ids = _json.loads(row["valid_candidate_ids"] or "[]")
            if not isinstance(cand_ids, list):
                cand_ids = []
        except Exception:
            cand_ids = []
        out.append({
            "job_id": int(row["job_id"]),
            "focus_type": str(row["focus_type"] or ""),
            "focus_id": str(row["focus_id"] or ""),
            "focus_name": str(row["focus_name"] or ""),
            "focus_ref_json": str(row["focus_ref_json"] or "{}"),
            "conv_type": str(row["focus_type"] or ""),
            "conv_id": str(row["focus_id"] or ""),
            "conv_name": str(row["focus_name"] or ""),
            "sender_id": str(row["sender_id"] or ""),
            "dialogue": str(row["dialogue"] or ""),
            "signature": str(row["signature"] or ""),
            "prev_signature": str(row["prev_signature"] or ""),
            "valid_candidate_ids": [int(x) for x in cand_ids if isinstance(x, (int, str)) and str(x).lstrip("-").isdigit()],
            "enqueued_at": int(row["enqueued_at"] or 0),
        })
    return out
