from __future__ import annotations

import asyncio
import json
import sqlite3

import database


def test_legacy_qq_conversation_rows_migrate_to_focus_refs(monkeypatch, tmp_path):
    db_path = tmp_path / "AICQ.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_path))

    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE chat_sessions (
                session_key TEXT PRIMARY KEY,
                conv_type TEXT NOT NULL DEFAULT '',
                conv_id TEXT NOT NULL DEFAULT '',
                conv_name TEXT NOT NULL DEFAULT '',
                temp_source_group_id TEXT NOT NULL DEFAULT '',
                temp_source_group_name TEXT NOT NULL DEFAULT '',
                last_active_at INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                role TEXT NOT NULL,
                message_id TEXT NOT NULL DEFAULT '',
                sender_id TEXT NOT NULL DEFAULT '',
                sender_name TEXT NOT NULL DEFAULT '',
                sender_card TEXT NOT NULL DEFAULT '',
                sender_nickname TEXT NOT NULL DEFAULT '',
                sender_role TEXT NOT NULL DEFAULT '',
                sender_title TEXT NOT NULL DEFAULT '',
                sender_level TEXT NOT NULL DEFAULT '',
                timestamp TEXT NOT NULL DEFAULT '',
                reply_to TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL DEFAULT '',
                content_type TEXT NOT NULL DEFAULT 'text',
                content_segments TEXT NOT NULL DEFAULT '[]',
                images TEXT NOT NULL DEFAULT '[]',
                delivery_state TEXT NOT NULL DEFAULT '',
                delivery_error TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE bot_turns (
                turn_id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL DEFAULT 0,
                conv_type TEXT NOT NULL DEFAULT '',
                conv_id TEXT NOT NULL DEFAULT '',
                result_json TEXT NOT NULL DEFAULT '{}',
                tool_calls TEXT NOT NULL DEFAULT '[]',
                world_xml TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE watcher_cycles (
                cycle_id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL DEFAULT 0,
                conv_type TEXT NOT NULL DEFAULT '',
                conv_id TEXT NOT NULL DEFAULT '',
                result_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE bot_goals (
                goal_id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL DEFAULT 0,
                title TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                conv_type TEXT NOT NULL DEFAULT '',
                conv_id TEXT NOT NULL DEFAULT '',
                conv_name TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                resolution TEXT NOT NULL DEFAULT '',
                is_deleted INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE archive_signatures (
                conv_key TEXT PRIMARY KEY,
                signature TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE pending_archive_jobs (
                job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_type TEXT NOT NULL DEFAULT '',
                conv_id TEXT NOT NULL DEFAULT '',
                conv_name TEXT NOT NULL DEFAULT '',
                sender_id TEXT NOT NULL DEFAULT '',
                dialogue TEXT NOT NULL DEFAULT '',
                signature TEXT NOT NULL DEFAULT '',
                prev_signature TEXT NOT NULL DEFAULT '',
                valid_candidate_ids TEXT NOT NULL DEFAULT '[]',
                enqueued_at INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO chat_sessions (session_key, conv_type, conv_id, conv_name, last_active_at)
                VALUES ('group_123', 'group', '123', '旧群', 10);
            INSERT INTO chat_messages (session_key, role, message_id, content)
                VALUES ('group_123', 'user', 'm1', 'hello');
            INSERT INTO bot_turns (turn_id, created_at, conv_type, conv_id, result_json, tool_calls)
                VALUES ('turn1', 1000, 'group', '123', '{"cognition":"old"}', '[]');
            INSERT INTO watcher_cycles (cycle_id, created_at, conv_type, conv_id, result_json)
                VALUES ('cycle1', 1000, 'group', '123', '{}');
            INSERT INTO bot_goals (goal_id, created_at, updated_at, title, conv_type, conv_id, conv_name)
                VALUES ('goal1', 1000, 1000, 'goal', 'group', '123', '旧群');
            INSERT INTO archive_signatures (conv_key, signature)
                VALUES ('group/123', 'sig-old');
            INSERT INTO pending_archive_jobs (conv_type, conv_id, conv_name, sender_id, dialogue, signature, prev_signature, valid_candidate_ids)
                VALUES ('group', '123', '旧群', 'u1', 'dialogue', 'sig-new', 'sig-old', '[1]');
            """
        )
        conn.commit()

    asyncio.run(database.init_db())

    async def scenario():
        sessions = await database.load_chat_sessions()
        messages = await database.load_chat_messages("qq:group:123", limit=5)
        turns = await database.load_recent_bot_turns(limit=5)
        goals = await database.load_goals(limit=5)
        signatures = await database.load_archive_signatures()
        jobs = await database.load_pending_archive_jobs()
        watcher, _created_at = await database.load_last_watcher_cycle("group", "123")
        return sessions, messages, turns, goals, signatures, jobs, watcher

    sessions, messages, turns, goals, signatures, jobs, watcher = asyncio.run(scenario())

    assert sessions[0]["session_key"] == "qq:group:123"
    assert sessions[0]["focus_platform"] == "qq"
    assert sessions[0]["focus_type"] == "group"
    assert sessions[0]["focus_id"] == "123"
    assert json.loads(sessions[0]["focus_ref_json"]) == {
        "platform": "qq",
        "target_type": "group",
        "target_id": "123",
        "target_name": "旧群",
    }
    assert messages[0]["message_id"] == "m1"
    assert turns[0]["session_key"] == "qq:group:123"
    assert goals[0]["focus_type"] == "group"
    assert goals[0]["focus_id"] == "123"
    assert signatures == {("group", "123"): "sig-old"}
    assert jobs[0]["focus_type"] == "group"
    assert jobs[0]["focus_id"] == "123"
    assert watcher == {}
