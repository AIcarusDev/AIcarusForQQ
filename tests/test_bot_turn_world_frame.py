from __future__ import annotations

import asyncio
import json
import sqlite3

import database


def test_bot_turn_world_xml_migrates_round_trips_and_pages(monkeypatch, tmp_path):
    db_path = tmp_path / "AICQ.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_path))

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE bot_turns (
                turn_id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL DEFAULT 0,
                conv_type TEXT NOT NULL DEFAULT '',
                conv_id TEXT NOT NULL DEFAULT '',
                result_json TEXT NOT NULL DEFAULT '{}',
                tool_calls TEXT NOT NULL DEFAULT '[]'
            )"""
        )
        conn.execute(
            """INSERT INTO bot_turns (turn_id, created_at, conv_type, conv_id, result_json, tool_calls)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("old", 1000, "group", "42", json.dumps({"cognition": "old"}), "[]"),
        )
        conn.commit()

    asyncio.run(database.init_db())

    async def scenario():
        await database.upsert_chat_session(
            "qq:group:42",
            "group",
            "42",
            "测试群",
        )
        await database.save_bot_turn(
            turn_id="new",
            conv_type="group",
            conv_id="42",
            result={
                "cognition": "new",
                "motive": "wait for a new event",
                "request_started_at": 10.0,
                "action_finished_at": 11.0,
                "tokens": {"in": 1, "out": 2},
                "elapsed_ms": 1234.5,
            },
            tool_calls_log=[
                {
                    "function": "runtime_manage",
                    "call_id": "call_1",
                    "arguments": {"action": "wait", "seconds": 1},
                    "result": {"ok": True},
                    "elapsed_ms": 12.5,
                }
            ],
            world_xml="<world><qq>hello</qq></world>",
        )
        latest = await database.load_recent_bot_turns(limit=1)
        older = await database.load_recent_bot_turns(limit=5, before=latest[0]["created_at"])
        return latest, older

    latest, older = asyncio.run(scenario())

    assert latest[0]["turn_id"] == "new"
    assert latest[0]["session_key"] == "qq:group:42"
    assert latest[0]["conv_name"] == "测试群"
    assert latest[0]["world_xml"] == "<world><qq>hello</qq></world>"
    assert latest[0]["result"]["elapsed_ms"] == 1234.5
    assert latest[0]["result"]["motive"] == "wait for a new event"
    assert latest[0]["result"]["request_started_at"] == 10.0
    assert latest[0]["result"]["action_finished_at"] == 11.0
    assert latest[0]["tool_calls"][0]["function"] == "runtime_manage"
    assert latest[0]["tool_calls"][0]["call_id"] == "call_1"
    assert latest[0]["tool_calls"][0]["elapsed_ms"] == 12.5
    assert [row["turn_id"] for row in older] == ["old"]
    assert older[0]["world_xml"] == ""
