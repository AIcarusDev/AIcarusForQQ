import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import aiosqlite

import database


class DatabaseSchemaMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_migrate_schema_adds_reply_to_to_legacy_chat_messages(self) -> None:
        async with aiosqlite.connect(":memory:") as db:
            await db.execute(
                """CREATE TABLE chat_messages (
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
                )"""
            )
            await db.commit()

            await database._migrate_schema(db)

            async with db.execute("PRAGMA table_info(chat_messages)") as cur:
                rows = await cur.fetchall()
            columns = {row[1] for row in rows}

        self.assertIn("reply_to", columns)


if __name__ == "__main__":
    unittest.main()
