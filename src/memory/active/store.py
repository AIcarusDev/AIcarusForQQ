"""Small transactional store for a single Core process's FIFO memory queue.

Only text and internal memory/batch identities are stored. No external-source
IDs, entity taxonomy, session scope, vector jobs or graph are part of this path.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from memory.tokenizer import tokenize


SCHEMA = """
CREATE TABLE IF NOT EXISTS ActiveMemoryBatches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cognition TEXT NOT NULL,
    memories_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    retry_at REAL NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    result_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    completed_at INTEGER,
    published INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS active_memory_queue ON ActiveMemoryBatches(status, id);
CREATE TABLE IF NOT EXISTS ActiveMemoryEntries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'active',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS ActiveMemoryRevisions (
    memory_id INTEGER NOT NULL REFERENCES ActiveMemoryEntries(id),
    revision INTEGER NOT NULL,
    batch_id INTEGER NOT NULL REFERENCES ActiveMemoryBatches(id),
    content TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY(memory_id, revision)
);
"""


def default_path() -> str:
    from database import DB_PATH
    return DB_PATH


@contextmanager
def connection(path: str):
    con = sqlite3.connect(path, timeout=5)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    try:
        with con:
            yield con
    finally:
        con.close()


def initialize(path: str) -> None:
    with connection(path) as con:
        con.executescript(SCHEMA)


def identity(prefix: str, value: int) -> str:
    return f"{prefix}{value:06d}"


def parse_id(value: str, prefix: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(prefix + r"[0-9]{6,}", value):
        raise ValueError(f"expected {prefix} memory identifier")
    return int(value[1:])


def _entry(row) -> dict:
    result = dict(row)
    result["id"] = identity("M", result["id"])
    return result


def enqueue(path: str, cognition: str, memories: list[str]) -> dict:
    if not isinstance(cognition, str) or not cognition.strip():
        raise ValueError("current cognition is required")
    if not isinstance(memories, list) or not memories or any(
        not isinstance(item, str) or not item.strip() for item in memories
    ):
        raise ValueError("memories must contain nonblank strings")
    with connection(path) as con:
        cur = con.execute(
            "INSERT INTO ActiveMemoryBatches(cognition, memories_json, created_at) VALUES (?, ?, ?)",
            (cognition, json.dumps(memories, ensure_ascii=False), _now()),
        )
        return {"saved": True, "batch_id": identity("B", cur.lastrowid), "status": "pending", "count": len(memories)}


def recover(path: str) -> None:
    with connection(path) as con:
        con.execute("UPDATE ActiveMemoryBatches SET status='pending', retry_at=0 WHERE status='running'")


def head(path: str) -> dict | None:
    with connection(path) as con:
        row = con.execute("SELECT * FROM ActiveMemoryBatches WHERE status <> 'completed' ORDER BY id LIMIT 1").fetchone()
        return dict(row) if row else None


def begin(path: str, batch_id: int) -> bool:
    with connection(path) as con:
        cur = con.execute(
            "UPDATE ActiveMemoryBatches SET status='running', attempts=attempts+1, error='' "
            "WHERE id=? AND status IN ('pending','retry') AND retry_at<=? "
            "AND id=(SELECT MIN(id) FROM ActiveMemoryBatches WHERE status<>'completed')",
            (batch_id, time.time()),
        )
        return cur.rowcount == 1


def retry(path: str, batch_id: int, error: str, delay: float) -> None:
    with connection(path) as con:
        con.execute(
            "UPDATE ActiveMemoryBatches SET status='retry', retry_at=?, error=? WHERE id=? AND status='running'",
            (time.time() + delay, error, batch_id),
        )


def search(path: str, query: str, *, limit: int = 10, offset: int = 0, literal: bool = False) -> list[dict]:
    """Simple keyword matching with deterministic ordering; never recent fallback."""
    query = query.strip()
    if not query:
        return []
    terms = [query] if literal else list(dict.fromkeys([query, *tokenize(query).split()]))[:32]
    terms = [term.casefold() for term in terms if term.strip()]
    # instr, not LIKE: '%' and '_' are literal user characters.
    match = " + ".join("(instr(lower(content), ?) > 0)" for _ in terms)
    ids = [] if literal else [int(item[1:]) for item in re.findall(r"\bM[0-9]{6,}\b", query)]
    id_expr = f"id IN ({','.join('?' for _ in ids)})" if ids else "0"
    with connection(path) as con:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE name='ActiveMemoryEntries'").fetchone():
            return []
        rows = con.execute(
            f"SELECT *, ({match}) + 100*({id_expr}) AS matches FROM ActiveMemoryEntries "
            "WHERE status='active' AND matches>0 ORDER BY matches DESC, updated_at DESC, id DESC LIMIT ? OFFSET ?",
            [*terms, *ids, limit, offset],
        ).fetchall()
        return [_entry(row) for row in rows]


def read(path: str, item_id: str, *, history: bool = False) -> dict | None:
    prefix = item_id[:1]
    number = parse_id(item_id, "B" if prefix == "B" else "M")
    with connection(path) as con:
        if prefix == "B":
            row = con.execute("SELECT * FROM ActiveMemoryBatches WHERE id=?", (number,)).fetchone()
            if not row:
                return None
            result = dict(row)
            result["id"] = identity("B", number)
            result["memories"] = json.loads(result.pop("memories_json"))
            result["result"] = json.loads(result.pop("result_json"))
            return result
        row = con.execute("SELECT * FROM ActiveMemoryEntries WHERE id=?", (number,)).fetchone()
        if not row:
            return None
        result = _entry(row)
        if history:
            result["history"] = [dict(item) for item in con.execute(
                "SELECT revision, content, status, created_at FROM ActiveMemoryRevisions WHERE memory_id=? ORDER BY revision",
                (number,),
            )]
        return result


def complete(path: str, batch_id: int, changes: list[dict], note: str, related: list[dict]) -> None:
    """All changes and queue completion commit together, or nothing does."""
    versions = {item["id"]: item["revision"] for item in related}
    with connection(path) as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT status FROM ActiveMemoryBatches WHERE id=?", (batch_id,)).fetchone()
        if not row or row["status"] != "running":
            raise ValueError("batch is no longer running")
        result_ids = []
        touched = set()
        for change in changes:
            item_id = change.get("id")
            status = "retracted" if change["action"] == "retract" else "active"
            content = change.get("content", "")
            now = _now()
            if item_id is None:
                cur = con.execute(
                    "INSERT INTO ActiveMemoryEntries(content, created_at, updated_at) VALUES (?, ?, ?)",
                    (content, now, now),
                )
                number, revision = cur.lastrowid, 1
            else:
                if item_id not in versions or item_id in touched:
                    raise ValueError("change target must be unique and present in related memories")
                touched.add(item_id)
                number = parse_id(item_id, "M")
                old = con.execute("SELECT * FROM ActiveMemoryEntries WHERE id=?", (number,)).fetchone()
                if not old or old["revision"] != versions[item_id] or old["status"] != "active":
                    raise ValueError("memory revision changed")
                revision = old["revision"] + 1
                if status == "retracted":
                    content = old["content"]
                con.execute(
                    "UPDATE ActiveMemoryEntries SET content=?, revision=?, status=?, updated_at=? WHERE id=?",
                    (content, revision, status, now, number),
                )
            con.execute(
                "INSERT INTO ActiveMemoryRevisions VALUES (?, ?, ?, ?, ?, ?)",
                (number, revision, batch_id, content, status, now),
            )
            result_ids.append(identity("M", number))
        con.execute(
            "UPDATE ActiveMemoryBatches SET status='completed', completed_at=?, error='', result_json=? WHERE id=?",
            (_now(), json.dumps({"memory_ids": result_ids, "note": note}, ensure_ascii=False), batch_id),
        )


def publish(path: str, *, force: bool = False) -> bool:
    """Rebuild owned Markdown views. Failure never replays a completed LLM job."""
    with connection(path) as con:
        dirty = con.execute("SELECT id FROM ActiveMemoryBatches WHERE status='completed' AND published=0").fetchall()
        if not dirty and not force:
            return False
        rows = con.execute("SELECT * FROM ActiveMemoryEntries ORDER BY id").fetchall()
        root = Path(path).parent / "memory" / "published"
        entries = root / "entries"
        entries.mkdir(parents=True, exist_ok=True)
        active_ids = set()
        index = ["# 主动记忆", "", "此目录为数据库生成的只读镜像；使用 memory_write 修改，memory_search / memory_read 查询。", ""]
        for row in rows:
            item_id = identity("M", row["id"])
            if row["status"] != "active":
                continue
            active_ids.add(item_id)
            _atomic_text(entries / f"{item_id}.md", f"---\nid: {item_id}\nrevision: {row['revision']}\nstatus: active\n---\n\n{row['content']}\n")
            index.append(f"- [{item_id}](entries/{item_id}.md)")
        # Only generated entry names belong to us; leave all other files alone.
        for file in entries.glob("M*.md"):
            if re.fullmatch(r"M[0-9]{6,}", file.stem) and file.stem not in active_ids:
                file.unlink()
        _atomic_text(root / "index.md", "\n".join(index) + "\n")
        con.executemany("UPDATE ActiveMemoryBatches SET published=1 WHERE id=?", [(row["id"],) for row in dirty])
        return True


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _now() -> int:
    return int(time.time() * 1000)
