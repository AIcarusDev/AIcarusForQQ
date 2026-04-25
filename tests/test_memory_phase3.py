"""test_memory_phase3.py — Phase 3 记忆系统测试套件

测试范围：
    1. database.decay_triple_confidence（置信度衰减，含边界检查）
    2. database.upsert_merge_suggestion（写入 + 幂等更新）
    3. database.list_pending_suggestions（过滤 + 排序）
    4. database.resolve_merge_suggestion（状态流转）
    5. suggest_person_merge 工具（参数校验 + DB 交互）
    6. Session confidence boost（召回命中后 +0.05）

所有测试使用临时 SQLite 数据库，互相隔离，不依赖生产 DB。
"""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── src/ 加入 sys.path ────────────────────────────────────────
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════

def _make_temp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db", prefix="test_p3_")
    os.close(fd)
    return path


def _patch_db(monkeypatch, db_path: str):
    import database
    monkeypatch.setattr(database, "DB_PATH", db_path)


def _run(coro):
    return asyncio.run(coro)


async def _seed_profiles(db_path: str, *profile_ids: str) -> None:
    """在 entity_profiles 表中插入测试用 profile_id 行（满足 FK 约束）。"""
    import aiosqlite
    now_ms = int(time.time() * 1000)
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA foreign_keys=ON")
        for pid in profile_ids:
            await db.execute(
                "INSERT OR IGNORE INTO entity_profiles (profile_id, created_at, updated_at)"
                " VALUES (?, ?, ?)",
                (pid, now_ms, now_ms),
            )
        await db.commit()


async def _init_and_write(db_path: str, triples: list[dict]) -> list[int]:
    """初始化数据库并批量写入 triples，返回各自的 id 列表。"""
    import database
    old = database.DB_PATH
    database.DB_PATH = db_path
    try:
        await database.init_db()
        ids = []
        for t in triples:
            tok = t.get("tok", t["object_text"])
            row_id = await database.write_triple(
                subject=t["subject"],
                predicate=t.get("predicate", "[note]"),
                object_text=t["object_text"],
                object_text_tok=tok,
                source=t.get("source", "test"),
                reason=t.get("reason", ""),
                confidence=t.get("confidence", 0.5),
            )
            ids.append(row_id)
        return ids
    finally:
        database.DB_PATH = old


async def _set_last_accessed(db_path: str, triple_id: int, ms_ago: int):
    """将指定 triple 的 last_accessed 设置为 ms_ago 毫秒以前。"""
    import aiosqlite
    now_ms = int(time.time() * 1000)
    target = now_ms - ms_ago
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE MemoryTriples SET last_accessed=? WHERE id=?",
            (target, triple_id),
        )
        await db.commit()


async def _get_confidence(db_path: str, triple_id: int) -> float:
    """读取指定 triple 的 confidence。"""
    import aiosqlite
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT confidence FROM MemoryTriples WHERE id=?", (triple_id,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else -1.0


# ═══════════════════════════════════════════════════════════════
# §1 decay_triple_confidence
# ═══════════════════════════════════════════════════════════════

class TestDecayTripleConfidence:

    def test_decay_idle_triple(self, monkeypatch):
        """超过阈值天数未访问的记忆应被降权。"""
        db = _make_temp_db()
        _patch_db(monkeypatch, db)

        import database
        ids = _run(_init_and_write(db, [
            {"subject": "Alice", "object_text": "喜欢猫", "confidence": 0.8},
        ]))
        # 设为 10 天前
        _run(_set_last_accessed(db, ids[0], 10 * 24 * 3600 * 1000))

        count = _run(database.decay_triple_confidence(
            min_confidence=0.05, decay_rate=0.1, idle_days_threshold=7.0
        ))

        assert count == 1
        conf = _run(_get_confidence(db, ids[0]))
        assert abs(conf - 0.7) < 1e-6  # 0.8 - 0.1

        os.unlink(db)

    def test_recent_triple_not_decayed(self, monkeypatch):
        """未超过阈值的记忆不应被降权。"""
        db = _make_temp_db()
        _patch_db(monkeypatch, db)

        import database
        ids = _run(_init_and_write(db, [
            {"subject": "Bob", "object_text": "喜欢狗", "confidence": 0.6},
        ]))
        # 设为 3 天前（阈值 7 天）
        _run(_set_last_accessed(db, ids[0], 3 * 24 * 3600 * 1000))

        count = _run(database.decay_triple_confidence(
            min_confidence=0.05, decay_rate=0.1, idle_days_threshold=7.0
        ))

        assert count == 0
        conf = _run(_get_confidence(db, ids[0]))
        assert abs(conf - 0.6) < 1e-6  # 未变化

        os.unlink(db)

    def test_confidence_floor_respected(self, monkeypatch):
        """置信度已达到下限（min_confidence）的记忆不继续降权。"""
        db = _make_temp_db()
        _patch_db(monkeypatch, db)

        import database
        ids = _run(_init_and_write(db, [
            {"subject": "Carlo", "object_text": "已被遗忘", "confidence": 0.05},
        ]))
        _run(_set_last_accessed(db, ids[0], 20 * 24 * 3600 * 1000))

        count = _run(database.decay_triple_confidence(
            min_confidence=0.05, decay_rate=0.1, idle_days_threshold=7.0
        ))

        # confidence == min_confidence，不满足 > min_confidence 条件
        assert count == 0
        conf = _run(_get_confidence(db, ids[0]))
        assert abs(conf - 0.05) < 1e-6

        os.unlink(db)

    def test_decay_does_not_go_below_min(self, monkeypatch):
        """decay_rate 大于 (confidence - min_confidence) 时，confidence 应精确夹到 min_confidence。"""
        db = _make_temp_db()
        _patch_db(monkeypatch, db)

        import database
        ids = _run(_init_and_write(db, [
            {"subject": "Dana", "object_text": "即将触底", "confidence": 0.08},
        ]))
        _run(_set_last_accessed(db, ids[0], 10 * 24 * 3600 * 1000))

        count = _run(database.decay_triple_confidence(
            min_confidence=0.05, decay_rate=0.1, idle_days_threshold=7.0
        ))

        assert count == 1
        conf = _run(_get_confidence(db, ids[0]))
        # MAX(0.05, 0.08 - 0.10) = MAX(0.05, -0.02) = 0.05
        assert abs(conf - 0.05) < 1e-6

        os.unlink(db)

    def test_deleted_triple_not_decayed(self, monkeypatch):
        """is_deleted=1 的记忆不应被降权。"""
        db = _make_temp_db()
        _patch_db(monkeypatch, db)

        import database
        ids = _run(_init_and_write(db, [
            {"subject": "Eve", "object_text": "已删除", "confidence": 0.9},
        ]))
        _run(_set_last_accessed(db, ids[0], 10 * 24 * 3600 * 1000))
        _run(database.soft_delete_triple(ids[0]))

        count = _run(database.decay_triple_confidence(
            min_confidence=0.05, decay_rate=0.1, idle_days_threshold=7.0
        ))

        assert count == 0

        os.unlink(db)

    def test_returns_correct_count(self, monkeypatch):
        """仅超阈值且 confidence > min 的记忆计入返回值。"""
        db = _make_temp_db()
        _patch_db(monkeypatch, db)

        import database
        ids = _run(_init_and_write(db, [
            {"subject": "F1", "object_text": "陈旧 A", "confidence": 0.8},
            {"subject": "F2", "object_text": "陈旧 B", "confidence": 0.7},
            {"subject": "F3", "object_text": "新鲜 C", "confidence": 0.6},
        ]))
        _run(_set_last_accessed(db, ids[0], 10 * 24 * 3600 * 1000))
        _run(_set_last_accessed(db, ids[1], 10 * 24 * 3600 * 1000))
        # ids[2] 使用默认（当前时间），不超阈值

        count = _run(database.decay_triple_confidence(
            min_confidence=0.05, decay_rate=0.1, idle_days_threshold=7.0
        ))

        assert count == 2

        os.unlink(db)


# ═══════════════════════════════════════════════════════════════
# §2 merge_suggestions DB 函数
# ═══════════════════════════════════════════════════════════════

class TestMergeSuggestionsDB:

    def test_upsert_creates_suggestion(self, monkeypatch):
        """首次提交应创建新记录，返回 UUID 字符串。"""
        db = _make_temp_db()
        _patch_db(monkeypatch, db)

        import database
        _run(database.init_db())
        _run(_seed_profiles(db, "111", "222"))
        sid = _run(database.upsert_merge_suggestion("111", "222", 0.95, "相同昵称"))

        assert isinstance(sid, str) and len(sid) == 36  # UUID4 格式

        rows = _run(database.list_pending_suggestions())
        assert len(rows) == 1
        assert rows[0]["suggestion_id"] == sid
        assert rows[0]["status"] == "pending"

        os.unlink(db)

    def test_upsert_normalizes_order(self, monkeypatch):
        """(B, A) 与 (A, B) 应被视为同一 pair，规范化为 A < B 存储。"""
        db = _make_temp_db()
        _patch_db(monkeypatch, db)

        import database
        _run(database.init_db())
        _run(_seed_profiles(db, "aaa", "zzz"))
        sid1 = _run(database.upsert_merge_suggestion("zzz", "aaa", 0.9, "顺序 B>A"))
        sid2 = _run(database.upsert_merge_suggestion("aaa", "zzz", 0.92, "顺序 A>B"))

        # 同一 pair，第二次应更新而非新建
        assert sid1 == sid2

        rows = _run(database.list_pending_suggestions())
        assert len(rows) == 1
        assert rows[0]["similarity"] == pytest.approx(0.92)
        assert rows[0]["reason"] == "顺序 A>B"

        os.unlink(db)

    def test_upsert_idempotent_different_pairs(self, monkeypatch):
        """不同 pair 应各自独立创建记录。"""
        db = _make_temp_db()
        _patch_db(monkeypatch, db)

        import database
        _run(database.init_db())
        _run(_seed_profiles(db, "A", "B", "C"))
        _run(database.upsert_merge_suggestion("A", "B", 0.9, "pair AB"))
        _run(database.upsert_merge_suggestion("A", "C", 0.8, "pair AC"))
        _run(database.upsert_merge_suggestion("B", "C", 0.7, "pair BC"))

        rows = _run(database.list_pending_suggestions(limit=10))
        assert len(rows) == 3
        # 应按 similarity 降序
        assert rows[0]["similarity"] >= rows[1]["similarity"] >= rows[2]["similarity"]

        os.unlink(db)

    def test_list_excludes_resolved(self, monkeypatch):
        """已 confirmed/rejected 的建议不应出现在 list_pending_suggestions 中。"""
        db = _make_temp_db()
        _patch_db(monkeypatch, db)

        import database
        _run(database.init_db())
        _run(_seed_profiles(db, "P1", "P2", "C1", "C2", "R1", "R2"))
        sid_p = _run(database.upsert_merge_suggestion("P1", "P2", 0.9, "待审核"))
        sid_c = _run(database.upsert_merge_suggestion("C1", "C2", 0.85, "已确认"))
        sid_r = _run(database.upsert_merge_suggestion("R1", "R2", 0.8, "已拒绝"))

        _run(database.resolve_merge_suggestion(sid_c, "confirmed"))
        _run(database.resolve_merge_suggestion(sid_r, "rejected"))

        rows = _run(database.list_pending_suggestions(limit=10))
        assert len(rows) == 1
        assert rows[0]["suggestion_id"] == sid_p

        os.unlink(db)

    def test_resolve_confirmed(self, monkeypatch):
        """resolve 为 confirmed 时应返回 True 并修改状态。"""
        db = _make_temp_db()
        _patch_db(monkeypatch, db)

        import database, aiosqlite

        _run(database.init_db())
        _run(_seed_profiles(db, "X1", "X2"))
        sid = _run(database.upsert_merge_suggestion("X1", "X2", 0.95, "确认测试"))
        result = _run(database.resolve_merge_suggestion(sid, "confirmed"))

        assert result is True

        async def _check():
            async with aiosqlite.connect(db) as conn:
                async with conn.execute(
                    "SELECT status FROM merge_suggestions WHERE suggestion_id=?", (sid,)
                ) as cur:
                    return (await cur.fetchone())[0]

        assert _run(_check()) == "confirmed"

        os.unlink(db)

    def test_resolve_rejected(self, monkeypatch):
        """resolve 为 rejected 时应返回 True 并修改状态。"""
        db = _make_temp_db()
        _patch_db(monkeypatch, db)

        import database, aiosqlite

        _run(database.init_db())
        _run(_seed_profiles(db, "Y1", "Y2"))
        sid = _run(database.upsert_merge_suggestion("Y1", "Y2", 0.9, "拒绝测试"))
        result = _run(database.resolve_merge_suggestion(sid, "rejected"))

        assert result is True

        async def _check():
            async with aiosqlite.connect(db) as conn:
                async with conn.execute(
                    "SELECT status FROM merge_suggestions WHERE suggestion_id=?", (sid,)
                ) as cur:
                    return (await cur.fetchone())[0]

        assert _run(_check()) == "rejected"

        os.unlink(db)

    def test_resolve_nonexistent_returns_false(self, monkeypatch):
        """不存在的 suggestion_id 应返回 False。"""
        db = _make_temp_db()
        _patch_db(monkeypatch, db)

        import database
        _run(database.init_db())
        result = _run(database.resolve_merge_suggestion("no-such-id", "confirmed"))

        assert result is False

        os.unlink(db)

    def test_resolve_invalid_status_raises(self, monkeypatch):
        """传入非法 status 应抛出 ValueError。"""
        db = _make_temp_db()
        _patch_db(monkeypatch, db)

        import database
        _run(database.init_db())
        _run(_seed_profiles(db, "Z1", "Z2"))
        sid = _run(database.upsert_merge_suggestion("Z1", "Z2", 0.9, "非法状态"))

        with pytest.raises(ValueError, match="confirmed.*rejected"):
            _run(database.resolve_merge_suggestion(sid, "maybe"))

        os.unlink(db)

    def test_resolve_already_resolved_returns_false(self, monkeypatch):
        """已 confirmed 的建议不能再次修改（返回 False）。"""
        db = _make_temp_db()
        _patch_db(monkeypatch, db)

        import database
        _run(database.init_db())
        _run(_seed_profiles(db, "W1", "W2"))
        sid = _run(database.upsert_merge_suggestion("W1", "W2", 0.9, "重复解决"))
        _run(database.resolve_merge_suggestion(sid, "confirmed"))

        result = _run(database.resolve_merge_suggestion(sid, "rejected"))
        assert result is False

        os.unlink(db)


# ═══════════════════════════════════════════════════════════════
# §3 suggest_person_merge 工具
# ═══════════════════════════════════════════════════════════════

class TestSuggestPersonMergeTool:

    def _make_session(self):
        return SimpleNamespace(
            context_messages=[{"role": "user", "sender_id": "99999"}],
        )

    def _make_mock_loop(self, return_value):
        """返回模拟事件循环，run_coroutine_threadsafe 直接返回 return_value。"""
        loop = MagicMock()
        future = MagicMock()
        future.result.return_value = return_value
        loop.is_running.return_value = True
        loop.run_until_complete = None
        loop_mock = loop
        loop_mock.is_running.return_value = True

        import concurrent.futures as _cf
        future_real = _cf.Future()
        future_real.set_result(return_value)

        loop_mock.run_coroutine_threadsafe = MagicMock(return_value=future_real)
        return loop_mock

    def test_valid_call_returns_suggestion_id(self, monkeypatch, tmp_path):
        """合法调用应返回 suggestion_id 和 status: pending。"""
        db = str(tmp_path / "test.db")
        monkeypatch.setattr("database.DB_PATH", db)
        _run(__import__("database").init_db())
        _run(_seed_profiles(db, "111", "222"))

        import tools.suggest_person_merge as _tool
        import app_state
        import concurrent.futures

        session = self._make_session()
        handler = _tool.make_handler(session)

        # 用 MagicMock 伪造「运行中」的循环，实际执行用独立 runner_loop
        runner_loop = asyncio.new_event_loop()
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True

        def _fake_rcf(coro, loop):
            f = concurrent.futures.Future()
            async def _run_and_set():
                result = await coro
                f.set_result(result)
            runner_loop.run_until_complete(_run_and_set())
            return f

        monkeypatch.setattr(app_state, "main_loop", mock_loop)
        monkeypatch.setattr("asyncio.run_coroutine_threadsafe", _fake_rcf)

        result = handler(
            profile_id_a="111",
            profile_id_b="222",
            similarity=0.95,
            reason="昵称相同",
        )
        runner_loop.close()

        assert "suggestion_id" in result
        assert result["status"] == "pending"

    def test_same_id_returns_error(self, monkeypatch, tmp_path):
        """profile_id_a == profile_id_b 应返回 error。"""
        import tools.suggest_person_merge as _tool
        import app_state

        loop = MagicMock()
        loop.is_running.return_value = True
        monkeypatch.setattr(app_state, "main_loop", loop)

        session = self._make_session()
        handler = _tool.make_handler(session)

        result = handler(
            profile_id_a="same",
            profile_id_b="same",
            similarity=0.99,
            reason="自己等于自己",
        )
        assert "error" in result
        assert "不能相同" in result["error"]

    def test_invalid_similarity_returns_error(self, monkeypatch, tmp_path):
        """similarity 超出 0-1 范围应返回 error。"""
        import tools.suggest_person_merge as _tool
        import app_state

        loop = MagicMock()
        loop.is_running.return_value = True
        monkeypatch.setattr(app_state, "main_loop", loop)

        session = self._make_session()
        handler = _tool.make_handler(session)

        result = handler(
            profile_id_a="aaa",
            profile_id_b="bbb",
            similarity=1.5,
            reason="超出范围",
        )
        assert "error" in result

    def test_no_loop_returns_error(self, monkeypatch):
        """main_loop 为 None 时应返回 error。"""
        import tools.suggest_person_merge as _tool
        import app_state

        monkeypatch.setattr(app_state, "main_loop", None)
        session = self._make_session()
        handler = _tool.make_handler(session)

        result = handler(
            profile_id_a="aaa",
            profile_id_b="bbb",
            similarity=0.95,
            reason="无循环",
        )
        assert "error" in result


# ═══════════════════════════════════════════════════════════════
# §4 Session confidence boost（召回命中后强化）
# ═══════════════════════════════════════════════════════════════

class TestSessionConfidenceBoost:

    def test_boost_called_when_memories_recalled(self, monkeypatch, tmp_path):
        """prepare_memory_recall 有结果时应调用 update_triple_confidence。"""
        db = str(tmp_path / "test.db")

        import database
        monkeypatch.setattr(database, "DB_PATH", db)
        _run(database.init_db())

        # 注入的记忆含 id 字段
        fake_memories = [{"id": 1, "object_text": "喜欢猫"}, {"id": 2, "object_text": "养了狗"}]
        boosted_ids = []

        async def _fake_recall(*args, **kwargs):
            return fake_memories

        async def _fake_boost(ids, delta):
            boosted_ids.extend(ids)

        import llm.session as _session_mod
        import app_state

        monkeypatch.setattr(app_state, "config", {"memory": {}})

        # 构建最简 session 对象
        from llm.session import ChatSession
        session = ChatSession.__new__(ChatSession)
        session.context_messages = [{"role": "user", "content": "你好", "sender_id": "12345"}]
        session.recalled_memories = []
        # last_sender_id 是 property，从 context_messages 自动推导

        with patch("memory.recall_memories", side_effect=_fake_recall), \
            patch("memory.repo.triples.update_triple_confidence", side_effect=_fake_boost):
            _run(session.prepare_memory_recall())

        assert session.recalled_memories == fake_memories
        assert boosted_ids == [1, 2]

    def test_boost_skipped_when_no_memories(self, monkeypatch, tmp_path):
        """召回结果为空时不调用 update_triple_confidence。"""
        db = str(tmp_path / "test.db")

        import database
        monkeypatch.setattr(database, "DB_PATH", db)
        _run(database.init_db())

        async def _fake_recall(*args, **kwargs):
            return []

        boost_called = []

        async def _fake_boost(ids, delta):
            boost_called.append(True)

        import app_state
        monkeypatch.setattr(app_state, "config", {"memory": {}})

        from llm.session import ChatSession
        session = ChatSession.__new__(ChatSession)
        session.context_messages = [{"role": "user", "content": "嗨", "sender_id": "12345"}]
        session.recalled_memories = []
        # last_sender_id 是 property，从 context_messages 自动推导

        with patch("memory.recall_memories", side_effect=_fake_recall), \
            patch("memory.repo.triples.update_triple_confidence", side_effect=_fake_boost):
            _run(session.prepare_memory_recall())

        assert boost_called == []
