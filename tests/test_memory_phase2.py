"""test_memory_phase2.py — Phase 2 记忆系统测试套件

测试范围：
    1. write_memory 工具三元组参数（predicate + object_text）
    2. write_memory 工具兼容旧 content 调用
    3. build_memory_xml 三元组格式展示（subject/predicate 字段）
  4. recall_memory 工具（FTS5 召回作为 tool_result 返回）
    5. database.update_person_profile（entity_profiles 表侧写更新）
  6. update_person_profile 工具

所有测试使用临时 SQLite 数据库，互相隔离，不依赖生产 DB。
"""

import asyncio
import importlib
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock

import pytest

# ── 将 src/ 加入 sys.path ─────────────────────────────────────
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ═══════════════════════════════════════════════════════════════
# 辅助 fixture
# ═══════════════════════════════════════════════════════════════

def _make_temp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db", prefix="test_p2_")
    os.close(fd)
    return path


def _patch_db(monkeypatch, db_path: str):
    import database
    monkeypatch.setattr(database, "DB_PATH", db_path)


def _fresh_memory(monkeypatch, db_path: str):
    """重置记忆运行时状态，并 patch DB 路径。"""
    import memory as _mem
    _mem.configure(15, 15, 50)
    _mem.restore([])
    _patch_db(monkeypatch, db_path)


# ═══════════════════════════════════════════════════════════════
# §1 write_memory 工具 — 三元组参数
# ═══════════════════════════════════════════════════════════════

class TestWriteMemoryTriple:

    def setup_method(self):
        """每个测试前重置记忆运行时状态。"""
        import memory as m
        m.configure(15, 15, 50)
        m.restore([])

    def _make_session(self, sender_id="12345"):
        session = SimpleNamespace(
            context_messages=[{"role": "user", "sender_id": sender_id}],
            conv_type="group",
            conv_id="9999",
            conv_name="测试群",
        )
        return session

    def _make_loop_and_run(self, coro):
        """在新事件循环中运行协程，模拟 run_coroutine_threadsafe。"""
        return asyncio.run(coro)

    def test_triple_params_write(self, monkeypatch, tmp_path):
        """predicate + object_text 参数写入后，DB 中 predicate 应为传入值。"""
        db_path = str(tmp_path / "test.db")
        _fresh_memory(monkeypatch, db_path)

        import database
        asyncio.run(database.init_db())

        import memory as _mem

        async def _run():
            return await _mem.add_memory(
                content="喜欢打乒乓球",
                predicate="喜欢",
                source="测试",
                reason="测试",
                subject="User:qq_12345",
            )

        tid = asyncio.run(_run())
        assert isinstance(tid, int)
        assert tid > 0

        rows = asyncio.run(database.load_all_triples())
        assert len(rows) == 1
        assert rows[0]["predicate"] == "喜欢"
        assert rows[0]["object_text"] == "喜欢打乒乓球"
        assert rows[0]["subject"] == "User:qq_12345"

    def test_content_fallback_uses_note_predicate(self, monkeypatch, tmp_path):
        """只传 content（旧接口）时，predicate 应自动填为 [note]。"""
        db_path = str(tmp_path / "test.db")
        _fresh_memory(monkeypatch, db_path)

        import database
        asyncio.run(database.init_db())

        import memory as _mem

        async def _run():
            return await _mem.add_memory(
                content="用户不喜欢吃香菜",
                predicate="[note]",   # 默认值
                source="测试",
                reason="测试",
            )

        tid = asyncio.run(_run())
        rows = asyncio.run(database.load_all_triples())
        assert rows[0]["predicate"] == "[note]"

    def test_write_memory_tool_with_triple(self, monkeypatch, tmp_path):
        """工具层 execute 接收 predicate+object_text，写入 predicate 正确。"""
        db_path = str(tmp_path / "test.db")
        _fresh_memory(monkeypatch, db_path)

        import database
        asyncio.run(database.init_db())

        import app_state
        # 构造一个可用的 fake loop
        loop = asyncio.new_event_loop()
        monkeypatch.setattr(app_state, "main_loop", loop)

        from tools.write_memory import make_handler
        session = self._make_session()
        handler = make_handler(session)

        async def _run():
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: handler(
                    predicate="职业是",
                    object_text="程序员",
                    source="自我介绍时",
                    motivation="用户告知了职业",
                ),
            )
            return result

        result = loop.run_until_complete(_run())
        loop.close()

        assert result.get("ok") is True
        rows = asyncio.run(database.load_all_triples())
        assert rows[0]["predicate"] == "职业是"
        assert rows[0]["object_text"] == "程序员"

    def test_write_memory_tool_content_only(self, monkeypatch, tmp_path):
        """工具层 execute 只传 content，写入 predicate 为 [note]。"""
        db_path = str(tmp_path / "test.db")
        _fresh_memory(monkeypatch, db_path)

        import database
        asyncio.run(database.init_db())

        import app_state
        loop = asyncio.new_event_loop()
        monkeypatch.setattr(app_state, "main_loop", loop)

        from tools.write_memory import make_handler
        session = self._make_session()
        handler = make_handler(session)

        async def _run():
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: handler(
                    content="旧格式写入测试",
                    source="旧调用",
                    motivation="兼容旧接口",
                ),
            )

        result = loop.run_until_complete(_run())
        loop.close()

        assert result.get("ok") is True
        rows = asyncio.run(database.load_all_triples())
        assert rows[0]["predicate"] == "[note]"

    def test_write_memory_tool_requires_content_or_triple(self, monkeypatch, tmp_path):
        """既不传 content，又不传 predicate+object_text，应返回 error。"""
        db_path = str(tmp_path / "test.db")
        _fresh_memory(monkeypatch, db_path)

        import database
        asyncio.run(database.init_db())

        import app_state
        loop = asyncio.new_event_loop()
        monkeypatch.setattr(app_state, "main_loop", loop)

        from tools.write_memory import make_handler
        session = self._make_session()
        handler = make_handler(session)

        async def _run():
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: handler(source="测试", motivation="测试"),
            )

        result = loop.run_until_complete(_run())
        loop.close()
        assert "error" in result


# ═══════════════════════════════════════════════════════════════
# §2 build_memory_xml — 三元组格式展示
# ═══════════════════════════════════════════════════════════════

class TestXmlTripleFormat:

    def _make_entry(self, subject="User:qq_1", predicate="喜欢", content="打球",
                    source="", reason="", created_at=0):
        import time
        return {
            "id": 1,
            "subject": subject,
            "predicate": predicate,
            "object_text": content,
            "confidence": 0.8,
            "context": "truth",
            "created_at": int(time.time() * 1000) if created_at == 0 else created_at,
            "last_accessed": int(time.time() * 1000),
            "source": source,
            "reason": reason,
            "conv_type": "",
            "conv_id": "",
            "conv_name": "",
        }

    def test_structural_predicate_shows_subject_and_predicate(self):
        """非 [bracket] 谓语时，XML 应包含 <subject> 和 <predicate> 标签。"""
        from memory import build_memory_xml, restore
        entry = self._make_entry(predicate="喜欢")
        restore([entry])
        xml = build_memory_xml(recalled=[entry])
        assert "<subject>" in xml
        assert "<predicate>" in xml
        assert "喜欢" in xml

    def test_note_predicate_hides_subject_and_predicate(self):
        """[note] 谓语时，XML 不应显示 <subject>/<predicate>（避免噪音）。"""
        from memory import build_memory_xml, restore
        entry = self._make_entry(predicate="[note]", content="这是旧格式记忆")
        restore([entry])
        xml = build_memory_xml(recalled=[entry])
        assert "<subject>" not in xml
        assert "<predicate>" not in xml
        assert "这是旧格式记忆" in xml

    def test_other_bracket_predicates_hidden(self):
        """[喜好] 这类 [bracket] 谓语也不显示 subject/predicate。"""
        from memory import build_memory_xml, restore
        entry = self._make_entry(predicate="[喜好]", content="唱歌")
        restore([entry])
        xml = build_memory_xml(recalled=[entry])
        assert "<subject>" not in xml
        assert "<predicate>" not in xml

    def test_mixed_predicates(self):
        """混合条目：结构化条目展示三元组，[note] 条目只展示 content。"""
        from memory import build_memory_xml, restore
        import time
        entries = [
            {**self._make_entry(predicate="喜欢", content="乒乓球"), "id": 1},
            {**self._make_entry(predicate="[note]", content="自由文本备注"), "id": 2},
        ]
        restore(entries)
        xml = build_memory_xml(recalled=entries)
        assert "乒乓球" in xml
        assert "自由文本备注" in xml
        # subject/predicate 出现一次（来自第一条）
        assert xml.count("<subject>") == 1
        assert xml.count("<predicate>") == 1

    def test_html_escaping_in_predicate(self):
        """predicate 字段含特殊字符时应被 HTML 转义。"""
        from memory import build_memory_xml, restore
        entry = self._make_entry(predicate='认为<很重要>', content="安全感")
        restore([entry])
        xml = build_memory_xml(recalled=[entry])
        assert "<very重要>" not in xml  # 不泄露未转义内容
        assert "&lt;" in xml or "认为" in xml  # 转义后存在


# ═══════════════════════════════════════════════════════════════
# §3 recall_memory 工具
# ═══════════════════════════════════════════════════════════════

class TestRecallMemoryTool:

    def _make_session(self, sender_id="12345"):
        return SimpleNamespace(
            context_messages=[{"role": "user", "sender_id": sender_id}],
            conv_type="private",
            conv_id="12345",
            conv_name="",
        )

    def test_recall_returns_relevant(self, monkeypatch, tmp_path):
        """搜索相关关键词时，应返回 found > 0 且 memories 包含相关内容。"""
        db_path = str(tmp_path / "test.db")
        import database
        _fresh_memory(monkeypatch, db_path)

        asyncio.run(database.init_db())

        from memory.tokenizer import tokenize
        asyncio.run(database.write_triple(
            subject="User:qq_12345", predicate="喜欢",
            object_text="喜欢乒乓球运动",
            object_text_tok=tokenize("喜欢乒乓球运动"),
        ))

        import app_state
        loop = asyncio.new_event_loop()
        monkeypatch.setattr(app_state, "main_loop", loop)
        monkeypatch.setattr(app_state, "config", {"memory": {}})

        from tools.recall_memory import make_handler
        session = self._make_session()
        handler = make_handler(session)

        async def _run():
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: handler(query="乒乓球", motivation="测试召回"),
            )

        result = loop.run_until_complete(_run())
        loop.close()

        assert result.get("found", 0) >= 1
        assert any("乒乓球" in m["content"] for m in result.get("memories", []))

    def test_recall_returns_empty_for_no_match(self, monkeypatch, tmp_path):
        """无相关记忆时应返回 found=0。"""
        db_path = str(tmp_path / "test.db")
        import database
        _fresh_memory(monkeypatch, db_path)

        asyncio.run(database.init_db())
        asyncio.run(database.write_triple(
            subject="Bot:self", predicate="[note]",
            object_text="完全无关的内容",
            object_text_tok="完全 无关",
        ))

        import app_state
        loop = asyncio.new_event_loop()
        monkeypatch.setattr(app_state, "main_loop", loop)
        monkeypatch.setattr(app_state, "config", {"memory": {}})

        from tools.recall_memory import make_handler
        handler = make_handler(self._make_session())

        async def _run():
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: handler(query="量子力学超弦理论", motivation="测试"),
            )

        result = loop.run_until_complete(_run())
        loop.close()

        assert result.get("found", 0) == 0

    def test_recall_result_has_required_fields(self, monkeypatch, tmp_path):
        """每条召回结果应包含 id/subject/predicate/content/confidence 字段。"""
        db_path = str(tmp_path / "test.db")
        import database
        _fresh_memory(monkeypatch, db_path)

        asyncio.run(database.init_db())
        from memory.tokenizer import tokenize
        asyncio.run(database.write_triple(
            subject="User:qq_12345", predicate="喜欢",
            object_text="苹果手机",
            object_text_tok=tokenize("苹果手机"),
        ))

        import app_state
        loop = asyncio.new_event_loop()
        monkeypatch.setattr(app_state, "main_loop", loop)
        monkeypatch.setattr(app_state, "config", {"memory": {}})

        from tools.recall_memory import make_handler
        handler = make_handler(self._make_session())

        async def _run():
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: handler(query="苹果", motivation="测试字段完整性"),
            )

        result = loop.run_until_complete(_run())
        loop.close()

        assert result.get("found", 0) >= 1
        for m in result["memories"]:
            assert "id" in m
            assert "subject" in m
            assert "predicate" in m
            assert "content" in m
            assert "confidence" in m


# ═══════════════════════════════════════════════════════════════
# §4 database.update_person_profile
# ═══════════════════════════════════════════════════════════════

class TestUpdatePersonProfile:

    def test_update_existing_person(self, monkeypatch, tmp_path):
        """已有账号时 update_person_profile 应更新 entity_profiles 字段并返回 True。"""
        db_path = str(tmp_path / "test.db")
        import database
        monkeypatch.setattr(database, "DB_PATH", db_path)
        asyncio.run(database.init_db())

        async def _run():
            # 先创建账号
            await database.upsert_account(platform="qq", platform_id="66666", nickname="小红")
            # 更新侧写
            ok = await database.update_person_profile(
                platform_id="66666",
                platform="qq",
                sex="female",
                age=22,
                area="北京",
                notes="喜欢唱歌",
            )
            return ok

        result = asyncio.run(_run())
        assert result is True

        # 验证 DB 中确实写入
        async def _check():
            import aiosqlite
            async with aiosqlite.connect(db_path) as db:
                async with db.execute(
                    "SELECT p.sex, p.age, p.area, p.notes FROM entity_profiles p "
                    "JOIN entities e ON p.profile_id = e.profile_id "
                    "WHERE e.platform='qq' AND e.platform_id='66666'"
                ) as cur:
                    return await cur.fetchone()

        row = asyncio.run(_check())
        assert row[0] == "female"
        assert row[1] == 22
        assert row[2] == "北京"
        assert row[3] == "喜欢唱歌"

    def test_update_nonexistent_person_returns_false(self, monkeypatch, tmp_path):
        """不存在的 platform_id 应返回 False。"""
        db_path = str(tmp_path / "test.db")
        import database
        monkeypatch.setattr(database, "DB_PATH", db_path)
        asyncio.run(database.init_db())

        result = asyncio.run(
            database.update_person_profile(platform_id="00000", sex="male")
        )
        assert result is False

    def test_update_partial_fields(self, monkeypatch, tmp_path):
        """只传部分字段时，只更新那些字段，其余保持 NULL。"""
        db_path = str(tmp_path / "test.db")
        import database
        monkeypatch.setattr(database, "DB_PATH", db_path)
        asyncio.run(database.init_db())

        async def _run():
            await database.upsert_account(platform="qq", platform_id="77777", nickname="测试")
            await database.update_person_profile(platform_id="77777", area="上海")

        asyncio.run(_run())

        async def _check():
            import aiosqlite
            async with aiosqlite.connect(db_path) as db:
                async with db.execute(
                    "SELECT p.sex, p.area FROM entity_profiles p "
                    "JOIN entities e ON p.profile_id = e.profile_id "
                    "WHERE e.platform_id='77777'"
                ) as cur:
                    return await cur.fetchone()

        row = asyncio.run(_check())
        assert row[0] is None   # sex 未更新，保持 NULL
        assert row[1] == "上海"

    def test_update_no_fields_returns_true(self, monkeypatch, tmp_path):
        """不传任何字段时，应直接返回 True（无需 UPDATE）。"""
        db_path = str(tmp_path / "test.db")
        import database
        monkeypatch.setattr(database, "DB_PATH", db_path)
        asyncio.run(database.init_db())

        async def _run():
            await database.upsert_account(platform="qq", platform_id="88888", nickname="空更新")
            return await database.update_person_profile(platform_id="88888")

        result = asyncio.run(_run())
        assert result is True


# ═══════════════════════════════════════════════════════════════
# §5 update_person_profile 工具
# ═══════════════════════════════════════════════════════════════

class TestUpdatePersonProfileTool:

    def _make_session(self):
        return SimpleNamespace(
            context_messages=[{"role": "user", "sender_id": "55555"}],
            conv_type="private",
            conv_id="55555",
            conv_name="",
        )

    def test_tool_updates_successfully(self, monkeypatch, tmp_path):
        """工具层端到端：调用后 DB 中 entity_profiles 字段正确更新。"""
        db_path = str(tmp_path / "test.db")
        import database
        monkeypatch.setattr(database, "DB_PATH", db_path)
        asyncio.run(database.init_db())
        asyncio.run(database.upsert_account(platform="qq", platform_id="55555", nickname="工具测试用户"))

        import app_state
        loop = asyncio.new_event_loop()
        monkeypatch.setattr(app_state, "main_loop", loop)

        from tools.update_person_profile import make_handler
        handler = make_handler(self._make_session())

        async def _run():
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: handler(
                    platform_id="55555",
                    updates={"sex": "male", "age": 28, "notes": "开朗健谈"},
                    motivation="从对话中推断",
                ),
            )

        result = loop.run_until_complete(_run())
        loop.close()

        assert result.get("ok") is True
        assert set(result.get("updated_fields", [])) == {"sex", "age", "notes"}

    def test_tool_rejects_unknown_fields(self, monkeypatch, tmp_path):
        """updates 中含非法字段时（如 platform_id 注入），应过滤后若无合法字段报 error。"""
        db_path = str(tmp_path / "test.db")
        import database
        monkeypatch.setattr(database, "DB_PATH", db_path)
        asyncio.run(database.init_db())

        import app_state
        loop = asyncio.new_event_loop()
        monkeypatch.setattr(app_state, "main_loop", loop)

        from tools.update_person_profile import make_handler
        handler = make_handler(self._make_session())

        async def _run():
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: handler(
                    platform_id="55555",
                    updates={"platform_id": "00000", "person_id": "hack"},  # 非法字段
                    motivation="测试安全过滤",
                ),
            )

        result = loop.run_until_complete(_run())
        loop.close()

        assert "error" in result

    def test_tool_empty_updates(self, monkeypatch, tmp_path):
        """updates 为空 dict 时，应返回 error。"""
        db_path = str(tmp_path / "test.db")
        import database
        monkeypatch.setattr(database, "DB_PATH", db_path)
        asyncio.run(database.init_db())

        import app_state
        loop = asyncio.new_event_loop()
        monkeypatch.setattr(app_state, "main_loop", loop)

        from tools.update_person_profile import make_handler
        handler = make_handler(self._make_session())

        async def _run():
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: handler(platform_id="55555", updates={}, motivation="测试"),
            )

        result = loop.run_until_complete(_run())
        loop.close()

        assert "error" in result

    def test_tool_not_found(self, monkeypatch, tmp_path):
        """不存在的 platform_id 返回 ok=False，不报异常。"""
        db_path = str(tmp_path / "test.db")
        import database
        monkeypatch.setattr(database, "DB_PATH", db_path)
        asyncio.run(database.init_db())

        import app_state
        loop = asyncio.new_event_loop()
        monkeypatch.setattr(app_state, "main_loop", loop)

        from tools.update_person_profile import make_handler
        handler = make_handler(self._make_session())

        async def _run():
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: handler(
                    platform_id="00000",
                    updates={"sex": "male"},
                    motivation="测试不存在",
                ),
            )

        result = loop.run_until_complete(_run())
        loop.close()

        assert result.get("ok") is False
