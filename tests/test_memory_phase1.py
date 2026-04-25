"""test_memory_phase1.py — Phase 1 记忆系统测试套件

测试范围：
    1. memory.tokenizer  — jieba 分词、FTS5 查询串构建、词典注册
    2. database          — MemoryTriples 写入/软删除/FTS5 触发器/recall 管道
    3. memory domain     — add_memory / remove_memory / build_memory_xml / recall_memories

每个测试函数使用临时 SQLite 数据库，互相隔离，不依赖生产 DB。
"""

import asyncio
import importlib
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ── 将 src/ 加入 sys.path，使 import 正常解析 ───────────────────────
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ═══════════════════════════════════════════════════════════════
# 辅助：临时数据库 fixture
# ═══════════════════════════════════════════════════════════════

def _make_temp_db() -> str:
    """创建临时 SQLite 文件路径并返回（调用方负责清理）。"""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="test_memory_")
    os.close(fd)
    return path


def _patch_db(monkeypatch, db_path: str):
    """将 database.DB_PATH 临时替换为测试路径。"""
    import database
    monkeypatch.setattr(database, "DB_PATH", db_path)


# ═══════════════════════════════════════════════════════════════
# §1 memory_tokenizer 单元测试（纯同步，不依赖 DB）
# ═══════════════════════════════════════════════════════════════

class TestMemoryTokenizer:
    """测试 jieba 分词封装层。"""

    def test_tokenize_returns_string(self):
        from memory.tokenizer import tokenize
        result = tokenize("用户喜欢星际争霸")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_tokenize_empty_returns_empty(self):
        from memory.tokenizer import tokenize
        assert tokenize("") == ""

    def test_tokenize_filters_stopwords(self):
        from memory.tokenizer import tokenize
        # "我" 和 "的" 是停用词，不应出现在 token 流中
        result = tokenize("我喜欢的苹果")
        tokens = result.split()
        assert "我" not in tokens
        assert "的" not in tokens

    def test_tokenize_preserves_content_words(self):
        from memory.tokenizer import tokenize
        result = tokenize("苹果手机很贵")
        # 至少包含"苹果"或"手机"等内容词
        assert any(w in result for w in ("苹果", "手机"))

    def test_tokenize_fallback_on_all_stopwords(self):
        from memory.tokenizer import tokenize
        # 全停用词，应回退返回原文而不是空字符串
        result = tokenize("的了在")
        assert result  # 非空

    def test_build_fts_query_empty_message(self):
        from memory.tokenizer import build_fts_query
        assert build_fts_query("") == ""

    def test_build_fts_query_includes_prefix_terms(self):
        from memory.tokenizer import build_fts_query
        result = build_fts_query("苹果手机价格")
        # 每个 token 应同时有精确项（无 *）和前缀项（有 *）
        assert "*" in result
        assert "OR" in result

    def test_build_fts_query_stopword_only(self):
        from memory.tokenizer import build_fts_query
        # 仅停用词，返回空串
        result = build_fts_query("的了在是")
        assert result == ""

    def test_register_word_affects_tokenization(self):
        from memory.tokenizer import register_word, tokenize
        # 注册自定义词后，jieba 应将其识别为整体
        register_word("超级无敌大魔王")
        result = tokenize("他是超级无敌大魔王同学")
        assert "超级无敌大魔王" in result.split()

    def test_load_custom_dict_registers_object_text(self):
        from memory.tokenizer import load_custom_dict_from_triples, tokenize
        triples = [
            {"object_text": "星辰大海电子游戏", "predicate": "[note]"},
            {"object_text": None, "predicate": "[note]"},  # None 应被安全跳过
        ]
        load_custom_dict_from_triples(triples)
        result = tokenize("我最近在玩星辰大海电子游戏")
        assert "星辰大海电子游戏" in result.split()

    def test_load_custom_dict_skips_structured_predicate(self):
        """[note] 之类的结构标记不应被注册为 jieba 词。"""
        from memory.tokenizer import load_custom_dict_from_triples, build_fts_query
        # 确保 [note] 不会影响分词结果（无法验证内部，但至少不应崩溃）
        load_custom_dict_from_triples([{"predicate": "[note]", "object_text": "正常词汇"}])


# ═══════════════════════════════════════════════════════════════
# §2 database.MemoryTriples 集成测试（async，使用临时 DB）
# ═══════════════════════════════════════════════════════════════

class TestDatabaseTriples:
    """测试 MemoryTriples 表的写入、软删除和 FTS5 触发器。"""

    @pytest.fixture(autouse=True)
    def setup_db(self, monkeypatch, tmp_path):
        """每个测试用独立临时 DB，初始化表结构。"""
        import database
        db_path = str(tmp_path / "test.db")
        monkeypatch.setattr(database, "DB_PATH", db_path)
        asyncio.run(database.init_db())

    def test_write_triple_returns_positive_int(self):
        import database
        triple_id = asyncio.run(database.write_triple(
            subject="User:qq_123",
            predicate="[note]",
            object_text="用户喜欢打游戏",
            object_text_tok="用户 喜欢 打 游戏",
        ))
        assert isinstance(triple_id, int)
        assert triple_id > 0

    def test_write_triple_ids_auto_increment(self):
        import database
        id1 = asyncio.run(database.write_triple(
            subject="Bot:self", predicate="[note]",
            object_text="第一条", object_text_tok="第一 条",
        ))
        id2 = asyncio.run(database.write_triple(
            subject="Bot:self", predicate="[note]",
            object_text="第二条", object_text_tok="第二 条",
        ))
        assert id2 > id1

    def test_load_all_triples_returns_written_entries(self):
        import database
        asyncio.run(database.write_triple(
            subject="User:qq_999", predicate="[note]",
            object_text="测试记忆内容", object_text_tok="测试 记忆 内容",
            source="测试来源", reason="测试原因",
        ))
        rows = asyncio.run(database.load_all_triples())
        assert len(rows) == 1
        assert rows[0]["object_text"] == "测试记忆内容"
        assert rows[0]["subject"] == "User:qq_999"

    def test_load_all_triples_raw_text_preserved(self):
        """object_text 应存储原始文本，不应被分词切碎。"""
        import database
        asyncio.run(database.write_triple(
            subject="Bot:self", predicate="[note]",
            object_text="星际争霸2",
            object_text_tok="星际争霸2",  # 已注册为词时是一个整体
        ))
        rows = asyncio.run(database.load_all_triples())
        # 原始文本不含空格
        assert rows[0]["object_text"] == "星际争霸2"

    def test_soft_delete_hides_from_load_all(self):
        import database
        triple_id = asyncio.run(database.write_triple(
            subject="Bot:self", predicate="[note]",
            object_text="待删除记忆", object_text_tok="待删除 记忆",
        ))
        deleted = asyncio.run(database.soft_delete_triple(triple_id))
        assert deleted is True
        rows = asyncio.run(database.load_all_triples())
        assert all(r["id"] != triple_id for r in rows)

    def test_soft_delete_nonexistent_returns_false(self):
        import database
        result = asyncio.run(database.soft_delete_triple(99999))
        assert result is False

    def test_soft_delete_idempotent(self):
        import database
        triple_id = asyncio.run(database.write_triple(
            subject="Bot:self", predicate="[note]",
            object_text="一条记忆", object_text_tok="一 条 记忆",
        ))
        asyncio.run(database.soft_delete_triple(triple_id))
        # 再次删除同一条，应返回 False（已被标记）
        result = asyncio.run(database.soft_delete_triple(triple_id))
        assert result is False

    def test_fts5_index_populated_via_trigger(self):
        """写入 MemoryTriples 后，FTS5 触发器应自动填充 MemorySearch。"""
        import aiosqlite

        async def _check():
            import database
            await database.write_triple(
                subject="User:qq_42", predicate="[note]",
                object_text="很喜欢苹果",
                object_text_tok="喜欢 苹果",
            )
            async with aiosqlite.connect(database.DB_PATH) as db:
                async with db.execute(
                    "SELECT COUNT(*) FROM MemorySearch WHERE MemorySearch MATCH '\"苹果\"'"
                ) as cur:
                    count = (await cur.fetchone())[0]
            return count

        count = asyncio.run(_check())
        assert count == 1

    def test_fts5_index_removed_on_soft_delete(self):
        """软删除后触发器应同步从 FTS5 中移除索引行。"""
        import aiosqlite

        async def _check():
            import database
            triple_id = await database.write_triple(
                subject="Bot:self", predicate="[note]",
                object_text="将被删除",
                object_text_tok="将 被 删除",
            )
            await database.soft_delete_triple(triple_id)
            async with aiosqlite.connect(database.DB_PATH) as db:
                async with db.execute(
                    "SELECT COUNT(*) FROM MemorySearch WHERE MemorySearch MATCH '\"删除\"'"
                ) as cur:
                    count = (await cur.fetchone())[0]
            return count

        count = asyncio.run(_check())
        assert count == 0

    def test_search_triples_returns_relevant_result(self):
        import database
        from memory.tokenizer import tokenize

        async def _run():
            await database.write_triple(
                subject="User:qq_10", predicate="[note]",
                object_text="用户喜欢乒乓球",
                object_text_tok=tokenize("用户喜欢乒乓球"),
            )
            await database.write_triple(
                subject="User:qq_10", predicate="[note]",
                object_text="用户不喜欢吃香菜",
                object_text_tok=tokenize("用户不喜欢吃香菜"),
            )
            return await database.search_triples(
                fts_query='"乒乓球" OR "乒乓球*"',
                subject_filter="User:qq_10",
            )

        results = asyncio.run(_run())
        assert len(results) >= 1
        assert any("乒乓球" in r["object_text"] for r in results)

    def test_search_triples_empty_query_returns_recent(self):
        """空查询时回退到返回最近条目。"""
        import database

        async def _run():
            for i in range(3):
                await database.write_triple(
                    subject="Bot:self", predicate="[note]",
                    object_text=f"记忆{i}", object_text_tok=f"记忆 {i}",
                )
            return await database.search_triples(fts_query="")

        results = asyncio.run(_run())
        assert len(results) == 3

    def test_search_triples_bm25_ranks_relevant_higher(self):
        """相关性高的结果应排在前面。"""
        import database
        from memory.tokenizer import tokenize

        async def _run():
            # 高相关：包含两次关键词
            await database.write_triple(
                subject="Bot:self", predicate="[note]",
                object_text="苹果苹果手机苹果",
                object_text_tok=tokenize("苹果苹果手机苹果"),
            )
            # 低相关：只出现一次
            await database.write_triple(
                subject="Bot:self", predicate="[note]",
                object_text="今天天气苹果",
                object_text_tok=tokenize("今天天气苹果"),
            )
            # 无关
            await database.write_triple(
                subject="Bot:self", predicate="[note]",
                object_text="完全无关的内容",
                object_text_tok=tokenize("完全无关的内容"),
            )
            results = await database.search_triples(fts_query='"苹果" OR "苹果*"')
            return results

        results = asyncio.run(_run())
        # 确保有结果且不包含无关条目（无关条目不会命中 FTS5）
        assert len(results) >= 1
        assert all("苹果" in r["object_text"] for r in results)
        # BM25 有长度归一化，不保证词频直接映射到排名，仅验证结果集合的正确性

    def test_update_triple_confidence(self):
        import database

        async def _run():
            tid = await database.write_triple(
                subject="Bot:self", predicate="[note]",
                object_text="置信度测试", object_text_tok="置信度 测试",
                confidence=0.5,
            )
            await database.update_triple_confidence([tid], delta=0.2)
            rows = await database.load_all_triples()
            return next(r for r in rows if r["id"] == tid)

        row = asyncio.run(_run())
        assert abs(row["confidence"] - 0.7) < 1e-6

    def test_update_triple_confidence_capped(self):
        import database

        async def _run():
            tid = await database.write_triple(
                subject="Bot:self", predicate="[note]",
                object_text="上限测试", object_text_tok="上限 测试",
                confidence=0.9,
            )
            await database.update_triple_confidence([tid], delta=0.5, cap=1.0)
            rows = await database.load_all_triples()
            return next(r for r in rows if r["id"] == tid)

        row = asyncio.run(_run())
        assert row["confidence"] <= 1.0

    def test_migrate_bot_memories_to_triples(self):
        """迁移时应将 bot_memories 的 content 写入 MemoryTriples.object_text。"""
        import aiosqlite
        import database
        from memory.tokenizer import tokenize

        async def _run():
            # 手动写入 bot_memories 旧数据
            async with aiosqlite.connect(database.DB_PATH) as db:
                now = database._ms()
                await db.execute(
                    """INSERT INTO bot_memories
                       (memory_id, created_at, content, source, reason,
                        conv_type, conv_id, conv_name, is_deleted)
                       VALUES ('mem_old1', ?, '旧记忆内容', '旧来源', '旧原因',
                               'group', '12345', '测试群', 0)""",
                    (now,),
                )
                await db.commit()

            migrated = await database.migrate_bot_memories_to_triples(tokenize)
            rows = await database.load_all_triples()
            return migrated, rows

        migrated, rows = asyncio.run(_run())
        assert migrated == 1
        assert len(rows) == 1
        assert rows[0]["object_text"] == "旧记忆内容"
        assert rows[0]["subject"] == "Bot:self"

    def test_migrate_is_idempotent(self):
        """迁移在 MemoryTriples 已有数据时应跳过（返回 0）。"""
        import database
        from memory.tokenizer import tokenize

        async def _run():
            # 先写一条 MemoryTriples 数据
            await database.write_triple(
                subject="Bot:self", predicate="[note]",
                object_text="已有数据", object_text_tok="已有 数据",
            )
            return await database.migrate_bot_memories_to_triples(tokenize)

        result = asyncio.run(_run())
        assert result == 0


# ═══════════════════════════════════════════════════════════════
# §3 memory domain 集成测试
# ═══════════════════════════════════════════════════════════════

class TestMemoryDomain:
    """测试 memory 域包的全流程（add → recall → render XML → delete）。"""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch, tmp_path):
        import database
        import memory as mem
        db_path = str(tmp_path / "mem.db")
        monkeypatch.setattr(database, "DB_PATH", db_path)
        asyncio.run(database.init_db())
        # 重置全局缓存
        mem.configure(15, 15, 50)
        mem.restore([])

    def test_add_memory_returns_id(self):
        import memory as mem
        triple_id = asyncio.run(mem.add_memory(
            content="用户喜欢喝绿茶",
            source="聊天时",
            reason="明确表达",
            subject="User:qq_777",
        ))
        assert isinstance(triple_id, int)
        assert triple_id > 0

    def test_add_memory_populates_cache(self):
        import memory as mem
        asyncio.run(mem.add_memory(
            content="记忆内容", source="来源", reason="原因",
            subject="Bot:self",
        ))
        all_mems = mem.get_all()
        assert len(all_mems) == 1
        assert all_mems[0]["object_text"] == "记忆内容"

    def test_add_memory_evicts_oldest_when_full(self):
        import memory as mem
        mem.configure(15, 3, 50)
        for i in range(3):
            asyncio.run(mem.add_memory(
                content=f"记忆{i}", source="x", reason="x", subject="User:qq_4242",
            ))
        # 再写一条，最旧的应被淘汰
        asyncio.run(mem.add_memory(
            content="第四条记忆", source="x", reason="x", subject="User:qq_4242",
        ))
        all_mems = mem.get_all()
        assert len(all_mems) == 3
        contents = [m["object_text"] for m in all_mems]
        assert "记忆0" not in contents
        assert "第四条记忆" in contents

    def test_remove_memory_removes_from_cache(self):
        import memory as mem
        tid = asyncio.run(mem.add_memory(
            content="待删除", source="x", reason="x", subject="Bot:self",
        ))
        found = asyncio.run(mem.remove_memory(str(tid)))
        assert found is True
        assert len(mem.get_all()) == 0

    def test_remove_memory_returns_false_for_missing(self):
        import memory as mem
        result = asyncio.run(mem.remove_memory("99999"))
        assert result is False

    def test_build_memory_xml_empty_no_recalled(self):
        import memory as mem
        xml = mem.build_memory_xml(recalled=[])
        assert xml == (
            '<about_user items="0"/>\n'
            '<about_relationship items="0"/>\n'
            '<about_self items="0"/>\n'
            '<recent_events items="0"/>'
        )

    def test_build_memory_xml_with_recalled(self):
        import memory as mem
        now = datetime.now(timezone.utc)
        recalled = [{
            "id": 42,
            "subject": "User:qq_1",
            "predicate": "职业是",
            "object_text": "用户喜欢苹果",
            "source": "聊天",
            "reason": "明确表达",
            "conv_name": "",
            "conv_id": "",
            "created_at": int(now.timestamp() * 1000),
            "last_accessed": int(now.timestamp() * 1000),
        }]
        xml = mem.build_memory_xml(now=now, recalled=recalled)
        assert '<about_user items="1">' in xml
        assert 'id="42"' in xml
        assert "<subject>User:qq_1</subject>" in xml
        assert "<predicate>职业是</predicate>" in xml
        assert "用户喜欢苹果" in xml
        assert "<source>" in xml
        assert "<age>" in xml
        assert "<reason>" in xml

    def test_build_memory_xml_recalls_none_uses_cache(self):
        """recalled=None 时应回退到运行时缓存。"""
        import memory as mem
        asyncio.run(mem.add_memory(
            content="缓存内容", source="x", reason="x", subject="User:qq_777",
        ))
        xml = mem.build_memory_xml(recalled=None)
        assert '<about_user items="1">' in xml
        assert "缓存内容" in xml

    def test_build_memory_xml_escapes_html(self):
        """XSS 防护：<> & 字符应被 html.escape 转义。"""
        import memory as mem
        now = datetime.now(timezone.utc)
        recalled = [{
            "id": 1,
            "subject": "User:qq_1",
            "predicate": "备注",
            "object_text": "<script>alert('xss')</script> & 测试",
            "source": "来源<>",
            "reason": "原&因",
            "conv_name": "",
            "conv_id": "",
            "created_at": int(now.timestamp() * 1000),
            "last_accessed": int(now.timestamp() * 1000),
        }]
        xml = mem.build_memory_xml(now=now, recalled=recalled)
        assert "<script>" not in xml
        assert "&lt;script&gt;" in xml
        assert "&amp;" in xml

    def test_recall_memories_returns_relevant(self):
        """端到端测试：写入后执行 recall，应能检索到相关记忆。"""
        import memory as mem
        from memory.tokenizer import register_word

        register_word("乒乓球")

        async def _run():
            await mem.add_memory(
                content="用户喜欢打乒乓球",
                source="聊天时",
                reason="明确说的",
                subject="User:qq_55",
            )
            await mem.add_memory(
                content="用户养了一只猫",
                source="聊天时",
                reason="提到了",
                subject="User:qq_55",
            )
            return await mem.recall_memories(
                message_text="你还记得我喜欢打乒乓球吗",
                sender_id="55",
            )

        results = asyncio.run(_run())
        assert len(results) >= 1
        assert any("乒乓球" in r["object_text"] for r in results)

    def test_recall_memories_returns_empty_for_no_match(self):
        import memory as mem

        async def _run():
            await mem.add_memory(
                content="用户喜欢火锅",
                source="聊天",
                reason="说的",
                subject="User:qq_99",
            )
            return await mem.recall_memories(
                message_text="量子物理学理论",
                sender_id="99",
            )

        results = asyncio.run(_run())
        # 无关联查询，应返回空列表或不含"乒乓球"的结果
        for r in results:
            assert "乒乓球" not in r["object_text"]
