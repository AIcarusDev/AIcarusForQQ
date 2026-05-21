"""tests/test_archiver_multimodal.py — 归档器多模态支持冒烟测试

验证 archiver.py 的以下改动：
1. _content_to_text / _append_to_content 辅助函数
2. archive_turn_memories 对 str / list 两种 build_chat_log_xml 返回值
   都能正确构建 payload["chat_content"]
3. _run_archive_job 把 chat_content（多模态）而非 dialogue（纯文本）
   传给 adapter._call_forced_tool
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memory.archiver import _append_to_content, _content_to_text


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 辅助函数单元测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestContentToText(unittest.TestCase):
    def test_str_passthrough(self):
        assert _content_to_text("hello world") == "hello world"

    def test_empty_str(self):
        assert _content_to_text("") == ""

    def test_list_single_text_part(self):
        parts = [{"type": "text", "text": "foo bar"}]
        assert _content_to_text(parts) == "foo bar"

    def test_list_skips_image_parts(self):
        parts = [
            {"type": "text", "text": "before "},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
            {"type": "text", "text": "after"},
        ]
        assert _content_to_text(parts) == "before after"

    def test_list_empty(self):
        assert _content_to_text([]) == ""


class TestAppendToContent(unittest.TestCase):
    def test_str_appends(self):
        result = _append_to_content("hello", "<block>x</block>")
        assert result == "hello\n\n<block>x</block>"

    def test_list_adds_new_text_part(self):
        parts = [{"type": "text", "text": "hello"}]
        result = _append_to_content(parts, "<block>x</block>")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[-1] == {"type": "text", "text": "\n\n<block>x</block>"}

    def test_list_does_not_mutate_original(self):
        original = [{"type": "text", "text": "hello"}]
        _append_to_content(original, "extra")
        assert len(original) == 1  # 原列表未被修改

    def test_list_with_image_part_appends_at_end(self):
        parts = [
            {"type": "text", "text": "A"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,xyz"}},
        ]
        result = _append_to_content(parts, "SUFFIX")
        assert result[-1]["type"] == "text"
        assert "SUFFIX" in result[-1]["text"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. archive_turn_memories：payload["chat_content"] 构建验证
# ═══════════════════════════════════════════════════════════════════════════════

def _make_session(chat_log_xml_return):
    """构造最小化 mock session，build_chat_log_xml 返回指定值。"""
    session = MagicMock()
    session.conv_type = "group"
    session.conv_id = "643700843"
    session.conv_name = "测试群"
    session.context_messages = [
        {
            "role": "user",
            "content": "hello world",
            "sender_id": "12345",
            "sender_name": "Alice",
            "message_id": "m1",
        },
    ]
    session.build_chat_log_xml = MagicMock(return_value=chat_log_xml_return)
    return session


def _mock_app_state():
    """构造最小化 mock app_state。"""
    adapter_mock = MagicMock()
    adapter_mock._call_forced_tool = MagicMock(return_value={"events": []})

    state = SimpleNamespace(
        config={
            "memory": {
                "auto_archive": {
                    "enabled": True,
                    "context_turns": 3,
                }
            }
        },
        archiver_adapter=adapter_mock,
        archive_tasks=set(),
    )
    return state


class TestArchiveTurnMemoriesPayload(unittest.IsolatedAsyncioTestCase):
    """验证 archive_turn_memories 在 str / list 两种情况下正确构建 payload。"""

    async def _run_and_capture_payload(self, chat_log_xml_return):
        """运行 archive_turn_memories 并捕获传给 _run_archive_job 的 payload。"""
        import memory.archiver as archiver_mod

        session = _make_session(chat_log_xml_return)
        app_state_mock = _mock_app_state()

        captured: dict = {}

        async def fake_run_archive_job(payload):
            captured.update(payload)

        with (
            patch.object(archiver_mod, "_run_archive_job", side_effect=fake_run_archive_job),
            patch("app_state.config", app_state_mock.config),
            patch("app_state.archiver_adapter", app_state_mock.archiver_adapter),
            patch("database.enqueue_archive_job", new=AsyncMock(return_value=42)),
            patch("database.load_archive_signatures", new=AsyncMock(return_value={})),
            patch("database.save_archive_signature", new=AsyncMock()),
            patch(
                "memory.repo.events.prefetch_candidates_for_archiver",
                new=AsyncMock(return_value=[]),
            ),
        ):
            # 重置签名缓存，确保窗口变化触发归档
            archiver_mod._LAST_ARCHIVED_SIG.clear()
            archiver_mod._sig_loaded = True

            await archiver_mod.archive_turn_memories(session, "12345", [])

        return captured

    async def test_str_chat_log_xml_builds_str_chat_content(self):
        xml_str = "<conversation><chat_logs>...</chat_logs></conversation>"
        payload = await self._run_and_capture_payload(xml_str)

        self.assertIn("chat_content", payload)
        # str 路径：scene prefix 注入后仍是 str
        self.assertIsInstance(payload["chat_content"], str)
        self.assertIn("[场景: group/643700843]", payload["chat_content"])
        self.assertIn(xml_str, payload["chat_content"])
        # dialogue（纯文本）与 chat_content 内容一致
        self.assertEqual(payload["dialogue"], payload["chat_content"])

    async def test_list_chat_log_xml_builds_list_chat_content(self):
        xml_parts = [
            {"type": "text", "text": "<conversation>text part</conversation>"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/AAA="}},
            {"type": "text", "text": "</chat_logs></conversation>"},
        ]
        payload = await self._run_and_capture_payload(xml_parts)

        self.assertIn("chat_content", payload)
        # list 路径：chat_content 仍是 list
        self.assertIsInstance(payload["chat_content"], list)

        # 第一个 text part 含场景前缀
        first_text = next(p for p in payload["chat_content"] if p.get("type") == "text")
        self.assertIn("[场景: group/643700843]", first_text["text"])

        # 图片 part 原样保留
        image_parts = [p for p in payload["chat_content"] if p.get("type") == "image_url"]
        self.assertEqual(len(image_parts), 1)
        self.assertIn("base64", image_parts[0]["image_url"]["url"])

        # dialogue 是纯文本（无图片 base64）
        self.assertIsInstance(payload["dialogue"], str)
        self.assertNotIn("base64", payload["dialogue"])

    async def test_aliases_block_appended_to_both(self):
        """<member_aliases> 同时追加到 chat_content 和 dialogue。"""
        xml_parts = [{"type": "text", "text": "<conversation>...</conversation>"}]
        payload = await self._run_and_capture_payload(xml_parts)

        alias_block = "<member_aliases>"
        # chat_content list 的最后若干 text parts 应含 aliases
        text_in_content = _content_to_text(payload["chat_content"])
        self.assertIn(alias_block, text_in_content)
        self.assertIn(alias_block, payload["dialogue"])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. _run_archive_job：adapter 接收 chat_content 而非 dialogue
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunArchiveJobUsesMultimodal(unittest.IsolatedAsyncioTestCase):

    async def test_adapter_receives_list_chat_content(self):
        """当 payload 含 list 类型的 chat_content 时，LLM 收到的是 list，不是 str。"""
        import memory.archiver as archiver_mod
        from concurrent.futures import Future

        multimodal_parts = [
            {"type": "text", "text": "[场景: group/123]\n<conversation>A: hi</conversation>"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/AAA="}},
        ]
        plain_dialogue = "[场景: group/123]\n<conversation>A: hi</conversation>"

        captured_user_content: list = []

        def fake_call_forced_tool(system, user_content, gen, tool_decl, tag):
            captured_user_content.append(user_content)
            fut: Future = Future()
            fut.set_result({"events": []})
            return fut

        adapter_mock = MagicMock()
        adapter_mock._call_forced_tool = fake_call_forced_tool

        payload = {
            "job_id": 999,
            "conv_type": "group",
            "conv_id": "123",
            "conv_name": "TestGroup",
            "sender_id": "12345",
            "dialogue": plain_dialogue,
            "chat_content": multimodal_parts,
            "signature": "abc123",
            "prev_signature": "",
            "valid_candidate_ids": [],
        }

        with (
            patch("app_state.archiver_adapter", adapter_mock),
            patch("database.delete_archive_job", new=AsyncMock()),
        ):
            archiver_mod._LAST_ARCHIVED_SIG[("group", "123")] = "abc123"
            await archiver_mod._run_archive_job(payload)

        self.assertEqual(len(captured_user_content), 1)
        received = captured_user_content[0]
        # 必须是 list（多模态），不是 str
        self.assertIsInstance(received, list)
        self.assertIs(received, multimodal_parts)

    async def test_adapter_falls_back_to_dialogue_when_no_chat_content(self):
        """payload 缺少 chat_content 时（DB 恢复路径），LLM 收到 dialogue str。"""
        import memory.archiver as archiver_mod
        from concurrent.futures import Future

        plain_dialogue = "[场景: group/123]\nA: hello"
        captured_user_content: list = []

        def fake_call_forced_tool(system, user_content, gen, tool_decl, tag):
            captured_user_content.append(user_content)
            fut: Future = Future()
            fut.set_result({"events": []})
            return fut

        adapter_mock = MagicMock()
        adapter_mock._call_forced_tool = fake_call_forced_tool

        payload = {
            "job_id": 998,
            "conv_type": "group",
            "conv_id": "123",
            "conv_name": "TestGroup",
            "sender_id": "12345",
            "dialogue": plain_dialogue,
            # chat_content 故意缺失（模拟从 DB 恢复的旧 job）
            "signature": "xyz789",
            "prev_signature": "",
            "valid_candidate_ids": [],
        }

        with (
            patch("app_state.archiver_adapter", adapter_mock),
            patch("database.delete_archive_job", new=AsyncMock()),
        ):
            archiver_mod._LAST_ARCHIVED_SIG[("group", "123")] = "xyz789"
            await archiver_mod._run_archive_job(payload)

        self.assertEqual(len(captured_user_content), 1)
        received = captured_user_content[0]
        # 降级路径：应为 str（dialogue）
        self.assertIsInstance(received, str)
        self.assertEqual(received, plain_dialogue)


if __name__ == "__main__":
    unittest.main()
