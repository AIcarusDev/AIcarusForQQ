import asyncio
import base64
import io
import os
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import database
from database import init_db
from llm.prompt.user_prompt_builder import build_main_user_prompt
from llm.session import create_session
from qq_adapter.events import expand_forward_previews
from tools.browse_forward_view import (
    DECLARATION as BROWSE_FORWARD_VIEW_DECLARATION,
    make_handler as make_browse_forward_view_handler,
)
from tools.open_forward_message import (
    DECLARATION as OPEN_FORWARD_MESSAGE_DECLARATION,
    make_handler as make_open_forward_message_handler,
)


def _forward_entry(message_id: str, forward_id: str) -> dict:
    return {
        "role": "user",
        "message_id": message_id,
        "sender_id": "42",
        "sender_name": "Alice",
        "sender_role": "member",
        "timestamp": "2026-04-29T00:00:00+08:00",
        "content": "[合并转发]",
        "content_type": "forward",
        "content_segments": [{"type": "forward", "forward_id": forward_id, "title": "合并转发", "preview": [], "total": 0}],
    }


def _image_b64(fmt: str) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (80, 120, 160)).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class FakeQQAdapterClient:
    connected = True
    bot_id = "999"

    def __init__(self, loop, responses):
        self._loop = loop
        self.responses = responses
        self.calls = []
        self.last_api_error = None

    async def send_api(self, action: str, params: dict, timeout: float = 15.0):
        self.calls.append((action, dict(params)))
        key = params.get("id")
        if key is None and action == "get_group_msg_history":
            key = ("history", str(params.get("group_id", "")))
        result = self.responses.get(key)
        if result is None:
            self.last_api_error = {
                "action": action,
                "status": "failed",
                "message": "消息已过期或者为内层消息，无法获取转发消息",
            }
        else:
            self.last_api_error = None
        return result


class ForwardBrowserTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._old_db_path = database.DB_PATH
        database.DB_PATH = os.path.join(self._tmpdir.name, "test_forward_browser.db")
        await init_db()

    async def asyncTearDown(self) -> None:
        database.DB_PATH = self._old_db_path
        try:
            self._tmpdir.cleanup()
        except PermissionError:
            pass

    def _session(self):
        session = create_session()
        session.set_conversation_meta("group", "1", "测试群")
        session._qq_id = "999"
        session._qq_name = "Bot"
        return session

    def test_forward_message_is_rendered_as_openable(self) -> None:
        session = self._session()
        session.add_to_context(_forward_entry("101", "root-fwd"))

        prompt = build_main_user_prompt(session, consume_unread=False)

        self.assertIn('<message id="101"', prompt)
        self.assertIn('<content type="forward" openable="true">', prompt)

    async def test_failed_forward_preview_is_marked_as_unavailable(self) -> None:
        loop = asyncio.get_running_loop()
        client = FakeQQAdapterClient(loop, {"expired-fwd": None})
        entry = _forward_entry("101", "expired-fwd")
        entry["content_segments"][0]["_needs_expand"] = True

        await expand_forward_previews(entry, client)

        session = self._session()
        session.add_to_context(entry)
        prompt = build_main_user_prompt(session, consume_unread=False)
        self.assertIn("<error>消息已过期或者为内层消息，无法获取转发消息</error>", prompt)

    async def test_forward_preview_uses_embedded_content_without_api_call(self) -> None:
        loop = asyncio.get_running_loop()
        client = FakeQQAdapterClient(loop, {})
        entry = _forward_entry("101", "root-fwd")
        entry["content_segments"][0]["content"] = [
            {
                "sender": {"user_id": "42", "nickname": "Alice"},
                "time": 1777392000,
                "message": [{"type": "text", "data": {"text": "inside"}}],
            }
        ]
        entry["content_segments"][0]["_needs_expand"] = True

        await expand_forward_previews(entry, client)

        self.assertEqual(client.calls, [])
        self.assertEqual(entry["content_segments"][0]["total"], 1)
        self.assertEqual(entry["content_segments"][0]["preview"][0]["content_text"], "inside")

    async def test_forward_preview_reads_llonebot_text_content_field(self) -> None:
        loop = asyncio.get_running_loop()
        client = FakeQQAdapterClient(loop, {
            "root-fwd": {
                "messages": [
                    {
                        "sender": {"user_id": "42", "nickname": "Alice"},
                        "time": 1777392000,
                        "message": [{"type": "text", "data": {"text": "", "content": "from content"}}],
                    }
                ]
            },
        })
        entry = _forward_entry("101", "root-fwd")
        entry["content_segments"][0]["_needs_expand"] = True

        await expand_forward_previews(entry, client)

        self.assertEqual(entry["content_segments"][0]["preview"][0]["content_text"], "from content")

    async def test_forward_preview_reads_llonebot_node_content_segments(self) -> None:
        loop = asyncio.get_running_loop()
        client = FakeQQAdapterClient(loop, {
            "root-fwd": {
                "messages": [
                    {
                        "sender": {"user_id": "42", "nickname": "Alice"},
                        "time": 1777392000,
                        "content": [{"type": "text", "data": {"text": "from node content"}}],
                    },
                    {
                        "sender": {"user_id": "43", "nickname": "Bob"},
                        "time": 1777392001,
                        "content": [{"type": "image", "data": {"subType": 1}}],
                    },
                ]
            },
        })
        entry = _forward_entry("101", "root-fwd")
        entry["content_segments"][0]["_needs_expand"] = True

        await expand_forward_previews(entry, client)

        self.assertEqual(entry["content_segments"][0]["total"], 2)
        self.assertEqual(entry["content_segments"][0]["preview"][0]["content_text"], "from node content")
        self.assertEqual(entry["content_segments"][0]["preview"][1]["content_type"], "sticker")

    async def test_forward_preview_marks_llonebot_empty_nodes_unreadable(self) -> None:
        loop = asyncio.get_running_loop()
        client = FakeQQAdapterClient(loop, {
            "root-fwd": {
                "messages": [
                    {"sender": {"user_id": "42", "nickname": "Alice"}},
                    {"sender": {"user_id": "43", "nickname": "Bob"}, "message": []},
                ]
            },
        })
        entry = _forward_entry("101", "root-fwd")
        entry["content_segments"][0]["_needs_expand"] = True

        await expand_forward_previews(entry, client)

        self.assertEqual(entry["content_segments"][0]["total"], 2)
        self.assertEqual(entry["content_segments"][0]["preview"], [])
        self.assertIn("未包含可读取正文", entry["content_segments"][0]["error"])

    def test_open_forward_message_declaration_is_fixed_and_requires_id(self) -> None:
        declaration = OPEN_FORWARD_MESSAGE_DECLARATION
        parameters = declaration["parameters"]

        self.assertEqual(declaration["name"], "open_forward_message")
        self.assertNotIn("action", parameters["properties"])
        self.assertIn("id", parameters["properties"])
        self.assertNotIn("motivation", parameters["properties"])
        self.assertEqual(parameters["required"], ["id"])

    def test_browse_forward_view_declaration_is_fixed_navigation_only(self) -> None:
        declaration = BROWSE_FORWARD_VIEW_DECLARATION
        parameters = declaration["parameters"]

        self.assertEqual(declaration["name"], "browse_forward_view")
        self.assertEqual(
            parameters["properties"]["action"]["enum"],
            ["next_page", "prev_page", "back", "close_all"],
        )
        self.assertNotIn("id", parameters["properties"])
        self.assertNotIn("motivation", parameters["properties"])
        self.assertEqual(parameters["required"], ["action"])

    def test_forward_browser_tools_are_registered_under_split_names(self) -> None:
        from tools import build_tools

        session = self._session()
        collection = build_tools({}, session=session, qq_adapter_client=FakeQQAdapterClient(None, {}))

        self.assertIn("open_forward_message", collection.active_specs)
        self.assertIn("browse_forward_view", collection.active_specs)
        self.assertNotIn("browse_forward_message", collection.active_specs)

    def test_forward_browser_renders_virtual_ids_and_registry(self) -> None:
        session = self._session()
        session.forward_browser_stack.append({
            "forward_id": "root-fwd",
            "root_message_id": "101",
            "path": [],
            "title": "合并转发",
            "nodes": [
                {
                    "role": "user",
                    "sender_id": "42",
                    "sender_name": "Alice",
                    "sender_role": "member",
                    "timestamp": "2026-04-29T00:00:00+08:00",
                    "content": "[合并转发]",
                    "content_type": "forward",
                    "content_segments": [{"type": "forward", "forward_id": "child-fwd"}],
                }
            ],
            "total": 1,
            "page_offset": 0,
            "page_size": 8,
        })

        prompt = build_main_user_prompt(session, consume_unread=False)

        self.assertIn('<forward_browser active="true">', prompt)
        self.assertIn('<message id="fwd:101:1" virtual="true"', prompt)
        self.assertEqual(session.forward_virtual_registry["fwd:101:1"]["forward_id"], "child-fwd")

    def test_forward_browser_uses_normal_image_status_labels(self) -> None:
        session = self._session()
        session.forward_browser_stack.append({
            "forward_id": "root-fwd",
            "root_message_id": "101",
            "path": [],
            "title": "合并转发",
            "nodes": [
                {
                    "role": "user",
                    "sender_id": "42",
                    "sender_name": "Alice",
                    "sender_role": "member",
                    "timestamp": "2026-04-29T00:00:00+08:00",
                    "content": "[图片]",
                    "content_type": "image",
                    "content_segments": [{"type": "image", "ref": "111111111111"}],
                    "images": {"111111111111": {"pending": True, "label": "图片"}},
                },
                {
                    "role": "user",
                    "sender_id": "43",
                    "sender_name": "Bob",
                    "sender_role": "member",
                    "timestamp": "2026-04-29T00:00:01+08:00",
                    "content": "[动画表情]",
                    "content_type": "sticker",
                    "content_segments": [{"type": "sticker", "ref": "222222222222"}],
                    "images": {"222222222222": {"failed": True, "label": "动画表情"}},
                },
            ],
            "total": 2,
            "page_offset": 0,
            "page_size": 8,
        })

        prompt = build_main_user_prompt(session, consume_unread=False)
        text = "\n".join(part.get("text", "") for part in prompt if part.get("type") == "text")
        image_parts = [part for part in prompt if part.get("type") == "image_url"]

        self.assertIn('[图片（加载中） ref="111111111111"]', text)
        self.assertIn('[动画表情（加载失败） ref="222222222222"]', text)
        self.assertNotIn('id="fwd:', text)
        self.assertEqual(session.forward_virtual_registry, {})
        self.assertEqual(image_parts, [])

    def test_prompt_has_empty_system_reminder_when_forward_browser_is_closed(self) -> None:
        session = self._session()
        session.add_to_context(_forward_entry("101", "root-fwd"))

        prompt = build_main_user_prompt(session, consume_unread=False)

        self.assertIn("<system_reminder/>", prompt)
        self.assertNotIn("<forward_browser_reminder>", prompt)

    def test_prompt_reminds_model_to_close_open_forward_browser(self) -> None:
        session = self._session()
        session.forward_browser_stack.append({
            "forward_id": "root-fwd",
            "root_message_id": "101",
            "path": [],
            "title": "合并转发",
            "nodes": [],
            "total": 0,
            "page_offset": 0,
            "page_size": 8,
        })

        prompt = build_main_user_prompt(session, consume_unread=False)

        self.assertIn("<system_reminder>", prompt)
        self.assertIn("<forward_browser_reminder>", prompt)
        self.assertIn("当前打开着一个合并转发浏览窗口", prompt)
        self.assertIn("用 `open_forward_message` 打开嵌套合并转发", prompt)
        self.assertIn("用 `browse_forward_view` 关闭它", prompt)
        self.assertIn("全部关闭用 `close_all`", prompt)
        self.assertIn("如果你使用 `shift` 切换到其它会话，当前合并转发浏览窗口会自动关闭", prompt)

    async def test_browse_forward_view_navigates_open_window(self) -> None:
        session = self._session()
        session.forward_browser_stack.append({
            "forward_id": "root-fwd",
            "root_message_id": "101",
            "path": [],
            "title": "合并转发",
            "nodes": [{"content": str(i)} for i in range(12)],
            "total": 12,
            "page_offset": 0,
            "page_size": 8,
        })
        handler = make_browse_forward_view_handler(session, object())

        result = await asyncio.to_thread(
            handler,
            action="next_page",
            motivation="继续查看后面的内容",
        )
        self.assertTrue(result["ok"])
        self.assertTrue(result["moved"])
        self.assertEqual(session.forward_browser_stack[-1]["page_offset"], 4)

        result = await asyncio.to_thread(
            handler,
            action="close_all",
            motivation="已经看完了",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(session.forward_browser_stack, [])

    async def test_tool_opens_real_and_virtual_forward_ids(self) -> None:
        session = self._session()
        session.add_to_context(_forward_entry("101", "root-fwd"))
        loop = asyncio.get_running_loop()
        client = FakeQQAdapterClient(loop, {
            "root-fwd": {
                "messages": [
                    {
                        "sender": {"user_id": "42", "nickname": "Alice"},
                        "time": 1777392000,
                        "message": [{"type": "text", "data": {"text": "inside"}}],
                    },
                    {
                        "sender": {"user_id": "43", "nickname": "Bob"},
                        "time": 1777392001,
                        "message": [{"type": "forward", "data": {"id": "child-fwd"}}],
                    },
                ]
            },
            "child-fwd": {
                "messages": [
                    {
                        "sender": {"user_id": "44", "nickname": "Carol"},
                        "time": 1777392002,
                        "message": [{"type": "text", "data": {"text": "deep"}}],
                    }
                ]
            },
        })
        handler = make_open_forward_message_handler(session, client)

        result = await asyncio.to_thread(
            handler,
            id="101",
            motivation="想查看里面的内容",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["view"]["depth"], 1)

        prompt = build_main_user_prompt(session, consume_unread=False)
        self.assertIn('<content type="text">inside</content>', prompt)
        self.assertNotIn('id="fwd:101:1"', prompt)
        self.assertIn('id="fwd:101:2"', prompt)

        result = await asyncio.to_thread(
            handler,
            id="fwd:101:2",
            motivation="继续深入",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["view"]["depth"], 2)

        prompt = build_main_user_prompt(session, consume_unread=False)
        self.assertIn('<path>', prompt)
        self.assertIn('<from_node depth="1" node_index="2"/>', prompt)
        self.assertIn('<content type="text">deep</content>', prompt)

    async def test_tool_opens_llonebot_text_content_field(self) -> None:
        session = self._session()
        session.add_to_context(_forward_entry("101", "root-fwd"))
        loop = asyncio.get_running_loop()
        client = FakeQQAdapterClient(loop, {
            "root-fwd": {
                "messages": [
                    {
                        "sender": {"user_id": "42", "nickname": "Alice"},
                        "time": 1777392000,
                        "message": [{"type": "text", "data": {"text": "", "content": "from content"}}],
                    },
                    {
                        "sender": {"user_id": "43", "nickname": "Bob"},
                        "time": 1777392001,
                        "raw_message": "from raw message",
                        "message": [{"type": "text", "data": {"text": ""}}],
                    },
                ]
            },
        })
        handler = make_open_forward_message_handler(session, client)

        result = await asyncio.to_thread(handler, id="101")

        self.assertTrue(result["ok"])
        prompt = build_main_user_prompt(session, consume_unread=False)
        self.assertIn('<content type="text">from content</content>', prompt)
        self.assertIn('<content type="text">from raw message</content>', prompt)

    async def test_tool_opens_llonebot_node_content_segments(self) -> None:
        session = self._session()
        session.add_to_context(_forward_entry("101", "root-fwd"))
        loop = asyncio.get_running_loop()
        client = FakeQQAdapterClient(loop, {
            "root-fwd": {
                "messages": [
                    {
                        "sender": {"user_id": "42", "nickname": "Alice"},
                        "time": 1777392000,
                        "content": [{"type": "text", "data": {"text": "from node content"}}],
                    },
                    {
                        "sender": {"user_id": "43", "nickname": "Bob"},
                        "time": 1777392001,
                        "content": [{"type": "image", "data": {"subType": 1, "base64": _image_b64("JPEG")}}],
                    },
                ]
            },
        })
        handler = make_open_forward_message_handler(session, client)

        result = await asyncio.to_thread(handler, id="101")

        self.assertTrue(result["ok"])
        prompt = build_main_user_prompt(session, consume_unread=False)
        text = "\n".join(part.get("text", "") for part in prompt if part.get("type") == "text")
        self.assertIn('<content type="text">from node content</content>', text)
        self.assertIn('<content type="sticker">[动画表情 ref="', text)

    async def test_tool_rejects_llonebot_empty_forward_nodes(self) -> None:
        session = self._session()
        session.add_to_context(_forward_entry("101", "root-fwd"))
        loop = asyncio.get_running_loop()
        client = FakeQQAdapterClient(loop, {
            "root-fwd": {
                "messages": [
                    {"sender": {"user_id": "42", "nickname": "Alice"}},
                    {"sender": {"user_id": "43", "nickname": "Bob"}, "message": []},
                ]
            },
            ("history", "1"): {
                "messages": [
                    {
                        "message_id": 101,
                        "message": [{"type": "forward", "data": {"id": "root-fwd"}}],
                    }
                ]
            },
        })
        handler = make_open_forward_message_handler(session, client)

        result = await asyncio.to_thread(handler, id="101")

        self.assertFalse(result["ok"])
        self.assertIn("未包含可读取正文", result["error"])
        self.assertEqual(session.forward_browser_stack, [])

    async def test_tool_falls_back_to_group_history_when_forward_api_expires(self) -> None:
        session = self._session()
        session.add_to_context(_forward_entry("101", "root-fwd"))
        loop = asyncio.get_running_loop()
        client = FakeQQAdapterClient(loop, {
            "root-fwd": None,
            ("history", "1"): {
                "messages": [
                    {
                        "message_id": 101,
                        "sender": {"user_id": "42", "nickname": "Alice"},
                        "time": 1777392000,
                        "message": [
                            {
                                "type": "forward",
                                "data": {
                                    "id": "root-fwd",
                                    "content": [
                                        {
                                            "sender": {"user_id": "44", "nickname": "Carol"},
                                            "time": 1777392002,
                                            "message": [{"type": "text", "data": {"text": "from history"}}],
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ]
            },
        })
        handler = make_open_forward_message_handler(session, client)

        result = await asyncio.to_thread(
            handler,
            id="101",
            motivation="查看里面的内容",
        )

        self.assertTrue(result["ok"])
        prompt = build_main_user_prompt(session, consume_unread=False)
        self.assertIn('<content type="text">from history</content>', prompt)

    async def test_tool_opens_nested_forward_from_embedded_content(self) -> None:
        session = self._session()
        session.add_to_context(_forward_entry("101", "root-fwd"))
        loop = asyncio.get_running_loop()
        client = FakeQQAdapterClient(loop, {
            "root-fwd": {
                "messages": [
                    {
                        "sender": {"user_id": "42", "nickname": "Alice"},
                        "time": 1777392000,
                        "message": [
                            {
                                "type": "forward",
                                "data": {
                                    "id": "child-fwd",
                                    "content": [
                                        {
                                            "sender": {"user_id": "44", "nickname": "Carol"},
                                            "time": 1777392002,
                                            "message": [{"type": "text", "data": {"text": "deep from embedded"}}],
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ]
            },
            "child-fwd": None,
        })
        handler = make_open_forward_message_handler(session, client)

        result = await asyncio.to_thread(
            handler,
            id="101",
            motivation="查看里面的内容",
        )
        self.assertTrue(result["ok"])
        prompt = build_main_user_prompt(session, consume_unread=False)
        self.assertIn('id="fwd:101:1"', prompt)

        result = await asyncio.to_thread(
            handler,
            id="fwd:101:1",
            motivation="继续深入",
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["view"]["depth"], 2)
        prompt = build_main_user_prompt(session, consume_unread=False)
        self.assertIn('<content type="text">deep from embedded</content>', prompt)
        self.assertNotIn(("get_forward_msg", {"id": "child-fwd"}), client.calls)

    async def test_nested_forward_api_failure_does_not_render_empty_window(self) -> None:
        session = self._session()
        session.add_to_context(_forward_entry("101", "root-fwd"))
        loop = asyncio.get_running_loop()
        client = FakeQQAdapterClient(loop, {
            "root-fwd": {
                "messages": [
                    {
                        "sender": {"user_id": "42", "nickname": "Alice"},
                        "time": 1777392000,
                        "message": [{"type": "forward", "data": {"id": "inner-fwd"}}],
                    },
                    {
                        "sender": {"user_id": "42", "nickname": "Alice"},
                        "time": 1777392001,
                        "message": [{"type": "text", "data": {"text": "测试一下"}}],
                    },
                ]
            },
            "inner-fwd": None,
        })
        handler = make_open_forward_message_handler(session, client)

        result = await asyncio.to_thread(
            handler,
            id="101",
            motivation="查看里面的内容",
        )
        self.assertTrue(result["ok"])

        prompt = build_main_user_prompt(session, consume_unread=False)
        self.assertIn('id="fwd:101:1"', prompt)

        result = await asyncio.to_thread(
            handler,
            id="fwd:101:1",
            motivation="继续深入",
        )

        self.assertFalse(result["ok"])
        self.assertIn("内层消息", result["error"])
        self.assertEqual(len(session.forward_browser_stack), 1)
        prompt = build_main_user_prompt(session, consume_unread=False)
        self.assertIn('<forward_view depth="1" title="合并转发" total="2"', prompt)
        self.assertNotIn('<forward_view depth="2" title="合并转发" total="0"', prompt)

    async def test_opening_another_real_forward_replaces_current_window(self) -> None:
        session = self._session()
        session.add_to_context(_forward_entry("101", "root-fwd"))
        session.add_to_context(_forward_entry("102", "other-fwd"))
        loop = asyncio.get_running_loop()
        client = FakeQQAdapterClient(loop, {
            "root-fwd": {
                "messages": [
                    {
                        "sender": {"user_id": "42", "nickname": "Alice"},
                        "time": 1777392000,
                        "message": [{"type": "text", "data": {"text": "first window"}}],
                    }
                ]
            },
            "other-fwd": {
                "messages": [
                    {
                        "sender": {"user_id": "43", "nickname": "Bob"},
                        "time": 1777392001,
                        "message": [{"type": "text", "data": {"text": "replacement window"}}],
                    }
                ]
            },
        })
        handler = make_open_forward_message_handler(session, client)

        result = await asyncio.to_thread(
            handler,
            id="101",
            motivation="先打开看看",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(session.forward_browser_stack[0]["forward_id"], "root-fwd")

        result = await asyncio.to_thread(
            handler,
            id="102",
            motivation="改看新的这条",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["view"]["depth"], 1)
        self.assertEqual(len(session.forward_browser_stack), 1)
        self.assertEqual(session.forward_browser_stack[0]["forward_id"], "other-fwd")

        prompt = build_main_user_prompt(session, consume_unread=False)
        self.assertIn("replacement window", prompt)
        self.assertNotIn("first window", prompt)

    async def test_forward_browser_renders_images_and_stickers_as_multimodal_parts(self) -> None:
        session = self._session()
        session.add_to_context(_forward_entry("101", "root-fwd"))
        loop = asyncio.get_running_loop()
        client = FakeQQAdapterClient(loop, {
            "root-fwd": {
                "messages": [
                    {
                        "sender": {"user_id": "42", "nickname": "Alice"},
                        "time": 1777392000,
                        "message": [{"type": "image", "data": {"base64": _image_b64("JPEG")}}],
                    },
                    {
                        "sender": {"user_id": "43", "nickname": "Bob"},
                        "time": 1777392001,
                        "message": [{"type": "mface", "data": {"base64": _image_b64("JPEG")}}],
                    },
                ]
            },
        })
        handler = make_open_forward_message_handler(session, client)

        result = await asyncio.to_thread(
            handler,
            id="101",
            motivation="查看里面的图片",
        )
        self.assertTrue(result["ok"])

        prompt = build_main_user_prompt(session, consume_unread=False)
        self.assertIsInstance(prompt, list)
        text = "\n".join(part.get("text", "") for part in prompt if part.get("type") == "text")
        image_parts = [part for part in prompt if part.get("type") == "image_url"]

        self.assertIn('<content type="image">[图片 ref="', text)
        self.assertIn('<content type="sticker">[动画表情 ref="', text)
        self.assertEqual(len(image_parts), 2)
        self.assertTrue(all(
            part["image_url"]["url"].startswith("data:image/jpeg;base64,")
            for part in image_parts
        ))

    async def test_forward_browser_converts_webp_images_for_llm(self) -> None:
        session = self._session()
        session.add_to_context(_forward_entry("101", "root-fwd"))
        loop = asyncio.get_running_loop()
        client = FakeQQAdapterClient(loop, {
            "root-fwd": {
                "messages": [
                    {
                        "sender": {"user_id": "42", "nickname": "Alice"},
                        "time": 1777392000,
                        "message": [{"type": "image", "data": {"base64": _image_b64("WEBP")}}],
                    },
                ]
            },
        })
        handler = make_open_forward_message_handler(session, client)

        result = await asyncio.to_thread(
            handler,
            id="101",
            motivation="查看里面的 webp 图片",
        )
        self.assertTrue(result["ok"])

        prompt = build_main_user_prompt(session, consume_unread=False)
        image_parts = [part for part in prompt if part.get("type") == "image_url"]

        self.assertEqual(len(image_parts), 1)
        url = image_parts[0]["image_url"]["url"]
        self.assertTrue(
            url.startswith("data:image/jpeg;base64,")
            or url.startswith("data:image/png;base64,")
        )
        self.assertNotIn("data:image/webp;base64,", url)


if __name__ == "__main__":
    unittest.main()
