import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm.prompt.xml_builder import _render_content_chunks, build_chat_log_xml
from napcat.segments import build_content_segments, napcat_segments_to_text


def _entry(segments: list[dict]) -> dict:
    return {
        "role": "user",
        "message_id": "1001",
        "sender_id": "42",
        "sender_name": "Alice",
        "sender_role": "member",
        "timestamp": "2026-05-15T12:00:00+08:00",
        "content": napcat_segments_to_text(segments),
        "content_type": "text",
        "content_segments": build_content_segments(segments),
    }


class CardSegmentTests(unittest.TestCase):
    def test_json_ark_card_preserves_payload_and_renders_summary(self) -> None:
        segments = [{
            "type": "json",
            "data": {
                "data": {
                    "app": "com.tencent.structmsg",
                    "prompt": "[分享] 示例文章",
                    "meta": {
                        "news": {
                            "title": "示例文章",
                            "desc": "这是一段摘要",
                            "jumpUrl": "https://example.com/article",
                        }
                    },
                }
            },
        }]

        content_segments = build_content_segments(segments)

        self.assertEqual(napcat_segments_to_text(segments), "[卡片消息]")
        self.assertEqual(content_segments[0]["type"], "card")
        self.assertEqual(content_segments[0]["kind"], "json")
        self.assertEqual(content_segments[0]["title"], "示例文章")
        self.assertEqual(content_segments[0]["summary"], "这是一段摘要")
        self.assertEqual(content_segments[0]["url"], "https://example.com/article")
        self.assertIn("com.tencent.structmsg", content_segments[0]["raw"])

        xml = build_chat_log_xml([_entry(segments)], {"type": "group", "id": "1"})
        self.assertIn('<content type="card" kind="json">', xml)
        self.assertIn("<title>示例文章</title>", xml)
        self.assertIn("<summary>这是一段摘要</summary>", xml)
        self.assertIn("<url>https://example.com/article</url>", xml)
        self.assertIn("<raw>", xml)

    def test_markdown_card_renders_markdown_body(self) -> None:
        segments = [{
            "type": "markdown",
            "data": {"content": "## 标题\n正文内容\n[查看详情](https://example.com)"},
        }]

        xml = build_chat_log_xml([_entry(segments)], {"type": "group", "id": "1"})

        self.assertIn('<content type="card" kind="markdown">', xml)
        self.assertIn("<summary>## 标题 正文内容 [查看详情](https://example.com)</summary>", xml)
        self.assertIn("<markdown>## 标题", xml)

    def test_xml_card_extracts_basic_fields_and_escapes_raw(self) -> None:
        segments = [{
            "type": "xml",
            "data": {"data": '<msg title="活动通知" brief="点击查看" url="https://example.com?a=1&b=2"></msg>'},
        }]

        xml = build_chat_log_xml([_entry(segments)], {"type": "group", "id": "1"})

        self.assertIn('<content type="card" kind="xml">', xml)
        self.assertIn("<title>活动通知</title>", xml)
        self.assertIn("<summary>点击查看</summary>", xml)
        self.assertIn("<url>https://example.com?a=1&amp;b=2</url>", xml)
        self.assertIn("&lt;msg title=", xml)

    def test_contact_and_location_cards_render_structured_fields(self) -> None:
        contact = _render_content_chunks(build_content_segments([{
            "type": "contact",
            "data": {"type": "group", "id": "987654321"},
        }]))
        location = _render_content_chunks(build_content_segments([{
            "type": "location",
            "data": {"lat": "31.2304", "lon": "121.4737", "title": "上海", "content": "位置分享"},
        }]))

        self.assertEqual(contact[0][0], "card:contact")
        self.assertIn('<contact type="group" id="987654321"/>', contact[0][1])
        self.assertEqual(location[0][0], "card:location")
        self.assertIn("<title>上海</title>", location[0][1])
        self.assertIn('<geo lat="31.2304" lon="121.4737"/>', location[0][1])

    def test_legacy_json_label_still_renders_as_placeholder(self) -> None:
        xml = build_chat_log_xml([{
            "role": "user",
            "message_id": "1001",
            "sender_id": "42",
            "sender_name": "Alice",
            "sender_role": "member",
            "timestamp": "2026-05-15T12:00:00+08:00",
            "content": "[卡片消息]",
            "content_type": "text",
            "content_segments": [{"type": "json", "label": "卡片消息"}],
        }], {"type": "group", "id": "1"})

        self.assertIn("<content type=\"text\">[卡片消息]</content>", xml)


if __name__ == "__main__":
    unittest.main()
