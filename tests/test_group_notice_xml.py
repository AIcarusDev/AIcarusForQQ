import os
import sys
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm.prompt.xml_builder import build_chat_log_xml
from napcat.events import build_group_notice_entry


class GroupNoticeXmlTests(unittest.TestCase):
    def test_group_ban_renders_structured_note_with_cards(self) -> None:
        entry = build_group_notice_entry(
            {
                "notice_type": "group_ban",
                "group_id": 123,
                "operator_id": 456,
                "user_id": 789,
                "duration": 600,
                "time": 1710000000,
            },
            operator_name="AdminNick",
            operator_card="管理员群名片",
            target_name="TargetNick",
            target_card="目标群名片",
            timezone=ZoneInfo("Asia/Shanghai"),
        )

        self.assertIsNotNone(entry)
        assert entry is not None
        xml = build_chat_log_xml(
            [entry],
            {"type": "group", "id": "123", "name": "test", "bot_id": "1", "bot_name": "Bot"},
        )

        self.assertIn('<note type="group_ban" sub_type="ban"', xml)
        self.assertIn('<operator id="456" card="管理员群名片" nickname="AdminNick"/>', xml)
        self.assertIn('<target id="789" card="目标群名片" nickname="TargetNick"/>', xml)
        self.assertIn('<duration seconds="600">10分钟</duration>', xml)
        self.assertIn("<content>管理员群名片 禁言了 目标群名片 10分钟</content>", xml)

    def test_group_admin_falls_back_to_nickname_then_id(self) -> None:
        entry = build_group_notice_entry(
            {
                "notice_type": "group_admin",
                "sub_type": "set",
                "group_id": 123,
                "user_id": 789,
                "time": 1710000000,
            },
            target_name="TargetNick",
            timezone=ZoneInfo("Asia/Shanghai"),
        )

        self.assertIsNotNone(entry)
        assert entry is not None
        xml = build_chat_log_xml(
            [entry],
            {"type": "group", "id": "123", "name": "test", "bot_id": "1", "bot_name": "Bot"},
        )

        self.assertIn('<target id="789" nickname="TargetNick"/>', xml)
        self.assertIn("<content>TargetNick 被设置为管理员</content>", xml)

        fallback = build_group_notice_entry(
            {
                "notice_type": "group_decrease",
                "sub_type": "leave",
                "group_id": 123,
                "user_id": 789,
                "time": 1710000000,
            },
            timezone=ZoneInfo("Asia/Shanghai"),
        )
        assert fallback is not None
        self.assertEqual(fallback["content"], "789 退出了群")


if __name__ == "__main__":
    unittest.main()
