import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qq_adapter.access_control import (
    is_session_allowed_by_config,
    is_whitelist_mode_enabled,
    whitelist_rejection_reason,
)


class QQAdapterAccessControlTests(unittest.TestCase):
    def test_whitelist_mode_defaults_to_enabled(self) -> None:
        cfg = {"whitelist": {"private_users": ["42"], "group_ids": []}}

        self.assertTrue(is_whitelist_mode_enabled(cfg))
        self.assertTrue(is_session_allowed_by_config(cfg, "private", "42"))
        self.assertFalse(is_session_allowed_by_config(cfg, "private", "43"))
        self.assertTrue(is_session_allowed_by_config(cfg, "temp", "42"))
        self.assertFalse(is_session_allowed_by_config(cfg, "temp", "43"))
        self.assertFalse(is_session_allowed_by_config(cfg, "group", "1"))
        self.assertEqual(
            whitelist_rejection_reason(cfg, "private", "43"),
            "私聊用户 43 不在白名单中",
        )
        self.assertEqual(
            whitelist_rejection_reason(cfg, "temp", "43"),
            "临时会话用户 43 不在白名单中",
        )

    def test_free_mode_allows_any_known_conversation_type(self) -> None:
        cfg = {
            "whitelist": {
                "enabled": False,
                "private_users": ["42"],
                "group_ids": ["1"],
            }
        }

        self.assertFalse(is_whitelist_mode_enabled(cfg))
        self.assertTrue(is_session_allowed_by_config(cfg, "private", "43"))
        self.assertTrue(is_session_allowed_by_config(cfg, "temp", "43"))
        self.assertTrue(is_session_allowed_by_config(cfg, "group", "2"))
        self.assertIsNone(whitelist_rejection_reason(cfg, "group", "2"))
        self.assertFalse(is_session_allowed_by_config(cfg, "channel", "2"))


if __name__ == "__main__":
    unittest.main()
