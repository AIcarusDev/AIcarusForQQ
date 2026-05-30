import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qq_adapter.segments import llm_segments_to_qq_adapter, qq_adapter_segments_to_text


class StickerAdapterSegmentTests(unittest.TestCase):
    def test_default_sticker_send_uses_onebot_snake_case_sub_type(self) -> None:
        with patch("llm.media.sticker_collection.load_sticker_bytes", return_value=(b"jpg", "image/jpeg")):
            segments = llm_segments_to_qq_adapter([
                {"command": "sticker", "params": {"sticker_id": "009"}},
            ])

        self.assertEqual(segments[0]["type"], "image")
        self.assertEqual(segments[0]["data"]["sub_type"], 1)
        self.assertNotIn("subType", segments[0]["data"])

    def test_llonebot_sticker_send_uses_camel_case_sub_type(self) -> None:
        with patch("llm.media.sticker_collection.load_sticker_bytes", return_value=(b"jpg", "image/jpeg")):
            segments = llm_segments_to_qq_adapter([
                {"command": "sticker", "params": {"sticker_id": "009"}},
            ], adapter="llonebot")

        self.assertEqual(segments[0]["type"], "image")
        self.assertEqual(segments[0]["data"]["subType"], 1)
        self.assertNotIn("sub_type", segments[0]["data"])

    def test_llonebot_image_sub_type_is_rendered_as_sticker_text(self) -> None:
        text = qq_adapter_segments_to_text([
            {"type": "image", "data": {"subType": 1}},
        ])

        self.assertEqual(text, "[动画表情]")


if __name__ == "__main__":
    unittest.main()
