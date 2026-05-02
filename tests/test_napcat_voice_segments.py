import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm.prompt.xml_builder import _render_content_chunks
from napcat.events import napcat_event_to_context
from napcat.segments import _determine_content_type, build_content_segments


class NapCatVoiceSegmentTests(unittest.IsolatedAsyncioTestCase):
    def test_record_segment_becomes_voice_with_duration(self) -> None:
        segments = [{"type": "record", "data": {"file": "voice.silk", "duration": 1.4}}]

        self.assertEqual(_determine_content_type(segments), "voice")
        self.assertEqual(
            build_content_segments(segments),
            [{"type": "voice", "label": "语音", "duration": 1.4}],
        )
        self.assertEqual(
            _render_content_chunks(build_content_segments(segments)),
            [("voice", "[语音 1'']")],
        )

    async def test_record_only_event_is_context_entry(self) -> None:
        entry = await napcat_event_to_context({
            "post_type": "message",
            "message_type": "group",
            "message_id": 123,
            "time": 1710000000,
            "group_id": 456,
            "sender": {"user_id": 789, "nickname": "Alice", "role": "member"},
            "message": [{"type": "record", "data": {"file": "voice.silk"}}],
        })

        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry["content"], "[语音]")
        self.assertEqual(entry["content_type"], "voice")
        self.assertEqual(entry["content_segments"], [{"type": "voice", "label": "语音"}])
        self.assertEqual(
            _render_content_chunks(entry["content_segments"]),
            [("voice", "[语音]")],
        )


if __name__ == "__main__":
    unittest.main()