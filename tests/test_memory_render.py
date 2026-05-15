import os
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memory.render import build_memory_xml


class MemoryRenderTests(unittest.TestCase):
    def test_recalled_event_prompt_hides_internal_fields(self) -> None:
        now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
        xml = build_memory_xml(
            now=now,
            recalled_events=[
                {
                    "event_id": 120,
                    "event_type": "say",
                    "context_type": "episodic",
                    "polarity": "positive",
                    "modality": "actual",
                    "confidence": 0.95,
                    "occurred_at": int(now.timestamp() * 1000) - 86400 * 1000,
                    "summary": "未來星織#qq_1321807442 向吹雪#qq_3533611951 发送亲吻表情",
                    "roles": [
                        {"role": "agent", "entity": "User:qq_1321807442"},
                        {"role": "recipient", "entity": "User:qq_3533611951"},
                        {"role": "theme", "value_text": "么么么（亲吻表情）"},
                    ],
                }
            ],
            nickname_map={"1321807442": "未來星織", "3533611951": "吹雪"},
        )

        self.assertIn('<event id="120" confidence="0.95" when="1天前">', xml)
        self.assertIn("未來星織#qq_1321807442 向吹雪#qq_3533611951 发送亲吻表情", xml)
        self.assertNotIn('type="say"', xml)
        self.assertNotIn('ctx="episodic"', xml)
        self.assertNotIn('pol="positive"', xml)
        self.assertNotIn('mod="actual"', xml)
        self.assertNotIn("agent=", xml)
        self.assertNotIn("recipient=", xml)
        self.assertNotIn("theme=", xml)


if __name__ == "__main__":
    unittest.main()
