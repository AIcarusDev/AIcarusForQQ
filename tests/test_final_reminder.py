from __future__ import annotations

from llm.prompt.final_reminder import append_final_reminder


class ReminderSession:
    pending_error_logger = "legacy error"

    def is_browsing_history(self) -> bool:
        return True

    def is_browsing_forward(self) -> bool:
        return True


def test_final_reminder_keeps_only_empty_placeholder():
    session = ReminderSession()

    result = append_final_reminder("<world/>", session)

    assert result == "<world/>\n<system_reminder/>"
    assert session.pending_error_logger == "legacy error"


def test_final_reminder_appends_to_last_multimodal_text_part():
    parts = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        {"type": "text", "text": "<world/>"},
    ]

    result = append_final_reminder(parts, ReminderSession())

    assert result == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        {"type": "text", "text": "<world/>\n<system_reminder/>"},
    ]


def test_final_reminder_adds_text_part_when_multimodal_tail_is_not_text():
    parts = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]

    result = append_final_reminder(parts, ReminderSession())

    assert result == parts + [{"type": "text", "text": "\n<system_reminder/>"}]
