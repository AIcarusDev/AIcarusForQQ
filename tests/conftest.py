from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for path in (ROOT, SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


@pytest.fixture
def fake_session():
    class FakeSession:
        conv_type = "group"
        conv_id = "1234"
        conv_name = "Sandbox Group"
        temp_source_group_id = ""
        temp_source_group_name = ""
        _qq_id = "bot"
        _qq_name = "Bot"

        def __init__(self):
            self.context_messages = []

        def add_to_context(self, entry):
            self.context_messages.append(entry)

    return FakeSession()
