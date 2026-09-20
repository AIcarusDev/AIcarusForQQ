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


@pytest.fixture(autouse=True)
def isolated_media_database(monkeypatch, tmp_path):
    """Never migrate the production database or persist test media in its store."""
    import database
    from llm.media import media_storage

    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "AICQ.db"))
    monkeypatch.setattr(media_storage, "MEDIA_ROOT", tmp_path / "media")


@pytest.fixture(autouse=True)
def isolated_sticker_store(monkeypatch, tmp_path):
    """Image tests must not share cache state or touch the user's collection."""
    from llm.media import media_cache, sticker_collection

    root = tmp_path / "stickers"
    monkeypatch.setattr(sticker_collection, "_STICKER_DIR", root)
    monkeypatch.setattr(sticker_collection, "_INDEX_PATH", root / "index.json")
    monkeypatch.setattr(sticker_collection, "_IMAGES_DIR", root / "images")
    monkeypatch.setattr(sticker_collection, "_GRID_CACHE_PATH", tmp_path / "grid.jpg")
    media_cache.clear_recent_media_cache()
    yield root
    media_cache.clear_recent_media_cache()


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
