from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import database
from llm.media import image_store, media_identity as identity, media_storage as storage, sticker_collection as stickers
from llm.media.media_cache import clear_recent_media_cache
from llm.media.image_resolver import ImageResolver, image_bytes
from platforms.qq.adapter.segments import build_message_content
from workspace.media import register_workspace_image
from test_sticker_collection import png


DAY = datetime(2026, 9, 7, tzinfo=timezone.utc)


def collide(monkeypatch):
    monkeypatch.setattr(image_store.uuid, "uuid4", lambda: SimpleNamespace(hex="a" * 32))


def test_collision_retries_at_same_length_before_expanding(monkeypatch):
    collide(monkeypatch)
    assert storage.generate_time_ref(DAY) == "260907_aaaaa"
    values = iter(["a" * 32] * 7 + ["b" * 32])
    monkeypatch.setattr(image_store.uuid, "uuid4", lambda: SimpleNamespace(hex=next(values)))
    assert storage.generate_time_ref(DAY) == "260907_bbbbb"
    collide(monkeypatch)
    assert storage.generate_time_ref(DAY) == "260907_aaaaaa"
    assert storage.parse_time_ref_date("260907_aaaaaa") == (2026, 9, 7)
    storage.save_media_bytes(png((1, 0, 0)), image_ref="260907_aaaaaa")
    assert storage.read_media_bytes("260907_aaaaaa")[0] == png((1, 0, 0))


def test_allocation_is_bounded_and_supports_four_digit_start(monkeypatch):
    collide(monkeypatch)
    refs = [identity.reserve_time_ref(DAY, initial_length=4) for _ in range(9)]
    assert refs[0] == "260907_aaaa"
    assert len(set(refs)) == 9
    with pytest.raises(identity.MediaRefConflict, match="exhausted"):
        identity.reserve_time_ref(DAY, initial_length=4)


@pytest.mark.parametrize("source", ["disk", "registry", "sticker_tombstone"])
def test_allocation_respects_pre_ledger_refs(monkeypatch, source):
    ref = "260907_aaaaa"
    if source == "disk":
        folder = storage.get_media_dir(2026, 9)
        folder.mkdir(parents=True)
        (folder / f"{ref}.gif").write_bytes(b"old")
    elif source == "registry":
        asyncio.run(database.init_db())
        with sqlite3.connect(database.DB_PATH) as conn:
            conn.execute("DROP VIEW media_registry")
            conn.execute("CREATE TABLE media_registry(image_ref TEXT PRIMARY KEY, source_type TEXT, locator TEXT, sha256 TEXT)")
            conn.execute("INSERT INTO media_registry(image_ref, source_type, locator) VALUES (?, 'chat', 'pending')", (ref,))
    else:
        stickers._STICKER_DIR.mkdir()
        stickers._INDEX_PATH.write_text(json.dumps({"version": 3, "stickers": {}, "ref_hashes": {ref: hashlib.sha256(b"old").hexdigest()}}))
    collide(monkeypatch)
    assert storage.generate_time_ref(DAY) == "260907_aaaaaa"


def test_pending_segments_have_distinct_reserved_refs(monkeypatch):
    collide(monkeypatch)
    entry = build_message_content([{"type": "image", "data": {"url": "https://example.invalid/image"}}] * 2)
    refs = [part["image_ref"] for part in entry["content_segments"]]
    assert len(set(refs)) == 2
    assert set(refs) == set(entry["images"])
    assert all(info["pending"] for info in entry["images"].values())


def test_cross_process_allocation_survives_restart(tmp_path):
    code = """
import sys
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
import database
from llm.media import media_identity as i, media_storage as s, sticker_collection as stickers
s.MEDIA_ROOT = Path(sys.argv[2])
database.DB_PATH = sys.argv[3]
stickers._INDEX_PATH = Path(sys.argv[4])
(__import__("llm.media.image_store", fromlist=["uuid"])).uuid.uuid4 = lambda: SimpleNamespace(hex='a' * 32)
print(s.generate_time_ref(datetime(2026, 9, 7, tzinfo=timezone.utc)))
"""
    args = [sys.executable, "-B", "-c", code, str(Path(__file__).resolve().parents[1] / "src"),
            str(storage.MEDIA_ROOT), database.DB_PATH, str(stickers._INDEX_PATH)]

    def run(_):
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()

    with ThreadPoolExecutor(max_workers=4) as pool:
        refs = list(pool.map(run, range(4)))
    assert len(set(refs)) == 4
    assert run(None) not in refs


def test_content_binding_is_atomic_across_competing_writers():
    ref = storage.generate_time_ref(DAY)

    def write(raw):
        try:
            storage.save_media_bytes(raw, image_ref=ref, mime="image/png")
            return raw
        except identity.MediaRefConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, [png((2, 0, 0)), png((3, 0, 0))]))
    winners = [raw for raw in results if raw is not None]
    assert len(winners) == 1
    assert storage.read_media_bytes(ref)[0] == winners[0]


def test_same_content_retries_share_path_across_extensions_and_deletion():
    ref, path = storage.save_media_bytes(png((4, 0, 0)), mime="image/png")
    assert storage.save_media_bytes(png((4, 0, 0)), mime="image/gif", image_ref=ref)[1] == path
    with pytest.raises(identity.MediaRefConflict):
        storage.save_media_bytes(png((5, 0, 0)), mime="image/gif", image_ref=ref)
    path.unlink()
    with pytest.raises(identity.MediaRefConflict):
        storage.save_media_bytes(png((5, 0, 0)), image_ref=ref)
    assert storage.save_media_bytes(png((4, 0, 0)), mime="image/gif", image_ref=ref)[1] == path
    assert path.read_bytes() == png((4, 0, 0))


def test_interrupted_write_keeps_binding_and_allows_same_content_retry(monkeypatch):
    ref = storage.generate_time_ref(DAY)
    with monkeypatch.context() as patch:
        def fail(*_):
            raise OSError("simulated disk error")
        patch.setattr(image_store.os, "link", fail)
        with pytest.raises(OSError):
            storage.save_media_bytes(png((4, 0, 0)), image_ref=ref)
    with pytest.raises(identity.MediaRefConflict):
        storage.save_media_bytes(png((5, 0, 0)), image_ref=ref)
    storage.save_media_bytes(png((4, 0, 0)), image_ref=ref)
    assert storage.read_media_bytes(ref)[0] == png((4, 0, 0))


def test_allocation_io_failure_is_not_retried_as_collision(monkeypatch):
    calls = []
    def fail(_):
        calls.append(True)
        raise OSError("simulated read error")
    monkeypatch.setattr(identity, "_historical_binding", fail)
    with pytest.raises(identity.MediaIdentityUnavailable):
        storage.generate_time_ref(DAY)
    assert len(calls) == 1


def test_registry_rebinding_cannot_overwrite_chat_bytes():
    async def run():
        await database.init_db()
        ref, path = storage.save_media_bytes(png((2, 0, 0)), mime="image/png")
        await database.register_media_ref(ref, "chat", str(path))
        conflicting = path.parent / "conflicting.png"
        conflicting.write_bytes(png((3, 0, 0)))
        with pytest.raises(identity.MediaRefConflict):
            await register_workspace_image(conflicting, image_ref=ref)
        assert database.lookup_media_ref_sync(ref)["locator"] == str(path)
        with pytest.raises(ValueError):
            await database.register_media_ref(ref, "chat", "pending")
        assert database.lookup_media_ref_sync(ref)["locator"] == str(path)
    asyncio.run(run())


def test_workspace_original_edits_do_not_change_published_ref(tmp_path):
    async def run():
        await database.init_db()
        path = tmp_path / "original.png"
        path.write_bytes(png((4, 0, 0)))
        ref = await register_workspace_image(path)
        path.write_bytes(png((6, 0, 0)))
        clear_recent_media_cache()
        found = ImageResolver(browser_image_reader=lambda _: None).resolve(ref)
        assert image_bytes(found[0])[0] == png((4, 0, 0))
        assert found[1] == "workspace"
        with pytest.raises(identity.MediaRefConflict):
            await register_workspace_image(path, image_ref=ref)
    asyncio.run(run())


def test_chat_conflict_rolls_back_message_and_registry():
    import base64
    async def run():
        await database.init_db()
        entry = {"role": "user", "message_id": "fixture", "images": {
            "legacy_ref": {"base64": base64.b64encode(png((2, 0, 0))).decode()},
        }}
        await database.save_chat_message("qq:private:fixture", entry)
        entry["images"]["legacy_ref"]["base64"] = base64.b64encode(png((3, 0, 0))).decode()
        with pytest.raises(identity.MediaRefConflict):
            await database.save_chat_message("qq:private:fixture", entry)
        assert image_bytes(database.load_chat_image_payload_sync("qq:private:fixture", "fixture", "legacy_ref"))[0] == png((2, 0, 0))
    asyncio.run(run())


def test_download_conflict_does_not_fall_back_to_new_base64(monkeypatch):
    from platforms.qq.adapter import events
    import base64
    ref, _ = storage.save_media_bytes(png((2, 0, 0)))
    entry = {"images": {ref: {"pending": True}}, "_pending_images": [(ref, "fixture", "image")]}
    async def download(_):
        return base64.b64encode(png((3, 0, 0))).decode(), "image/png"
    monkeypatch.setattr(events, "_fetch_image_b64", download)
    with pytest.raises(identity.MediaRefConflict):
        asyncio.run(events.download_pending_images(entry))
    assert "base64" not in entry["images"][ref]
    assert storage.read_media_bytes(ref)[0] == png((2, 0, 0))
