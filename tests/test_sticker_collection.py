from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from llm.media import sticker_collection as stickers


def png(color="red") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (6, 4), color).save(output, format="PNG")
    return output.getvalue()


def gif() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (6, 4), "red").save(
        output, format="GIF", save_all=True,
        append_images=[Image.new("RGB", (6, 4), "blue")], duration=[100, 200], loop=0,
    )
    return output.getvalue()


def legacy_collection():
    stickers._IMAGES_DIR.mkdir(parents=True)
    entries = {}
    for i, (raw, mime, ext) in enumerate(((png(), "image/png", "png"), (gif(), "image/gif", "gif"))):
        filename = f"{i:03d}.{ext}"
        (stickers._IMAGES_DIR / filename).write_bytes(raw)
        entries[f"{i:03d}"] = {
            "filename": filename, "mime": mime, "description": f"impression-{i}",
            "created_at": f"2025-01-0{i + 1}T00:00:00+00:00",
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    raw_index = json.dumps(entries).encode()
    stickers._INDEX_PATH.write_bytes(raw_index)
    return entries, raw_index


def test_migration_preserves_originals_and_refs_survive_process_restart():
    old, raw_index = legacy_collection()
    items = stickers.list_all()
    refs = [item["image_ref"] for item in items]
    assert len(set(refs)) == 2
    assert all(len(ref) == 12 and ref not in old for ref in refs)
    assert stickers._INDEX_PATH.with_name("index.v1.backup.json").read_bytes() == raw_index
    for item, source in zip(items, old.values()):
        assert all(item[key] == source[key] for key in source)
        assert stickers.load_sticker_bytes(item["image_ref"])[0] == (stickers._IMAGES_DIR / item["filename"]).read_bytes()
    code = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from llm.media import sticker_collection as s
s._STICKER_DIR = Path(sys.argv[2])
s._INDEX_PATH = s._STICKER_DIR / "index.json"
s._IMAGES_DIR = s._STICKER_DIR / "images"
print(json.dumps(s.list_all()))
"""
    process = subprocess.run(
        [sys.executable, "-B", "-c", code, str(Path(__file__).resolve().parents[1] / "src"), str(stickers._STICKER_DIR)],
        check=True, text=True, capture_output=True,
    )
    assert json.loads(process.stdout) == items
    stickers.reconcile_stickers()
    assert [item["image_ref"] for item in stickers.list_all()] == refs
    assert stickers.load_sticker_bytes("000") is None


@pytest.mark.parametrize("raw", [b"{", b"[]", b'{"version":3}', b'{"not-an-entry":{}}'])
def test_corrupt_index_is_not_treated_as_empty(raw):
    stickers._STICKER_DIR.mkdir()
    stickers._INDEX_PATH.write_bytes(raw)
    with pytest.raises(stickers.StickerCollectionError):
        stickers.save_sticker(png(), "image/png", "new", image_ref="a" * 12)
    assert stickers._INDEX_PATH.read_bytes() == raw
    assert not stickers._IMAGES_DIR.exists()


def test_migration_commit_failure_is_retryable_without_losing_originals(monkeypatch):
    old, raw_index = legacy_collection()
    replace = stickers.os.replace

    def fail_index(source, destination):
        if destination == stickers._INDEX_PATH:
            raise OSError("simulated index replace failure")
        return replace(source, destination)

    monkeypatch.setattr(stickers.os, "replace", fail_index)
    with pytest.raises(stickers.StickerCollectionError) as error:
        stickers.list_all()
    assert error.value.code == "migration_failed"
    assert stickers._INDEX_PATH.read_bytes() == raw_index
    assert all((stickers._IMAGES_DIR / item["filename"]).is_file() for item in old.values())
    monkeypatch.setattr(stickers.os, "replace", replace)
    assert len(stickers.list_all()) == 2


def test_missing_legacy_file_leaves_index_untouched():
    _, raw_index = legacy_collection()
    (stickers._IMAGES_DIR / "000.png").unlink()
    with pytest.raises(stickers.StickerCollectionError):
        stickers.list_all()
    assert stickers._INDEX_PATH.read_bytes() == raw_index


def test_aliases_edit_and_delete_one_collection_without_affecting_other_refs():
    raw = png()
    main, alias, other = "a" * 12, "b" * 12, "c" * 12
    assert stickers.save_sticker(raw, "image/png", "first", image_ref=main) == (main, False)
    assert stickers.save_sticker(raw, "image/png", "ignored", image_ref=alias) == (main, True)
    stickers.save_sticker(png("blue"), "image/png", "other", image_ref=other)
    assert stickers.list_all()[0]["description"] == "first"
    assert stickers.load_sticker_bytes(alias) == (raw, "image/png")
    assert stickers.update_sticker_description(alias, "revised") == main
    assert stickers.list_all()[0]["description"] == "revised"
    other_item = stickers.list_all()[1]
    assert stickers.delete_sticker(alias) == main
    assert stickers.load_sticker_bytes(main) is None
    assert stickers.load_sticker_bytes(alias) is None
    assert stickers.list_all() == [other_item]
    assert not (stickers._IMAGES_DIR / f"{main}.png").exists()
    assert stickers.delete_sticker(alias) is None


@pytest.mark.parametrize("deleted", [False, True])
def test_reference_cannot_be_rebound_to_different_contents(deleted):
    ref = "a" * 12
    stickers.save_sticker(png(), "image/png", "first", image_ref=ref)
    if deleted:
        stickers.delete_sticker(ref)
    with pytest.raises(stickers.StickerCollectionError) as error:
        stickers.save_sticker(png("blue"), "image/png", "wrong", image_ref=ref)
    assert error.value.code == "ref_conflict"
    assert len(stickers.list_all()) == (0 if deleted else 1)


def test_concurrent_duplicate_collection_preserves_every_alias():
    main = "a" * 12
    raw = png()
    stickers.save_sticker(raw, "image/png", "first", image_ref=main)
    refs = [f"{i:012x}" for i in range(20)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda ref: stickers.save_sticker(raw, "image/png", "ignored", image_ref=ref), refs))
    assert results == [(main, True)] * len(refs)
    assert len(stickers.list_all()) == 1
    assert set(stickers.list_all()[0]["aliases"]) == set(refs)
    assert all(stickers.load_sticker_bytes(ref) == (raw, "image/png") for ref in refs)


def test_concurrent_different_saves_and_edits_do_not_lose_entries():
    refs = [f"{i:012x}" for i in range(16)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: stickers.save_sticker(png((i, 0, 0)), "image/png", "before", image_ref=refs[i]), range(len(refs))))
        list(pool.map(lambda ref: stickers.update_sticker_description(ref, ref), refs))
    assert {item["image_ref"]: item["description"] for item in stickers.list_all()} == {ref: ref for ref in refs}


def test_save_and_delete_commit_failures_restore_previous_collection(monkeypatch):
    main = "a" * 12
    raw = png()
    stickers.save_sticker(raw, "image/png", "first", image_ref=main)
    original = stickers._INDEX_PATH.read_bytes()
    replace = stickers.os.replace

    def fail_index(source, destination):
        if destination == stickers._INDEX_PATH:
            raise OSError("simulated commit failure")
        return replace(source, destination)

    monkeypatch.setattr(stickers.os, "replace", fail_index)
    with pytest.raises(stickers.StickerCollectionError):
        stickers.save_sticker(png("blue"), "image/png", "new", image_ref="b" * 12)
    assert not (stickers._IMAGES_DIR / ("b" * 12 + ".png")).exists()
    with pytest.raises(stickers.StickerCollectionError):
        stickers.delete_sticker(main)
    assert stickers._INDEX_PATH.read_bytes() == original
    assert stickers.load_sticker_bytes(main) == (raw, "image/png")


def test_reconcile_rename_and_duplicates_keep_identity_and_impression():
    main = "a" * 12
    raw = png()
    stickers.save_sticker(raw, "image/png", "first", image_ref=main)
    path = stickers._IMAGES_DIR / f"{main}.png"
    path.rename(stickers._IMAGES_DIR / "renamed.png")
    (stickers._IMAGES_DIR / "duplicate.png").write_bytes(raw)
    stickers.reconcile_stickers()
    item, = stickers.list_all()
    assert item["image_ref"] == main
    assert item["description"] == "first"
    assert stickers.load_sticker_bytes(main) == (raw, "image/png")
    assert len(list(stickers._IMAGES_DIR.iterdir())) == 1


def test_interrupted_delete_before_commit_restores_the_indexed_original():
    ref, raw = "a" * 12, png()
    stickers.save_sticker(raw, "image/png", "first", image_ref=ref)
    trash = stickers._STICKER_DIR / "trash"
    trash.mkdir()
    (stickers._IMAGES_DIR / f"{ref}.png").rename(trash / f"{ref}-interrupted.png")
    assert stickers.load_sticker_bytes(ref) == (raw, "image/png")
    assert (stickers._IMAGES_DIR / f"{ref}.png").exists()
    assert not list(trash.iterdir())


def test_replaced_contents_never_inherit_old_reference_or_impression():
    main, alias = "a" * 12, "b" * 12
    stickers.save_sticker(png(), "image/png", "old impression", image_ref=main)
    stickers.save_sticker(png(), "image/png", "ignored", image_ref=alias)
    replacement = png("blue")
    (stickers._IMAGES_DIR / f"{main}.png").write_bytes(replacement)
    assert stickers.load_sticker_bytes(main) is None
    assert stickers.get_sticker_image(alias)["unavailable_status"] == "image_changed"
    stickers.reconcile_stickers()
    item, = stickers.list_all()
    assert item["image_ref"] not in {main, alias}
    assert item["description"] != "old impression"
    assert stickers.load_sticker_bytes(item["image_ref"]) == (replacement, "image/png")
    assert stickers.load_sticker_bytes(main) is None
    assert stickers.load_sticker_bytes(alias) is None


def test_swapped_files_are_repaired_by_exact_content_instead_of_changing_refs():
    main, other = "a" * 12, "b" * 12
    first, second = png(), png("blue")
    stickers.save_sticker(first, "image/png", "red", image_ref=main)
    stickers.save_sticker(second, "image/png", "blue", image_ref=other)
    (stickers._IMAGES_DIR / f"{main}.png").write_bytes(second)
    (stickers._IMAGES_DIR / f"{other}.png").write_bytes(first)
    stickers.reconcile_stickers()
    assert stickers.load_sticker_bytes(main) == (first, "image/png")
    assert stickers.load_sticker_bytes(other) == (second, "image/png")
    assert len(stickers.list_all()) == 2


def test_capacity_uses_exact_contents_and_still_accepts_aliases_at_limit():
    for i in range(stickers.MAX_STICKERS):
        assert stickers.save_sticker(png((i, 0, 0)), "image/png", str(i), image_ref=f"{i:012x}")
    assert stickers.save_sticker(png("blue"), "image/png", "overflow") is None
    assert stickers.save_sticker(png((0, 0, 0)), "image/png", "ignored", image_ref="alias-first") == ("0" * 12, True)
    assert len(stickers.list_all()) == stickers.MAX_STICKERS
    orphan = stickers._IMAGES_DIR / "user-added.png"
    orphan.write_bytes(png("blue"))
    assert stickers.reconcile_stickers()["skipped_overflow"] == 1
    assert len(stickers.list_all()) == stickers.MAX_STICKERS
    assert orphan.read_bytes() == png("blue")


@pytest.mark.parametrize("ref", ["000", "../unsafe", "/home/agent/image.png", "bad\\ref"])
def test_collection_rejects_legacy_ids_and_paths(ref):
    with pytest.raises(stickers.StickerCollectionError):
        stickers.save_sticker(png(), "image/png", "bad", image_ref=ref)
    assert stickers.list_all() == []
