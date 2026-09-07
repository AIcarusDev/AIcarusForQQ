from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor

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




@pytest.mark.parametrize("raw", [b"{", b"[]", b'{"version":3}', b'{"not-an-entry":{}}'])
def test_corrupt_index_is_not_treated_as_empty(raw):
    stickers._STICKER_DIR.mkdir()
    stickers._INDEX_PATH.write_bytes(raw)
    with pytest.raises(stickers.StickerCollectionError):
        stickers.save_sticker(png(), "image/png", "new", image_ref="a" * 12)
    assert stickers._INDEX_PATH.read_bytes() == raw
    assert not stickers._IMAGES_DIR.exists()






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
    from llm.media.image_store import read_image
    assert all(read_image(ref)["image_ref"] == main for ref in refs)
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










def test_capacity_uses_exact_contents_and_still_accepts_aliases_at_limit():
    for i in range(stickers.MAX_STICKERS):
        assert stickers.save_sticker(png((i, 0, 0)), "image/png", str(i), image_ref=f"{i:012x}")
    assert stickers.save_sticker(png("blue"), "image/png", "overflow") is None
    assert stickers.save_sticker(png((0, 0, 0)), "image/png", "ignored", image_ref="alias-first") == ("0" * 12, True)
    assert len(stickers.list_all()) == stickers.MAX_STICKERS



@pytest.mark.parametrize("ref", ["000", "../unsafe", "/home/agent/image.png", "bad\\ref"])
def test_collection_rejects_legacy_ids_and_paths(ref):
    with pytest.raises(stickers.StickerCollectionError):
        stickers.save_sticker(png(), "image/png", "bad", image_ref=ref)
    assert stickers.list_all() == []
