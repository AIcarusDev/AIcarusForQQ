"""Persistent sticker images addressed by their original image references."""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("AICQ.llm.media.sticker")
_STICKER_DIR = Path(__file__).resolve().parents[3] / "data" / "stickers"
_INDEX_PATH = _STICKER_DIR / "index.json"
_IMAGES_DIR = _STICKER_DIR / "images"
_GRID_CACHE_PATH = Path(__file__).resolve().parents[3] / "cache" / "stickers" / "stickers_grid.jpg"
_LOCK = threading.RLock()
_REF_PATTERN = re.compile(r"[A-Za-z0-9_-]{4,128}\Z")
_HASH_PATTERN = re.compile(r"[a-f0-9]{64}\Z")
_MIME_TO_EXT = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
    "image/gif": ".gif", "image/bmp": ".bmp",
}
_VALID_EXTENSIONS = frozenset((*_MIME_TO_EXT.values(), ".jpeg"))
MAX_STICKERS = 30
MAX_STICKER_BYTES = 20 * 1024 * 1024
_THUMB_SIZE = 96
_GRID_COLS = 5
_GRID_SPACING = 10
_GRID_MARGIN = 14
_LABEL_FONT_SIZE = 16


class StickerCollectionError(ValueError):
    """A bounded error suitable for tool/API results, without host paths."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def valid_image_ref(value: object) -> bool:
    return isinstance(value, str) and _REF_PATTERN.fullmatch(value) is not None


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".sticker-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _save_index(document: dict) -> None:
    try:
        _atomic_write(_INDEX_PATH, json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8"))
    except OSError as exc:
        raise StickerCollectionError("write_failed", "表情包索引保存失败") from exc


def _image_path(filename: str) -> Path:
    if not isinstance(filename, str) or not filename or Path(filename).name != filename or "\\" in filename:
        raise StickerCollectionError("invalid_index", "表情包索引包含无效文件名")
    path = _IMAGES_DIR / filename
    if path.resolve().parent != _IMAGES_DIR.resolve():
        raise StickerCollectionError("invalid_index", "表情包文件不在收藏目录内")
    return path


def _validate_index(document: dict) -> None:
    try:
        if document.get("version") not in (2, 3, 4) or not isinstance(document["stickers"], dict):
            raise ValueError
        if document.get("version") == 4:
            for ref, info in document['stickers'].items():
                if not valid_image_ref(ref) or not isinstance(info.get('description'), str) or not isinstance(info.get('created_at'), str):
                    raise ValueError
            return
        bindings = document["ref_hashes"]
        if not isinstance(bindings, dict):
            raise ValueError
        for ref, digest in bindings.items():
            if not valid_image_ref(ref) or not isinstance(digest, str) or not _HASH_PATTERN.fullmatch(digest):
                raise ValueError
        claimed = set()
        for ref, info in document["stickers"].items():
            if document.get("version") != 4:
                _image_path(info["filename"])
            if not isinstance(info["sha256"], str) or not _HASH_PATTERN.fullmatch(info["sha256"]):
                raise ValueError
            if info["mime"] not in _MIME_TO_EXT or not isinstance(info["description"], str):
                raise ValueError
            if not isinstance(info["created_at"], str) or not isinstance(info["aliases"], list):
                raise ValueError
            for candidate in [ref, *info["aliases"]]:
                if not valid_image_ref(candidate) or candidate in claimed or bindings.get(candidate) != info["sha256"]:
                    raise ValueError
                claimed.add(candidate)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise StickerCollectionError("invalid_index", "表情包索引损坏，未修改收藏数据") from exc






def _inspect(raw: bytes) -> str:
    from .image_resolver import ImagePayloadError, inspect_image_payload

    try:
        info = inspect_image_payload(raw, max_bytes=MAX_STICKER_BYTES, max_pixels=100_000_000)
    except ImagePayloadError as exc:
        raise StickerCollectionError(exc.code, "表情包图片内容无效或超出限制") from exc
    if info.mime_type not in _MIME_TO_EXT:
        raise StickerCollectionError("unsupported_image_format", "不支持此表情包图片格式")
    return info.mime_type


def _load_index() -> dict:
    """Called under _LOCK. Migrate legacy metadata before any collection access."""
    if not _INDEX_PATH.exists():
        return {"version": 4, "stickers": {}}
    try:
        raw = _INDEX_PATH.read_bytes()
        document = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise StickerCollectionError("invalid_index", "表情包索引读取失败，未修改收藏数据") from exc
    if not isinstance(document, dict):
        raise StickerCollectionError("invalid_index", "表情包索引格式错误")
    if document.get("version") not in (2, 3, 4):
        raise StickerCollectionError("migration_required", "请先执行统一图片迁移，旧收藏索引已保留")
    _validate_index(document)
    return document


def _find_ref(document: dict, image_ref: str) -> str | None:
    if not valid_image_ref(image_ref):
        return None
    from .image_store import lookup_image
    record = lookup_image(image_ref)
    if record and not record.get("unavailable_status"):
        image_ref = record["image_ref"]
    entries = document["stickers"]
    if image_ref in entries:
        return image_ref
    return next((ref for ref, info in entries.items() if image_ref in info.get("aliases", [])), None)




def _entries(document: dict) -> list[tuple[str, dict]]:
    return sorted(document["stickers"].items(), key=lambda item: (item[1]["created_at"], item[0]))


def _read_entry(ref: str, info: dict) -> dict:
    from .image_store import read_image
    record = read_image(ref)
    if not record:
        return {"image_ref": ref, "unavailable_status": "missing_image"}
    return {**record, "collection_description": info["description"]}


def get_sticker_image(image_ref: str) -> dict | None:
    """Known but unavailable images return a status, preventing source fallback."""
    if not valid_image_ref(image_ref):
        return None
    with _LOCK:
        document = _load_index()
        ref = _find_ref(document, image_ref)
        return _read_entry(ref, document["stickers"][ref]) if ref else None


def load_sticker_bytes(image_ref: str) -> tuple[bytes, str] | None:
    image = get_sticker_image(image_ref)
    if image is None or "data" not in image:
        return None
    return image["data"], image["mime"]


def save_sticker(raw_bytes: bytes, mime: str, description: str, *, image_ref: str | None = None) -> tuple[str, bool] | None:
    from .image_store import register_image
    from .media_identity import MediaRefConflict
    if image_ref is not None and not valid_image_ref(image_ref):
        raise StickerCollectionError("invalid_ref", "请提供有效的 image_ref，旧表情包编号已停用")
    with _LOCK:
        _load_index()  # Validate collection metadata before storing new content.
    _inspect(raw_bytes)
    try:
        record = register_image(raw_bytes, "sticker", image_ref)
    except MediaRefConflict as exc:
        raise StickerCollectionError("ref_conflict", "此 image_ref 已绑定其他图片内容") from exc
    ref = record["image_ref"]
    with _LOCK:
        document = _load_index()
        existing = _find_ref(document, ref)
        if existing:
            return existing, True
        if len(document["stickers"]) >= MAX_STICKERS:
            return None
        document["version"] = 4
        document["stickers"][ref] = {
            "description": description, "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_index(document)
    return ref, False


def update_sticker_description(image_ref: str, new_description: str) -> str | None:
    with _LOCK:
        document = _load_index()
        ref = _find_ref(document, image_ref)
        if ref is None:
            return None
        document["stickers"][ref]["description"] = new_description
        _save_index(document)
        return ref


def delete_sticker(image_ref: str) -> str | None:
    with _LOCK:
        document = _load_index()
        ref = _find_ref(document, image_ref)
        if ref is None:
            return None
        document["stickers"].pop(ref)
        _save_index(document)
        return ref


def list_all() -> list[dict]:
    with _LOCK:
        return [{"image_ref": ref, **info} for ref, info in _entries(_load_index())]


def _get_grid_font():
    for name in ("C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, _LABEL_FONT_SIZE)
        except OSError:
            pass
    return ImageFont.load_default(size=_LABEL_FONT_SIZE)


def _rebuild_grid_cache(document: dict) -> bytes | None:
    entries = _entries(document)
    if not entries:
        _GRID_CACHE_PATH.unlink(missing_ok=True)
        return None
    font = _get_grid_font()
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    boxes = [measure.textbbox((0, 0), ref, font=font) for ref, _ in entries]
    cell_w = max(_THUMB_SIZE, max(box[2] - box[0] for box in boxes) + 8)
    label_h = max(box[3] - box[1] for box in boxes)
    cell_h = _THUMB_SIZE + 5 + label_h
    cols = min(_GRID_COLS, len(entries))
    rows = math.ceil(len(entries) / cols)
    canvas = Image.new("RGB", (
        2 * _GRID_MARGIN + cols * cell_w + (cols - 1) * _GRID_SPACING,
        2 * _GRID_MARGIN + rows * cell_h + (rows - 1) * _GRID_SPACING,
    ), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (ref, info) in enumerate(entries):
        x = _GRID_MARGIN + i % cols * (cell_w + _GRID_SPACING)
        y = _GRID_MARGIN + i // cols * (cell_h + _GRID_SPACING)
        image = _read_entry(ref, info)
        try:
            with Image.open(io.BytesIO(image.get("data", b""))) as original:
                thumb = original.convert("RGBA")
                thumb.thumbnail((_THUMB_SIZE, _THUMB_SIZE), Image.Resampling.LANCZOS)
                canvas.paste(thumb, (x + (cell_w - thumb.width) // 2, y + (_THUMB_SIZE - thumb.height) // 2), thumb)
        except (OSError, ValueError):
            draw.rectangle((x, y, x + cell_w - 1, y + _THUMB_SIZE - 1), outline="gray")
        box = boxes[i]
        draw.text((x + (cell_w - box[2] + box[0]) // 2, y + _THUMB_SIZE + 5 - box[1]), ref, font=font, fill=(40, 40, 40))
    output = io.BytesIO()
    canvas.save(output, format="JPEG", quality=92)
    raw = output.getvalue()
    _atomic_write(_GRID_CACHE_PATH, raw)
    return raw


def get_sticker_snapshot(*, include_grid: bool = False) -> tuple[list[dict], bytes | None]:
    """One locked snapshot keeps grid labels and the returned list in agreement."""
    with _LOCK:
        document = _load_index()
        items = [{"image_ref": ref, **info} for ref, info in _entries(document)]
        grid = _rebuild_grid_cache(document) if include_grid else None
        return items, grid


def get_sticker_grid_bytes() -> bytes | None:
    return get_sticker_snapshot(include_grid=True)[1]


def reconcile_stickers() -> dict:
    """Check collection links; original-file migration is an explicit maintenance operation."""
    with _LOCK:
        document = _load_index()
        stats = dict.fromkeys(("removed_stale", "updated_hash", "fixed_rename", "adopted_orphans", "removed_duplicates", "skipped_overflow"), 0)
        stats["unavailable"] = sum(bool(_read_entry(ref, info).get("unavailable_status")) for ref, info in _entries(document))
        return stats
