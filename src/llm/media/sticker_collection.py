"""Persistent sticker images addressed by their original image references."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import re
import tempfile
import threading
import uuid
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
        if document.get("version") not in (2, 3) or not isinstance(document["stickers"], dict):
            raise ValueError
        bindings = document["ref_hashes"]
        if not isinstance(bindings, dict):
            raise ValueError
        for ref, digest in bindings.items():
            if not valid_image_ref(ref) or not isinstance(digest, str) or not _HASH_PATTERN.fullmatch(digest):
                raise ValueError
        claimed = set()
        for ref, info in document["stickers"].items():
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


def _new_ref(document: dict, dt: datetime | None = None) -> str:
    from .media_storage import generate_time_ref
    from .media_identity import bind_media_identity
    for ref, digest in document["ref_hashes"].items():
        bind_media_identity(ref, digest)
    return generate_time_ref(dt)


def _migrate_v2_to_v3(document: dict) -> None:
    from .media_storage import _TIME_REF_PATTERN
    _validate_index(document)
    backup = _INDEX_PATH.with_name("index.v2.backup.json")
    if not backup.exists() and _INDEX_PATH.exists():
        _atomic_write(backup, _INDEX_PATH.read_bytes())

    new_stickers = {}
    bindings = document.setdefault("ref_hashes", {})

    for old_ref, info in list(document.get("stickers", {}).items()):
        if _TIME_REF_PATTERN.match(old_ref):
            new_stickers[old_ref] = info
            continue

        created_at_str = str(info.get("created_at") or "")
        try:
            created_dt = datetime.fromisoformat(created_at_str)
        except Exception:
            created_dt = None

        new_ref = _new_ref(document, created_dt)

        # References are metadata, not filenames. Keep the original file so
        # either the old or the atomically committed new index remains usable
        # after an interrupted migration (including the v2 backup).

        aliases = list(info.get("aliases") or [])
        if old_ref not in aliases:
            aliases.append(old_ref)

        info["aliases"] = aliases
        new_stickers[new_ref] = info

        digest = info["sha256"]
        bindings[new_ref] = digest
        bindings[old_ref] = digest

    document["stickers"] = new_stickers
    document["version"] = 3
    _validate_index(document)
    _save_index(document)
    logger.info("[sticker_collection] 已迁移表情包至 v3 时序 ref (共 %d 个)", len(new_stickers))


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
        return {"version": 3, "stickers": {}, "ref_hashes": {}}
    try:
        raw = _INDEX_PATH.read_bytes()
        document = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise StickerCollectionError("invalid_index", "表情包索引读取失败，未修改收藏数据") from exc
    if not isinstance(document, dict):
        raise StickerCollectionError("invalid_index", "表情包索引格式错误")
    if "version" in document:
        if document.get("version") == 2 and isinstance(document.get("stickers"), dict):
            _migrate_v2_to_v3(document)
        _validate_index(document)
        _recover_interrupted_deletions(document)
        return document
    migrated = {"version": 3, "stickers": {}, "ref_hashes": {}}
    try:
        for old_id, info in sorted(document.items()):
            if not re.fullmatch(r"\d{3}", old_id) or not isinstance(info, dict):
                raise ValueError("invalid legacy entry")
            image = _image_path(info["filename"]).read_bytes()
            digest = hashlib.sha256(image).hexdigest()
            if info.get("sha256") and info["sha256"] != digest:
                raise ValueError("legacy image changed")
            ref = _new_ref(migrated)
            migrated["stickers"][ref] = {
                "filename": info["filename"], "mime": _inspect(image),
                "description": info["description"], "created_at": info["created_at"],
                "sha256": digest, "aliases": [],
            }
            migrated["ref_hashes"][ref] = digest
        _validate_index(migrated)
        backup = _INDEX_PATH.with_name("index.v1.backup.json")
        if not backup.exists():
            _atomic_write(backup, raw)
        _save_index(migrated)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise StickerCollectionError("migration_failed", "表情包迁移失败，旧索引和图片已保留") from exc
    logger.info("[sticker_collection] 已迁移 %d 个收藏到 image_ref", len(migrated["stickers"]))
    return migrated


def _find_ref(document: dict, image_ref: str) -> str | None:
    if not valid_image_ref(image_ref):
        return None
    entries = document["stickers"]
    if image_ref in entries:
        return image_ref
    return next((ref for ref, info in entries.items() if image_ref in info["aliases"]), None)


def _recover_interrupted_deletions(document: dict) -> None:
    """A crash before index commit must leave the indexed original recoverable."""
    for ref, info in document["stickers"].items():
        path = _image_path(info["filename"])
        if path.exists():
            continue
        for staged in (_STICKER_DIR / "trash").glob(f"{ref}-*"):
            try:
                if hashlib.sha256(staged.read_bytes()).hexdigest() == info["sha256"]:
                    staged.rename(path)
                    break
            except OSError as exc:
                raise StickerCollectionError("recovery_failed", "表情包中断操作恢复失败，文件已保留") from exc


def _entries(document: dict) -> list[tuple[str, dict]]:
    return sorted(document["stickers"].items(), key=lambda item: (item[1]["created_at"], item[0]))


def _read_entry(ref: str, info: dict) -> dict:
    result = {"image_ref": ref, "mime": info["mime"], "description": info["description"]}
    try:
        path = _image_path(info["filename"])
        if path.stat().st_size > MAX_STICKER_BYTES:
            result["unavailable_status"] = "image_too_large"
            return result
        raw = path.read_bytes()
    except OSError:
        result["unavailable_status"] = "missing_image"
        return result
    if hashlib.sha256(raw).hexdigest() != info["sha256"]:
        result["unavailable_status"] = "image_changed"
        return result
    result["data"] = raw
    return result


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
    """Store exact bytes; retain the first reference and register duplicate aliases."""
    if image_ref is not None and not valid_image_ref(image_ref):
        raise StickerCollectionError("invalid_ref", "请提供有效的 image_ref，旧表情包编号已停用")
    mime = _inspect(raw_bytes)
    digest = hashlib.sha256(raw_bytes).hexdigest()
    from .media_identity import bind_media_identity, MediaRefConflict

    def bind(ref):
        try:
            bind_media_identity(ref, digest)
        except MediaRefConflict as exc:
            raise StickerCollectionError("ref_conflict", "此 image_ref 已绑定其他图片内容") from exc

    with _LOCK:
        document = _load_index()
        entries, bindings = document["stickers"], document["ref_hashes"]
        if image_ref in bindings and bindings[image_ref] != digest:
            raise StickerCollectionError("ref_conflict", "此 image_ref 已绑定其他图片内容")
        if image_ref:
            bind(image_ref)
        for ref, info in _entries(document):
            if info["sha256"] != digest:
                continue
            if "data" not in _read_entry(ref, info):
                raise StickerCollectionError("image_unavailable", "收藏图片不可用，请先整理收藏")
            if image_ref and image_ref != ref and image_ref not in info["aliases"]:
                info["aliases"].append(image_ref)
                bindings[image_ref] = digest
                _save_index(document)
            return ref, True
        if len(entries) >= MAX_STICKERS:
            return None
        ref = image_ref or _new_ref(document)
        bind(ref)
        filename = ref + _MIME_TO_EXT[mime]
        path = _image_path(filename)
        _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise StickerCollectionError("file_exists", "收藏目标文件已存在，请先整理收藏")
        try:
            _atomic_write(path, raw_bytes)
            entries[ref] = {
                "description": description, "created_at": datetime.now(timezone.utc).isoformat(),
                "filename": filename, "mime": mime, "sha256": digest, "aliases": [],
            }
            bindings[ref] = digest
            _save_index(document)
        except (OSError, StickerCollectionError):
            path.unlink(missing_ok=True)
            raise
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
        info = document["stickers"].pop(ref)
        path = _image_path(info["filename"])
        # Quarantine before committing: an interrupted deletion cannot be adopted as an orphan.
        trash = _STICKER_DIR / "trash" / f"{ref}-{uuid.uuid4().hex}-{path.name}"
        moved = False
        try:
            if path.exists() and not any(item["filename"] == info["filename"] for item in document["stickers"].values()):
                trash.parent.mkdir(parents=True, exist_ok=True)
                path.rename(trash)
                moved = True
            _save_index(document)
        except (OSError, StickerCollectionError):
            if moved:
                trash.rename(path)
            raise
        if moved:
            try:
                trash.unlink()
            except OSError:
                logger.warning("[sticker_collection] 已撤销收藏引用，但隔离文件清理失败 ref=%s", ref)
        # Historical hash bindings prevent a deleted reference being rebound to different bytes.
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
    """Repair filenames and deduplicate exact contents without reassigning references."""
    with _LOCK:
        document = _load_index()
        entries = document["stickers"]
        stats = dict.fromkeys(("removed_stale", "updated_hash", "fixed_rename", "adopted_orphans", "removed_duplicates", "skipped_overflow"), 0)
        _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        disk = {}
        for path in sorted(_IMAGES_DIR.iterdir()):
            if path.suffix.lower() not in _VALID_EXTENSIONS or not path.is_file():
                continue
            try:
                _image_path(path.name)
                if path.stat().st_size > MAX_STICKER_BYTES:
                    continue
                raw = path.read_bytes()
                disk[path.name] = (hashlib.sha256(raw).hexdigest(), _inspect(raw))
            except (OSError, StickerCollectionError):
                logger.warning("[sticker_collection] 整理跳过无效图片 %s", path.name)
        claimed = set()
        seen = {}
        for ref, info in _entries(document):
            original = info["filename"]
            matching = [name for name, (digest, _) in disk.items() if digest == info["sha256"]]
            if not matching:
                del entries[ref]
                stats["removed_stale"] += 1
                continue
            filename = original if original in matching else matching[0]
            if filename != original:
                info["filename"] = filename
                stats["fixed_rename"] += 1
            digest = info["sha256"]
            if digest in seen:
                kept = entries[seen[digest]]
                kept["aliases"].extend([ref, *info["aliases"]])
                del entries[ref]
                stats["removed_duplicates"] += 1
            else:
                seen[digest] = ref
                claimed.add(filename)
        duplicate_files = []
        for filename, (digest, mime) in disk.items():
            if filename in claimed:
                continue
            if digest in seen:
                duplicate_files.append(filename)
                stats["removed_duplicates"] += 1
                continue
            if len(entries) >= MAX_STICKERS:
                stats["skipped_overflow"] += 1
                continue
            ref = _new_ref(document)
            entries[ref] = {
                "filename": filename, "mime": mime, "sha256": digest,
                "description": "（用户手动添加或替换，暂无描述）",
                "created_at": datetime.now(timezone.utc).isoformat(), "aliases": [],
            }
            document["ref_hashes"][ref] = digest
            seen[digest] = ref
            stats["adopted_orphans"] += 1
        _validate_index(document)
        _save_index(document)
        for filename in duplicate_files:
            try:
                _image_path(filename).unlink(missing_ok=True)
            except OSError:
                logger.warning("[sticker_collection] 整理未能清除重复图片 %s", filename)
        return stats
