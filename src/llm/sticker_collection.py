"""sticker_collection.py — 表情包收藏管理

布局：
  data/stickers/
    index.json          ← 表情包索引（id → {description, created_at, filename, mime}）
    images/             ← 表情包图片文件
      000.jpg
      001.png
      ...
    cache/              ← 自动生成的缓存（勿手动修改）
      stickers_grid.jpg ← 缩略图网格缓存
"""

import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("AICQ.sticker_collection")

# 表情包目录：项目根 / data / stickers
_STICKER_DIR = Path(__file__).parent.parent.parent / "data" / "stickers"
_INDEX_PATH = _STICKER_DIR / "index.json"
_IMAGES_DIR = _STICKER_DIR / "images"
_GRID_CACHE_PATH = _STICKER_DIR / "cache" / "stickers_grid.jpg"

_MIME_TO_EXT: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}

# ── 网格布局常量 ──────────────────────────────────────────
MAX_STICKERS: int = 30   # 单张网格图支持的最大表情包数量（5 列 × 6 行）
_THUMB_SIZE: int = 96
_GRID_COLS: int = 5
_GRID_SPACING: int = 10  # 单元格之间的间距
_GRID_MARGIN: int = 14   # 整体边距
_LABEL_FONT_SIZE: int = 16
_LABEL_SPACING: int = 5  # 缩略图底部到标签文字的间距
_GRID_BG_COLOR: tuple[int, int, int] = (255, 255, 255)
_GRID_LABEL_COLOR: tuple[int, int, int] = (40, 40, 40)


# ── 内部工具 ──────────────────────────────────────────────

def _load_index() -> dict:
    if _INDEX_PATH.exists():
        try:
            return json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[sticker_collection] 读取 index.json 失败: %s", e)
    return {}


def _save_index(index: dict) -> None:
    _STICKER_DIR.mkdir(parents=True, exist_ok=True)
    _INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _next_id(index: dict) -> str:
    """生成下一个可用的三位表情包 ID（如 '000', '001'）。"""
    n = len(index)
    while True:
        sid = f"{n:03d}"
        if sid not in index:
            return sid
        n += 1


# ── 公共 API ──────────────────────────────────────────────

def _get_grid_font():
    """获取网格标签字体。优先尝试系统等宽字体，再 fallback 到 Pillow 内置字体。"""
    from PIL import ImageFont
    candidates = [
        "C:/Windows/Fonts/consola.ttf",       # Consolas（Windows，等宽，数字极清晰）
        "C:/Windows/Fonts/cour.ttf",           # Courier New（Windows）
        "C:/Windows/Fonts/arial.ttf",          # Arial（Windows，通用）
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",   # Linux
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, _LABEL_FONT_SIZE)
        except OSError:
            continue
    # 最后 fallback：Pillow 内置位图字体
    try:
        return ImageFont.load_default(size=_LABEL_FONT_SIZE)
    except TypeError:
        return ImageFont.load_default()


def _rebuild_grid_cache() -> None:
    """根据当前索引重新生成网格缩略图并写入磁盘缓存。"""
    from PIL import Image, ImageDraw

    index = _load_index()
    entries = sorted(index.items())[:MAX_STICKERS]

    if not entries:
        _GRID_CACHE_PATH.unlink(missing_ok=True)
        logger.info("[sticker_collection] 无表情包，已移除网格缓存")
        return

    font = _get_grid_font()

    # 计算标签实际高度
    _probe = Image.new("RGB", (1, 1))
    _probe_draw = ImageDraw.Draw(_probe)
    _bbox = _probe_draw.textbbox((0, 0), "000", font=font)
    label_h = int(_bbox[3] - _bbox[1])

    cell_h = _THUMB_SIZE + _LABEL_SPACING + label_h
    num_rows = math.ceil(len(entries) / _GRID_COLS)
    grid_w = _GRID_MARGIN * 2 + _GRID_COLS * _THUMB_SIZE + (_GRID_COLS - 1) * _GRID_SPACING
    grid_h = _GRID_MARGIN * 2 + num_rows * cell_h + (num_rows - 1) * _GRID_SPACING

    canvas = Image.new("RGB", (grid_w, grid_h), _GRID_BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    for idx, (sid, info) in enumerate(entries):
        row = idx // _GRID_COLS
        col = idx % _GRID_COLS
        x = _GRID_MARGIN + col * (_THUMB_SIZE + _GRID_SPACING)
        y = _GRID_MARGIN + row * (cell_h + _GRID_SPACING)

        img_path = _IMAGES_DIR / info["filename"]
        try:
            with Image.open(img_path) as img:
                if hasattr(img, "seek"):
                    try:
                        img.seek(0)
                    except EOFError:
                        pass
                img_rgba = img.convert("RGBA")
                img_rgba.thumbnail((_THUMB_SIZE, _THUMB_SIZE), Image.Resampling.LANCZOS)

                # 将缩略图居中合成到白底方块上，处理透明通道
                cell_canvas = Image.new("RGBA", (_THUMB_SIZE, _THUMB_SIZE), (255, 255, 255, 255))
                paste_x = (_THUMB_SIZE - img_rgba.width) // 2
                paste_y = (_THUMB_SIZE - img_rgba.height) // 2
                cell_canvas.paste(img_rgba, (paste_x, paste_y), img_rgba)
                canvas.paste(cell_canvas.convert("RGB"), (x, y))
        except Exception as e:
            logger.warning("[sticker_collection] 生成网格时跳过图片 id=%s: %s", sid, e)
            continue

        # ID 标签居中绘制
        label_bbox = draw.textbbox((0, 0), sid, font=font)
        label_w = label_bbox[2] - label_bbox[0]
        label_x = x + (_THUMB_SIZE - label_w) // 2
        label_y = y + _THUMB_SIZE + _LABEL_SPACING
        draw.text((label_x, label_y), sid, fill=_GRID_LABEL_COLOR, font=font)

    _GRID_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(_GRID_CACHE_PATH, format="JPEG", quality=92)
    logger.info("[sticker_collection] 网格缓存已更新，共 %d 个表情包", len(entries))


def get_sticker_grid_bytes() -> Optional[bytes]:
    """返回网格缩略图的 JPEG 字节流。缓存不存在时自动重建。"""
    if not _GRID_CACHE_PATH.exists():
        _rebuild_grid_cache()
    if _GRID_CACHE_PATH.exists():
        try:
            return _GRID_CACHE_PATH.read_bytes()
        except OSError as e:
            logger.warning("[sticker_collection] 读取网格缓存失败: %s", e)
    return None


def save_sticker(raw_bytes: bytes, mime: str, description: str) -> Optional[tuple[str, bool]]:
    """将图片存为新表情包，返回 (id, is_duplicate)。

    - 若收藏已达 MAX_STICKERS 上限，返回 None。
    - 若图片与已有表情包完全相同（SHA-256 一致），返回 (已有id, True)。
    - 正常保存时返回 (新id, False)。
    """
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    index = _load_index()

    # 查重：遍历已有条目，比较 sha256
    for sid, info in index.items():
        if info.get("sha256") == sha256:
            logger.info(
                "[sticker_collection] 表情包重复，跳过保存 id=%s sha256=%.16s…",
                sid, sha256,
            )
            return sid, True

    if len(index) >= MAX_STICKERS:
        logger.warning("[sticker_collection] 表情包数量已达上限 (%d)，拒绝添加", MAX_STICKERS)
        return None

    sid = _next_id(index)
    ext = _MIME_TO_EXT.get(mime, ".jpg")
    filename = f"{sid}{ext}"

    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    (_IMAGES_DIR / filename).write_bytes(raw_bytes)

    index[sid] = {
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "mime": mime,
        "sha256": sha256,
    }
    _save_index(index)
    logger.info("[sticker_collection] 已保存表情包 id=%s desc=%r", sid, description)

    try:
        _rebuild_grid_cache()
    except Exception as e:
        logger.warning("[sticker_collection] 更新网格缓存失败: %s", e)

    return sid, False


def load_sticker_bytes(sticker_id: str) -> Optional[tuple[bytes, str]]:
    """读取表情包原始字节，返回 (bytes, mime)，不存在返回 None。"""
    index = _load_index()
    if sticker_id not in index:
        return None
    entry = index[sticker_id]
    p = _IMAGES_DIR / entry["filename"]
    try:
        return p.read_bytes(), entry.get("mime", "image/jpeg")
    except OSError as e:
        logger.warning("[sticker_collection] 读取表情包文件失败 id=%s: %s", sticker_id, e)
        return None


def list_all() -> list[dict]:
    """返回所有表情包的元数据列表（id + info），不含图片字节。
    若发现对应文件不存在，自动清理该索引条目。
    """
    index = _load_index()
    if stale_ids := [
        sid
        for sid, info in index.items()
        if not (_IMAGES_DIR / info["filename"]).exists()
    ]:
        for sid in stale_ids:
            logger.warning(
                "[sticker_collection] 清理过期索引 id=%s (文件不存在)", sid
                )
            del index[sid]
        _save_index(index)
    return [{"id": sid, **info} for sid, info in sorted(index.items())]


def _compact_index(index: dict) -> tuple[dict, bool]:
    """对 index 进行连续重编号，消除空洞（如 000, 002 → 000, 001）。

    按旧 id 升序处理，新编号始终 ≤ 旧编号，因此文件重命名不会互相覆盖。
    返回 (新 index, 是否发生了变更)。
    """
    sorted_items = sorted(index.items())
    # 快速检查：若编号已连续，直接返回
    if all(sid == f"{i:03d}" for i, (sid, _) in enumerate(sorted_items)):
        return index, False

    new_index: dict = {}
    for new_num, (old_sid, info) in enumerate(sorted_items):
        new_sid = f"{new_num:03d}"
        if old_sid != new_sid:
            old_ext = Path(info["filename"]).suffix
            new_filename = f"{new_sid}{old_ext}"
            try:
                (_IMAGES_DIR / info["filename"]).rename(_IMAGES_DIR / new_filename)
                info = {**info, "filename": new_filename}
                logger.info("[sticker_collection] 重编号 %s → %s", old_sid, new_sid)
            except OSError as e:
                logger.warning(
                    "[sticker_collection] 重命名文件失败 %s → %s: %s",
                    old_sid, new_sid, e,
                )
                new_sid = old_sid  # 重命名失败则保留原 id
        new_index[new_sid] = info
    return new_index, True


def deduplicate_stickers() -> int:
    """扫描已有表情包，删除内容完全相同的重复条目，并修复编号空洞。

    - 对没有 sha256 字段的旧条目，按需计算并回填。
    - 无论是否有重复，都会检查并修复编号空洞（支持用户手动删除文件的场景）。
    返回本次删除的重复条目数量。
    """
    index = _load_index()
    if not index:
        return 0

    seen: dict[str, str] = {}   # sha256 → 首个 sid
    to_delete: list[str] = []
    updated = False

    for sid, info in sorted(index.items()):   # 按 id 升序，优先保留最早的
        img_path = _IMAGES_DIR / info["filename"]
        sha256 = info.get("sha256")

        if not sha256:
            # 旧条目没有 sha256，现场计算并回填
            try:
                sha256 = hashlib.sha256(img_path.read_bytes()).hexdigest()
                index[sid]["sha256"] = sha256
                updated = True
            except OSError as e:
                logger.warning("[sticker_collection] 去重跳过 id=%s，无法读取文件: %s", sid, e)
                continue

        if sha256 in seen:
            first_sid = seen[sha256]
            logger.info(
                "[sticker_collection] 发现重复表情包 id=%s 与 id=%s 相同，删除 id=%s",
                sid, first_sid, sid,
            )
            to_delete.append(sid)
        else:
            seen[sha256] = sid

    if to_delete:
        for sid in to_delete:
            filename = index[sid]["filename"]
            try:
                (_IMAGES_DIR / filename).unlink(missing_ok=True)
            except OSError as e:
                logger.warning("[sticker_collection] 删除重复文件失败 id=%s: %s", sid, e)
            del index[sid]
        updated = True

    # 无论是否有去重，都检查并修复编号空洞
    index, compacted = _compact_index(index)
    if compacted:
        updated = True
        logger.info("[sticker_collection] 已修复编号空洞，当前共 %d 个表情包", len(index))

    if updated:
        _save_index(index)
        try:
            _rebuild_grid_cache()
        except Exception as e:
            logger.warning("[sticker_collection] 去重后更新网格缓存失败: %s", e)

    logger.info(
        "[sticker_collection] 去重完成，删除 %d 个重复表情包，剩余 %d 个",
        len(to_delete), len(index),
    )
    return len(to_delete)
