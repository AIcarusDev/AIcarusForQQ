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
_VALID_EXTENSIONS: frozenset[str] = frozenset(_MIME_TO_EXT.values())
_EXT_TO_MIME: dict[str, str] = {v: k for k, v in _MIME_TO_EXT.items()}

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


def reconcile_stickers() -> dict:
    """全面检查并修复表情包收藏，返回操作摘要 dict。

    按顺序处理以下问题：
    0. 预扫描 images/ 中所有有效图片，建立 filename→sha256 及 sha256→filename 映射
    1. index 记录的文件存在 → 校验 SHA-256，回填或更新
    2. index 记录的文件不存在，但磁盘上有相同 SHA-256 的文件（被改名）→ 修正 filename
    3. index 记录的文件不存在且磁盘上也找不到相同内容 → 清除 index 条目
    4. images/ 中有未被 index 认领的图片（含非标准文件名）→ 纳入 index
    5. SHA-256 重复 → 保留最早的（标准 id 优先），删除多余的
    6. 两步重命名法重编号为连续的 000.ext、001.ext…（防止改名时互相覆盖）
    """
    index = _load_index()
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    stats = {
        "removed_stale": 0,      # index 有记录但内容彻底消失
        "updated_hash": 0,       # SHA-256 回填或更新
        "fixed_rename": 0,       # 文件被改名，已修正 filename
        "adopted_orphans": 0,    # 孤儿文件纳入 index
        "removed_duplicates": 0, # 去除重复
    }
    changed = False

    # ── 0：预扫描磁盘，建立双向映射 ──────────────────────────────
    # disk_files:       filename  → sha256（已读取的文件）
    # disk_sha256_map:  sha256    → filename（用于通过哈希反查文件位置）
    disk_files: dict[str, str] = {}
    disk_sha256_map: dict[str, str] = {}
    for img_path in _IMAGES_DIR.iterdir():
        if img_path.suffix.lower() not in _VALID_EXTENSIONS:
            continue
        try:
            h = hashlib.sha256(img_path.read_bytes()).hexdigest()
            disk_files[img_path.name] = h
            # 同一 sha256 有多个文件时，保留字典序最小的（即编号最早的）
            if h not in disk_sha256_map or img_path.name < disk_sha256_map[h]:
                disk_sha256_map[h] = img_path.name
        except OSError as e:
            logger.warning("[sticker_collection] 预扫描跳过 %s: %s", img_path.name, e)

    # 已被 index 认领的文件名集合（防止孤儿扫描重复纳入）
    claimed: set[str] = set()

    # ── 1/2/3：校验 index 中每条记录 ──────────────────────────
    for sid in list(index.keys()):
        info = index[sid]
        current_filename = info["filename"]
        stored_sha256 = info.get("sha256", "")

        if current_filename in disk_files:
            # 文件存在：校验 SHA-256
            actual_sha256 = disk_files[current_filename]
            if not stored_sha256:
                index[sid]["sha256"] = actual_sha256
                stats["updated_hash"] += 1
                changed = True
                logger.info("[sticker_collection] 回填 SHA-256 id=%s", sid)
            elif stored_sha256 != actual_sha256:
                old_desc = index[sid].get("description", "")
                index[sid]["sha256"] = actual_sha256
                index[sid]["description"] = "（图片已被替换，暂无描述）"
                stats["updated_hash"] += 1
                changed = True
                logger.info(
                    "[sticker_collection] 文件内容已变更，更新 SHA-256 并清空描述 id=%s（原描述: %r）",
                    sid, old_desc,
                )
            claimed.add(current_filename)

        elif stored_sha256 and stored_sha256 in disk_sha256_map:
            # 文件不在原路径，但磁盘上有相同内容的文件 → 被改名了
            found_filename = disk_sha256_map[stored_sha256]
            index[sid]["filename"] = found_filename
            claimed.add(found_filename)
            stats["fixed_rename"] += 1
            changed = True
            logger.info(
                "[sticker_collection] id=%s 文件已被改名（%s → %s），已修正",
                sid, current_filename, found_filename,
            )

        else:
            # 文件彻底消失：index 条目无效
            logger.warning(
                "[sticker_collection] 清除失效条目 id=%s（文件不存在且无法通过 SHA-256 找回）", sid
            )
            del index[sid]
            stats["removed_stale"] += 1
            changed = True

    # ── 4：扫描 images/，纳入孤儿文件（含非标准文件名）─────────
    # 使用 "~" 前缀作为临时 key：~ (ASCII 126) > 所有数字和字母，
    # 保证标准 id 在后续 sorted() 中始终排在孤儿之前
    for filename, sha256 in sorted(disk_files.items()):
        if filename in claimed:
            continue
        mime = _EXT_TO_MIME.get(Path(filename).suffix.lower(), "image/jpeg")
        tmp_key = f"~orphan~{filename}"
        index[tmp_key] = {
            "description": "（用户手动添加，暂无描述）",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "filename": filename,
            "mime": mime,
            "sha256": sha256,
        }
        claimed.add(filename)
        stats["adopted_orphans"] += 1
        changed = True
        logger.info("[sticker_collection] 纳入孤儿文件: %s", filename)

    # ── 5：去重（SHA-256 相同只保留最早的，标准 id 优先）────────
    # sorted() 自然序下标准 id（"000"...）< "~orphan~..."，所以标准 id 先遍历，得以保留
    seen_sha256: dict[str, str] = {}
    for sid, info in sorted(index.items()):
        h = info.get("sha256", "")
        if not h:
            continue
        if h in seen_sha256:
            logger.info(
                "[sticker_collection] 删除重复 id=%s（同 id=%s）", sid, seen_sha256[h]
            )
            try:
                (_IMAGES_DIR / info["filename"]).unlink(missing_ok=True)
            except OSError as e:
                logger.warning("[sticker_collection] 删除重复文件失败: %s", e)
            del index[sid]
            stats["removed_duplicates"] += 1
            changed = True
        else:
            seen_sha256[h] = sid

    # ── 6：两步重命名法，重编号为连续的 000、001… ────────────────
    # sorted() 保证标准 id 在前、孤儿在后，结果：现有表情包保持相对顺序，孤儿追加在末尾
    sorted_entries = sorted(index.items())
    need_rename = not all(
        sid == f"{i:03d}" and Path(info["filename"]).stem == f"{i:03d}"
        for i, (sid, info) in enumerate(sorted_entries)
    )
    if need_rename:
        # 第一步：全部改为 __tmp_NNN.ext，彻底消除任何潜在的名称冲突
        tmp_entries: list[tuple[str, str, dict]] = []
        for new_num, (old_sid, info) in enumerate(sorted_entries):
            new_sid = f"{new_num:03d}"
            ext = Path(info["filename"]).suffix.lower()
            tmp_name = f"__tmp_{new_num:03d}{ext}"
            final_name = f"{new_sid}{ext}"
            try:
                (_IMAGES_DIR / info["filename"]).rename(_IMAGES_DIR / tmp_name)
                if old_sid != new_sid or info["filename"] != final_name:
                    logger.info(
                        "[sticker_collection] 重编号 id=%s(%s) → %s",
                        old_sid, info["filename"], new_sid,
                    )
                tmp_entries.append((tmp_name, new_sid, {**info, "filename": final_name}))
            except OSError as e:
                logger.warning(
                    "[sticker_collection] 第一步重命名失败 %s → %s: %s",
                    info["filename"], tmp_name, e,
                )
                # 重命名失败：保留原文件名，id 也不变以避免 index 损坏
                tmp_entries.append((info["filename"], old_sid, info))

        # 第二步：从 __tmp_NNN.ext 改为最终名
        new_index: dict = {}
        for tmp_name, new_sid, info in tmp_entries:
            final_name = info["filename"]
            tmp_path = _IMAGES_DIR / tmp_name
            if tmp_path.exists() and tmp_name != final_name:
                try:
                    tmp_path.rename(_IMAGES_DIR / final_name)
                except OSError as e:
                    logger.warning(
                        "[sticker_collection] 第二步重命名失败 %s → %s: %s",
                        tmp_name, final_name, e,
                    )
                    info = {**info, "filename": tmp_name}
            new_index[new_sid] = info
        index = new_index
        changed = True

    if changed:
        _save_index(index)
        try:
            _rebuild_grid_cache()
        except Exception as e:
            logger.warning("[sticker_collection] 重建网格缓存失败: %s", e)

    logger.info(
        "[sticker_collection] reconcile 完成："
        "清除失效=%d 更新哈希=%d 纳入孤儿=%d 删除重复=%d 剩余=%d",
        stats["removed_stale"], stats["updated_hash"],
        stats["adopted_orphans"], stats["removed_duplicates"], len(index),
    )
    return stats
