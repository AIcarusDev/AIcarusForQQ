"""sticker_collection.py — 表情包收藏管理

布局：
  data/stickers/
    index.json       ← 表情包索引（id → {description, created_at, filename, mime}）
    000.jpg          ← 表情包图片文件
    001.png
    ...
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("AICQ.sticker_collection")

# 表情包目录：项目根 / data / stickers
_STICKER_DIR = Path(__file__).parent.parent.parent / "data" / "stickers"
_INDEX_PATH = _STICKER_DIR / "index.json"

_MIME_TO_EXT: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}


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

def save_sticker(raw_bytes: bytes, mime: str, description: str) -> str:
    """将图片存为新表情包，返回生成的 ID（如 '000'）。"""
    index = _load_index()
    sid = _next_id(index)
    ext = _MIME_TO_EXT.get(mime, ".jpg")
    filename = f"{sid}{ext}"

    _STICKER_DIR.mkdir(parents=True, exist_ok=True)
    (_STICKER_DIR / filename).write_bytes(raw_bytes)

    index[sid] = {
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "mime": mime,
    }
    _save_index(index)
    logger.info("[sticker_collection] 已保存表情包 id=%s desc=%r", sid, description)
    return sid


def load_sticker_bytes(sticker_id: str) -> Optional[tuple[bytes, str]]:
    """读取表情包原始字节，返回 (bytes, mime)，不存在返回 None。"""
    index = _load_index()
    if sticker_id not in index:
        return None
    entry = index[sticker_id]
    p = _STICKER_DIR / entry["filename"]
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
        if not (_STICKER_DIR / info["filename"]).exists()
    ]:
        for sid in stale_ids:
            logger.warning(
                "[sticker_collection] 清理过期索引 id=%s (文件不存在)", sid
                )
            del index[sid]
        _save_index(index)
    return [{"id": sid, **info} for sid, info in sorted(index.items())]
