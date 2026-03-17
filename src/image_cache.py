"""image_cache.py — 图片落盘 + pHash 去重 + sidecar 描述缓存

将收到的图片以感知哈希（pHash）为键持久化到磁盘，避免：
  - 同一张图（经过 JPEG 重压缩/加水印）反复存储和描述
  - 重启后图片数据丢失（描述依然可复用）

布局：
  data/image_cache/
    {phash[:2]}/
      {phash}.jpg          ← 原始图片字节
      {phash}.meta.json    ← sidecar，含 description + examinations

需要 imagehash + Pillow：
  pip install imagehash Pillow
当库未安装时，所有函数优雅降级（缓存/相似查找均返回 None）。
"""

import base64
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("AICQ.image_cache")

# 缓存目录：src/ 上一级 / data / image_cache
_CACHE_DIR = Path(__file__).parent.parent / "data" / "image_cache"

_MIME_TO_EXT: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}

# 懒加载 — 避免强制依赖
try:
    import imagehash
    from PIL import Image as _PILImage
    _PHASH_AVAILABLE = True
except ImportError:
    _PHASH_AVAILABLE = False
    logger.debug("[image_cache] imagehash/Pillow 未安装，pHash 功能已禁用")


# ── 内部工具 ──────────────────────────────────────────────

def _phash_str(raw_bytes: bytes) -> Optional[str]:
    """从原始字节计算感知哈希（64位，hex 字符串）。失败返回 None。"""
    if not _PHASH_AVAILABLE:
        return None
    try:
        img = _PILImage.open(io.BytesIO(raw_bytes))
        ph = imagehash.phash(img)
        return str(ph)  # e.g. "f8e0c0a0b8d0e8f0"
    except Exception as exc:
        logger.debug("[image_cache] pHash 计算失败: %s", exc)
        return None


def _image_path(phash: str, mime: str) -> Path:
    ext = _MIME_TO_EXT.get(mime, ".jpg")
    return _CACHE_DIR / phash[:2] / f"{phash}{ext}"


def _meta_path(phash: str) -> Path:
    return _CACHE_DIR / phash[:2] / f"{phash}.meta.json"


def _load_meta_raw(phash: str) -> dict:
    p = _meta_path(phash)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_meta_raw(phash: str, meta: dict) -> None:
    p = _meta_path(phash)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 公共 API ──────────────────────────────────────────────

def cache_image(raw_bytes: bytes, mime: str) -> tuple[Optional[str], bool]:
    """将图片字节落盘并返回 (phash, is_new)。

    - 若 imagehash 不可用，返回 (None, False)。
    - 若存在汉明距离 ≤ 默认阈值的缓存，视为同一张图，返回已有 phash，is_new=False。
    - 否则写入磁盘并初始化 .meta.json，is_new=True。
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    phash = _phash_str(raw_bytes)
    if phash is None:
        return None, False

    # 检查是否已有相似图片
    existing = find_similar(phash)
    if existing is not None:
        logger.debug("[image_cache] 相似图已存在: %s（新 %s）", existing[:8], phash[:8])
        return existing, False

    # 写入图片文件
    img_p = _image_path(phash, mime)
    img_p.parent.mkdir(parents=True, exist_ok=True)
    if not img_p.exists():
        img_p.write_bytes(raw_bytes)

    # 初始化 sidecar
    meta = {
        "phash": phash,
        "mime": mime,
        "size": len(raw_bytes),
        "first_seen_at": datetime.now(timezone.utc).isoformat(),
        "description": None,
        "examinations": [],
    }
    _save_meta_raw(phash, meta)
    logger.debug("[image_cache] 新图片已缓存: %s", phash[:8])
    return phash, True


def find_similar(phash: str, threshold: int = 10) -> Optional[str]:
    """在同前缀子目录中查找汉明距离 ≤ threshold 的已缓存图片。

    返回匹配的 phash 字符串，未找到返回 None。
    精确匹配（距离=0）也在此处处理。
    """
    if not _PHASH_AVAILABLE:
        return None
    try:
        query_h = imagehash.hex_to_hash(phash)
    except Exception:
        return None

    prefix_dir = _CACHE_DIR / phash[:2]
    if not prefix_dir.exists():
        return None

    best_phash: Optional[str] = None
    best_dist = threshold + 1

    for meta_p in prefix_dir.glob("*.meta.json"):
        candidate = meta_p.stem
        try:
            candidate_h = imagehash.hex_to_hash(candidate)
            dist = query_h - candidate_h
            if dist <= threshold and dist < best_dist:
                best_dist = dist
                best_phash = candidate
        except Exception:
            continue

    return best_phash


def load_meta(phash: str) -> dict:
    """加载指定 phash 的 sidecar meta，不存在时返回空 dict。"""
    return _load_meta_raw(phash)


def update_description(phash: str, description: str) -> None:
    """设置或更新 description 字段，其他字段保持不变。"""
    meta = _load_meta_raw(phash)
    meta["description"] = description
    _save_meta_raw(phash, meta)


def append_examination(phash: str, focus: str, result_text: str) -> None:
    """向 examinations 列表追加一条精查记录。"""
    meta = _load_meta_raw(phash)
    if "examinations" not in meta:
        meta["examinations"] = []
    meta["examinations"].append({
        "focus": focus,
        "result": result_text,
        "examined_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_meta_raw(phash, meta)


def read_image_bytes(phash: str) -> Optional[bytes]:
    """从磁盘读取缓存图片的原始字节，失败返回 None。"""
    meta = _load_meta_raw(phash)
    mime = meta.get("mime", "image/jpeg")
    p = _image_path(phash, mime)
    try:
        return p.read_bytes() if p.exists() else None
    except Exception:
        return None


def read_image_b64(phash: str) -> Optional[tuple[str, str]]:
    """从磁盘读取缓存图片，返回 (base64字符串, mime)，失败返回 None。"""
    meta = _load_meta_raw(phash)
    mime = meta.get("mime", "image/jpeg")
    raw = read_image_bytes(phash)
    if raw is None:
        return None
    return base64.b64encode(raw).decode("ascii"), mime
