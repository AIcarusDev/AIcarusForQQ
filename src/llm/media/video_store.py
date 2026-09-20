"""Durable video sources and bounded, immutable on-demand originals."""
from __future__ import annotations

import hashlib
import asyncio
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urljoin

from . import media_storage as storage
from .image_store import connection, _reserve
from .media_identity import bind_media_identity

MAX_VIDEO_BYTES = 128 * 1024 * 1024
_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".flv"}
_SCHEMA = """CREATE TABLE IF NOT EXISTS media_video_sources (
 video_ref TEXT PRIMARY KEY, source_key TEXT UNIQUE, url TEXT NOT NULL,
 source TEXT NOT NULL
)"""


class VideoStoreError(RuntimeError):
    pass


def _valid_ref(value: str) -> str:
    ref = str(value or "").strip()
    if not storage._SAFE_REF_PATTERN.fullmatch(ref):
        raise VideoStoreError("无效的视频引用")
    return ref


def register_video_source(url: str, *, source: str, source_key: str | None = None,
                          video_ref: str | None = None) -> str:
    """Keep raw URLs internal. A source key preserves identity across snapshots."""
    key = hashlib.sha256(source_key.encode()).hexdigest() if source_key else None
    with connection(write=True) as db:
        db.execute(_SCHEMA)
        if key:
            row = db.execute("SELECT video_ref FROM media_video_sources WHERE source_key=?", (key,)).fetchone()
            if row:
                return row[0]
        ref = _valid_ref(video_ref) if video_ref else _reserve(db)
        db.execute("INSERT OR IGNORE INTO media_video_sources VALUES (?,?,?,?)",
                   (ref, key, str(url or "").strip(), source))
        return ref


def get_video_source(video_ref: str) -> dict | None:
    ref = _valid_ref(video_ref)
    with connection() as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "media_video_sources" in tables:
            row = db.execute("SELECT * FROM media_video_sources WHERE video_ref=?", (ref,)).fetchone()
            if row:
                return dict(row)
        # Recover refs issued before the source registry was introduced.
        if "chat_messages" in tables:
            for row in db.execute("SELECT content_segments FROM chat_messages WHERE content_segments LIKE ?",
                                  (f'%"{ref}"%',)):
                try:
                    segments = json.loads(row[0] or "[]")
                except (TypeError, ValueError):
                    continue
                for seg in segments if isinstance(segments, list) else []:
                    if isinstance(seg, dict) and seg.get("type") == "video" and seg.get("video_ref") == ref:
                        return {"video_ref": ref, "url": seg.get("url", ""), "source": "qq"}
    return None


def locate_video(video_ref: str) -> Path | None:
    ref = _valid_ref(video_ref)
    path = storage.locate_media_file(ref)
    return path if path is not None and path.suffix.lower() in _EXTENSIONS else None


def _container_extension(header: bytes) -> str:
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return ".mov" if header[8:12] == b"qt  " else ".mp4"
    if header.startswith(b"\x1aE\xdf\xa3"):
        return ".webm" if b"webm" in header[:4096] else ".mkv"
    if header.startswith(b"RIFF") and header[8:12] == b"AVI ":
        return ".avi"
    if header.startswith(b"FLV"):
        return ".flv"
    raise VideoStoreError("下载内容不是支持的视频容器，可能是登录页或已失效的地址")


async def download_video_for_ref(video_ref: str, url: str, *, timeout: float = 60.0,
                                 max_bytes: int = MAX_VIDEO_BYTES) -> Path:
    """Download through the existing public-network gateway; never overwrite refs."""
    from .image_importer import _default_http_client, _safe_public_http_url

    ref = _valid_ref(video_ref)
    limit = min(int(max_bytes), MAX_VIDEO_BYTES)
    if limit <= 0:
        raise VideoStoreError("视频大小限制必须为正数")
    existing = locate_video(ref)
    if existing is not None:
        if existing.stat().st_size > limit:
            raise VideoStoreError("视频超过大小限制，请先在 Agent 电脑中压缩或截取片段")
        return existing
    if not str(url).startswith(("http://", "https://")):
        raise VideoStoreError("视频没有可下载的 HTTP(S) 来源（blob/MSE 不支持）；请用 computer 将视频保存到 /home/agent 后传 path")
    date = storage.parse_time_ref_date(ref)
    folder = storage.get_media_dir(date[0], date[1]) if date else storage.MEDIA_ROOT
    if not storage._inside_media_root(folder):
        raise VideoStoreError("视频存储路径超出媒体目录")
    folder.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        current = _safe_public_http_url(url)
        async with asyncio.timeout(timeout), _default_http_client() as http:
            for redirects in range(6):
                current = _safe_public_http_url(current)
                async with http.stream("GET", current, timeout=timeout, headers={
                    "Accept": "video/*,application/octet-stream;q=0.9,*/*;q=0.1",
                    "User-Agent": "AIcarusForQQ/video", "Accept-Encoding": "identity",
                }) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location or redirects == 5:
                            raise VideoStoreError("视频下载重定向失败")
                        current = urljoin(current, location)
                        continue
                    if response.status_code != 200:
                        raise VideoStoreError(f"视频下载失败 (HTTP {response.status_code})；来源可能需要登录或已过期，请重新获取视频或传 Agent 电脑 path")
                    length = response.headers.get("content-length", "")
                    if response.headers.get("content-encoding", "identity").lower() not in {"", "identity"}:
                        raise VideoStoreError("视频下载响应使用了不支持的压缩编码")
                    if length.isdigit() and int(length) > limit:
                        raise VideoStoreError("视频超过下载大小限制")
                    total = 0
                    hasher = hashlib.sha256()
                    with tempfile.NamedTemporaryFile(dir=folder, prefix=".video-", delete=False) as stream:
                        temporary = Path(stream.name)
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            total += len(chunk)
                            if total > limit:
                                raise VideoStoreError("视频超过下载大小限制")
                            stream.write(chunk)
                            hasher.update(chunk)
                        stream.flush()
                        os.fsync(stream.fileno())
                    with temporary.open("rb") as stream:
                        extension = _container_extension(stream.read(4096))
                    target = folder / f"{ref}{extension}"
                    if not storage._inside_media_root(target):
                        raise VideoStoreError("视频存储路径超出媒体目录")
                    bind_media_identity(ref, hasher.hexdigest())
                    try:
                        os.link(temporary, target)
                    except FileExistsError:
                        with target.open("rb") as stream:
                            if hashlib.file_digest(stream, "sha256").hexdigest() != hasher.hexdigest():
                                raise VideoStoreError("视频引用已绑定其他内容")
                    return target
        raise VideoStoreError("视频下载重定向失败")
    except VideoStoreError:
        raise
    except Exception as exc:
        raise VideoStoreError("视频下载或持久化失败，请重新获取来源后重试") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
