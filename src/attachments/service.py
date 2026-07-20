"""Background attachment downloads and constrained host-side readers."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import mimetypes
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import aiosqlite
import httpx

from browser.gateway import get_browser_gateway, resolve_public_addresses, validate_browser_url
from database import DB_PATH

from .models import AttachmentResult

logger = logging.getLogger("AICQ.attachments")
CACHE_ROOT = Path(__file__).resolve().parents[2] / "cache" / "attachments"
MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024
MAX_CACHE_BYTES = 1024 * 1024 * 1024
MAX_TEXT_READ_BYTES = 10 * 1024 * 1024
MAX_PDF_READ_BYTES = 25 * 1024 * 1024
OBSERVATION_SECONDS = 15.0
MAX_ACTIVE_DOWNLOADS = 4
MAX_DOWNLOAD_SECONDS = 300.0
MAX_TEXT_CHARS = 5000
_BAD_FILENAME = re.compile(r"[\\/:*?\"<>|\x00-\x1f]")
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml", ".xml",
    ".html", ".htm", ".css", ".js", ".ts", ".py", ".java", ".c", ".h", ".cpp",
    ".hpp", ".cs", ".go", ".rs", ".toml", ".ini", ".cfg", ".log", ".sql", ".sh",
    ".ps1", ".bat",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_filename(value: str | None, fallback: str = "attachment.bin") -> str:
    name = Path(str(value or "").replace("\\", "/")).name
    name = _BAD_FILENAME.sub("_", name).strip(" .")
    return (name or fallback)[:240]


def public_source_label(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, "", "", ""))


def _detect_mime(path: Path, filename: str) -> str:
    with path.open("rb") as stream:
        head = stream.read(32)
    if head.startswith(b"%PDF-"):
        return "application/pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "image/webp"
    guessed = mimetypes.guess_type(filename)[0]
    if guessed:
        return guessed
    if b"\x00" not in head:
        return "text/plain"
    return "application/octet-stream"


async def _validate_public_url(url: str) -> str:
    value = validate_browser_url(url)
    parsed = urlsplit(value)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL credentials are not allowed")
    port = parsed.port or (443 if parsed.scheme.casefold() == "https" else 80)
    await asyncio.to_thread(resolve_public_addresses, parsed.hostname or "", port)
    return value


async def _fetch_url(url: str, destination: Path, progress) -> tuple[str, int]:
    current = url
    downloaded = 0
    filename = safe_filename(unquote(Path(urlsplit(url).path).name))
    timeout = httpx.Timeout(connect=15.0, read=60.0, write=15.0, pool=15.0)
    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=timeout,
        trust_env=False,
        proxy=get_browser_gateway().proxy_url,
        headers={"User-Agent": "AICQ-Attachments/1", "Accept-Encoding": "identity"},
    ) as client:
        for _ in range(6):
            current = await _validate_public_url(current)
            async with client.stream("GET", current) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("download redirect has no location")
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()
                content_encoding = response.headers.get("content-encoding", "").strip().casefold()
                if content_encoding not in {"", "identity"}:
                    raise ValueError("compressed HTTP attachments are not accepted")
                length = response.headers.get("content-length", "")
                total = int(length) if length.isdigit() else None
                if total is not None and total > MAX_ATTACHMENT_BYTES:
                    raise ValueError("attachment exceeds the 100 MiB limit")
                disposition = response.headers.get("content-disposition", "")
                match = re.search(r"filename\*?=(?:UTF-8''|\")?([^\";]+)", disposition, re.I)
                if match:
                    filename = safe_filename(unquote(match.group(1).strip()), filename)
                with destination.open("wb") as output:
                    async for chunk in response.aiter_raw(1024 * 1024):
                        downloaded += len(chunk)
                        if downloaded > MAX_ATTACHMENT_BYTES:
                            raise ValueError("attachment exceeds the 100 MiB limit")
                        output.write(chunk)
                        await progress(downloaded, total)
                return filename, downloaded
        raise ValueError("download has too many redirects")


async def _thread_boundary(func, *args):
    """Run one bounded blocking operation and finish it before propagating cancellation."""
    task = asyncio.create_task(asyncio.to_thread(func, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


async def _copy_file_limited(source: Path, destination: Path, progress) -> int:
    copied = 0
    source_stream = source.open("rb")
    try:
        source_stat = await _thread_boundary(lambda: __import__("os").fstat(source_stream.fileno()))
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError("adapter source is not a regular file")
        with destination.open("xb") as output:
            while True:
                chunk = await _thread_boundary(source_stream.read, 1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > MAX_ATTACHMENT_BYTES:
                    raise ValueError("attachment exceeds the 100 MiB limit")
                await _thread_boundary(output.write, chunk)
                await progress(copied, source_stat.st_size)
    finally:
        await _thread_boundary(source_stream.close)
    return copied


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_download_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP download failed with status {exc.response.status_code}"
    if isinstance(exc, httpx.RequestError):
        return f"HTTP download failed ({type(exc).__name__})"
    if isinstance(exc, OSError):
        return f"attachment I/O failed ({type(exc).__name__})"
    return str(exc)


class AttachmentService:
    def __init__(self, cache_root: Path | None = None) -> None:
        self.cache_root = (cache_root or CACHE_ROOT).resolve()
        self._results: dict[str, AttachmentResult] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._futures: dict[str, asyncio.Future[AttachmentResult]] = {}
        self._deadlines: dict[str, asyncio.TimerHandle] = {}
        self._deadline_expired: set[str] = set()
        self._publish_lock = asyncio.Lock()
        self._cache_prepared = False
        self._terminal_callback = None
        self._closed = False

    def set_terminal_callback(self, callback) -> None:
        self._terminal_callback = callback

    async def _prepare_cache(self) -> None:
        if self._cache_prepared:
            return
        async with self._publish_lock:
            if self._cache_prepared:
                return
            await asyncio.to_thread(self.cache_root.mkdir, parents=True, exist_ok=True)
            for stale in self.cache_root.glob("download-*"):
                if stale.is_dir():
                    await asyncio.to_thread(shutil.rmtree, stale, True)
            self._cache_prepared = True

    def _blob_usage(self) -> int:
        total = 0
        for path in self.cache_root.rglob("*"):
            if path.is_file() and not any(part.startswith("download-") for part in path.parts):
                total += path.stat().st_size
        return total

    async def _publish_blob(self, temp_path: Path, digest: str, size: int) -> Path:
        stored = self.cache_root / digest[:2] / digest
        async with self._publish_lock:
            await asyncio.to_thread(stored.parent.mkdir, parents=True, exist_ok=True)
            if stored.exists():
                if stored.is_symlink() or not stored.is_file():
                    raise ValueError("attachment cache entry is not a regular file")
                existing_digest = await asyncio.to_thread(_hash_file, stored)
                if existing_digest != digest:
                    raise ValueError("attachment cache integrity check failed")
                await asyncio.to_thread(temp_path.unlink, missing_ok=True)
                return stored
            usage = await asyncio.to_thread(self._blob_usage)
            if usage + size > MAX_CACHE_BYTES:
                raise ValueError("attachment cache exceeds the 1 GiB limit")
            await asyncio.to_thread(temp_path.replace, stored)
        return stored

    async def _persist(self, result: AttachmentResult) -> bool:
        try:
            async with aiosqlite.connect(DB_PATH, timeout=30.0) as db:
                await db.execute(
                    """INSERT INTO attachment_tasks
                       (task_id,attachment_id,source_type,source,status,path,filename,mime,image_ref,
                        bytes_downloaded,bytes_total,sha256,error,started_at,finished_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(task_id) DO UPDATE SET status=excluded.status,path=excluded.path,
                       filename=excluded.filename,mime=excluded.mime,image_ref=excluded.image_ref,
                       bytes_downloaded=excluded.bytes_downloaded,bytes_total=excluded.bytes_total,
                       sha256=excluded.sha256,error=excluded.error,finished_at=excluded.finished_at""",
                    (
                        result.task_id, result.attachment_id, result.source_type, result.source,
                        result.status, result.path, result.filename, result.mime, result.image_ref,
                        result.bytes_downloaded, result.bytes_total, result.sha256, result.error,
                        result.started_at, result.finished_at,
                    ),
                )
                await db.commit()
            return True
        except Exception:
            logger.warning("failed to persist attachment task %s", result.task_id, exc_info=True)
            return False

    async def start(
        self,
        *,
        source_type: str,
        source: str,
        url: str | None = None,
        resolver: Callable[[], Awaitable[Mapping[str, object]]] | None = None,
    ) -> AttachmentResult:
        if self._closed:
            raise RuntimeError("attachment service is closed")
        await self._prepare_cache()
        if len(self._tasks) >= MAX_ACTIVE_DOWNLOADS:
            raise ValueError("too many active attachment downloads")
        if (url is None) == (resolver is None):
            raise ValueError("provide exactly one attachment source")
        task_id = uuid.uuid4().hex
        result = AttachmentResult(
            task_id=task_id,
            attachment_id=uuid.uuid4().hex,
            status="running",
            source_type=source_type,
            source=source,
            started_at=_utc_now(),
        )
        self._results[task_id] = result
        self._futures[task_id] = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(
            self._run(task_id, url=url, resolver=resolver),
            name=f"attachment-download-{task_id[:8]}",
        )
        self._tasks[task_id] = task

        def expire() -> None:
            if not task.done():
                self._deadline_expired.add(task_id)
                task.cancel()

        self._deadlines[task_id] = asyncio.get_running_loop().call_later(
            MAX_DOWNLOAD_SECONDS, expire
        )
        return result

    async def _run(self, task_id: str, *, url: str | None, resolver) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="download-", dir=str(self.cache_root)))
        temp_path = temp_dir / "payload"
        result = self._results[task_id]
        try:
            await self._persist(result)

            async def progress(done: int, total: int | None) -> None:
                self._results[task_id] = replace(
                    self._results[task_id], bytes_downloaded=done, bytes_total=total
                )

            resolved: dict[str, object] = dict(await resolver()) if resolver is not None else {}
            image_ref = str(resolved.get("image_ref") or "") or None
            if image_ref and resolved.get("data") is None and not resolved.get("url") and not resolved.get("host_path"):
                final = replace(
                    self._results[task_id], status="completed", image_ref=image_ref,
                    filename=str(resolved.get("filename") or f"qq_image_{image_ref}"),
                    mime=str(resolved.get("mime") or "image/jpeg"),
                    bytes_downloaded=int(resolved.get("size") or 0),
                    bytes_total=int(resolved.get("size") or 0), finished_at=_utc_now(),
                )
            else:
                filename = str(resolved.get("filename") or "") or None
                actual_url = str(resolved.get("url") or url or "") or None
                data = resolved.get("data")
                host_path = str(resolved.get("host_path") or "") or None
                if sum(value is not None for value in (actual_url, data, host_path)) != 1:
                    raise ValueError("attachment ref did not resolve to one source")
                if actual_url is not None:
                    discovered, size = await _fetch_url(actual_url, temp_path, progress)
                    filename = safe_filename(filename or discovered)
                elif data is not None:
                    raw = bytes(data)
                    if len(raw) > MAX_ATTACHMENT_BYTES:
                        raise ValueError("attachment exceeds the 100 MiB limit")
                    await asyncio.to_thread(temp_path.write_bytes, raw)
                    size = len(raw)
                    filename = safe_filename(filename)
                    await progress(size, size)
                else:
                    source_path = Path(str(host_path)).resolve(strict=True)
                    if not source_path.is_file():
                        raise ValueError("adapter source is not a regular file")
                    size = await _copy_file_limited(source_path, temp_path, progress)
                    filename = safe_filename(filename or source_path.name)
                digest = await asyncio.to_thread(_hash_file, temp_path)
                # Physical identity is content-only. Original filename/MIME stay in task metadata,
                # so identical bytes cannot be duplicated under different extensions.
                stored = await self._publish_blob(temp_path, digest, size)
                mime = await asyncio.to_thread(_detect_mime, stored, filename or stored.name)
                final = replace(
                    self._results[task_id], status="completed", path=str(stored),
                    filename=filename, mime=mime, image_ref=image_ref,
                    bytes_downloaded=size, bytes_total=size, sha256=digest, finished_at=_utc_now(),
                )
        except asyncio.CancelledError:
            expired = task_id in self._deadline_expired
            final = replace(
                self._results[task_id],
                status="failed" if expired else "stopped",
                error="download exceeded the 300 second limit" if expired else "download stopped",
                finished_at=_utc_now(),
            )
        except Exception as exc:
            final = replace(
                self._results[task_id],
                status="failed",
                error=_safe_download_error(exc),
                finished_at=_utc_now(),
            )
        finally:
            await asyncio.to_thread(shutil.rmtree, temp_dir, True)

        self._results[task_id] = final

        persisted = False

        async def finalize() -> None:
            nonlocal persisted
            persisted = await self._persist(final)
            callback = self._terminal_callback
            if callback is not None:
                try:
                    value = callback(final)
                    if inspect.isawaitable(value):
                        await value
                except Exception:
                    logger.warning("attachment terminal callback failed", exc_info=True)
            future = self._futures.get(task_id)
            if future is not None and not future.done():
                future.set_result(final)

        finalize_task = asyncio.create_task(finalize(), name=f"attachment-finalize-{task_id[:8]}")
        try:
            await asyncio.shield(finalize_task)
        except asyncio.CancelledError:
            # A stop/shutdown racing with persistence must not strand waiters or DB state.
            await finalize_task
        finally:
            deadline = self._deadlines.pop(task_id, None)
            if deadline is not None:
                deadline.cancel()
            self._deadline_expired.discard(task_id)
            self._tasks.pop(task_id, None)
            # Keep the terminal result available in memory during a transient DB outage.
            # Normal operation remains bounded because persisted jobs are released here.
            if persisted:
                self._results.pop(task_id, None)
                self._futures.pop(task_id, None)

    async def poll(self, task_id: str) -> AttachmentResult:
        if task_id in self._results:
            return self._results[task_id]
        async with aiosqlite.connect(DB_PATH, timeout=30.0) as db:
            db.row_factory = aiosqlite.Row
            row = await (await db.execute("SELECT * FROM attachment_tasks WHERE task_id=?", (task_id,))).fetchone()
        if row is None:
            raise ValueError("unknown attachment task_id")
        return AttachmentResult(**{key: row[key] for key in AttachmentResult.__dataclass_fields__})

    async def wait(self, task_id: str, timeout: float = OBSERVATION_SECONDS) -> AttachmentResult | None:
        if task_id not in self._futures:
            return await self.poll(task_id)
        try:
            return await asyncio.wait_for(asyncio.shield(self._futures[task_id]), timeout=max(0.0, timeout))
        except asyncio.TimeoutError:
            return None

    async def stop(self, task_id: str) -> AttachmentResult:
        result = await self.poll(task_id)
        task = self._tasks.get(task_id)
        if task is not None and not result.terminal:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return await self.poll(task_id)

    def mark_delivered(self, task_id: str) -> None:
        # Kept as a compatibility hook; terminal state is persisted and memory is bounded.
        return None

    async def _attachment(self, attachment_id: str) -> AttachmentResult:
        for result in self._results.values():
            if result.attachment_id == attachment_id and result.status == "completed":
                return result
        async with aiosqlite.connect(DB_PATH, timeout=30.0) as db:
            db.row_factory = aiosqlite.Row
            row = await (await db.execute(
                "SELECT * FROM attachment_tasks WHERE attachment_id=? AND status='completed' ORDER BY started_at DESC LIMIT 1",
                (attachment_id,),
            )).fetchone()
        if row is None:
            raise ValueError("unknown or incomplete attachment_id")
        result = AttachmentResult(**{key: row[key] for key in AttachmentResult.__dataclass_fields__})
        if result.image_ref and not result.path:
            raise ValueError("ephemeral QQ image attachment expired; download its message ref again")
        return result

    async def read(self, attachment_id: str, *, offset: int = 0, limit: int = MAX_TEXT_CHARS,
                   page_start: int = 1, page_count: int = 5) -> dict[str, object]:
        result = await self._attachment(attachment_id)
        if result.image_ref and not result.path:
            return {"ok": True, "attachment_id": attachment_id, "kind": "image",
                    "image_ref": result.image_ref, "mime": result.mime, "filename": result.filename,
                    "message": "图片复用当前会话缓存；可用 view_image_by_ref 或 examine_image 查看。"}
        if not result.path:
            raise ValueError("attachment has no readable file")
        path = Path(result.path).resolve(strict=True)
        try:
            path.relative_to(self.cache_root)
        except ValueError as exc:
            raise ValueError("attachment path escaped the managed cache") from exc
        if path.is_symlink() or not path.is_file():
            raise ValueError("attachment is not a regular cached file")
        mime = result.mime or await asyncio.to_thread(_detect_mime, path, result.filename or path.name)
        if mime.startswith("image/"):
            from PIL import Image

            def image_info() -> tuple[int, int, str]:
                with Image.open(path) as image:
                    return image.width, image.height, image.format or ""

            width, height, image_format = await asyncio.wait_for(asyncio.to_thread(image_info), timeout=10.0)
            return {"ok": True, "attachment_id": attachment_id, "kind": "image", "mime": mime,
                    "filename": result.filename, "width": width, "height": height, "format": image_format,
                    "message": "已识别为图片；当前视觉功能关闭时只能读取图片元数据。"}
        if mime == "application/pdf":
            if path.stat().st_size > MAX_PDF_READ_BYTES:
                raise ValueError("PDF exceeds the 25 MiB read limit")
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise ValueError("PDF reader dependency is not installed") from exc

            def pdf_text() -> tuple[int, str]:
                reader = PdfReader(str(path), strict=False)
                start = max(0, int(page_start) - 1)
                end = min(len(reader.pages), start + max(1, min(int(page_count), 10)))
                text = "\n\n".join(
                    f"--- page {index + 1} ---\n{reader.pages[index].extract_text() or ''}"
                    for index in range(start, end)
                )
                return len(reader.pages), text[:MAX_TEXT_CHARS]

            total_pages, content = await asyncio.wait_for(asyncio.to_thread(pdf_text), timeout=15.0)
            return {"ok": True, "attachment_id": attachment_id, "kind": "pdf", "filename": result.filename,
                    "page_start": max(1, page_start), "page_count": min(page_count, 10),
                    "total_pages": total_pages, "content": content}
        extension = Path(result.filename or path.name).suffix.lower()
        with path.open("rb") as stream:
            head = stream.read(4096)
        if extension not in _TEXT_EXTENSIONS and mime != "text/plain" and b"\x00" in head:
            return {"ok": True, "attachment_id": attachment_id, "kind": "binary", "filename": result.filename,
                    "mime": mime, "size": result.bytes_total, "sha256": result.sha256,
                    "message": "这是二进制文件；只返回元数据，不执行或反编译。"}
        if path.stat().st_size > MAX_TEXT_READ_BYTES:
            raise ValueError("text attachment exceeds the 10 MiB read limit")
        raw = await asyncio.to_thread(path.read_bytes)
        text = None
        encoding = "utf-8"
        for candidate in ("utf-8-sig", "utf-16", "gb18030"):
            try:
                text = raw.decode(candidate)
                encoding = candidate
                break
            except UnicodeError:
                continue
        if text is None:
            raise ValueError("attachment is not decodable text")
        start = max(0, int(offset))
        count = max(1, min(int(limit), MAX_TEXT_CHARS))
        content = text[start:start + count]
        return {"ok": True, "attachment_id": attachment_id, "kind": "text", "filename": result.filename,
                "encoding": encoding, "offset": start, "next_offset": start + len(content),
                "has_more": start + len(content) < len(text), "total_chars": len(text), "content": content}

    async def close(self) -> None:
        self._closed = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
