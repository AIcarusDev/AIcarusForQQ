"""Resolve or download validated images and atomically import them into Agent Linux."""

from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, AsyncContextManager, Callable
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from browser.gateway import BrowserNetworkError, classify_target, get_browser_gateway
from workspace.errors import WorkspaceError

from .image_resolver import ImagePayloadError, ImageResolver, inspect_image_payload


MAX_SAVED_IMAGE_BYTES = 20 * 1024 * 1024
MAX_SAVED_IMAGE_PIXELS = 100_000_000
MAX_IMAGE_URL_CHARS = 8192
_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_DOWNLOADS = asyncio.Semaphore(4)
_MIME_EXTENSIONS = {
    "image/avif": (".avif",),
    "image/bmp": (".bmp",),
    "image/gif": (".gif",),
    "image/x-icon": (".ico",),
    "image/jpeg": (".jpg", ".jpeg"),
    "image/png": (".png",),
    "image/webp": (".webp",),
}

HttpClientFactory = Callable[[], AsyncContextManager[httpx.AsyncClient]]


class ImageImportError(RuntimeError):
    """Stable error returned by the save-image tool boundary."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = str(code)
        self.message = str(message)
        self.retryable = bool(retryable)
        self.details = dict(details or {})
        super().__init__(self.message)


@dataclass(frozen=True, slots=True)
class SavedImage:
    path: str
    mime_type: str
    size_bytes: int


def _default_http_client() -> AsyncContextManager[httpx.AsyncClient]:
    return httpx.AsyncClient(
        proxy=get_browser_gateway().proxy_url,
        headers={
            "User-Agent": "AIcarusForQQ/save-image",
            "Accept": "image/*,*/*;q=0.1",
            "Accept-Encoding": "identity",
        },
        timeout=httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=30.0),
        follow_redirects=False,
        trust_env=False,
        verify=True,
    )


def _safe_public_http_url(value: object) -> str:
    url = str(value or "").strip()
    if not url or len(url) > MAX_IMAGE_URL_CHARS:
        raise ImageImportError("invalid_url", "图片 URL 无效")
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.casefold()
        if (
            scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ImageImportError("invalid_url", "图片 URL 必须是无凭据的 HTTP(S) 地址")
        port = parsed.port or (443 if scheme == "https" else 80)
        target = classify_target(parsed.hostname, port)
    except ImageImportError:
        raise
    except (BrowserNetworkError, ValueError) as exc:
        raise ImageImportError("invalid_url", "图片 URL 无效") from exc

    if target.workspace_loopback:
        raise ImageImportError("unsafe_url", "图片 URL 不能指向本机地址")
    try:
        literal = ipaddress.ip_address(target.host)
    except ValueError:
        literal = None
    if literal is not None and not literal.is_global:
        raise ImageImportError("unsafe_url", "图片 URL 不能指向非公开地址")
    return urlunsplit((scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))


def _content_length(response: httpx.Response) -> int | None:
    value = str(response.headers.get("content-length") or "").strip()
    if not value:
        return None
    try:
        size = int(value)
    except ValueError:
        return None
    return size if size >= 0 else None


async def download_image_bytes(
    url: object,
    *,
    max_bytes: int = MAX_SAVED_IMAGE_BYTES,
    http_client_factory: HttpClientFactory = _default_http_client,
) -> bytes:
    """Download one public HTTP(S) resource through the isolated browser gateway."""

    current = _safe_public_http_url(url)
    try:
        async with _DOWNLOADS, http_client_factory() as http:
            for redirect_count in range(_MAX_REDIRECTS + 1):
                current = _safe_public_http_url(current)
                async with http.stream("GET", current) as response:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = str(response.headers.get("location") or "").strip()
                        if not location or redirect_count >= _MAX_REDIRECTS:
                            raise ImageImportError(
                                "redirect_error",
                                "图片 URL 跳转异常",
                                retryable=True,
                            )
                        current = urljoin(current, location)
                        continue
                    if response.status_code < 200 or response.status_code >= 300:
                        raise ImageImportError(
                            "download_failed",
                            "图片下载失败",
                            retryable=response.status_code >= 500 or response.status_code in {408, 429},
                            details={"status_code": int(response.status_code)},
                        )
                    encoding = str(response.headers.get("content-encoding") or "").strip().lower()
                    if encoding not in {"", "identity"}:
                        raise ImageImportError(
                            "unsupported_encoding",
                            "图片响应使用了不支持的内容编码",
                        )
                    declared_size = _content_length(response)
                    if declared_size is not None and declared_size > max_bytes:
                        raise ImageImportError(
                            "image_too_large",
                            "图片超过大小限制",
                            details={"size_bytes": declared_size, "limit_bytes": max_bytes},
                        )
                    data = bytearray()
                    async for chunk in response.aiter_raw(256 * 1024):
                        if not chunk:
                            continue
                        data.extend(chunk)
                        if len(data) > max_bytes:
                            raise ImageImportError(
                                "image_too_large",
                                "图片超过大小限制",
                                details={"size_bytes": len(data), "limit_bytes": max_bytes},
                            )
                    if declared_size is not None and len(data) != declared_size:
                        raise ImageImportError(
                            "size_mismatch",
                            "图片下载大小与响应声明不一致",
                            retryable=True,
                        )
                    return bytes(data)
    except ImageImportError:
        raise
    except (httpx.HTTPError, BrowserNetworkError, OSError) as exc:
        raise ImageImportError(
            "download_failed",
            "图片下载失败",
            retryable=True,
        ) from exc
    raise ImageImportError("redirect_error", "图片 URL 跳转异常", retryable=True)


def _decode_image_payload(payload: tuple[str | bytes, str]) -> bytes:
    value, _mime_type = payload
    if isinstance(value, bytes):
        return value
    encoded = value.strip()
    max_encoded_chars = ((MAX_SAVED_IMAGE_BYTES + 2) // 3) * 4 + 4
    if len(encoded) > max_encoded_chars:
        raise ImageImportError(
            "image_too_large",
            "图片超过大小限制",
            details={"limit_bytes": MAX_SAVED_IMAGE_BYTES},
        )
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageImportError("invalid_image_data", "图片数据无效") from exc


def _validate_destination_extension(path: str, mime_type: str) -> None:
    expected = _MIME_EXTENSIONS.get(mime_type, ())
    suffix = PurePosixPath(path).suffix.casefold()
    if suffix not in expected:
        raise ImageImportError(
            "extension_mismatch",
            "目标路径扩展名与图片格式不一致",
            details={"mime_type": mime_type, "expected_extensions": list(expected)},
        )


async def _import_bytes(workspace_service: Any, path: str, raw: bytes) -> None:
    session = None
    try:
        session = await workspace_service.begin_file_import(path, len(raw))
        for offset in range(0, len(raw), 256 * 1024):
            await session.write(raw[offset:offset + 256 * 1024])
        result = await session.finish()
    except WorkspaceError as exc:
        if session is not None:
            await asyncio.shield(session.abort())
        raise ImageImportError(
            exc.code.value,
            "Agent 电脑无法保存图片",
            retryable=exc.code.value not in {"invalid_argument", "path_error"},
        ) from exc
    except BaseException:
        if session is not None:
            await asyncio.shield(session.abort())
        raise
    if not result.get("ok"):
        code = str(result.get("code") or "write_failed")
        raise ImageImportError(
            code,
            "Agent 电脑无法保存图片",
            retryable=code not in {"already_exists", "invalid_path", "permission_denied"},
        )
    committed = int(result.get("size_bytes", -1))
    if committed != len(raw):
        raise ImageImportError("size_mismatch", "图片写入大小不一致", retryable=True)


class ImageImporter:
    """Save model-visible or public remote images into the Agent workspace."""

    def __init__(
        self,
        session: Any,
        workspace_service: Any,
        *,
        resolver: ImageResolver | None = None,
        http_client_factory: HttpClientFactory = _default_http_client,
    ) -> None:
        self.resolver = resolver or ImageResolver(session)
        self.workspace_service = workspace_service
        self.http_client_factory = http_client_factory

    async def save(
        self,
        *,
        path: str,
        image_ref: str | None = None,
        url: str | None = None,
    ) -> SavedImage:
        if (image_ref is None) == (url is None):
            raise ImageImportError("invalid_arguments", "image_ref 与 url 必须且只能提供一个")

        if image_ref is not None:
            found = self.resolver.resolve(image_ref)
            if found is None:
                raise ImageImportError("not_found", "未找到指定的 image_ref")
            image, _source = found
            payload = self.resolver.payload(image)
            if payload is None:
                status = self.resolver.unavailable_status(image)
                raise ImageImportError(status, "图片当前不可用", retryable=status == "pending")
            raw = _decode_image_payload(payload)
        else:
            raw = await download_image_bytes(
                url,
                max_bytes=MAX_SAVED_IMAGE_BYTES,
                http_client_factory=self.http_client_factory,
            )

        try:
            image_info = inspect_image_payload(
                raw,
                max_bytes=MAX_SAVED_IMAGE_BYTES,
                max_pixels=MAX_SAVED_IMAGE_PIXELS,
            )
        except ImagePayloadError as exc:
            raise ImageImportError(exc.code, "图片内容无效", details=exc.details) from exc
        _validate_destination_extension(path, image_info.mime_type)
        await _import_bytes(self.workspace_service, path, raw)
        return SavedImage(path=path, mime_type=image_info.mime_type, size_bytes=len(raw))


__all__ = [
    "ImageImportError",
    "ImageImporter",
    "MAX_IMAGE_URL_CHARS",
    "MAX_SAVED_IMAGE_BYTES",
    "MAX_SAVED_IMAGE_PIXELS",
    "SavedImage",
    "download_image_bytes",
]
