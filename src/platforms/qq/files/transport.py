"""Adapter-specific transports for receiving QQ file message bodies."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit

import httpx

from ..adapter.errors import QQFileStreamError


logger = logging.getLogger("AICQ.qq_file.transport")
_HTTP_DOWNLOADS = asyncio.Semaphore(4)
_MAX_REDIRECTS = 5
_REFRESHABLE_STATUSES = {403, 404}
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def adapter_for_download(client: Any) -> str:
    """Freeze the configured/detected adapter without guessing in auto mode."""

    configured_value = getattr(client, "configured_adapter", None)
    if configured_value is None:
        legacy_adapter = str(getattr(client, "adapter", "") or "").strip().lower()
        if legacy_adapter in {"napcat", "llonebot"}:
            return legacy_adapter
        if callable(getattr(client, "download_file_stream", None)):
            return "napcat"
        return ""
    configured = str(configured_value or "auto").strip().lower()
    detected = str(getattr(client, "detected_adapter", "") or "").strip().lower()
    if configured == "auto":
        return detected
    return configured


async def download_qq_file(
    client: Any,
    file_id: str,
    destination: Any,
    *,
    adapter: str,
    conversation_type: str,
    conversation_id: str,
    declared_size: int | None = None,
    max_bytes: int,
    on_progress: Callable[[int, int | None], Any] | None = None,
) -> dict[str, Any]:
    """Download one ordinary QQ file through the frozen adapter dialect."""

    normalized_adapter = str(adapter or "").strip().lower()
    if normalized_adapter == "napcat":
        return await client.download_file_stream(
            file_id,
            destination,
            max_bytes=max_bytes,
            on_progress=on_progress,
        )
    if normalized_adapter == "llonebot":
        return await _download_llonebot_file(
            client,
            file_id,
            destination,
            conversation_type=conversation_type,
            conversation_id=conversation_id,
            declared_size=declared_size,
            max_bytes=max_bytes,
            on_progress=on_progress,
        )
    raise QQFileStreamError(
        "source_unavailable",
        "当前 QQ 适配器类型未识别，暂不支持文件下载",
        retryable=True,
    )


async def _resolve_llonebot_url(
    client: Any,
    file_id: str,
    *,
    conversation_type: str,
    conversation_id: str,
) -> str:
    if conversation_type == "group":
        action = "get_group_file_url"
        params = {"group_id": str(conversation_id), "file_id": str(file_id)}
    elif conversation_type == "private":
        action = "get_private_file_url"
        params = {"file_id": str(file_id)}
    else:
        raise QQFileStreamError(
            "source_unavailable",
            "当前 QQ 会话不支持文件下载",
            retryable=False,
        )

    response = await client.send_api(action, params, timeout=30.0)
    url = str(response.get("url") or "").strip() if isinstance(response, dict) else ""
    if not url:
        raise QQFileStreamError(
            "source_unavailable",
            "无法取得 QQ 文件下载地址",
            retryable=True,
        )
    return url


def _is_public_address(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_global
    except ValueError:
        return False


async def _validate_public_http_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise QQFileStreamError(
            "source_unavailable",
            "QQ 文件下载地址无效",
            retryable=True,
        )
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise QQFileStreamError(
            "source_unavailable",
            "QQ 文件下载地址不安全",
            retryable=False,
        )
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        addresses = {str(literal)}
    else:
        try:
            infos = await asyncio.to_thread(
                socket.getaddrinfo,
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except OSError as exc:
            raise QQFileStreamError(
                "transport_error",
                "QQ 文件下载地址暂时无法连接",
                retryable=True,
            ) from exc
        addresses = {str(item[4][0]).split("%", 1)[0] for item in infos if item[4]}
    if not addresses or any(not _is_public_address(address) for address in addresses):
        raise QQFileStreamError(
            "source_unavailable",
            "QQ 文件下载地址不安全",
            retryable=False,
        )


def _content_length(response: httpx.Response) -> int | None:
    value = str(response.headers.get("content-length") or "").strip()
    if not value:
        return None
    try:
        size = int(value)
    except ValueError:
        return None
    return size if size >= 0 else None


async def _send_with_safe_redirects(http: httpx.AsyncClient, url: str) -> httpx.Response:
    current = url
    for redirect_count in range(_MAX_REDIRECTS + 1):
        await _validate_public_http_url(current)
        response = await http.send(http.build_request("GET", current), stream=True)
        if response.status_code not in _REDIRECT_STATUSES:
            return response
        location = str(response.headers.get("location") or "").strip()
        await response.aclose()
        if not location or redirect_count >= _MAX_REDIRECTS:
            raise QQFileStreamError(
                "transport_error",
                "QQ 文件下载地址跳转异常",
                retryable=True,
            )
        current = urljoin(current, location)
    raise QQFileStreamError("transport_error", "QQ 文件下载地址跳转异常", retryable=True)


async def _write_http_response(
    response: httpx.Response,
    destination: Any,
    *,
    declared_size: int | None,
    max_bytes: int,
    on_progress: Callable[[int, int | None], Any] | None,
) -> int:
    content_encoding = str(response.headers.get("content-encoding") or "").strip().lower()
    if content_encoding not in {"", "identity"}:
        raise QQFileStreamError(
            "verification_failed",
            "QQ 文件下载响应使用了不支持的内容编码",
            retryable=True,
        )
    response_size = _content_length(response)
    if response_size is not None and response_size > max_bytes:
        raise QQFileStreamError("file_too_large", "QQ 文件超过下载大小限制", retryable=False)
    if declared_size is not None and response_size is not None and declared_size != response_size:
        raise QQFileStreamError("size_mismatch", "QQ 文件下载大小与声明不一致")

    expected_size = declared_size if declared_size is not None else response_size
    sink = destination if hasattr(destination, "begin") else None
    output = None
    sink_started = False
    sink_finished = False
    returning_success = False
    received = 0
    try:
        if sink is not None:
            if expected_size is None:
                raise QQFileStreamError("verification_failed", "QQ 文件下载流没有声明文件大小")
            await sink.begin(expected_size)
            sink_started = True
        else:
            output = Path(destination).open("wb")
        if on_progress:
            result = on_progress(0, expected_size)
            if asyncio.iscoroutine(result):
                await result

        async for chunk in response.aiter_raw(256 * 1024):
            if not chunk:
                continue
            received += len(chunk)
            if received > max_bytes:
                raise QQFileStreamError("file_too_large", "QQ 文件超过下载大小限制", retryable=False)
            if expected_size is not None and received > expected_size:
                raise QQFileStreamError("size_mismatch", "QQ 文件下载数据超过声明大小")
            if sink is not None:
                await sink.write(chunk)
            else:
                output.write(chunk)
            if on_progress:
                result = on_progress(received, expected_size)
                if asyncio.iscoroutine(result):
                    await result

        if expected_size is not None and received != expected_size:
            raise QQFileStreamError("size_mismatch", "QQ 文件实际大小与声明不一致")
        if sink is not None:
            sink_finished = True
            committed_size = await sink.finish()
            if int(committed_size) != received:
                raise QQFileStreamError("size_mismatch", "QQ 文件写入大小不一致")
        elif output is not None:
            output.flush()
        returning_success = True
        return received
    except OSError as exc:
        raise QQFileStreamError(
            "write_failed",
            "QQ 文件写入失败",
            retryable=True,
        ) from exc
    finally:
        if output is not None:
            output.close()
        if sink is not None and sink_started and not returning_success:
            try:
                if sink_finished and hasattr(sink, "rollback"):
                    await asyncio.shield(sink.rollback())
                elif not sink_finished:
                    await asyncio.shield(sink.abort())
            except Exception:
                logger.warning("QQ 文件下载流清理失败", exc_info=True)


async def _download_llonebot_file(
    client: Any,
    file_id: str,
    destination: Any,
    *,
    conversation_type: str,
    conversation_id: str,
    declared_size: int | None,
    max_bytes: int,
    on_progress: Callable[[int, int | None], Any] | None,
) -> dict[str, Any]:
    timeout = httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=30.0)
    async with _HTTP_DOWNLOADS, httpx.AsyncClient(
        headers={"User-Agent": "AIcarusForQQ/qq-file", "Accept-Encoding": "identity"},
        timeout=timeout,
        follow_redirects=False,
        verify=True,
    ) as http:
        for attempt in range(2):
            url = await _resolve_llonebot_url(
                client,
                file_id,
                conversation_type=conversation_type,
                conversation_id=conversation_id,
            )
            response = None
            try:
                response = await _send_with_safe_redirects(http, url)
                if response.status_code in _REFRESHABLE_STATUSES and attempt == 0:
                    await response.aclose()
                    response = None
                    continue
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    message = (
                        "QQ 文件下载地址已失效"
                        if response.status_code in _REFRESHABLE_STATUSES
                        else "QQ 文件传输连接失败"
                    )
                    raise QQFileStreamError("transport_error", message, retryable=True) from exc
                size = await _write_http_response(
                    response,
                    destination,
                    declared_size=declared_size,
                    max_bytes=max_bytes,
                    on_progress=on_progress,
                )
                return {"file_name": "", "size_bytes": size}
            except QQFileStreamError:
                raise
            except (httpx.HTTPError, OSError) as exc:
                raise QQFileStreamError(
                    "transport_error",
                    "QQ 文件传输连接失败",
                    retryable=True,
                ) from exc
            finally:
                if response is not None:
                    await response.aclose()
    raise QQFileStreamError("transport_error", "QQ 文件下载地址已失效", retryable=True)
