"""Download QQ refs or public URLs into the managed attachment cache."""

from __future__ import annotations

import asyncio
import base64
import ntpath
import stat
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from attachments.service import (
    MAX_ATTACHMENT_BYTES,
    OBSERVATION_SECONDS,
    public_source_label,
    safe_filename,
)
from platforms.focus import current_focus_key
from tools.contract import ToolArgsModel, ToolContract

from ._common import acknowledge, run_on_main_loop


class DownloadArgs(ToolArgsModel):
    action: Literal["start", "poll", "stop"] = Field(default="start", description="启动、查询或停止下载。")
    ref: str | None = Field(default=None, description="当前 QQ 上下文中附件、图片、语音或视频的 ref。")
    url: str | None = Field(default=None, description="公开的 HTTP(S) 下载地址。")
    task_id: str | None = Field(default=None, description="查询或停止时使用的任务 ID。")

    @model_validator(mode="after")
    def validate_action(self):
        if self.action == "start":
            if (self.ref is None) == (self.url is None) or self.task_id is not None:
                raise ValueError("start 必须且只能提供 ref 或 url")
        elif not self.task_id or self.ref is not None or self.url is not None:
            raise ValueError("poll/stop 必须且只能提供 task_id")
        return self


TOOL_CONTRACT = ToolContract(
    name="download",
    description=("把当前或历史 QQ 消息中的 ref、或公开 URL 下载到只读附件缓存。"
                 "QQ ref 在进程重启后会尝试按原消息 ID 重新解析。"
                 "最多观察 15 秒，未完成会返回 running 和 task_id，可继续 poll 或 stop。"
                 "图片复用现有图片缓存；单个附件上限 100 MiB。"),
    args_model=DownloadArgs,
)
REQUIRES_CONTEXT = ["attachment_service", "runtime_event_hub", "main_loop", "qq_session_provider"]


def _decode_base64_attachment(value: object) -> bytes:
    encoded = str(value)
    if encoded.startswith("base64://"):
        encoded = encoded[9:]
    # Reject before decoding so an adapter cannot force a much larger temporary allocation.
    max_encoded_chars = ((MAX_ATTACHMENT_BYTES + 2) // 3) * 4
    if len(encoded) > max_encoded_chars:
        raise ValueError("attachment exceeds the 100 MiB limit")
    decoded = base64.b64decode(encoded, validate=True)
    if len(decoded) > MAX_ATTACHMENT_BYTES:
        raise ValueError("attachment exceeds the 100 MiB limit")
    return decoded


def _allowed_adapter_host_path(client, value: str) -> str:
    config = getattr(client, "file_transfer", None)
    config = config if isinstance(config, dict) else {}
    host_directory = str(config.get("host_directory") or "").strip()
    adapter_directory = str(config.get("adapter_directory") or "").strip()
    if not host_directory:
        raise ValueError("adapter local file paths require an explicit host_directory allowlist")

    host_root = Path(host_directory).expanduser().resolve(strict=True)
    raw = str(value or "").strip()
    candidate: Path
    if adapter_directory:
        adapter_root = PurePosixPath(adapter_directory.replace("\\", "/"))
        adapter_path = PurePosixPath(raw.replace("\\", "/"))
        try:
            relative = adapter_path.relative_to(adapter_root)
        except ValueError:
            candidate = Path(raw)
        else:
            candidate = host_root.joinpath(*relative.parts)
    else:
        candidate = Path(raw)
    if not candidate.is_absolute() and not ntpath.isabs(raw):
        raise ValueError("adapter local file path is not absolute")
    resolved = candidate.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(host_root)
    except ValueError as exc:
        raise ValueError("adapter local file path is outside the configured allowlist") from exc
    for path in (resolved, *resolved.parents):
        info = path.lstat()
        if path.is_symlink() or (
            getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            and getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError("adapter local file path crosses a link or reparse point")
        if path == host_root:
            break
    return str(resolved)


def _filename_from_source(source: dict, fallback: str) -> str:
    return safe_filename(
        source.get("name")
        or source.get("file_name")
        or source.get("filename")
        or source.get("path")
        or source.get("file"),
        fallback,
    )


async def _resolve_document_url(client, file_id: str, qq_context: dict) -> str:
    """Resolve ordinary QQ files through the adapter's document-file API."""
    conv_type = str(qq_context.get("conv_type") or "")
    conv_id = str(qq_context.get("conv_id") or "")
    attempts: list[tuple[str, dict]] = []
    if conv_type == "group" and conv_id:
        attempts.append(("get_group_file_url", {"group_id": conv_id, "file_id": file_id}))
        attempts.append(("get_group_file_download_url", {"group_id": conv_id, "file_id": file_id}))
    elif conv_type in {"private", "temp"}:
        attempts.append(("get_private_file_url", {"file_id": file_id}))
        if conv_id:
            # Milky-style fallback used by some adapters. LLBot accepts an empty
            # file_hash here and derives the URL from file_id internally.
            attempts.append((
                "get_private_file_download_url",
                {"user_id": conv_id, "file_id": file_id, "file_hash": ""},
            ))
    for action, params in attempts:
        response = await client.send_api(action, params, timeout=15.0)
        if not isinstance(response, dict):
            continue
        url = response.get("url") or response.get("download_url")
        if url:
            return str(url)
    return ""


async def _resolve_adapter_file(
    client,
    source: dict,
    filename: str,
    *,
    media_type: str = "",
    qq_context: dict | None = None,
) -> dict:
    """Resolve a raw OneBot media segment to bytes, URL, or a host path."""
    if source.get("base64"):
        return {"data": _decode_base64_attachment(source["base64"]), "filename": filename}
    if source.get("url"):
        return {"url": str(source["url"]), "filename": filename}
    raw_file = str(source.get("file") or "")
    if raw_file.startswith(("http://", "https://")):
        return {"url": raw_file, "filename": filename}
    if raw_file.startswith("base64://"):
        return {"data": _decode_base64_attachment(raw_file), "filename": filename}
    path = str(source.get("path") or "")
    if path or raw_file.startswith("/") or ntpath.isabs(raw_file):
        if client is None:
            raise ValueError("QQ adapter 当前不可用")
        return {
            "host_path": _allowed_adapter_host_path(client, path or raw_file),
            "filename": filename,
        }

    file_id = str(source.get("file_id") or raw_file or "")
    if not file_id:
        raise ValueError("QQ adapter 没有提供此 ref 的可下载地址")
    if client is None:
        raise ValueError("QQ adapter 当前不可用")
    if media_type == "file" and qq_context:
        document_url = await _resolve_document_url(client, file_id, qq_context)
        if document_url:
            return {"url": document_url, "filename": filename}
    response = await client.send_api("get_file", {"file_id": file_id}, timeout=15.0)
    if not isinstance(response, dict):
        response = await client.send_api("get_file", {"file": file_id}, timeout=15.0)
    if not isinstance(response, dict):
        raise ValueError("QQ adapter 无法解析 file_id")
    resolved_name = _filename_from_source(response, filename)
    if response.get("base64"):
        return {
            "data": _decode_base64_attachment(response["base64"]),
            "filename": resolved_name,
        }
    if response.get("url"):
        return {"url": str(response["url"]), "filename": resolved_name}
    resolved_path = response.get("path") or response.get("file")
    if resolved_path:
        return {
            "host_path": _allowed_adapter_host_path(client, str(resolved_path)),
            "filename": resolved_name,
        }
    raise ValueError("QQ adapter get_file 响应中没有可用文件地址")


def _qq_client():
    try:
        from platforms.registry import get_platform

        return getattr(get_platform("qq"), "client", None)
    except Exception:
        return None


def _history_ref_resolver(
    entry: dict,
    ref: str,
    qq_context: dict,
) -> Callable[[], Awaitable[dict]] | None:
    """Re-resolve a persisted ref through get_msg after a process restart."""
    content_segments = entry.get("content_segments") or []
    wanted: dict | None = None
    wanted_raw_type = ""
    wanted_ordinal = 0
    type_map = {"file": "file", "voice": "record", "video": "video"}
    wanted_index = -1
    for index, segment in enumerate(content_segments):
        raw_type = type_map.get(str(segment.get("type") or ""), "")
        if not raw_type:
            continue
        if str(segment.get("ref") or "") == ref:
            wanted = segment
            wanted_raw_type = raw_type
            wanted_index = index
            break
    if wanted is None:
        return None
    wanted_ordinal = sum(
        1
        for segment in content_segments[:wanted_index]
        if type_map.get(str(segment.get("type") or ""), "") == wanted_raw_type
    )
    message_id = str(entry.get("message_id") or "").strip()
    if not message_id:
        return None
    fallback = safe_filename(
        wanted.get("filename") or wanted.get("label"), f"qq_attachment_{ref}.bin"
    )

    def verify_message_scope(message: dict) -> None:
        returned_id = str(message.get("message_id") or "").strip()
        if returned_id and returned_id != message_id:
            raise ValueError("QQ adapter 返回了不匹配的附件消息")
        conv_type = str(qq_context.get("conv_type") or "")
        conv_id = str(qq_context.get("conv_id") or "")
        message_type = str(message.get("message_type") or "").casefold()
        if conv_type == "group":
            returned_group = str(message.get("group_id") or "").strip()
            if message_type and message_type != "group":
                raise ValueError("QQ adapter 返回了其他会话的附件消息")
            if returned_group and returned_group != conv_id:
                raise ValueError("QQ adapter 返回了其他群的附件消息")
        elif conv_type in {"private", "temp"}:
            if message_type == "group":
                raise ValueError("QQ adapter 返回了其他会话的附件消息")
            sender = message.get("sender") if isinstance(message.get("sender"), dict) else {}
            peer_ids = {
                str(value).strip()
                for value in (
                    message.get("user_id"), message.get("target_id"), sender.get("user_id")
                )
                if value not in (None, "")
            }
            if peer_ids and conv_id and conv_id not in peer_ids:
                raise ValueError("QQ adapter 返回了其他私聊的附件消息")

    async def resolve_from_message() -> dict:
        client = _qq_client()
        if client is None:
            raise ValueError("QQ adapter 当前不可用")
        try:
            adapter_message_id: int | str = int(message_id)
        except ValueError:
            adapter_message_id = message_id
        message = await client.send_api(
            "get_msg", {"message_id": adapter_message_id}, timeout=15.0
        )
        if not isinstance(message, dict):
            raise ValueError("QQ adapter 无法重新获取该附件消息，消息可能已过期")
        verify_message_scope(message)
        candidates = [
            seg for seg in (message.get("message") or [])
            if isinstance(seg, dict) and seg.get("type") == wanted_raw_type
        ]
        if wanted_ordinal >= len(candidates):
            raise ValueError("重新获取的 QQ 消息中没有对应附件")
        data = candidates[wanted_ordinal].get("data") or {}
        if not isinstance(data, dict):
            raise ValueError("QQ adapter 返回的附件信息格式无效")
        return await _resolve_adapter_file(
            client,
            data,
            _filename_from_source(data, fallback),
            media_type=wanted_raw_type,
            qq_context=qq_context,
        )

    return resolve_from_message


def _ref_source(session, ref: str) -> dict:
    if session is None:
        raise ValueError("ref 只能在当前 QQ 会话中解析")
    qq_context = {
        "conv_type": str(getattr(session, "conv_type", "") or ""),
        "conv_id": str(getattr(session, "conv_id", "") or ""),
    }
    for entry in reversed(getattr(session, "context_messages", []) or []):
        images = entry.get("images") or {}
        if ref in images:
            image = images[ref] or {}
            if image.get("pending") or image.get("failed") or image.get("expired"):
                raise ValueError("ref 的图片尚未加载或已经失效")
            return {
                "image_ref": ref,
                "filename": f"qq_image_{ref}",
                "mime": str(image.get("mime") or "image/jpeg"),
                "size": 0,
            }
        source = (entry.get("_attachment_sources") or {}).get(ref)
        if not source:
            resolver = _history_ref_resolver(entry, ref, qq_context)
            if resolver is not None:
                return {"resolver": resolver}
            continue
        filename = _filename_from_source(source, f"qq_attachment_{ref}.bin")

        async def resolve_from_adapter():
            client = _qq_client()
            return await _resolve_adapter_file(
                client,
                source,
                filename,
                media_type=str(source.get("_segment_type") or ""),
                qq_context=qq_context,
            )

        return {"resolver": resolve_from_adapter}
    raise ValueError("当前 QQ 上下文中找不到这个 ref")


def make_handler(attachment_service, runtime_event_hub, main_loop, qq_session_provider):
    async def execute_async(**kwargs):
        action = kwargs.get("action", "start")
        app_state = __import__("app_state")
        target = current_focus_key(getattr(app_state, "current_focus", None)) or ""
        if action == "poll":
            result = await attachment_service.poll(kwargs["task_id"])
        elif action == "stop":
            result = await attachment_service.stop(kwargs["task_id"])
        else:
            if kwargs.get("ref"):
                ref = str(kwargs["ref"])
                session = qq_session_provider()

                async def resolve_ref():
                    resolved = _ref_source(session, ref)
                    nested = resolved.pop("resolver", None)
                    return await nested() if nested is not None else resolved

                started = await attachment_service.start(
                    source_type="ref", source=ref, resolver=resolve_ref
                )
            else:
                url = str(kwargs["url"])
                started = await attachment_service.start(
                    source_type="url", source=public_source_label(url), url=url
                )
            terminal_task = asyncio.create_task(
                attachment_service.wait(started.task_id, timeout=OBSERVATION_SECONDS)
            )
            tasks = {terminal_task}
            if runtime_event_hub is not None:
                tasks.add(asyncio.create_task(runtime_event_hub.wait(
                    timeout=OBSERVATION_SECONDS, target=target, event_types={"attention"},
                    consume=False,
                )))
            _, pending = await asyncio.wait(
                tasks, timeout=OBSERVATION_SECONDS, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            result = await attachment_service.poll(started.task_id)
        if result.terminal:
            attachment_service.mark_delivered(result.task_id)
            await acknowledge(runtime_event_hub, result.task_id)
        return result.to_payload()

    def handler(**kwargs):
        return run_on_main_loop(execute_async(**kwargs), main_loop)

    return handler
