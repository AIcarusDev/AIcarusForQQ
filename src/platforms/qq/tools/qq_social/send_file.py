"""Send an existing Agent-home file or generated UTF-8 text to QQ."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, cast

from pydantic import Field, field_validator, model_validator

from platforms.qq.files.logical import normalized_filename, sanitize_filename
from platforms.qq.files.service import QQFileError, get_qq_file_service
from platforms.qq.session_context import NO_CURRENT_SESSION_ERROR, ensure_session_provider
from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, ToolContract
from workspace.config import workspace_enabled


logger = logging.getLogger("AICQ.qq.send_file")
MAX_INLINE_TEXT_BYTES = 1024 * 1024
SENT_EVENT_OBSERVATION_SECONDS = 3.0
_ALLOWED_TEXT_FORMATS = frozenset(
    {
        "ass", "bat", "c", "cfg", "cmd", "cpp", "css", "csv", "go", "h",
        "hpp", "htm", "html", "ics", "ini", "java", "js", "json", "jsonl",
        "jsx", "kt", "kts", "log", "markdown", "md", "php", "ps1", "py",
        "rb", "rs", "rst", "sh", "sql", "srt", "svg", "swift", "tex", "toml",
        "ts", "tsv", "tsx", "txt", "vcf", "xml", "yaml", "yml",
    }
)
_RECONCILE_TASKS: set[asyncio.Task[Any]] = set()


class FileTransferConfigError(ValueError):
    """A controlled configuration error whose message is safe for the model."""


class SendFileArgs(ToolArgsModel):
    path: str | None = Field(
        default=None,
        min_length=1,
        description="已有 Linux 文件路径，必须位于 /home/agent 内。",
    )
    content: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_INLINE_TEXT_BYTES,
        description="要现场生成并发送的 UTF-8 文本内容。",
    )
    filename: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="content 模式的文件名；可省略与 format 相同的后缀。",
    )
    format: str | None = Field(
        default=None,
        min_length=1,
        max_length=16,
        description="content 模式的纯文本格式/扩展名，例如 txt、md、json、csv、html、yaml 或 py；不执行格式转换。",
    )

    @field_validator("format", mode="before")
    @classmethod
    def normalize_format(cls, value: Any) -> Any:
        if value is None:
            return None
        return str(value).strip().lstrip(".").casefold()

    @model_validator(mode="after")
    def validate_source(self):
        has_path = self.path is not None
        has_content = self.content is not None
        if has_path == has_content:
            raise ValueError("path 与 content 必须且只能提供一个")
        if has_path:
            if self.filename is not None or self.format is not None:
                raise ValueError("path 模式不能提供 filename 或 format")
            return self
        if self.filename is None or self.format is None:
            raise ValueError("content 模式必须同时提供 filename 与 format")
        if self.format not in _ALLOWED_TEXT_FORMATS:
            raise ValueError("format 必须是受支持的纯文本格式")
        return self


TOOL_CONTRACT = ToolContract(
    name="send_file",
    description="发送qq文件，可用 `path` 选择已有的 Linux 文件，或用 `content` 直接生成 UTF-8 文本文件发送到当前 QQ 会话中，两者二选一；若 Linux 存在，生成文件将同时保存为该会话的本地 QQ 文件。",
    args_model=SendFileArgs,
)

EXTERNALLY_PERCEPTIBLE: bool = True
TOOL_EFFECT: dict[str, str] = {"surface": "qq", "kind": "session_write"}
REQUIRES_CONTEXT: list[str] = [
    "qq_client",
    "qq_session_provider",
    "workspace_service",
    "main_loop",
    "config",
]


def _adapter_error(action: str, response: dict[str, Any] | None) -> str:
    """Return only stable adapter metadata that is safe to expose to the model."""

    if not response:
        return f"QQ adapter 未响应: {action}"
    raw_status = str(response.get("status") or "failed")
    status = raw_status if raw_status in {"failed", "error", "timeout"} else "failed"
    parts = [action, status]
    raw_retcode = response.get("retcode")
    if isinstance(raw_retcode, int) or (
        isinstance(raw_retcode, str)
        and raw_retcode.removeprefix("-").isdigit()
        and len(raw_retcode) <= 12
    ):
        parts.append(f"retcode={raw_retcode}")
    return "QQ adapter 返回错误: " + " / ".join(parts)


def _file_transfer_config(qq_client: Any) -> tuple[str, str]:
    raw = getattr(qq_client, "file_transfer", None)
    config = raw if isinstance(raw, dict) else {}
    host_directory = str(config.get("host_directory") or "").strip()
    adapter_directory = str(config.get("adapter_directory") or "").strip()
    if bool(host_directory) != bool(adapter_directory):
        raise FileTransferConfigError("QQ 文件共享目录配置不完整：宿主目录与 Adapter 目录必须同时填写")
    return host_directory, adapter_directory


def _adapter_file_path(prepared: Any, host_directory: str, adapter_directory: str) -> str:
    if not host_directory:
        return str(prepared.host_path)
    host_root = Path(host_directory).expanduser().resolve()
    host_path = Path(prepared.host_path).resolve()
    try:
        relative = host_path.relative_to(host_root)
    except ValueError as exc:
        raise ValueError("暂存文件不在配置的 QQ 文件共享目录内") from exc
    return str(PurePosixPath(adapter_directory).joinpath(*relative.parts))


def _generated_filename(filename: str, file_format: str) -> str:
    raw_name = str(filename or "").strip()
    extension = str(file_format or "").strip().lstrip(".").casefold()
    if not raw_name.casefold().endswith(f".{extension}"):
        raw_name = f"{raw_name}.{extension}"
    return sanitize_filename(raw_name)


def _event_file_data(event: dict[str, Any]) -> dict[str, Any] | None:
    message = event.get("message")
    if not isinstance(message, list):
        return None
    files = [
        segment.get("data") or {}
        for segment in message
        if isinstance(segment, dict) and segment.get("type") == "file"
    ]
    if len(files) != 1 or not isinstance(files[0], dict):
        return None
    return files[0]


def _file_data_name(data: dict[str, Any]) -> str:
    for key in ("name", "file_name", "filename", "file"):
        value = str(data.get(key) or "").strip()
        if value:
            return PurePosixPath(value.replace("\\", "/")).name
    return ""


def _file_data_size(data: dict[str, Any]) -> int | None:
    for key in ("file_size", "size"):
        value = data.get(key)
        if value in (None, ""):
            continue
        try:
            result = int(value)
        except (TypeError, ValueError):
            continue
        if result >= 0:
            return result
    return None


def _event_matches_generated_file(
    event: dict[str, Any],
    *,
    conv_type: str,
    conv_id: str,
    bot_id: str,
    filename: str,
    size_bytes: int,
    sent_started_at: float,
) -> bool:
    message_id = str(event.get("message_id") or "").strip()
    if not message_id:
        return False
    message_type = str(event.get("message_type") or "").strip()
    if message_type and message_type != conv_type:
        return False
    if conv_type == "group":
        if str(event.get("group_id") or "") != conv_id:
            return False
    else:
        peer_ids = {
            str(event.get("user_id") or ""),
            str(event.get("target_id") or ""),
            str((event.get("sender") or {}).get("user_id") or ""),
        }
        peer_ids.discard("")
        peer_ids.discard(bot_id)
        if peer_ids and conv_id not in peer_ids:
            return False
    sender_id = str((event.get("sender") or {}).get("user_id") or "").strip()
    if sender_id and bot_id and sender_id != bot_id:
        return False
    try:
        event_time = float(event.get("time") or 0)
    except (TypeError, ValueError):
        event_time = 0.0
    if event_time and event_time < sent_started_at - 5.0:
        return False
    data = _event_file_data(event)
    if data is None or normalized_filename(_file_data_name(data)) != normalized_filename(filename):
        return False
    event_size = _file_data_size(data)
    return event_size is None or event_size == size_bytes


async def _fetch_recent_history(
    qq_client: Any,
    *,
    conv_type: str,
    target_id: int,
) -> list[dict[str, Any]]:
    if not getattr(qq_client, "connected", False):
        return []
    action = "get_group_msg_history" if conv_type == "group" else "get_friend_msg_history"
    key = "group_id" if conv_type == "group" else "user_id"
    response = await qq_client.send_api_raw(
        action,
        {key: target_id, "count": 50},
        timeout=20.0,
    )
    if not isinstance(response, dict) or response.get("status") != "ok":
        return []
    messages = (response.get("data") or {}).get("messages") or []
    return messages if isinstance(messages, list) else []


async def _history_match(
    qq_client: Any,
    *,
    conv_type: str,
    conv_id: str,
    target_id: int,
    bot_id: str,
    filename: str,
    size_bytes: int,
    sent_started_at: float,
) -> dict[str, Any] | None:
    messages = await _fetch_recent_history(
        qq_client,
        conv_type=conv_type,
        target_id=target_id,
    )
    matches: dict[str, dict[str, Any]] = {}
    for event in messages:
        if not isinstance(event, dict):
            continue
        if _event_matches_generated_file(
            event,
            conv_type=conv_type,
            conv_id=conv_id,
            bot_id=bot_id,
            filename=filename,
            size_bytes=size_bytes,
            sent_started_at=sent_started_at,
        ):
            matches[str(event["message_id"])] = event
    if len(matches) == 1:
        return next(iter(matches.values()))
    return None


def _event_file_id(event: dict[str, Any] | None) -> str:
    data = _event_file_data(event or {})
    if data is None:
        return ""
    return str(data.get("file_id") or data.get("id") or "").strip()


async def _persist_sent_file_message(
    *,
    service: Any,
    record_id: str,
    session: Any,
    qq_client: Any,
    event: dict[str, Any],
    filename: str,
    size_bytes: int,
    local_path: str,
    fallback_file_id: str,
) -> str:
    import app_state
    from database import save_chat_message
    from llm.core.round_context import get_current_inner_state
    from web.debug_server import broadcast_chat_event

    message_id = str(event.get("message_id") or "").strip()
    if not message_id:
        return ""
    file_id = _event_file_id(event) or fallback_file_id
    await service.attach_generated_delivery(
        record_id,
        message_id=message_id,
        file_id=file_id,
    )
    try:
        event_time = float(event.get("time") or 0)
        if event_time <= 0:
            raise ValueError("missing event time")
        timestamp = datetime.fromtimestamp(event_time, app_state.TIMEZONE).isoformat()
    except (TypeError, ValueError, OSError):
        timestamp = datetime.now(app_state.TIMEZONE).isoformat()
    bot_id = str(getattr(qq_client, "bot_id", None) or getattr(session, "_qq_id", "") or "")
    bot_name = str(
        getattr(session, "_qq_card", "")
        or getattr(session, "_qq_name", "")
        or app_state.SELF_NAME
        or bot_id
    )
    file_segment: dict[str, Any] = {
        "type": "file",
        "filename": filename,
        "size_bytes": size_bytes,
        "is_downloaded": True,
        "local_path": local_path,
    }
    if file_id:
        file_segment["file_id"] = file_id
    entry = {
        "role": "bot",
        "agent_qq": bot_id,
        "message_id": message_id,
        "sender_id": bot_id,
        "sender_name": bot_name,
        "sender_role": "",
        "timestamp": timestamp,
        "content": f"[文件:{filename}]",
        "content_type": "file",
        "content_segments": [file_segment],
    }
    await save_chat_message(str(getattr(session, "key", "") or ""), entry)
    if not any(
        str(item.get("message_id") or "") == message_id
        for item in getattr(session, "context_messages", [])
        if isinstance(item, dict)
    ):
        session.add_to_context(entry)
    await broadcast_chat_event(
        {
            "type": "bot_turn",
            "conv_id": str(getattr(session, "key", "") or ""),
            "conv_name": getattr(session, "conv_name", "") or str(getattr(session, "key", "") or ""),
            "conv_type": str(getattr(session, "conv_type", "") or "unknown"),
            "entries": [entry],
            "inner_state": get_current_inner_state(),
        }
    )
    return message_id


def _cancel_waiter(qq_client: Any, token: str | None) -> None:
    if not token:
        return
    cancel = getattr(qq_client, "cancel_sent_event_waiter", None)
    if callable(cancel):
        cancel(token)


async def _background_reconcile(
    *,
    qq_client: Any,
    service: Any,
    record_id: str,
    session: Any,
    token: str | None,
    future: asyncio.Future | None,
    conv_type: str,
    conv_id: str,
    target_id: int,
    bot_id: str,
    filename: str,
    size_bytes: int,
    local_path: str,
    file_id: str,
    sent_started_at: float,
) -> None:
    try:
        for _attempt in range(8):
            event = None
            if future is not None and not future.done():
                try:
                    event = await asyncio.wait_for(asyncio.shield(future), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
            elif future is not None and not future.cancelled():
                try:
                    event = future.result()
                except Exception:
                    event = None
            if not isinstance(event, dict):
                event = await _history_match(
                    qq_client,
                    conv_type=conv_type,
                    conv_id=conv_id,
                    target_id=target_id,
                    bot_id=bot_id,
                    filename=filename,
                    size_bytes=size_bytes,
                    sent_started_at=sent_started_at,
                )
            if isinstance(event, dict):
                message_id = await _persist_sent_file_message(
                    service=service,
                    record_id=record_id,
                    session=session,
                    qq_client=qq_client,
                    event=event,
                    filename=filename,
                    size_bytes=size_bytes,
                    local_path=local_path,
                    fallback_file_id=file_id,
                )
                if message_id:
                    logger.info("已回填 Agent 生成文件 message_id=%s path=%s", message_id, local_path)
                    return
            await asyncio.sleep(1.0)
        logger.warning("Agent 生成文件已发送，但暂未取得真实 message_id: %s", local_path)
    except Exception:
        logger.exception("Agent 生成文件 message_id 后台回填失败: %s", local_path)
    finally:
        _cancel_waiter(qq_client, token)


def make_handler(
    qq_client: Any,
    qq_session_provider: Callable[[], Any | None],
    workspace_service: Any,
    main_loop: asyncio.AbstractEventLoop,
    config: dict | None = None,
) -> Callable:
    qq_session_provider = ensure_session_provider(qq_session_provider)

    def execute(
        path: str | None = None,
        content: str | None = None,
        filename: str | None = None,
        format: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not qq_client or not qq_client.connected:
            return {"error": "QQ adapter 未连接"}
        if main_loop is None or not main_loop.is_running():
            return {"error": "主事件循环不可用"}

        session = qq_session_provider()
        if session is None:
            return {"error": NO_CURRENT_SESSION_ERROR}

        conv_type = str(getattr(session, "conv_type", "") or "")
        conv_id = str(getattr(session, "conv_id", "") or "")
        if conv_type == "group":
            action = "upload_group_file"
            target_key = "group_id"
        elif conv_type == "private":
            action = "upload_private_file"
            target_key = "user_id"
        elif conv_type == "temp":
            return {"error": "QQ 临时会话不支持独立文件上传，未发送文件。"}
        else:
            return {"error": f"当前会话类型不支持发送 QQ 文件: {conv_type or 'unknown'}"}
        try:
            target_id = int(conv_id)
        except (TypeError, ValueError):
            return {"error": f"会话 ID 无效: {conv_id}"}

        if (path is not None and content is not None) or (path is None and content is None):
            return {"error": "path 与 content 必须且只能提供一个"}

        if path is not None:
            if config is not None and not workspace_enabled(config):
                return {"error": "Linux 电脑未启用，不能发送 path 指定的已有文件。"}

            async def _upload_path() -> tuple[Any, dict[str, Any] | None]:
                host_directory, adapter_directory = _file_transfer_config(qq_client)
                async with workspace_service.stage_host_file(
                    path,
                    staging_root=host_directory or None,
                ) as prepared:
                    file_value = _adapter_file_path(prepared, host_directory, adapter_directory)
                    response = await qq_client.send_api_raw(
                        action,
                        {
                            target_key: target_id,
                            "file": file_value,
                            "name": prepared.name,
                        },
                        timeout=None,
                    )
                return prepared, response

            try:
                prepared, response = run_coroutine_sync(_upload_path(), main_loop, timeout=None)
            except FileTransferConfigError as exc:
                return {"error": str(exc)}
            except Exception:
                logger.exception("已有 QQ 文件发送失败")
                return {"error": "文件发送失败，请检查 QQ 文件传输配置或 Adapter 状态。"}
            if not response or response.get("status") != "ok":
                return {"error": _adapter_error(action, response)}
            return {
                "success": True,
                "path": prepared.workspace_path,
                "name": prepared.name,
                "size": prepared.size,
                "target": f"{conv_type}_{conv_id}",
            }

        if filename is None or format is None:
            return {"error": "content 模式必须同时提供 filename 与 format"}
        normalized_format = str(format).strip().lstrip(".").casefold()
        if normalized_format not in _ALLOWED_TEXT_FORMATS:
            return {"error": "format 必须是受支持的纯文本格式"}
        payload = str(content).encode("utf-8")
        if len(payload) > MAX_INLINE_TEXT_BYTES:
            return {
                "error": "现场生成的文本文件超过 1 MiB 上限",
                "size": len(payload),
                "limit": MAX_INLINE_TEXT_BYTES,
            }
        generated_name = _generated_filename(filename, normalized_format)

        async def _generate_and_upload() -> dict[str, Any]:
            service = get_qq_file_service(qq_client, workspace_service)
            stored = await service.store_generated_text(
                content=payload,
                filename=generated_name,
                session=session,
            )
            bot_id = str(getattr(qq_client, "bot_id", None) or stored["agent_qq"])
            sent_started_at = time.time()

            def matcher(event: dict[str, Any]) -> bool:
                return _event_matches_generated_file(
                    event,
                    conv_type=conv_type,
                    conv_id=conv_id,
                    bot_id=bot_id,
                    filename=stored["name"],
                    size_bytes=stored["size_bytes"],
                    sent_started_at=sent_started_at,
                )

            register = getattr(qq_client, "register_sent_event_waiter", None)
            token: str | None = None
            future: asyncio.Future | None = None
            if callable(register):
                token, future = cast(
                    tuple[str | None, asyncio.Future | None],
                    register(matcher),
                )
            try:
                file_value = "base64://" + base64.b64encode(payload).decode("ascii")
                response = await qq_client.send_api_raw(
                    action,
                    {
                        target_key: target_id,
                        "file": file_value,
                        "name": stored["name"],
                    },
                    timeout=None,
                )
            except Exception:
                _cancel_waiter(qq_client, token)
                logger.exception("生成的 QQ 文件上传失败: record_id=%s", stored["record_id"])
                return {
                    "error": "文件已保存到会话目录，但 QQ 上传失败，请稍后重试。",
                    "stored": True,
                    **stored,
                    "target": f"{conv_type}_{conv_id}",
                }
            if not response or response.get("status") != "ok":
                _cancel_waiter(qq_client, token)
                return {
                    "error": _adapter_error(action, response),
                    "stored": True,
                    **stored,
                    "target": f"{conv_type}_{conv_id}",
                }

            response_data = response.get("data") or {}
            file_id = str(response_data.get("file_id") or "").strip() if isinstance(response_data, dict) else ""
            await service.attach_generated_delivery(stored["record_id"], file_id=file_id)
            event: dict[str, Any] | None = None
            if future is not None:
                try:
                    candidate = await asyncio.wait_for(
                        asyncio.shield(future),
                        timeout=SENT_EVENT_OBSERVATION_SECONDS,
                    )
                    event = candidate if isinstance(candidate, dict) else None
                except asyncio.TimeoutError:
                    pass
            if event is None:
                event = await _history_match(
                    qq_client,
                    conv_type=conv_type,
                    conv_id=conv_id,
                    target_id=target_id,
                    bot_id=bot_id,
                    filename=stored["name"],
                    size_bytes=stored["size_bytes"],
                    sent_started_at=sent_started_at,
                )

            message_id = ""
            if event is not None:
                _cancel_waiter(qq_client, token)
                message_id = await _persist_sent_file_message(
                    service=service,
                    record_id=stored["record_id"],
                    session=session,
                    qq_client=qq_client,
                    event=event,
                    filename=stored["name"],
                    size_bytes=stored["size_bytes"],
                    local_path=stored["local_path"],
                    fallback_file_id=file_id,
                )
            else:
                task = asyncio.create_task(
                    _background_reconcile(
                        qq_client=qq_client,
                        service=service,
                        record_id=stored["record_id"],
                        session=session,
                        token=token,
                        future=future,
                        conv_type=conv_type,
                        conv_id=conv_id,
                        target_id=target_id,
                        bot_id=bot_id,
                        filename=stored["name"],
                        size_bytes=stored["size_bytes"],
                        local_path=stored["local_path"],
                        file_id=file_id,
                        sent_started_at=sent_started_at,
                    ),
                    name=f"qq-send-file-reconcile-{stored['record_id']}",
                )
                _RECONCILE_TASKS.add(task)
                task.add_done_callback(_RECONCILE_TASKS.discard)

            result = {
                "success": True,
                "path": stored["local_path"],
                "name": stored["name"],
                "size": stored["size_bytes"],
                "target": f"{conv_type}_{conv_id}",
                "source": "generated_text",
                "storage_backend": stored["storage_backend"],
                "record_id": stored["record_id"],
                "message_id_pending": not bool(message_id),
            }
            if file_id:
                result["file_id"] = file_id
            if message_id:
                result["message_id"] = message_id
            return result

        try:
            return run_coroutine_sync(_generate_and_upload(), main_loop, timeout=None)
        except QQFileError as exc:
            return {
                "error": str(exc),
                "code": exc.code,
                "retryable": exc.retryable,
                "details": exc.details,
            }
        except Exception:
            logger.exception("QQ 文件生成或发送失败")
            return {"error": "文件生成或发送失败，请稍后重试。"}

    return execute
