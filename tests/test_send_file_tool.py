from __future__ import annotations

import asyncio
import base64
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from platforms.qq.tools.qq_social import send_file
from tools import build_tools
from tools.namespaces import NamespaceRuntimeState, load_namespace_registry
class _LoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2)
        self.loop.close()


class _WorkspaceService:
    def __init__(self, content: bytes = b"test report") -> None:
        self.staged = []
        self.released = []
        self.content = content
        self.reported_size = len(content)
        self.staging_roots = []

    @asynccontextmanager
    async def stage_host_file(self, path: str, *, staging_root=None):
        assert path == "/home/agent/report.pdf"
        self.staging_roots.append(staging_root)
        if staging_root:
            Path(staging_root).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="aicq-send-file-test-",
            dir=staging_root,
        ) as directory:
            host_path = Path(directory) / "payload.bin"
            host_path.write_bytes(self.content)
            prepared = SimpleNamespace(
                workspace_path=path,
                host_path=str(host_path),
                name="report.pdf",
                size=self.reported_size,
            )
            self.staged.append(prepared.host_path)
            try:
                yield prepared
            finally:
                self.released.append(prepared.host_path)


class _QQClient:
    connected = True

    def __init__(self, response=None, *, adapter="napcat", file_transfer=None) -> None:
        self.response = response or {"status": "ok", "data": None}
        self.adapter = adapter
        self.file_transfer = file_transfer or {}
        self.calls = []

    async def send_api_raw(self, action, params, timeout=15.0):
        self.calls.append((action, params, timeout))
        if isinstance(self.response, list):
            return self.response.pop(0)
        return self.response


class _GeneratedFileService:
    def __init__(self) -> None:
        self.stored = []
        self.delivery_updates = []

    async def store_generated_text(self, *, content, filename, session):
        self.stored.append((content, filename, session))
        return {
            "record_id": "generated-record",
            "agent_qq": "4242",
            "session_key": session.key,
            "conversation": {"type": session.conv_type, "id": session.conv_id},
            "name": filename,
            "local_path": f"/home/agent/qq/4242/file/{session.conv_type}_{session.conv_id}/{filename}",
            "size_bytes": len(content),
            "storage_backend": "linux",
            "origin": "agent_generated",
            "created_at": "2026-09-01T10:00:00+08:00",
        }

    async def attach_generated_delivery(self, record_id, *, message_id=None, file_id=None):
        self.delivery_updates.append(
            {"record_id": record_id, "message_id": message_id, "file_id": file_id}
        )
        return self.delivery_updates[-1]


class _SentEventQQClient(_QQClient):
    bot_id = "4242"

    def __init__(self) -> None:
        super().__init__()
        self.waiters = {}
        self.uploaded_bytes = b""

    def register_sent_event_waiter(self, matcher):
        token = f"waiter-{len(self.waiters) + 1}"
        future = asyncio.get_running_loop().create_future()
        self.waiters[token] = (matcher, future)
        return token, future

    def cancel_sent_event_waiter(self, token):
        waiter = self.waiters.pop(token, None)
        if waiter is not None and not waiter[1].done():
            waiter[1].cancel()

    async def send_api_raw(self, action, params, timeout=15.0):
        self.calls.append((action, params, timeout))
        assert action in {"upload_group_file", "upload_private_file"}
        assert params["file"].startswith("base64://")
        self.uploaded_bytes = base64.b64decode(params["file"].removeprefix("base64://"))
        conv_type = "group" if action == "upload_group_file" else "private"
        event = {
            "post_type": "message_sent",
            "message_type": conv_type,
            "message_id": 9001,
            "time": time.time(),
            "sender": {"user_id": 4242},
            "message": [
                {
                    "type": "file",
                    "data": {
                        "file_id": "event-file-id",
                        "file_name": params["name"],
                        "file_size": len(self.uploaded_bytes),
                    },
                }
            ],
        }
        event["group_id" if conv_type == "group" else "user_id"] = params[
            "group_id" if conv_type == "group" else "user_id"
        ]
        for token, (matcher, future) in tuple(self.waiters.items()):
            if matcher(event):
                self.waiters.pop(token)
                future.set_result(event)
        return {"status": "ok", "data": {"file_id": "response-file-id"}}


class _RejectingGeneratedQQClient(_SentEventQQClient):
    async def send_api_raw(self, action, params, timeout=15.0):
        self.calls.append((action, params, timeout))
        assert params["file"].startswith("base64://")
        self.uploaded_bytes = base64.b64decode(params["file"].removeprefix("base64://"))
        return {
            "status": "failed",
            "retcode": 1200,
            "message": r"识别URL失败, uri= C:\Users\private-user\AppData\Local\Temp\secret\payload.bin",
            "wording": "/app/napcat/private/secret.bin",
        }


class _Session:
    def __init__(self, conv_type="group", conv_id="123") -> None:
        self.key = f"qq:{conv_type}:{conv_id}"
        self.conv_type = conv_type
        self.conv_id = conv_id
        self.conv_name = "测试会话"
        self.context_messages = []

    def add_to_context(self, entry):
        self.context_messages.append(entry)


def test_send_file_is_available_without_linux_and_exposes_both_sources() -> None:
    state = NamespaceRuntimeState()
    state.open("qq_social", load_namespace_registry(), 0)
    context = {
        "namespace_state": state,
        "current_round": 0,
        "current_platform": "qq",
        "session": SimpleNamespace(conv_type="group", conv_id="123"),
        "qq_client": _QQClient(),
        "workspace_service": _WorkspaceService(),
        "main_loop": object(),
    }

    disabled = build_tools({"platforms": {"qq": {"enabled": True}}}, **context)
    disabled_spec = disabled.all_specs["qq_social.send_file"]
    assert disabled_spec.declaration["name"] == "send_file"

    enabled = build_tools({
        "platforms": {"qq": {"enabled": True}},
        "workspace": {"enabled": True},
    }, **context)
    spec = enabled.all_specs["qq_social.send_file"]
    assert spec.declaration["name"] == "send_file"
    assert set(spec.declaration["parameters"]["properties"]) == {
        "path", "content", "filename", "format"
    }
    assert "qq_social.send_message" in enabled.all_specs


def test_send_file_argument_contract_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError):
        send_file.SendFileArgs.model_validate({})
    with pytest.raises(ValueError):
        send_file.SendFileArgs.model_validate({"path": "/home/agent/a.txt", "content": "x"})
    with pytest.raises(ValueError):
        send_file.SendFileArgs.model_validate(
            {"content": "x", "filename": "payload", "format": "exe"}
        )
    args = send_file.SendFileArgs.model_validate(
        {"content": "中文", "filename": "notes", "format": ".MD"}
    )
    assert args.format == "md"


def test_send_file_adapter_error_exposes_only_allowlisted_metadata() -> None:
    result = send_file._adapter_error(
        "upload_group_file",
        {
            "status": r"C:\Users\private\status",
            "retcode": r"C:\Users\private\retcode",
            "message": r"C:\Users\private\message",
            "wording": "/app/napcat/private/payload.bin",
        },
    )

    assert result
    assert "upload_group_file" in result
    assert "private" not in result
    assert "payload.bin" not in result


def test_send_file_generates_utf8_file_and_uses_real_sent_message_id(monkeypatch) -> None:
    import database
    import web.debug_server as debug_server

    loop_thread = _LoopThread()
    service = _GeneratedFileService()
    saved_messages = []
    broadcasts = []

    async def fake_save_chat_message(session_key, entry):
        saved_messages.append((session_key, entry))

    async def fake_broadcast_chat_event(event):
        broadcasts.append(event)

    monkeypatch.setattr(send_file, "get_qq_file_service", lambda *_args: service)
    monkeypatch.setattr(database, "save_chat_message", fake_save_chat_message)
    monkeypatch.setattr(debug_server, "broadcast_chat_event", fake_broadcast_chat_event)
    try:
        client = _SentEventQQClient()
        session = _Session()
        handler = send_file.make_handler(
            client,
            lambda: session,
            object(),
            loop_thread.loop,
            config={},
        )

        result = handler(content="# 标题\n正文\n", filename="现场文档", format="md")

        expected = "# 标题\n正文\n".encode("utf-8")
        assert client.uploaded_bytes == expected
        assert service.stored[0][0] == expected
        assert service.stored[0][1] == "现场文档.md"
        assert result == {
            "success": True,
            "path": "/home/agent/qq/4242/file/group_123/现场文档.md",
            "name": "现场文档.md",
            "size": len(expected),
            "target": "group_123",
            "source": "generated_text",
            "storage_backend": "linux",
            "record_id": "generated-record",
            "message_id_pending": False,
            "file_id": "response-file-id",
            "message_id": "9001",
        }
        assert service.delivery_updates == [
            {
                "record_id": "generated-record",
                "message_id": None,
                "file_id": "response-file-id",
            },
            {
                "record_id": "generated-record",
                "message_id": "9001",
                "file_id": "event-file-id",
            },
        ]
        assert saved_messages[0][0] == "qq:group:123"
        saved = saved_messages[0][1]
        assert saved["role"] == "bot"
        assert saved["message_id"] == "9001"
        assert saved["content_segments"] == [
            {
                "type": "file",
                "filename": "现场文档.md",
                "size_bytes": len(expected),
                "is_downloaded": True,
                "local_path": "/home/agent/qq/4242/file/group_123/现场文档.md",
                "file_id": "event-file-id",
            }
        ]
        assert session.context_messages == [saved]
        assert broadcasts[0]["entries"] == [saved]
        assert client.waiters == {}
    finally:
        loop_thread.close()


def test_generated_send_failure_never_returns_adapter_local_paths(monkeypatch) -> None:
    loop_thread = _LoopThread()
    service = _GeneratedFileService()
    monkeypatch.setattr(send_file, "get_qq_file_service", lambda *_args: service)
    try:
        client = _RejectingGeneratedQQClient()
        session = _Session()
        handler = send_file.make_handler(
            client,
            lambda: session,
            object(),
            loop_thread.loop,
            config={},
        )

        result = handler(content="private content", filename="safe", format="txt")

        assert result["error"]
        assert "upload_group_file" in result["error"]
        assert "retcode=1200" in result["error"]
        rendered = repr(result)
        assert "private-user" not in rendered
        assert "AppData" not in rendered
        assert "payload.bin" not in rendered
        assert "/app/napcat/private" not in rendered
        assert result["stored"] is True
        assert result["local_path"] == "/home/agent/qq/4242/file/group_123/safe.txt"
        assert client.uploaded_bytes == b"private content"
        assert client.waiters == {}
    finally:
        loop_thread.close()


def test_send_file_uses_local_path_when_native_napcat_can_read_it() -> None:
    loop_thread = _LoopThread()
    try:
        client = _QQClient()
        workspace = _WorkspaceService()
        handler = send_file.make_handler(
            client,
            lambda: SimpleNamespace(conv_type="group", conv_id="123"),
            workspace,
            loop_thread.loop,
        )

        result = handler("/home/agent/report.pdf")

        assert result == {
            "success": True,
            "path": "/home/agent/report.pdf",
            "name": "report.pdf",
            "size": len(workspace.content),
            "target": "group_123",
        }
        assert len(client.calls) == 1
        action, params, timeout = client.calls[0]
        assert action == "upload_group_file"
        assert params["group_id"] == 123
        assert params["name"] == "report.pdf"
        assert params["file"] == workspace.staged[0]
        assert timeout is None
        assert workspace.staging_roots == [None]
        assert workspace.staged == workspace.released
    finally:
        loop_thread.close()


def test_send_file_uses_private_upload_and_rejects_temp_session() -> None:
    loop_thread = _LoopThread()
    try:
        client = _QQClient()
        workspace = _WorkspaceService()
        private_handler = send_file.make_handler(
            client,
            lambda: SimpleNamespace(conv_type="private", conv_id="456"),
            workspace,
            loop_thread.loop,
        )
        assert private_handler("/home/agent/report.pdf")["success"] is True
        assert client.calls[0][0] == "upload_private_file"
        assert client.calls[0][1]["user_id"] == 456

        temp_handler = send_file.make_handler(
            client,
            lambda: SimpleNamespace(conv_type="temp", conv_id="789"),
            workspace,
            loop_thread.loop,
        )
        result = temp_handler("/home/agent/report.pdf")
        assert result["error"]
        assert len(client.calls) == 1
        assert workspace.staged == workspace.released
    finally:
        loop_thread.close()


def test_send_file_releases_staging_when_adapter_rejects_upload() -> None:
    loop_thread = _LoopThread()
    try:
        workspace = _WorkspaceService()
        client = _QQClient({"status": "failed", "retcode": 1200, "wording": "upload rejected"})
        handler = send_file.make_handler(
            client,
            lambda: SimpleNamespace(conv_type="group", conv_id="123"),
            workspace,
            loop_thread.loop,
        )

        result = handler("/home/agent/report.pdf")

        assert "upload_group_file" in result["error"]
        assert "retcode=1200" in result["error"]
        assert workspace.staged == workspace.released
    finally:
        loop_thread.close()


def test_send_file_preserves_local_path_transport_for_non_napcat_adapter() -> None:
    loop_thread = _LoopThread()
    try:
        workspace = _WorkspaceService()
        client = _QQClient(adapter="llonebot")
        handler = send_file.make_handler(
            client,
            lambda: SimpleNamespace(conv_type="private", conv_id="456"),
            workspace,
            loop_thread.loop,
        )

        assert handler("/home/agent/report.pdf")["success"] is True
        assert client.calls[0][1]["file"] == workspace.staged[0]
        assert workspace.staged == workspace.released
    finally:
        loop_thread.close()


def test_send_file_maps_shared_host_staging_path_for_docker_adapter() -> None:
    loop_thread = _LoopThread()
    try:
        workspace = _WorkspaceService()
        with tempfile.TemporaryDirectory(prefix="aicq-shared-root-test-") as shared_root:
            client = _QQClient(
                file_transfer={
                    "host_directory": shared_root,
                    "adapter_directory": "/app/napcat/transfer",
                }
            )
            handler = send_file.make_handler(
                client,
                lambda: SimpleNamespace(conv_type="group", conv_id="123"),
                workspace,
                loop_thread.loop,
            )

            assert handler("/home/agent/report.pdf")["success"] is True
            adapter_path = client.calls[0][1]["file"]
            assert adapter_path.startswith("/app/napcat/transfer/aicq-send-file-test-")
            assert adapter_path.endswith("/payload.bin")
            assert workspace.staging_roots == [shared_root]
            assert workspace.staged == workspace.released
    finally:
        loop_thread.close()


def test_send_file_shared_transport_does_not_impose_an_application_size_limit() -> None:
    loop_thread = _LoopThread()
    try:
        workspace = _WorkspaceService()
        workspace.reported_size = 8 * 1024**3
        with tempfile.TemporaryDirectory(prefix="aicq-shared-root-test-") as shared_root:
            client = _QQClient(
                file_transfer={
                    "host_directory": shared_root,
                    "adapter_directory": "/app/napcat/transfer",
                }
            )
            handler = send_file.make_handler(
                client,
                lambda: SimpleNamespace(conv_type="group", conv_id="123"),
                workspace,
                loop_thread.loop,
            )

            result = handler("/home/agent/report.pdf")

            assert result["success"] is True
            assert result["size"] == 8 * 1024**3
            assert len(client.calls) == 1
            assert client.calls[0][2] is None
            assert not client.calls[0][1]["file"].startswith("base64:")
            assert workspace.staged == workspace.released
    finally:
        loop_thread.close()


def test_send_file_rejects_incomplete_shared_directory_mapping() -> None:
    loop_thread = _LoopThread()
    try:
        workspace = _WorkspaceService()
        client = _QQClient(file_transfer={"host_directory": r"C:\transfer"})
        handler = send_file.make_handler(
            client,
            lambda: SimpleNamespace(conv_type="group", conv_id="123"),
            workspace,
            loop_thread.loop,
        )

        result = handler("/home/agent/report.pdf")

        assert result["error"]
        assert client.calls == []
        assert workspace.staged == workspace.released == []
    finally:
        loop_thread.close()
