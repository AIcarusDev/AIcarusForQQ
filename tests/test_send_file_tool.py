from __future__ import annotations

import asyncio
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

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


def test_send_file_is_an_independent_workspace_gated_tool_contract() -> None:
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

    disabled = build_tools({}, **context)
    assert "qq_social.send_file" not in disabled.all_specs

    enabled = build_tools({"workspace": {"enabled": True}}, **context)
    spec = enabled.all_specs["qq_social.send_file"]
    assert spec.declaration["name"] == "send_file"
    assert set(spec.declaration["parameters"]["properties"]) == {"path"}
    assert "qq_social.send_message" in enabled.all_specs


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
        assert "不支持" in result["error"]
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

        assert "共享目录配置不完整" in result["error"]
        assert client.calls == []
        assert workspace.staged == workspace.released == []
    finally:
        loop_thread.close()
