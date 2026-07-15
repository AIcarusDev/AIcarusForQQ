from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
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
    def __init__(self) -> None:
        self.staged = []
        self.released = []

    @asynccontextmanager
    async def stage_host_file(self, path: str):
        assert path == "/workspace/report.pdf"
        prepared = SimpleNamespace(
            workspace_path=path,
            host_path=r"C:\Temp\aicq-workspace-send\payload.bin",
            name="report.pdf",
            size=1234,
        )
        self.staged.append(prepared.host_path)
        try:
            yield prepared
        finally:
            self.released.append(prepared.host_path)


class _QQClient:
    connected = True

    def __init__(self, response=None) -> None:
        self.response = response or {"status": "ok", "data": None}
        self.calls = []

    async def send_api_raw(self, action, params, timeout=15.0):
        self.calls.append((action, params, timeout))
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
    assert "segments" not in spec.prompt_signature
    assert "qq_social.send_message" in enabled.all_specs


def test_send_file_uses_independent_group_upload_api() -> None:
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

        result = handler("/workspace/report.pdf")

        assert result == {
            "success": True,
            "path": "/workspace/report.pdf",
            "name": "report.pdf",
            "size": 1234,
            "target": "group_123",
        }
        assert client.calls == [
            (
                "upload_group_file",
                {
                    "group_id": 123,
                    "file": r"C:\Temp\aicq-workspace-send\payload.bin",
                    "name": "report.pdf",
                },
                120.0,
            )
        ]
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
        assert private_handler("/workspace/report.pdf")["success"] is True
        assert client.calls[0][0] == "upload_private_file"
        assert client.calls[0][1]["user_id"] == 456

        temp_handler = send_file.make_handler(
            client,
            lambda: SimpleNamespace(conv_type="temp", conv_id="789"),
            workspace,
            loop_thread.loop,
        )
        result = temp_handler("/workspace/report.pdf")
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

        result = handler("/workspace/report.pdf")

        assert "upload_group_file" in result["error"]
        assert "retcode=1200" in result["error"]
        assert workspace.staged == workspace.released
    finally:
        loop_thread.close()
