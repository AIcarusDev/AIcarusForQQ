from __future__ import annotations

import asyncio
import sqlite3

import app_state
import attachments
import database
from attachments.models import AttachmentResult
from attachments.service import AttachmentService, safe_filename
from attachments import service as attachment_service_module
from attachments.tools import download as download_tool
from llm.session import ConversationSession
from platforms.focus import FocusRef
from runtime.events import RuntimeEventHub


def result(status: str) -> AttachmentResult:
    return AttachmentResult(
        task_id="d" * 32,
        attachment_id="a" * 32,
        status=status,
        source_type="url",
        source="https://example.com/file",
        started_at="now",
    )


def test_attachment_service_lifecycle_follows_workspace_config(tmp_path, monkeypatch) -> None:
    constructed: list[object] = []
    real_service = attachments.AttachmentService

    def tracked_service(cache_root):
        constructed.append(cache_root)
        return real_service(cache_root)

    monkeypatch.setattr(attachments, "AttachmentService", tracked_service)

    assert (
        attachments.create_attachment_service(
            {"workspace": {"enabled": False}},
            cache_root=tmp_path,
        )
        is None
    )
    assert constructed == []

    service = attachments.create_attachment_service(
        {"workspace": {"enabled": True}},
        cache_root=tmp_path,
    )
    assert isinstance(service, real_service)
    assert constructed == [tmp_path]


def test_service_downloads_to_content_addressed_cache_and_reads_text(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        service = AttachmentService(tmp_path)

        async def no_persist(_result):
            return True

        monkeypatch.setattr(service, "_persist", no_persist)

        async def resolver():
            return {"data": b"hello\nworld", "filename": "hello.txt"}

        started = await service.start(source_type="ref", source="abc", resolver=resolver)
        completed = await service.wait(started.task_id, timeout=2)
        assert completed is not None and completed.status == "completed"
        assert completed.path is not None and completed.path.startswith(str(tmp_path))
        payload = await service.read(completed.attachment_id, limit=5)
        assert payload["content"] == "hello"
        assert payload["has_more"] is True
        await service.close()

    asyncio.run(scenario())


def test_attachment_result_persists_and_reloads_from_database(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "aicq.sqlite3"
    cache_path = tmp_path / "cache"
    monkeypatch.setattr(database, "DB_PATH", str(db_path))
    monkeypatch.setattr(attachment_service_module, "DB_PATH", str(db_path))

    async def scenario() -> None:
        await database.init_db()
        service = AttachmentService(cache_path)

        async def resolver():
            return {"data": b"persisted", "filename": "persisted.txt"}

        started = await service.start(source_type="ref", source="persisted-ref", resolver=resolver)
        completed = await service.wait(started.task_id, timeout=2)
        assert completed is not None and completed.status == "completed"
        await service.close()

        reloaded = AttachmentService(cache_path)
        stored = await reloaded.poll(started.task_id)
        assert stored.status == "completed"
        assert stored.mime == "text/plain"
        assert stored.path == completed.path
        await reloaded.close()

    asyncio.run(scenario())

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(attachment_tasks)")}
    assert {"mime", "image_ref"}.issubset(columns)


def test_qq_image_ref_is_reused_without_attachment_copy(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        service = AttachmentService(tmp_path)

        async def no_persist(_result):
            return True

        monkeypatch.setattr(service, "_persist", no_persist)

        async def resolver():
            return {"image_ref": "abc123", "mime": "image/png", "filename": "image.png"}

        started = await service.start(source_type="ref", source="abc123", resolver=resolver)
        completed = await service.wait(started.task_id, timeout=2)
        assert completed is not None and completed.path is None
        payload = await service.read(completed.attachment_id)
        assert payload["image_ref"] == "abc123"
        assert not list(tmp_path.rglob("*.png"))
        await service.close()

    asyncio.run(scenario())


def test_content_cache_deduplicates_same_bytes_across_extensions(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        service = AttachmentService(tmp_path)

        async def no_persist(_result):
            return True

        monkeypatch.setattr(service, "_persist", no_persist)
        for filename in ("same.txt", "same.bin"):
            async def resolver(name=filename):
                return {"data": b"same-content", "filename": name}

            started = await service.start(source_type="ref", source=filename, resolver=resolver)
            completed = await service.wait(started.task_id, timeout=2)
            assert completed is not None and completed.status == "completed"
        assert len([path for path in tmp_path.rglob("*") if path.is_file()]) == 1
        await service.close()

    asyncio.run(scenario())


def test_download_tool_returns_running_on_attention(monkeypatch) -> None:
    class Service:
        async def start(self, **kwargs):
            return result("running")

        async def wait(self, task_id, *, timeout):
            await asyncio.Event().wait()

        async def poll(self, task_id):
            return result("running")

        def mark_delivered(self, task_id):
            raise AssertionError("running task must not be acknowledged")

    async def scenario() -> None:
        hub = RuntimeEventHub()
        monkeypatch.setattr(app_state, "current_focus", FocusRef("qq", "group", "focus"))
        monkeypatch.setattr(download_tool, "run_on_main_loop", lambda coro, _loop: coro)
        await hub.publish({"type": "attention"}, target="qq:group:focus")
        handler = download_tool.make_handler(Service(), hub, object(), lambda: ConversationSession())
        payload = await handler(action="start", url="https://example.com/a", ref=None, task_id=None)
        assert payload["status"] == "running"
        assert payload["attachment_id"] == "a" * 32
        assert await hub.wait(
            timeout=0, target="qq:group:focus", event_types={"attention"}
        ) == [{"type": "attention"}]

    asyncio.run(scenario())


def test_ref_resolves_current_context_image_without_copy() -> None:
    session = ConversationSession()
    session.context_messages.append({"images": {"abc": {"base64": "aGVsbG8=", "mime": "image/png"}}})
    source = download_tool._ref_source(session, "abc")
    assert source["image_ref"] == "abc"
    assert "data" not in source


def test_persisted_qq_ref_is_resolved_again_through_message_id(monkeypatch) -> None:
    class Client:
        async def send_api(self, action, params, timeout=15.0):
            assert action == "get_msg"
            assert params == {"message_id": 2458}
            return {
                "message": [
                    {"type": "text", "data": {"text": "附件"}},
                    {
                        "type": "file",
                        "data": {"url": "https://example.com/report.txt", "file_name": "report.txt"},
                    },
                ]
            }

    session = ConversationSession()
    session.context_messages.append(
        {
            "message_id": "2458",
            "content_segments": [
                {"type": "text", "text": "附件"},
                {"type": "file", "filename": "未知", "ref": "persisted-ref"},
            ],
        }
    )
    monkeypatch.setattr(download_tool, "_qq_client", lambda: Client())

    source = download_tool._ref_source(session, "persisted-ref")
    resolved = asyncio.run(source["resolver"]())

    assert resolved == {
        "url": "https://example.com/report.txt",
        "filename": "report.txt",
    }


def test_persisted_qq_ref_rejects_message_from_another_group(monkeypatch) -> None:
    class Client:
        async def send_api(self, action, params, timeout=15.0):
            return {
                "message_id": 2458,
                "message_type": "group",
                "group_id": 999,
                "message": [{"type": "file", "data": {"url": "https://example.com/x"}}],
            }

    session = ConversationSession()
    session.conv_type = "group"
    session.conv_id = "84"
    session.context_messages.append(
        {
            "message_id": "2458",
            "content_segments": [{"type": "file", "filename": "x", "ref": "scope-ref"}],
        }
    )
    monkeypatch.setattr(download_tool, "_qq_client", lambda: Client())
    source = download_tool._ref_source(session, "scope-ref")
    try:
        asyncio.run(source["resolver"]())
    except ValueError as exc:
        assert "其他群" in str(exc)
    else:
        raise AssertionError("history resolver must reject a message from another group")


def test_private_qq_document_uses_adapter_file_url_api() -> None:
    class Client:
        calls = []

        async def send_api(self, action, params, timeout=15.0):
            self.calls.append((action, params))
            if action == "get_private_file_url":
                return {"url": "https://example.com/private.txt"}
            raise AssertionError(f"unexpected action: {action}")

    client = Client()
    resolved = asyncio.run(
        download_tool._resolve_adapter_file(
            client,
            {"file_id": "private-file-id"},
            "private.txt",
            media_type="file",
            qq_context={"conv_type": "private", "conv_id": "42"},
        )
    )

    assert resolved["url"] == "https://example.com/private.txt"
    assert client.calls == [("get_private_file_url", {"file_id": "private-file-id"})]


def test_group_qq_document_uses_adapter_file_url_api() -> None:
    class Client:
        async def send_api(self, action, params, timeout=15.0):
            assert action == "get_group_file_url"
            assert params == {"group_id": "84", "file_id": "group-file-id"}
            return {"url": "https://example.com/group.pdf"}

    resolved = asyncio.run(
        download_tool._resolve_adapter_file(
            Client(),
            {"file_id": "group-file-id"},
            "group.pdf",
            media_type="file",
            qq_context={"conv_type": "group", "conv_id": "84"},
        )
    )

    assert resolved["url"] == "https://example.com/group.pdf"


def test_safe_filename_handles_windows_adapter_paths() -> None:
    assert safe_filename(r"D:\QQ\cache\report.pdf") == "report.pdf"


def test_base64_attachment_is_rejected_before_oversized_decode(monkeypatch) -> None:
    monkeypatch.setattr(download_tool, "MAX_ATTACHMENT_BYTES", 3)

    try:
        download_tool._decode_base64_attachment("aGVsbG8=")
    except ValueError as exc:
        assert "100 MiB" in str(exc)
    else:
        raise AssertionError("oversized base64 attachment must be rejected")


def test_adapter_host_path_requires_configured_allowlist(tmp_path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    inside = allowed / "inside.txt"
    inside.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    class Client:
        file_transfer = {"host_directory": str(allowed), "adapter_directory": ""}

    assert download_tool._allowed_adapter_host_path(Client(), str(inside)) == str(inside.resolve())
    try:
        download_tool._allowed_adapter_host_path(Client(), str(outside))
    except ValueError as exc:
        assert "allowlist" in str(exc)
    else:
        raise AssertionError("path outside adapter allowlist must be rejected")
