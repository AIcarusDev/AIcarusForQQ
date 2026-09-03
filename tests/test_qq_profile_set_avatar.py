from __future__ import annotations

import asyncio
import base64
import io
from contextlib import asynccontextmanager
from types import SimpleNamespace

import app_state
from llm.core.tool_calling.schema import validate_arguments_by_declaration
from PIL import Image
from platforms.qq.tools.qq_profile import set_avatar
from tools import build_tools
from tools.namespaces import NamespaceRuntimeState, load_namespace_registry


def _png_bytes(*, width: int = 5, height: int = 9) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), (20, 40, 60)).save(output, format="PNG")
    return output.getvalue()


class _Loop:
    def is_running(self) -> bool:
        return True


class _QQClient:
    connected = True
    bot_id = "10000"
    _loop = _Loop()

    def __init__(self, response: dict | None = None) -> None:
        self.response = response or {"status": "ok", "retcode": 0, "data": None}
        self.calls: list[tuple[str, dict, float]] = []

    async def send_api_raw(self, action: str, params: dict, timeout: float = 15.0):
        self.calls.append((action, params, timeout))
        return self.response


def _run_coroutine(coro, _loop, *, timeout=None, **_kwargs):
    return asyncio.run(coro)


def _session_with_image(image_ref: str, raw: bytes):
    return SimpleNamespace(
        context_messages=[
            {
                "images": {
                    image_ref: {
                        "base64": base64.b64encode(raw).decode("ascii"),
                        "mime": "image/png",
                    }
                }
            }
        ],
        is_browsing_history=lambda: False,
        forward_browser_stack=[],
    )


def test_set_avatar_schema_requires_exactly_one_image_source() -> None:
    declaration = set_avatar.TOOL_CONTRACT.declaration()

    assert validate_arguments_by_declaration({"image_ref": "image-1"}, declaration)[0]
    assert validate_arguments_by_declaration({"path": "/home/agent/avatar.png"}, declaration)[0]
    assert not validate_arguments_by_declaration({}, declaration)[0]
    assert not validate_arguments_by_declaration(
        {"image_ref": "image-1", "path": "/home/agent/avatar.png"},
        declaration,
    )[0]
    assert not validate_arguments_by_declaration({"path": "/tmp/avatar.png"}, declaration)[0]


def test_set_avatar_uploads_image_ref_bytes_without_cropping(monkeypatch) -> None:
    raw = _png_bytes(width=5, height=9)
    client = _QQClient()
    monkeypatch.setattr(set_avatar, "run_coroutine_sync", _run_coroutine)

    result = set_avatar.make_handler(client, _session_with_image("image-1", raw))(
        image_ref="image_ref='image-1'"
    )

    assert result["ok"] is True
    assert result["width"] == 5
    assert result["height"] == 9
    assert result["size_bytes"] == len(raw)
    action, params, _timeout = client.calls[0]
    assert action == "set_qq_avatar"
    assert base64.b64decode(params["file"].removeprefix("base64://")) == raw


def test_set_avatar_reads_agent_linux_path_and_uploads_exact_bytes(monkeypatch, tmp_path) -> None:
    raw = _png_bytes(width=8, height=3)
    staged_file = tmp_path / "payload.bin"
    staged_file.write_bytes(raw)

    class _WorkspaceService:
        @asynccontextmanager
        async def stage_host_file(self, path: str):
            assert path == "/home/agent/avatar/source.png"
            yield SimpleNamespace(
                workspace_path=path,
                host_path=str(staged_file),
                name="source.png",
                size=len(raw),
            )

    client = _QQClient()
    monkeypatch.setattr(app_state, "workspace_service", _WorkspaceService())
    monkeypatch.setattr(set_avatar, "run_coroutine_sync", _run_coroutine)

    result = set_avatar.make_handler(
        client,
        SimpleNamespace(context_messages=[], is_browsing_history=lambda: False, forward_browser_stack=[]),
    )(path="/home/agent/avatar/source.png")

    assert result["ok"] is True
    assert result["width"] == 8
    assert result["height"] == 3
    assert len(client.calls) == 1
    assert base64.b64decode(client.calls[0][1]["file"].removeprefix("base64://")) == raw


def test_set_avatar_rejects_invalid_path_content_before_adapter_call(monkeypatch, tmp_path) -> None:
    staged_file = tmp_path / "not-image.bin"
    staged_file.write_bytes(b"not an image")

    class _WorkspaceService:
        @asynccontextmanager
        async def stage_host_file(self, path: str):
            yield SimpleNamespace(
                workspace_path=path,
                host_path=str(staged_file),
                name="not-image.bin",
                size=staged_file.stat().st_size,
            )

    client = _QQClient()
    monkeypatch.setattr(app_state, "workspace_service", _WorkspaceService())
    monkeypatch.setattr(set_avatar, "run_coroutine_sync", _run_coroutine)

    result = set_avatar.make_handler(
        client,
        SimpleNamespace(context_messages=[], is_browsing_history=lambda: False, forward_browser_stack=[]),
    )(path="/home/agent/not-image.bin")

    assert result["ok"] is False
    assert result["status"] == "invalid_image"
    assert client.calls == []


def test_set_avatar_does_not_expose_adapter_paths_or_payloads(monkeypatch) -> None:
    raw = _png_bytes()
    client = _QQClient(
        {
            "status": "failed",
            "retcode": 1004022,
            "message": r"failed, uri=C:\Users\private\avatar.png",
            "wording": "/app/napcat/private/avatar.png",
        }
    )
    monkeypatch.setattr(set_avatar, "run_coroutine_sync", _run_coroutine)

    result = set_avatar.make_handler(client, _session_with_image("image-1", raw))(
        image_ref="image-1"
    )

    rendered = repr(result)
    assert result["ok"] is False
    assert result["status"] == "adapter_error"
    assert "retcode=1004022" in result["error"]
    assert "C:\\Users" not in rendered
    assert "/app/napcat" not in rendered
    assert "base64://" not in rendered


def test_set_avatar_is_registered_as_guarded_qq_profile_write() -> None:
    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    state.open("qq_profile", registry, 1)
    collection = build_tools(
        {
            "platforms": {"qq": {"enabled": True}},
            "vision": False,
            "tts": {"enabled": False},
        },
        namespace_state=state,
        current_round=1,
        default_ttl_rounds=5,
        current_platform="qq",
        qq_client=_QQClient(),
        session=SimpleNamespace(context_messages=[]),
    )

    spec = collection.all_specs["qq_profile.set_avatar"]
    assert "qq_profile.set_avatar" in collection.active_names()
    assert spec.externally_perceptible is True
    assert spec.effect is not None
    assert spec.effect.surface == "qq"
    assert spec.effect.kind == "profile_write"
