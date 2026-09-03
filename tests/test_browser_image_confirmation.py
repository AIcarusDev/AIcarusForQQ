from __future__ import annotations

from types import SimpleNamespace

import browser
from browser.image_confirmation import (
    build_pending_confirmation_content,
    current_pending,
    consume_pending,
    expire_pending_after_round,
    stage_pending,
)
from llm.session import ConversationSession
from platforms.focus import FocusRef
from platforms.qq.tools.qq_social import (
    cancel_browser_image_send,
    confirm_browser_image_send,
)
from platforms.qq.tools.qq_social.send_message import send_message as send_mod
from tools import build_tools
from tools.namespaces import NamespaceRuntimeState, load_namespace_registry


def _session() -> ConversationSession:
    return ConversationSession(focus=FocusRef("qq", "group", "1234", "test"))


def _artifacts(count: int = 2) -> list[dict]:
    return [
        {
            "image_ref": f"img_{index:032x}",
            "resource_ref": f"br_{index:020x}",
            "sha256": f"{index:064x}",
            "confirmation_reasons": ["very_small_preview"],
        }
        for index in range(count)
    ]


def test_pending_confirmation_is_bound_to_session_revision_and_auto_expires() -> None:
    session = _session()
    pending = stage_pending(
        session,
        messages=[{"segments": [{"command": "image", "image_ref": _artifacts(1)[0]["image_ref"]}]}],
        artifacts=_artifacts(1),
    )

    assert current_pending(session) == pending
    assert expire_pending_after_round(session, "another-batch") is False
    assert current_pending(session) == pending
    assert expire_pending_after_round(session, pending.batch_id) is True
    assert current_pending(session) is None


def test_confirmation_consumes_batch_once_and_revalidates_bound_hash(monkeypatch) -> None:
    session = _session()
    pending = stage_pending(
        session,
        messages=[{"segments": [{"command": "text", "content": "caption"}]}],
        artifacts=_artifacts(1),
    )
    monkeypatch.setattr(
        "browser.image_confirmation.read_sendable_browser_image_file",
        lambda _ref: (b"original", "image/png", {"sha256": "0" * 64}),
    )

    assert consume_pending(session, pending.batch_id) == pending
    assert consume_pending(session, pending.batch_id) is None

    changed = stage_pending(
        session,
        messages=[{"segments": [{"command": "text", "content": "changed"}]}],
        artifacts=_artifacts(1),
    )
    monkeypatch.setattr(
        "browser.image_confirmation.read_sendable_browser_image_file",
        lambda _ref: (b"changed", "image/png", {"sha256": "f" * 64}),
    )
    assert consume_pending(session, changed.batch_id) is None
    assert session.browser_image_confirmation_last_status["reason"] == "artifact_changed"


def test_new_inbound_message_invalidates_pending_confirmation() -> None:
    session = _session()
    pending = stage_pending(
        session,
        messages=[{"segments": [{"command": "text", "content": "caption"}]}],
        artifacts=_artifacts(1),
    )

    session.mark_inbound_received()

    assert current_pending(session) is None
    assert session.browser_image_confirmation_last_status == {
        "status": "cancelled",
        "batch_id": pending.batch_id,
        "reason": "inbound_revision_changed",
    }


def test_leaving_session_cancels_pending_confirmation() -> None:
    session = _session()
    pending = stage_pending(
        session,
        messages=[{"segments": [{"command": "text", "content": "caption"}]}],
        artifacts=_artifacts(1),
    )

    session.reset_transient_views()

    assert current_pending(session) is None
    assert session.browser_image_confirmation_last_status == {
        "status": "cancelled",
        "batch_id": pending.batch_id,
        "reason": "session_focus_changed",
    }


def test_confirmation_preview_is_one_composite_block_without_internal_fields(monkeypatch) -> None:
    session = _session()
    pending = stage_pending(
        session,
        messages=[{"segments": [{"command": "text", "content": "caption"}]}],
        artifacts=_artifacts(2),
    )
    monkeypatch.setattr(
        "browser.image_confirmation.read_sendable_browser_image_file",
        lambda ref: (
            b"original",
            "image/png",
            {
                "local_path": "must-not-leak",
                "sha256": f"{int(ref.removeprefix('img_'), 16):064x}",
            },
        ),
    )

    content = build_pending_confirmation_content(session)

    assert isinstance(content, list)
    assert sum(part.get("type") == "image_url" for part in content) == 2
    text = "".join(part.get("text", "") for part in content if part.get("type") == "text")
    assert pending.batch_id in text
    assert "very_small_preview" in text
    assert "image_ref" not in text
    assert "sha256" not in text
    assert "local_path" not in text
    assert "request_id" not in text
    assert "frame_id" not in text


def test_confirm_cancel_tools_exist_only_while_batch_is_pending() -> None:
    session = _session()

    def provider():
        return session

    assert confirm_browser_image_send.is_available(provider) is False
    assert cancel_browser_image_send.is_available(provider) is False

    stage_pending(
        session,
        messages=[{"segments": [{"command": "text", "content": "caption"}]}],
        artifacts=_artifacts(1),
    )

    assert confirm_browser_image_send.is_available(provider) is True
    assert cancel_browser_image_send.is_available(provider) is True


def test_confirmation_tools_are_added_to_normal_composite_round() -> None:
    session = _session()
    state = NamespaceRuntimeState()
    state.open("qq_social", load_namespace_registry(), 1)

    without_pending = build_tools(
        {"platforms": {"qq": {"enabled": True}}},
        namespace_state=state,
        current_round=1,
        current_platform="qq",
        session=session,
        qq_client=SimpleNamespace(connected=True),
    )
    assert "qq_social.send_message" in without_pending.active_specs
    assert "qq_social.confirm_browser_image_send" not in without_pending.active_specs
    assert "qq_social.cancel_browser_image_send" not in without_pending.active_specs

    stage_pending(
        session,
        messages=[{"segments": [{"command": "text", "content": "caption"}]}],
        artifacts=_artifacts(1),
    )
    with_pending = build_tools(
        {"platforms": {"qq": {"enabled": True}}},
        namespace_state=state,
        current_round=2,
        current_platform="qq",
        session=session,
        qq_client=SimpleNamespace(connected=True),
    )

    assert "qq_social.send_message" in with_pending.active_specs
    assert "qq_social.confirm_browser_image_send" in with_pending.active_specs
    assert "qq_social.cancel_browser_image_send" in with_pending.active_specs
    assert len(with_pending.active_specs) > 3


def test_browser_resource_batch_materialization_preserves_order_and_maximum(monkeypatch) -> None:
    selected: list[str] = []

    def materialize(refs: list[str]) -> list[dict]:
        selected.extend(refs)
        return [
            {
                "resource_ref": ref,
                "image_ref": f"img_{index:032x}",
                "confirmation_reasons": [],
            }
            for index, ref in enumerate(refs)
        ]

    monkeypatch.setattr(browser, "materialize_browser_resources", materialize)
    messages = [{
        "segments": [
            {"command": "text", "content": "before"},
            {"command": "image", "resource_ref": "br_00000000000000000002"},
            {"command": "image", "resource_ref": "br_00000000000000000001"},
        ]
    }]

    prepared, artifacts, error = send_mod._materialize_selected_browser_resources(messages)

    assert error is None
    assert selected == ["br_00000000000000000002", "br_00000000000000000001"]
    assert [segment.get("image_ref") for segment in prepared[0]["segments"][1:]] == [
        "img_00000000000000000000000000000000",
        "img_00000000000000000000000000000001",
    ]
    assert all("resource_ref" not in segment for segment in prepared[0]["segments"][1:])
    assert len(artifacts) == 2

    too_many = [{
        "segments": [
            {"command": "image", "resource_ref": f"br_{index:020x}"}
            for index in range(5)
        ]
    }]
    prepared, artifacts, error = send_mod._materialize_selected_browser_resources(too_many)
    assert prepared is None
    assert artifacts == []
    assert error


def test_pending_high_risk_artifact_cannot_bypass_confirmation() -> None:
    session = _session()
    artifacts = _artifacts(1)
    stage_pending(
        session,
        messages=[{
            "segments": [{"command": "image", "image_ref": artifacts[0]["image_ref"]}],
        }],
        artifacts=artifacts,
    )

    error = send_mod._unconfirmed_high_risk_image_error(
        [{
            "segments": [{"command": "image", "image_ref": artifacts[0]["image_ref"]}],
        }],
        session,
        {"browser_control": {"image_send_confirmation": "high_risk"}},
    )

    assert error
