"""One-round, session-scoped confirmation state for high-risk image sends."""

from __future__ import annotations

import base64
import copy
import html
import time
import uuid
from dataclasses import dataclass
from typing import Any

from browser.session import read_sendable_browser_image_file


@dataclass(frozen=True, slots=True)
class PendingBrowserImageSend:
    batch_id: str
    target: tuple[str, str, str]
    inbound_revision: int
    messages: tuple[dict[str, Any], ...]
    artifacts: tuple[dict[str, Any], ...]
    created_at: float


def _target(session: Any) -> tuple[str, str, str]:
    return (
        str(getattr(session, "conv_type", "") or ""),
        str(getattr(session, "conv_id", "") or ""),
        str(getattr(session, "temp_source_group_id", "") or ""),
    )


def current_pending(session: Any) -> PendingBrowserImageSend | None:
    pending = getattr(session, "pending_browser_image_send", None)
    if not isinstance(pending, PendingBrowserImageSend):
        return None
    if pending.target != _target(session):
        cancel_pending(session, reason="target_changed")
        return None
    if pending.inbound_revision != int(getattr(session, "inbound_revision", 0) or 0):
        cancel_pending(session, reason="inbound_revision_changed")
        return None
    return pending


def stage_pending(
    session: Any,
    *,
    messages: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> PendingBrowserImageSend:
    if not artifacts or len(artifacts) > 4:
        raise ValueError("confirmation batch must contain 1..4 images")
    cancel_pending(session, reason="replaced_by_new_batch")
    pending = PendingBrowserImageSend(
        batch_id=f"bic_{uuid.uuid4().hex[:16]}",
        target=_target(session),
        inbound_revision=int(getattr(session, "inbound_revision", 0) or 0),
        messages=tuple(copy.deepcopy(messages)),
        artifacts=tuple(copy.deepcopy(artifacts)),
        created_at=time.time(),
    )
    session.pending_browser_image_send = pending
    session.browser_image_confirmation_last_status = {
        "status": "pending",
        "batch_id": pending.batch_id,
    }
    return pending


def cancel_pending(
    session: Any,
    *,
    batch_id: str | None = None,
    reason: str = "cancelled",
) -> bool:
    pending = getattr(session, "pending_browser_image_send", None)
    if not isinstance(pending, PendingBrowserImageSend):
        return False
    if batch_id is not None and pending.batch_id != str(batch_id):
        return False
    session.pending_browser_image_send = None
    session.browser_image_confirmation_last_status = {
        "status": "cancelled",
        "batch_id": pending.batch_id,
        "reason": reason,
    }
    return True


def consume_pending(session: Any, batch_id: str) -> PendingBrowserImageSend | None:
    pending = current_pending(session)
    if pending is None or pending.batch_id != str(batch_id or ""):
        return None
    for artifact in pending.artifacts:
        item = read_sendable_browser_image_file(str(artifact.get("image_ref") or ""))
        expected_hash = str(artifact.get("sha256") or "")
        if item is None or not expected_hash or str(item[2].get("sha256") or "") != expected_hash:
            cancel_pending(session, batch_id=pending.batch_id, reason="artifact_changed")
            return None
    session.pending_browser_image_send = None
    session.browser_image_confirmation_last_status = {
        "status": "confirmed",
        "batch_id": pending.batch_id,
    }
    return pending


def expire_pending_after_round(session: Any, batch_id_at_round_start: str | None) -> bool:
    if not batch_id_at_round_start:
        return False
    return cancel_pending(
        session,
        batch_id=batch_id_at_round_start,
        reason="confirmation_round_expired",
    )


def build_pending_confirmation_content(session: Any) -> str | list[dict[str, Any]]:
    """Build one-round model content from final immutable originals."""
    pending = current_pending(session)
    if pending is None:
        return ""
    parts: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            f'<image_send_confirmation batch_id="{html.escape(pending.batch_id, quote=True)}" '
            f'count="{len(pending.artifacts)}" policy="confirm_or_cancel_this_round">\n'
        ),
    }]
    for index, artifact in enumerate(pending.artifacts):
        image_ref = str(artifact.get("image_ref") or "")
        item = read_sendable_browser_image_file(image_ref)
        expected_hash = str(artifact.get("sha256") or "")
        if (
            item is None
            or not expected_hash
            or str(item[2].get("sha256") or "") != expected_hash
        ):
            cancel_pending(session, reason="artifact_unavailable")
            return ""
        raw, mime, _manifest = item
        reasons = ",".join(str(reason) for reason in artifact.get("confirmation_reasons") or [])
        parts.append({
            "type": "text",
            "text": (
                f'  <image index="{index}" '
                f'confirmation_reasons="{html.escape(reasons, quote=True)}">\n'
            ),
        })
        parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}",
            },
        })
        parts.append({"type": "text", "text": "\n  </image>\n"})
    parts.append({
        "type": "text",
        "text": (
            "</image_send_confirmation>\n"
            "These are the final immutable originals. Use confirm_browser_image_send "
            "or cancel_browser_image_send in this round; otherwise the batch expires."
        ),
    })
    return parts


__all__ = [
    "PendingBrowserImageSend",
    "build_pending_confirmation_content",
    "cancel_pending",
    "consume_pending",
    "current_pending",
    "expire_pending_after_round",
    "stage_pending",
]
