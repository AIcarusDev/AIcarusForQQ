from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AttachmentResult:
    task_id: str
    attachment_id: str
    status: str
    source_type: str
    source: str
    filename: str | None = None
    path: str | None = None
    mime: str | None = None
    image_ref: str | None = None
    bytes_downloaded: int = 0
    bytes_total: int | None = None
    sha256: str | None = None
    error: str | None = None
    started_at: str = ""
    finished_at: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status in {"completed", "failed", "stopped", "interrupted"}

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.status in {"running", "completed"},
            "task_id": self.task_id,
            "attachment_id": self.attachment_id,
            "status": self.status,
            "source_type": self.source_type,
            "bytes_downloaded": self.bytes_downloaded,
            "started_at": self.started_at,
        }
        for key in (
            "filename", "mime", "image_ref", "bytes_total", "sha256", "error", "finished_at"
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload
