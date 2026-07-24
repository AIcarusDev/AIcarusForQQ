"""Host-side attachment download and read-only inspection."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import AttachmentResult
from .service import AttachmentService
from workspace.config import workspace_enabled


def create_attachment_service(
    config: Mapping[str, Any],
    *,
    cache_root: Path | None = None,
) -> AttachmentService | None:
    """Create the host-side attachment service only for an enabled workspace."""

    if not workspace_enabled(config):
        return None
    return AttachmentService(cache_root)


__all__ = ["AttachmentResult", "AttachmentService", "create_attachment_service"]
