"""Host-side attachment download and read-only inspection."""

from .models import AttachmentResult
from .service import AttachmentService

__all__ = ["AttachmentResult", "AttachmentService"]
