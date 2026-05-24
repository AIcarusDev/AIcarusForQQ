"""napcat — NapCat WebSocket 连接层与消息翻译包

子模块：
  client   — NapcatClient（WebSocket 服务器，连接管理，API 调用）
  segments — 消息段格式互转（NapCat ↔ 文本/结构化/LLM segments）
  events   — 事件解析（→ context entry，should_respond，get_conversation_id）
  debug    — 调试 XML 生成
"""

from .client import NapcatClient
from .segments import (
    QQ_FACE,
    napcat_segments_to_text,
    build_content_segments,
    get_reply_message_id,
    llm_segments_to_napcat,
    ImageDownloadError,
)
from .events import (
    build_group_notice_entry,
    build_recall_notice_entry,
    napcat_event_to_context,
    download_pending_images,
    expand_forward_previews,
    get_conversation_id,
    should_respond,
)
from .debug import napcat_event_to_debug_xml

__all__ = [
    "NapcatClient",
    "QQ_FACE",
    "napcat_segments_to_text",
    "build_content_segments",
    "get_reply_message_id",
    "llm_segments_to_napcat",
    "build_group_notice_entry",
    "build_recall_notice_entry",
    "napcat_event_to_context",
    "download_pending_images",
    "expand_forward_previews",
    "get_conversation_id",
    "should_respond",
    "napcat_event_to_debug_xml",
]
