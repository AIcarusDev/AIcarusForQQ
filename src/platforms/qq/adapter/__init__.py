"""QQ adapter — QQ adapter WebSocket 连接层与消息翻译包

子模块：
  client   — QQAdapterClient（WebSocket 服务器，连接管理，API 调用）
  segments — 消息段格式互转（QQ adapter ↔ 文本/结构化/LLM segments）
  events   — 事件解析（→ context entry，should_respond，get_conversation_id）
  debug    — 调试 XML 生成
"""

from .client import QQAdapterClient
from . import segments as _segments
from . import events as _events
from .conversation import (
    is_temp_private_event,
    make_temp_session_key,
    parse_session_key,
)
from . import debug as _debug

QQ_FACE = _segments.QQ_FACE
qq_adapter_segments_to_text = _segments.qq_adapter_segments_to_text
build_content_segments = _segments.build_content_segments
get_reply_message_id = _segments.get_reply_message_id
llm_segments_to_qq_adapter = _segments.llm_segments_to_qq_adapter
ImageLoadError = _segments.ImageLoadError
build_group_notice_entry = _events.build_group_notice_entry
build_recall_notice_entry = _events.build_recall_notice_entry
qq_adapter_event_to_context = _events.qq_adapter_event_to_context
download_pending_images = _events.download_pending_images
expand_forward_previews = _events.expand_forward_previews
get_conversation_id = _events.get_conversation_id
should_respond = _events.should_respond
qq_adapter_event_to_debug_xml = _debug.qq_adapter_event_to_debug_xml

__all__ = [
    "QQAdapterClient",
    "QQ_FACE",
    "qq_adapter_segments_to_text",
    "build_content_segments",
    "get_reply_message_id",
    "llm_segments_to_qq_adapter",
    "ImageLoadError",
    "build_group_notice_entry",
    "build_recall_notice_entry",
    "qq_adapter_event_to_context",
    "download_pending_images",
    "expand_forward_previews",
    "get_conversation_id",
    "should_respond",
    "is_temp_private_event",
    "make_temp_session_key",
    "parse_session_key",
    "qq_adapter_event_to_debug_xml",
]

