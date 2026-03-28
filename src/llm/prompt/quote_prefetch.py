"""quote_prefetch.py — 引用消息预取

对上下文窗口外的引用消息，依次尝试：
  1. 查本地 DB（chat_messages 全局搜索）
  2. 查 NapCat get_msg API
缓存结果写入 session.quoted_extra，供 xml_builder 渲染正常预览。
全部找不到时留空，xml_builder 会输出 [ERROR: Message_lost]。
"""

import logging
from typing import Any

logger = logging.getLogger("AICQ.app")


async def prefetch_quoted_messages(session: Any, napcat_client: Any = None) -> None:
    """预取 session 上下文中所有窗口外引用消息，填入 session.quoted_extra。

    幂等：已在 quoted_extra 中的 ref_id 不重复查询。
    """
    from database import get_chat_message_by_id

    context_ids = {str(m.get("message_id", "")) for m in session.context_messages}
    needed = [
        str(msg["reply_to"])
        for msg in session.context_messages
        if msg.get("reply_to")
        and str(msg["reply_to"]) not in context_ids
        and str(msg["reply_to"]) not in session.quoted_extra
    ]
    if not needed:
        return

    for ref_id in needed:
        # ── 1. 先查 DB（跨所有 session）──────────────────────────
        entry = await get_chat_message_by_id(ref_id)
        if entry:
            session.quoted_extra[ref_id] = entry
            logger.debug(
                "[quote_prefetch] DB 命中 ref_id=%s sender=%s",
                ref_id, entry.get("sender_name"),
            )
            continue

        # ── 2. DB 未命中，尝试 NapCat get_msg ────────────────────
        if napcat_client is None or not napcat_client.connected:
            logger.debug("[quote_prefetch] NapCat 不可用，ref_id=%s 跳过", ref_id)
            continue

        try:
            msg_id_int = int(ref_id)
        except (ValueError, TypeError):
            logger.debug("[quote_prefetch] ref_id=%s 非整数，跳过", ref_id)
            continue

        try:
            msg_data = await napcat_client.send_api("get_msg", {"message_id": msg_id_int})
        except Exception as e:
            logger.warning("[quote_prefetch] NapCat get_msg 失败 ref_id=%s: %s", ref_id, e)
            continue

        if not msg_data:
            logger.debug("[quote_prefetch] NapCat 返回空 ref_id=%s", ref_id)
            continue

        sender = msg_data.get("sender", {})
        sender_name = (
            sender.get("card") or sender.get("nickname") or str(sender.get("user_id", "未知"))
        )
        segs = msg_data.get("message") or []
        from napcat.segments import napcat_segments_to_text
        content = napcat_segments_to_text(segs)
        session.quoted_extra[ref_id] = {
            "message_id": ref_id,
            "sender_name": sender_name,
            "content": content,
            "content_type": "text",
        }
        logger.debug(
            "[quote_prefetch] NapCat 命中 ref_id=%s sender=%s",
            ref_id, sender_name,
        )
