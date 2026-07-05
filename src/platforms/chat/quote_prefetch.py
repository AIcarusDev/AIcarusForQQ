"""quote_prefetch.py — chat quote prefetch support.

对上下文窗口外的引用消息，依次尝试：
  1. 查本地 DB（chat_messages 全局搜索）
  2. 调用平台提供的缺失引用补查回调
缓存结果写入 session.quoted_extra，供 xml_builder 渲染正常预览。
全部找不到时留空，xml_builder 会输出 [ERROR: Message_lost]。
"""

import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger("AICQ.platforms.chat.quote_prefetch")

QuoteFetcher = Callable[[str], Awaitable[dict | None]]


async def prefetch_quoted_messages(
    session: Any,
    fetch_missing: QuoteFetcher | None = None,
) -> None:
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

        # ── 2. DB 未命中，交给平台补查 ───────────────────────────────
        if fetch_missing is None:
            continue

        try:
            entry = await fetch_missing(ref_id)
        except Exception as e:
            logger.warning("[quote_prefetch] 平台补查失败 ref_id=%s: %s", ref_id, e)
            continue

        if entry:
            session.quoted_extra[ref_id] = entry
            logger.debug(
                "[quote_prefetch] 平台补查命中 ref_id=%s sender=%s",
                ref_id, entry.get("sender_name"),
            )


