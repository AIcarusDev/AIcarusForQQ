"""napcat/debug.py — NapCat 调试辅助

生成调试用 XML，展示原始 NapCat 事件结构和 LLM 视角的 context entry。
"""

import html as html_mod
import logging
from datetime import datetime
from typing import Any

from .events import napcat_event_to_context

logger = logging.getLogger("AICQ.napcat")


async def napcat_event_to_debug_xml(
    event: dict,
    bot_id: str | None = None,
    timezone: Any = None,
) -> str:
    """将 NapCat 消息事件转为完整的调试 XML，展示原始结构和 LLM 视角。"""
    esc = html_mod.escape

    from zoneinfo import ZoneInfo
    tz = timezone or ZoneInfo("Asia/Shanghai")
    ts = datetime.fromtimestamp(event.get("time", 0), tz=tz).isoformat()

    msg_type = event.get("message_type", "unknown")
    msg_id = str(event.get("message_id", ""))
    sender = event.get("sender", {})
    sender_id = str(sender.get("user_id", "unknown"))
    nickname = esc(sender.get("nickname", ""))
    card = esc(sender.get("card", ""))
    group_id = str(event.get("group_id", "")) if msg_type == "group" else ""

    lines = [
        f'<napcat_event type="message" message_type="{esc(msg_type)}" timestamp="{esc(ts)}">',
        "  <source>",
    ]
    if msg_type == "group" and group_id:
        lines.append(f'    <group id="{esc(group_id)}" />')
    lines.append(
        f'    <sender id="{esc(sender_id)}" nickname="{nickname}" card="{card}" />'
    )
    lines.append("  </source>")

    # 原始消息段
    lines.append(f'  <raw_message id="{esc(msg_id)}">')
    for seg in event.get("message", []):
        seg_type = seg.get("type", "")
        data = seg.get("data", {})
        attrs = " ".join(f'{esc(k)}="{esc(str(v))}"' for k, v in data.items())
        if seg_type == "text":
            lines.append(f"    <segment type=\"text\">{esc(data.get('text', ''))}</segment>")
        else:
            lines.append(f"    <segment type=\"{esc(seg_type)}\" {attrs} />")
    lines.append("  </raw_message>")

    # LLM 看到的 context_entry 视角
    ctx = await napcat_event_to_context(event, bot_id=bot_id, timezone=timezone)
    lines.append("  <context_entry>")
    if ctx:
        safe_name = esc(ctx["sender_name"])
        safe_content = esc(ctx["content"], quote=False)
        lines.append(
            f'    <message id="{ctx["message_id"]}" '
            f'sender_id="{ctx["sender_id"]}" '
            f'sender_name="{safe_name}" '
            f'timestamp="{ctx["timestamp"]}">'
        )
        lines.append(f"      {safe_content}")
        lines.append("    </message>")
    else:
        lines.append("    <!-- 该事件未产生有效 context entry（可能内容为空） -->")
    lines.append("  </context_entry>")

    lines.append("</napcat_event>")
    return "\n".join(lines)
