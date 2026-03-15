"""xml_builder.py — 聊天记录 → XML 转化

将内部上下文消息列表转为 XML 格式，供 LLM 上下文使用。
未来也可用于导出完整的聊天记录存档。
"""

import html


def build_chat_log_xml(context_messages: list[dict]) -> str:
    """将上下文消息列表转为 XML 字符串。

    每条消息格式:
        {"message_id", "sender_id", "sender_name", "timestamp", "content"}

    输出示例:
        <chat_log>
          <message id="msg_001" sender_id="10001"
                   sender_name="张三" timestamp="...">
            早上好各位
          </message>
        </chat_log>
    """
    if not context_messages:
        return "<chat_log>\n</chat_log>"

    lines = ["<chat_log>"]
    for msg in context_messages:
        safe_content = html.escape(msg["content"], quote=False)
        safe_name = html.escape(msg["sender_name"])
        lines.append(
            f'  <message id="{msg["message_id"]}" '
            f'sender_id="{msg["sender_id"]}" '
            f'sender_name="{safe_name}" '
            f'timestamp="{msg["timestamp"]}">'
        )
        lines.append(f"    {safe_content}")
        lines.append("  </message>")
    lines.append("</chat_log>")
    return "\n".join(lines)
