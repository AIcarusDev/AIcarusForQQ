ARRAY_DESCRIPTION = """
向当前打开的会话窗口发送一条或多条消息。
"messages" 参数是一个列表，每个列表项都是一条独立消息，会按顺序依次发送。
每条消息内部的 "segments" 字段是内容片段列表，用于将文字、@某人、表情包、图片等不同类型片段拼合为单条消息发送。

注意：
  - 同一条消息内的多个 segment 只会被拼接为一条消息，并不会变成多条。若要发送多条独立消息，请在 messages 数组中添加多个元素。
  - 私聊和临时会话无法发送 @某人（at）片段。当前会话是私聊/临时会话时，如果某条消息包含 at，该条消息会发送失败。
  - 消息会发送到当前会话，如果你想回应的是其它会话的未读消息，需先 shift 到指定会话。
"""


SINGLE_DESCRIPTION = """
向当前打开的会话窗口发送一条消息。
内部的 "segments" 字段是内容片段列表，用于将文字、@某人、表情包、图片等不同类型片段拼合为单条消息发送。
如需发送多条消息，按顺序多次调用该工具即可。

注意：
  - 私聊和临时会话无法发送 @某人（at）片段。当前会话是私聊/临时会话时，如果消息包含 at，会发送失败。
  - 消息会发送到当前会话，如果你想回应的是其它会话的未读消息，需先 shift 到指定会话。
"""


DESCRIPTION = ARRAY_DESCRIPTION


def get_description(message_shape: str) -> str:
    return SINGLE_DESCRIPTION if message_shape == "single" else ARRAY_DESCRIPTION
