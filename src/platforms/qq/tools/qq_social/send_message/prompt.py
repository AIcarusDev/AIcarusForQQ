ARRAY_DESCRIPTION = """
向当前打开的会话窗口发送一条或多条消息。
"messages" 参数是一个列表，每个列表项都是一条独立消息，会按顺序依次发送。
每条消息内部的 "segments" 字段是内容片段列表，用于将文字、@某人、表情包、图片等不同类型片段拼合为单条消息发送。
图片 segment 可以直接填写 image_ref；Linux 中已有图片的 path（须位于 /home/agent 下）；或 resource_ref。

注意：
  - 同一条消息内的多个 segment 只会被拼接为一条消息，并不会变成多条。若要发送多条独立消息，请在 messages 数组中添加多个元素。
  - 私聊和临时会话无法发送 @某人（at）片段。当前会话是私聊/临时会话时，如果某条消息包含 at，该条消息会发送失败。
  - 发送图片时，image_ref、path、resource_ref 三者只能选一个，不能同时使用。
  - 消息会发送到当前会话，如果需要向其它窗口发送消息，需先 enter_qq_session 进入目标会话。
"""


DESCRIPTION = ARRAY_DESCRIPTION


def get_description(message_shape: str = "array") -> str:
    return ARRAY_DESCRIPTION

