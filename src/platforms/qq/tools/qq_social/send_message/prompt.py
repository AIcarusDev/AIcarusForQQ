ARRAY_DESCRIPTION = """
向当前打开的会话窗口发送一条或多条消息。
messages 按顺序发送，每项中的 segments 拼成一条消息。
私聊和临时会话不支持 at 片段；图片的 image_ref、path、resource_ref 只能选一个。
"""


DESCRIPTION = ARRAY_DESCRIPTION


def get_description(message_shape: str = "array") -> str:
    return ARRAY_DESCRIPTION

