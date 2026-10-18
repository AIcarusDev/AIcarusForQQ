ARRAY_DESCRIPTION = """
向当前打开的会话窗口发送一条或多条消息。
messages 按顺序发送，每项中的 segments 拼成一条消息。
默认按正常聊天节奏模拟打字；整段内容是从别处原样粘贴的（如日志、新闻、代码）时可传 simulate_typing=false，跳过模拟打字、按顺序立即发送。
私聊和临时会话不支持 at 片段；图片的 image_ref、path、resource_ref 只能选一个。
原生 QQ 表情使用 face 片段和 list_faces 查到的 id。
"""


DESCRIPTION = ARRAY_DESCRIPTION


def get_description() -> str:
    return ARRAY_DESCRIPTION

