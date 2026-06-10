DESCRIPTION = """
向当前打开的会话窗口发送一条或多条消息。
"messages" 参数是一个列表，每个列表项（item）都是一条独立的消息对象。
用法：
  - messages 数组中的每一个元素代表一条独立发送的消息，会按顺序依次发送。
  - 支持一次调用发送多条消息（即 messages 中包含多个元素）。
  - 每条消息内部的 "segments" 字段是内容片段列表，用于将文字、@某人、表情包、图片等**不同类型**的片段拼合为单条消息发送。

图片发送：
  - 可用 command="image" 且 params.url 为 HTTP/HTTPS 图片直链。
  - 更推荐使用 <world><browser> 中的 image 或 viewport_image ref：
      {"command":"image","params":{"image_ref":"3a686ed196bf"}}
    这会直接发送浏览器实际加载并缓存到本地的图片字节，避免防盗链、cookie、Referer 或二次下载失败。
  - 对 Pixiv、Pinterest 等图片站，优先用 browser_control 打开页面、滚动/点击确认内容，再从下一轮 <browser> 状态中选 image_ref 发送。

注意：
  - 同一条消息内的多个 segment 只会被拼接为一条消息，并不会变成多条。若要发送多条独立消息，请在 messages 数组中添加多个元素，而不是在同一消息内堆叠多个 segment。
  - 私聊和临时会话无法发送 @某人（at）片段。当前会话是私聊/临时会话时，如果某条消息包含 at，该条消息会发送失败。
  - 消息会发送到**当前会话**，如果你想回应的是其它会话的未读消息，需先 shift 到指定会话。
"""
