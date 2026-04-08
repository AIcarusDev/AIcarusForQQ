DESCRIPTION = """
向当前打开的会话窗口发送一条或多条消息。
"messages" 参数是一个列表，每个列表项（item）都是一条独立的消息对象。
用法：
  - messages 数组中的每一个元素代表一条独立发送的消息，会按顺序依次发送。
  - 支持一次调用发送多条消息（即 messages 中包含多个元素）。
  - 每条消息内部的 "segments" 字段是内容片段列表，用于将文字、@某人、表情包等**不同类型**的片段拼合为单条消息发送。
  - 注意：同一条消息内的多个 segment 只会被拼接为一条消息，并不会变成多条。若要发送多条独立消息，请在 messages 数组中添加多个元素，而不是在同一消息内堆叠多个  segment。

建议：
  - 将长消息拆分为多条短消息发送。
  - 每条消息可以非常简短（甚至 5 个字以下），且不需要严谨的标点符号（例如句号结尾）。

示例 — 将回复拆成三条分开发送（正确做法）：
```json
{
  "motivation": "分条回复用户的问题",
  "messages": [
    { "segments": [{ "command": "text", "params": { "content": "哦对了" } }] },
    { "segments": [{ "command": "text", "params": { "content": "这个问题我之前想过" } }] },
    { "segments": [{ "command": "text", "params": { "content": "答案是42" } }] }
  ]
}
```
*这样将会依次发出："哦对了"、"这个问题我之前想过"、"答案是42"三条消息*√

示例 — 在同一条消息里 @ 某人并附文字（同一条消息内拼合片段）：
```json
{
  "motivation": "提醒某人",
  "messages": [
    {
      "segments": [
        { "command": "at",   "params": { "user_id": "12345678" } },
        { "command": "text", "params": { "content": " 记得看一下这个" } }
      ]
    }
  ]
}
```
*这样将会发出："@小明 记得看一下这个"一条消息（假设用户名称为小明）*√

反例 — 把多条内容堆在同一消息的多个 text segment（错误，只会合并成一条）：
```json
{
  "messages": [
    {
      "segments": [
        { "command": "text", "params": { "content": "第一句话" } },
        { "command": "text", "params": { "content": "第二句话" } }
      ]
    }
  ]
}
```
*这样将会发出："第一句话第二句话"一条消息，通常情况下是不符合预期的*×
"""