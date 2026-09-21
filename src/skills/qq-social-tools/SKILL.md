---
name: qq-social-tools
description: "QQ 会话中发送、引用、@、图片与表情包、撤回及会话切换的操作方法。"
---

## 发送与组合消息

`send_message` 向当前打开的 QQ 会话发送消息。`messages` 中每个元素是一条独立消息，按顺序发送；同一元素中的 `segments` 会拼成一条消息。

- 文字：`{"command":"text","content":"消息内容"}`。
- 引用：在消息元素的 `quote` 填入聊天记录中的目标消息 ID，不放进 `segments`。
- @：使用 `{"command":"at","user_id":"QQ号"}`；只有群聊支持，私聊和临时会话中含 `at` 的消息会失败。

例如，引用一条消息并发送两条独立文字消息：

```json
{"messages":[{"quote":"目标消息ID","segments":[{"command":"text","content":"第一条"}]},{"segments":[{"command":"text","content":"第二条"}]}]}
```

QQ 消息不渲染 Markdown，文本中的 Markdown 标记会作为原文显示。

## 图片与表情包

发送普通图片使用 `command: "image"`，以下来源只选一个：

- `image_ref`：上下文或工具结果中已有的图片引用。
- `path`：Agent Linux 中 `/home/agent` 下已有图片的绝对路径。
- `resource_ref`：`<browser><images>` 中的原图候选引用，发送时系统按需固化原图。

发送收藏表情包时，调用 `list_stickers` 获取列表中的 `image_ref`，再用 `{"command":"sticker","image_ref":"取得的引用"}` 作为消息片段。可见聊天图片的 `image_ref` 也可用于 sticker 片段。列表提供应用场景描述，支持视觉的模型还会收到标注引用的网格预览。

## 查找消息与切换会话

获取当前窗口之前的聊天记录使用 `scroll_chat_log`，从返回记录中取得引用或撤回所需的消息 ID。

向其他会话发送消息时，先用 `enter_qq_session` 进入目标会话，再发送。`type` 为 `group` 或 `private`，`id` 为群号或 QQ 号；临时会话也使用 `private`。目标 ID 可从会话列表查看或搜索取得。

切换与发送的执行顺序决定消息投递到哪个会话：先发送再切换，消息发往原会话；先切换再发送，消息发往目标会话。需要确定投递目标时，顺序调用这两个动作。

## 撤回与替换文字

`recall_message` 接收目标 `message_id`。只撤回时不填 `edited_text`；撤回后补发纯文本时填入 `edited_text`。这是撤回旧消息并发送新消息，不是原地修改。

通常可撤回自己两分钟内发送的消息；群管理员还可撤回普通成员的群消息，实际结果由 QQ 返回。