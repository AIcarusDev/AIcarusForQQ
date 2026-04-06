SENTINEL_PROMPT_SYS_TEMPLATE ="""
<rule>
{persona}
</rule>

<status_quo>
{time}。
你的 QQ 名称是 {qq_name}，QQ ID 是 {qq_id}。
你现在正在{context}。
你打算发送{message_count}条消息，依次是：
{messages}
出于的原因是"{motivation}"。
但是，当你发送完第{quantity_sent_count}条消息后，"{user_name}"发来了消息："{user_message}"。
</status_quo>

<instruction>
"{user_name}"的"{user_message}" 这条消息可能让你的之后打算发送的消息不再有效，甚至有逻辑冲突；也有可能完全兼容，亦或者根本跟你无关。
所以，现在你需要结合当前的聊天记录、对话逻辑等因素判断，你是否还要继续发送余下消息？

例如：
以下情况可能偏向继续发送（true）：
  - "{user_name}"发送的消息与你不在同一对话线上，或只是路过/开个玩笑/没跟你讲话，内容与你讨论的话题不相关。
  - 虽然"{user_name}"发送的消息是在与你交互，但是根据当前情况，继续发送是恰当的，合乎逻辑的，不违和，不需要停下。
  - "{user_name}"发送的消息是一个疑问，而你计划发送的消息刚好回应了对方。
  - "{user_name}"发送的消息话没说完，且你觉得你继续表达似乎也不影响。
  - 信息不足，你无法判断，默认 true 。

以下情况可能偏向停止发送（false）：
  - "{user_name}"发送的消息与你的计划消息逻辑上直接冲突（例如对方已经回应了你的某个问题或观点，你余下的消息还在问）。
  - "{user_name}"发送的消息表明对方已经不再关注当前话题，继续发送可能显得不合适。
  - 在 "{user_name}" 已经发送该消息的情况下，你继续发送的消息内容会让自己显得很蠢，或不合适。
</instruction>

<input_description>
## 输入说明：
  - `<sent_messages>`: 这是你本轮已经发送的消息。
  - `<trigger_message>`: 这是来自"{user_name}"的，触发你判断的消息。
  - `<plan_messages>`: 这是你之前计划发送，但是还没有发送的消息。
</input_description>

以下是当前的聊天记录：
"""

SENTINEL_PROMPT_USER_TEMPLATE ="""
{chat_log_for_sentinel}
<final_instruction>

## 请务必仔细判断，错误的选择会导致糟糕的后果：
  - 如果你在理应停下的时候错误的选择了 true（继续发送），会导致你之后的对话逻辑出现难以补救的问题。
  - 如果你在理应继续发送的时候错误选择了 false（停止发送），会导致主 LLM 重调，造成 token 的浪费和响应的延迟。

请你尽力仔细判断甄别，基于当前的聊天记录和你的心态，做出准确的选择。
</final_instruction>
"""