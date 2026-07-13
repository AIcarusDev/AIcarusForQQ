COMPRESSION_PROMPT_SYS_TEMPLATE = """
你的任务是上下文压缩，把已经发生的，来自你自己的动作、动机压缩成一份自然，流畅的`<summary>`摘要，供自己读取。

## 输入

user 消息是一个 `<compression_input>`：

- `generated_at`：本次压缩快照生成时的本地 ISO 8601 时间。
- `previous_summary`：更早历史的上一份摘要；为空表示首次压缩。
- `cycle`：按时间顺序排列的行动周期，`start_at` 是模型请求时间，`end_at` 是该周期全部动作结束的时间。
- `motive`：我发起本轮行动的原因。
- `action`：我尝试执行的全部工具调用。
- `action_response`：行动反馈；`result` 是工具结果，`feedback` 是格式或系统反馈。

## 压缩规则

1. 以 `previous_summary` 为底稿，融合全部 `cycle`，重新生成一份完整摘要。只有最新的摘要会被保留。
2. `motive` 只说明行动原因，`action` 只说明尝试做了什么；是否成功以及产生了什么，以 `action_response` 为准。
3. 只记录输入中能够确定的信息。发生冲突时，以时间更晚、结果更明确的内容为准。
4. 使用第一人称“我”，保留对后续理解有用的事实、决定、关系、状态变化和未完成事项，不额外制定计划或给出建议。
5. 重复等待、空转、普通成功回执和无后续价值的细节可以合并或省略。
6. 时间确有价值时，依据 `generated_at`、`start_at` 和 `end_at` 写成绝对时间。
7. `<summary>` 应自然、连贯、非空，理想长度在 1000 字以内。没有重要新增时，保留 `previous_summary` 中仍然有效的内容。

## 输出

只返回以下两个 XML 块：

<analysis>
用一两句话分析本次保留、合并或省略的重点。
</analysis>

<summary>
以第一人称写成的、自然流畅的完整的摘要。
</summary>
"""
