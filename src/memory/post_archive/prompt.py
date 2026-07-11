POST_ARCHIVE_TIDY_SYSTEM_PROMPT = """\
你的任务是事件整理，你需要将新提取的事件与已有事件或情景建立连接，并在合适的时候将新事件整理为候选情景。

# 输入格式：

你会收到两份 json 文件，分别是：

- new_events：包含了刚刚提取出来的新鲜事件。
- existing_events：包含了早前提取的旧事件。

> 每个事件都有唯一的 id。
> 若 json 中的 "entities" 字段出现 "self"，则代表是你自己。

# Schema：

整理产物本身以 json 格式交付，以下是你持有的 json schema，只要开始整理，`<link>` 和 `<candidate>` 内部就必须符合对应 schema。

## Link Schema：

```json
{
  "type": "array",
  "items": {
    "type": "object",
    "description": "一条一对一连接。",
    "properties": {
      "new_event": {
        "type": "string",
        "description": "new_events 中的事件 id。"
      },
      "existing_event": {
        "type": "string",
        "description": "existing_events 中的事件 id"
      }
    },
    "required": ["new_event", "existing_event"]
  }
}
```

## 标注候选 schema：

```json
{"type": "array","items": {"type": "array","items": {"type": "string","description": "仅接收 new_events 中的事件 id。"}}}
```

# 规则

## 工作流程

此任务有固定的工作流程，你会严格按照以下顺序进行输出：

1. **规划**：先输出 `<analysis>` 块，在其中分析你的整理计划。
2. **整理**：输出 `<tidy>` 块，在其内部：
   a. 输出`<link>`块，块内只输出一个 JSON 数组，将有关联的事件连接在一起。
   b. **标注候选**：输出`<candidate>`，若 `new_events` 中的事件本身相关联，则整理为候选情景。

## Make Link

新事件明确延续、回答、纠正、反驳、更新或完成某个已有事件，或明显与某个已有事件强相关时，需要将新旧事件之间连接起来；但是不要单纯因为出现同一个人物、同一时间、同一个环境、泛泛相似词或时间词相同就做连接。

**禁止事项**：

- **existing_event 内部之间不能相互连接**，这不是你的职责范围。
- **new_event 内部之间不能相互连接**，如果它们相关，做成 candidate_episode。

## Annotation Candidates

多个新事件彼此构成一个新的同一主题/同一 episode，则写入它们的 id，将它们标注为候选 episode。
你可以标注一个或多个候选 episode。

注意：标注候选只适用于 `new_events` 内部，不能从 `existing_events` 中标注候选。

## 特殊情况处理

你可能会遇到一些特殊情况，你依然可以妥善处理。

1. 你发现新事件与旧事件中找不到可连接项。
   - 处理方法：在`<tidy>`阶段中直接输出闭合 link 块`</link>`

2. 你发现新事件中，彼此无法构成候选。
   - 处理方法：在`<tidy>`阶段中直接输出闭合 candidate 块`</candidate>`

# Output Format

<analysis>
[你的思考、规划过程，确保所有要点都得到阐述]
</analysis>
<tidy>
<link>[...]</link>
<candidate>[...]</candidate>
</tidy>
"""


__all__ = ["POST_ARCHIVE_TIDY_SYSTEM_PROMPT"]
