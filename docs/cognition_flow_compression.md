# Cognition Flow Compression Draft

本文档描述一种用于认知流的异步滚动压缩机制。目标不是把无限历史原样塞进主模型上下文，而是在有限窗口内保持最近认知的精确性，同时用连续摘要承接已经离开窗口的旧认知。

## 背景

当前主模型可见的意识流由 `ConsciousnessFlow.to_xml_messages()` 生成，并受 `generation.llm_contents_max_rounds` 限制。超过窗口的旧轮次会从主模型上下文中移除，但原始轮次仍可通过 `bot_turns` 等日志持久化。

这意味着系统已经有原始认知的持久化基础，但缺少一个模型可见的、连续更新的压缩层。

## 核心目标

- 主 agent 不因压缩而阻塞。
- 最近若干轮认知保持原文，避免摘要损失短期细节。
- 已经稳定、不再变化的旧轮次可以提前压缩。
- 摘要只在其覆盖的原文即将或已经离开主上下文时生效，避免原文和摘要重复污染 prompt。
- 下一次压缩必须接收上一次摘要与新增封存轮次，形成连续检查点。
- 压缩落后时允许主 agent 正常裁剪；压缩 worker 后台追赶，完成后再应用摘要。

## 术语

- `raw round`: 一轮完整的 `FlowRound`，包含 cognition、tool calls、tool responses 和 timestamp。
- `sealed round`: 已完成且不会再被修改的 raw round。未完成的 deferred tool 轮次不应被压缩。
- `hot window`: 主模型当前直接可见的最近 raw rounds。
- `ready summary`: 后台已经生成、可供后续接替旧 raw rounds，但尚未注入 prompt 的摘要。
- `active summary`: 已经接替旧 raw rounds、会进入主模型 prompt 的连续摘要。
- `coverage`: 摘要覆盖的原始轮次范围，例如 `round_seq=1..5`。
- `compactor`: 异步压缩 worker。

## 基本不变量

1. 压缩永远不是主 agent 的同步前置条件。
2. 主 agent 到达 `max_rounds` 时可以照常裁剪旧 raw rounds。
3. 摘要只能覆盖 sealed rounds。
4. 同一段历史在 prompt 中只能出现一次：要么是 raw rounds，要么是 active summary。
5. ready summary 不进入 prompt；只有当 raw window 即将超过 `llm_contents_max_rounds` 时，才提升最早一个足够让窗口回到上限内的 ready summary。
6. 新摘要必须基于上一份 active summary 加上新增 sealed rounds，而不是只摘要新增片段。
7. active summary 表达的是机器人主观认知连续性，不把未经验证的主观判断升级为客观事实。
8. 原始 raw rounds 永远保留在持久日志中，摘要必须能追溯到来源范围。

## 示例时序

假设 `max_rounds = 8`，`compress_watermark = 5`。

```text
T1:
hot raw: 1 2 3 4 5
compactor 生成 pending_summary(1..5)
prompt: raw 1..5

T2:
hot raw: 1 2 3 4 5 6 7 8
pending_summary(1..5) 已存在，但仍不重复注入
prompt: raw 1..8

T3:
第 9 轮到来，主 agent 需要裁剪
active_summary = pending_summary(1..5)
hot raw: 6 7 8 9
prompt: summary(1..5) + raw 6..9

T4:
hot raw 累积到 6 7 8 9 10
compactor 生成 pending_summary(1..10)
输入为: active_summary(1..5) + raw 6..10

T5:
下一次需要裁剪时
active_summary = pending_summary(1..10)
hot raw: 11 ...
prompt: summary(1..10) + raw 11...
```

## 异步追赶语义

压缩可能慢于主 agent。此时主 agent 不等待压缩结果：

```text
hot raw: 1 2 3 4 5 6 7 8
pending_summary(1..5) 尚未完成

第 9 轮到来:
主 agent 照常裁剪旧 raw
hot raw: 2 3 4 5 6 7 8 9 或按当前策略保留尾部窗口
active_summary 暂时不变

稍后 compactor 完成 pending_summary(1..5):
如果 coverage 仍然有效，提升为 active_summary
prompt 之后变为 summary(1..5) + 当前 hot raw 中未被覆盖的部分
```

极端情况下，如果 compactor 长时间落后，它仍然按持久日志中的顺序追赶。主 agent 总会出现 sleep/wait 或低活动期，compactor 可以在后台补齐缺口。系统退化结果是“暂时只有短期窗口”，而不是阻塞主 agent。

## 推荐状态机

```text
raw sealed rounds written
        |
        v
eligible segment found
        |
        v
pending compression job
        |
        | success
        v
ready_summary stored
        |
        | raw prompt window would exceed llm_contents_max_rounds
        v
active_summary promoted
        |
        v
next compression uses active_summary + newly sealed raw rounds
```

失败或超时不改变 active summary。重复调度同一 coverage 应保持幂等。

## Prompt 形态

注入主模型时使用独立 `<summary>` 块，而不是混进现有 `<memory>`：

```xml
<summary>
...
</summary>
```

原因：

- `<memory>` 当前偏向事件召回，语义上接近长期记忆。
- 认知连续摘要是机器人自己的主观状态检查点，应该和客观世界、聊天事实、事件记忆分开。
- coverage、更新时间和压缩状态属于运行时控制/调试信息，不需要暴露给主模型。
- 分开后更容易做开关、token 预算和调试展示。

## 摘要内容建议

摘要应保留：

- 当前仍未解决的意图、目标、承诺。
- 对用户、群聊、当前任务的主观理解变化。
- 工具结果带来的重要观察。
- 自我修正、策略改变、风险判断。
- 需要后续验证的不确定判断。
- 最近情绪或倾向中确实影响后续行为的部分。

摘要应丢弃：

- “继续观察”“没有新消息”“等待中”等重复空转文本。
- 已被后续轮次明确解决且无长期影响的瞬时计划。
- 不可追溯、无证据的事实断言。
- 仅用于当轮工具调用格式的临时推理。

## 持久化草案

可新增一个轻量表或 JSON 状态。表结构暂定：

```text
cognition_summaries
- id
- status: pending | active | superseded | failed
- coverage_start_seq
- coverage_end_seq
- source_turn_ids_json
- input_active_summary_id
- summary_text
- summary_json
- created_at
- updated_at
- model_provider
- model_name
- error
```

另需给 raw rounds 建立稳定序号。可以从 `bot_turns.created_at` 派生，但更稳的是在意识流写入时记录单调递增的 `round_seq`。

## 调度策略

配置项先落在 `generation` 下，和现有意识流回放窗口同步校验：

```yaml
generation:
  llm_contents_max_rounds: 10              # 最小 6
  cognition_compression_trigger_rounds: 5  # 最小 3，且必须小于 llm_contents_max_rounds
```

调度条件：

- active summary 之后的新 sealed rounds 数量达到 `cognition_compression_trigger_rounds`。
- 或当前 active summary 后的未压缩 sealed rounds 达到阈值。
- 或启动恢复时发现存在缺口。
- 单次压缩任务只冻结 `cognition_compression_trigger_rounds` 个 turn；如果仍有足够未压缩轮次，由 worker 串行继续追赶，避免一次任务输入数量漂移。

提升条件：

- ready summary 已完成。
- active summary 之后的 raw prompt window 即将超过 `llm_contents_max_rounds`。
- 存在 coverage 大于当前 active summary 的 ready summary。
- 选择最早一个足够让 raw prompt window 回到上限内的 ready summary，而不是总是选择最新 summary。

## 与现有代码的接入点

- `src/consciousness/flow.py`: 保持 hot raw window 和 XML 回放职责。
- `src/consciousness/main_loop.py`: 每轮持久化后只负责通知或调度 compactor，不等待结果。
- `src/database.py`: 保存 raw round 序号、summary 状态和 coverage。
- `src/llm/prompt/user_prompt_builder.py`: 在 `<memory>` 附近注入 active cognition continuity。
- `src/templates/focus.html` 或日志页：展示 active summary、pending summary 和覆盖范围，方便调试。

## 风险

- 摘要漂移：多次 summary-of-summary 后可能丢细节或改写语义。
- 事实污染：认知里的猜测被摘要写成事实。
- 工具状态丢失：自然语言摘要不能替代工具调用历史中的运行时状态。
- 覆盖错位：summary 覆盖范围和 raw prompt 裁剪范围不一致会导致重复或缺口。
- 压缩模型失败：需要保证失败不影响主 agent。

## 初步结论

这个机制成立。它应被视为“意识流连续压缩检查点”，而不是普通长期记忆，也不是同步上下文压缩。

最关键的设计点是：压缩提前做，应用延后做；压缩可落后，主 agent 不等待；下一次压缩沿用上一份摘要，保证连续性。
