# Memory Consolidation LLM Concept Design

日期: 2026-07-08

## 目标

记忆整合 LLM 只负责辅助判断和生成 summary refresh 草稿，不拥有记忆事实写入权。SQLite Memory 仍是唯一事实来源，`MemoryEvents` 和 `MemoryParticipants` 不允许被整合流程修改或删除。

生产路径采用 entitySystem 参考实现中的 RNA-style online mount + sleep solidification:

1. 事件抽取阶段只输出 `<extract><event>{...}</event></extract>`，不输出 mount、不输出 cluster summary、不改变原有 prompt 解析契约。
2. 归档程序先把有效事件写入 `MemoryEvents` / `MemoryParticipants`。
3. 归档后第二步读取“本批刚写入事件 + 本轮召回候选关联的 cluster summary”，把相关性判断写成 `MemoryMounts(status='pending')`。
4. sleep 阶段读取 pending mounts、cluster summaries、已有 cluster/thread relations，产出可审计决策。
5. 只有显式 `solidify=true` 且 `dry_run=false` 时，程序逻辑才把 accepted mount 写入 `MemoryClusterRelations`、`MemoryThreadStates`，并把对应 summary 标记为 stale。
6. summary refresh 通过 `MemorySummaryInputs` 排队，后续单独生成新的 `MemorySummaryCache`。

核心约束: 相关性挂载不能挤进第一步抽取输出。这样即使第二步模型失败、card 缺失或候选不足，最多只是少一个 pending mount，不会影响事实事件抽取质量。

## LLM 边界

LLM 可以做:

- 在第二步对“新事件 atom vs 已召回 cluster summary”的挂载关系给出候选。
- 对 pending mount 的关系类型、置信度和不确定点给出建议。
- 对 correction/refutation 指出可能需要失效的旧关系候选。
- 对 thread state 的自然语言状态变化给出结构化建议。
- 为 summary refresh 输入草拟新的 cluster summary 内容。

LLM 不可以做:

- 要求事件抽取阶段输出 `memory_mount` 或 JSON bundle。
- 直接修改 `MemoryEvents`、`MemoryParticipants`、`MemoryEventSources`。
- 删除旧证据。
- 把 pending mount 放入正式召回。
- 在没有 source event id、anchor summary id、anchor revision 的情况下创造事实。
- 绕过程序侧阈值、revision 检查和 solidify 开关。

## 输入契约

每个整合任务至少包含:

- `task_id`: 本次整合任务 id。
- `anchor_summary`: cluster summary，包括 `summary_id`、`source_kind`、`source_id`、`revision`、`core_entities`、`open_slots`、`source_event_ids`。
- `mount`: pending mount，包括 `mount_id`、`new_event_id`、`anchor_summary_id`、`anchor_revision`、`relation_type`、`confidence`、`evidence_text`、`evidence_json`。
- `existing_relations`: anchor 下当前 active/weak/rejected relation 的简表。
- `source_events`: 仅包含必要事件摘要、事件类型、参与实体和 source ids。
- `policy`: 当前 `accept_threshold`、允许的 relation types、是否允许 solidify。

输入必须保留原始 id。LLM 的输出只能引用这些 id。

归档后第二步 mount task 的输入更窄:

- `new_atoms`: 本批刚写入事件的 `event_id`、`summary`、`event_type_norm`、参与实体、`occurred_at`。
- `candidate_cluster_summaries`: 由本轮召回候选 event id 映射出来的 ready cluster summary。
- `policy`: 允许的 relation types、每个 atom 的 mount 上限、是否只允许 pending。

该任务只允许产生 mount candidate；程序侧再校验 anchor revision、source ids 和 relation type 后写入 pending mount。

## Summary Refresh 窗口

summary refresh 不直接把所有事件簇成员全量塞给模型，而是先构造一个可控窗口:

1. `previous_cluster_summary_stale_prior`: 旧 cluster summary，放在任务前部，只作为待修订草稿。
2. `events`: 按 `occurred_at` 从旧到新排列，仿照聊天记录形式，让新证据靠后出现。
3. `relations`: 当前 cluster/thread 下 active、weak、rejected relation 简表。

事件窗口的选择和排列分开处理:

- 选择阶段优先保留本轮 accepted/weak mount 形成的新 delta 事件。
- 剩余名额按 activation score 选择旧事件；activation score 由新近度、`access_count`、`last_accessed`、`occurrences` 和 confidence 组合而成。
- 排列阶段不再按分数，而是按时间旧到新输出，避免 prompt 顺序打乱叙事。
- activation score 只表示“值得进入窗口”，不表示事实更可信；事实可信度仍由 status、confidence、correction/refutation relation 和 source evidence 决定。

窗口按预算截断，当前程序侧同时使用事件条数上限和近似 token budget；后续 summary worker 接入时应把近似估算替换成模型 tokenizer。

## 输出契约

LLM 输出应是结构化对象，概念字段如下:

- `decision`: `accept_attach`、`accept_with_uncertainty`、`revise_existing_relation`、`reject_background`、`reject_wrong_anchor`、`needs_human_review`。
- `relation_type`: 允许集合中的关系类型。
- `confidence`: 0 到 1。
- `required_source_event_ids`: 支撑该决策的事件 id 列表。
- `revised_relation_ids`: 被 correction/refutation 影响的旧关系 id。
- `stale_summary_ids`: 需要 refresh 的 summary id。
- `thread_state_patch`: 可选，包含新状态、milestones、open_slots。
- `uncertainty_reason`: 必填，当置信度不足或证据边界不清时说明原因。

程序侧必须重新校验:

- `anchor_revision` 是否仍匹配。
- 引用的 source ids 是否存在于输入。
- 关系类型是否在白名单中。
- `confidence >= accept_threshold` 才能成为 active，否则最多 weak。
- `solidify=true` 且 `dry_run=false` 才能写正式表。

## 失败与边界情况

- anchor summary 缺失或 revision 过期: mount 只能标记为 obsolete。
- relation 只是背景评论: 不能进入正式 recall，只能 reject 或保留 pending review。
- correction/refutation: 不删除旧关系，只把可被纠正的旧关系改为 rejected，并写 revision 证据。
- 多个 mount 指向同一 anchor: 先处理 correction/refutation，再处理 progress/update/background。
- pending mount: 不参与正式召回，只能作为 sleep 输入。
- LLM 输出无法解析或引用未知 id: 整个任务保持 pending 或进入 review，不写正式表。

## 配置

WebUI 中的 `memory.consolidation` 只配置整合模型和运行开关:

- `enabled`: 是否启用专用 LLM 配置。
- `dry_run`: 默认 true，只生成审计结果。
- `solidify`: 默认 false，必须显式打开才允许写正式整合表。
- `max_mounts_per_sleep`: 每次 sleep 处理的 mount 上限。
- `accept_threshold`: 程序侧接受阈值。
- `provider` / `model` / `generation`: 专用模型绑定。

prompt 文本、few-shot 样例和模型输出修复策略不在本设计范围内。
