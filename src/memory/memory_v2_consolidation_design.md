# Memory V2 整理层设计文档

## 1. 定位

Memory V2 整理层是 sleep 动作期间运行的慢速记忆整理系统。

它不属于抽取侧，也不属于主循环实时召回侧。它的任务是在不修改原始抽取结果的前提下，对长期记忆中的事件、实体提及、值文本、谓词和时间关系进行整理，生成更适合主循环使用的结构。

整理层服务的核心目标是：

1. 减少长期记忆碎片化。
2. 发现重复、相似、演化、冲突和别名关系。
3. 将零散事件沉淀成更稳定的语义记忆。
4. 保留原始记忆和认知变化轨迹。
5. 保证整理动作可追溯、可撤销、可重新计算。

Memory V2 不是一个绝对正确的知识图库，而是一个随时间演进的认知数据库。因此整理层不能以“删除旧信息、只留下最新正确答案”为目标。旧记忆是认知历史的一部分，变化本身也应该成为记忆结构。

## 2. 基本原则

### 2.1 不修改抽取侧 prompt

抽取侧继续开放抽取：

```text
MemoryV2Events
MemoryV2Participants.entity
MemoryV2Participants.value_text
MemoryV2Events.event_type
```

整理层不要求抽取侧输出固定实体类型、固定角色名、固定谓词集合或稳定格式。

例如，抽取侧可能输出：

```text
Platform:星のgalgame资源社
Group:星のgalgame资源社
Location:星のgalgame资源社
星のgalgame资源社
QQ群里的星社
```

整理层必须接受这种开放、漂移、不稳定的输入。

### 2.2 原始记忆不物理删除

整理层默认不删除原始事件，不直接改写原始 participant，不覆盖原始 entity 或 value_text。

整理动作通过新增结构表达：

1. 新增事件簇。
2. 新增整理决策。
3. 新增事件关系。
4. 新增规范实体映射。
5. 新增抽象事件或变化事件。

原始事件始终保留，用作证据、历史和回滚基础。

### 2.3 整理动作默认可逆

整理层的所有合并、映射、抽象、冲突判断都必须通过 decision 记录。

如果后续发现整理错误，应通过撤销 decision 实现回滚，而不是尝试恢复已被破坏的原始数据。

### 2.4 不预设封闭本体

整理层不应该硬编码：

```text
Person
Group
Platform
Location
Tool
```

这些词可以作为文本特征或弱证据，但不能作为基础本体规则。

同理，数字片段、括号 ID、QQ 群号、平台名称也只能作为证据，不能直接决定合并。

### 2.5 使用方式比类型标签更重要

整理层判断两个节点是否相似，不应只看名字或前缀，而应看它们在记忆图中的使用方式。

核心思想：

```text
不要先定义它是什么；
先观察它如何被使用。
```

这称为“节点使用侧写”。

## 3. 专名说明

本文使用以下术语：

1. **整理层**  
   sleep 期间运行的记忆整理系统，用于合并、抽象、建立关系和修正召回结构。

2. **原始事件**  
   抽取侧直接写入的 `MemoryV2Events`。它记录当时模型从上下文中抽到的记忆。

3. **事件簇**  
   一组相关事件。它们可能是重复观察、同一主题的证据、同一认知变化链，或围绕同一对象的经历。

4. **规范实体**  
   整理层认为多个原始实体提及可能指向的同一个对象。规范实体不替换原始 entity，只作为召回和整理时的映射目标。

5. **原始提及**  
   来自 `entity` 或 `value_text` 的原始文本节点。它不一定是干净实体，也可能是一段描述。

6. **节点使用侧写**  
   一个节点在记忆图中出现时积累的使用模式，例如常见角色、常见谓词、共现对象、上下文分布、时间分布和相关摘要向量。

7. **整合决策**  
   整理层对候选关系做出的判断，例如重复、抽象、变化、冲突、别名、无需处理。整合决策必须可追溯、可撤销。

8. **软合并**  
   不删除原始数据，只建立“这些记忆目前被视为一组”的关系。Memory V2 默认只做软合并。

9. **硬合并**  
   物理删除或改写原始数据。Memory V2 整理层第一阶段不做硬合并。

10. **泼溅逻辑**  
    从一个节点向周围事件、角色、上下文、邻居和值文本扩散，形成节点使用侧写，再比较两个侧写是否相似。这里迁移的是“通过周围证据判断相似”的思想，而不是侧写档案系统的表结构。

## 4. 整理层输入

整理层主要读取以下数据：

1. `MemoryV2Events`
   - `summary`
   - `event_type`
   - `event_type_norm`
   - `status`
   - `is_negated`
   - `confidence`
   - `occurred_at`
   - `created_at`
   - `last_seen_at`
   - `occurrences`
   - `conv_type`
   - `conv_id`
   - `conv_name`
   - `raw_event_json`

2. `MemoryV2Participants`
   - `role`
   - `entity`
   - `value_text`
   - `value_tok`
   - `raw_participant_json`

3. `MemoryV2Predicates`
   - `event_type_norm`
   - `display_event_type`
   - `occurrences`

4. `MemoryV2EventSources`
   - `source_uid`
   - `source_id`
   - `source_timestamp`

5. `MemoryV2Vectors`
   - summary vector
   - predicate vector

整理层不要求所有数据都完整。缺失向量时可以退回文本、角色分布和上下文分布。

## 5. 整理层输出

整理层不改写原始抽取结果，而是新增以下结构。

### 5.1 整合运行记录

用于记录一次 sleep 整理任务。

建议表：

```text
MemoryV2ConsolidationRuns
```

字段建议：

```text
run_id
started_at
finished_at
status
trigger_reason
input_event_min_id
input_event_max_id
new_event_count
candidate_count
decision_count
error
metadata_json
```

用途：

1. 追踪每次 sleep 整理运行。
2. 允许中断后恢复或跳过已处理范围。
3. 便于排查整理行为。

### 5.2 整合候选

用于记录系统发现的“可能需要整理”的对象。

建议表：

```text
MemoryV2ConsolidationCandidates
```

字段建议：

```text
candidate_id
run_id
candidate_type
subject_json
evidence_json
score
status
created_at
```

`candidate_type` 示例：

```text
duplicate_event
semantic_duplicate_event
abstraction_cluster
evolution_chain
conflict_pair
entity_alias
predicate_cluster
low_value_trace_cluster
```

候选不是决策，只表示“值得进一步判断”。

### 5.3 整合决策

用于记录整理层最终采用或暂缓的判断。

建议表：

```text
MemoryV2ConsolidationDecisions
```

字段建议：

```text
decision_id
run_id
candidate_id
decision_type
action
confidence
reason
input_json
output_json
applied
revoked
revoked_at
revoked_reason
created_at
```

`decision_type` 示例：

```text
duplicate
abstract
evolved
corrected
conflict
alias
predicate_similar
ignore
defer
```

`action` 示例：

```text
create_cluster
create_abstract_event
create_evolution_event
create_relation
create_canonical_map
mark_low_priority
no_action
```

### 5.4 事件簇

用于表达一组事件之间的整理关系。

建议表：

```text
MemoryV2Clusters
MemoryV2ClusterMembers
```

`MemoryV2Clusters` 字段建议：

```text
cluster_id
cluster_type
title
summary
canonical_event_id
created_by_decision_id
status
created_at
updated_at
```

`cluster_type` 示例：

```text
duplicate
topic
evolution
conflict
evidence
low_value_trace
```

`MemoryV2ClusterMembers` 字段建议：

```text
cluster_member_id
cluster_id
event_id
member_role
weight
created_at
```

`member_role` 示例：

```text
original
supporting_evidence
old_state
new_state
conflicting_claim
canonical
low_priority_trace
```

### 5.5 规范实体映射

用于表达开放抽取出来的多个原始提及可能指向同一个对象。

建议表：

```text
MemoryV2CanonicalEntities
MemoryV2EntityCanonicalMap
```

`MemoryV2CanonicalEntities` 字段建议：

```text
canonical_entity_id
canonical_key
display_name
summary
created_by_decision_id
status
created_at
updated_at
```

`MemoryV2EntityCanonicalMap` 字段建议：

```text
map_id
raw_text
raw_kind
canonical_entity_id
usage_label
confidence
created_by_decision_id
status
created_at
revoked_at
revoked_reason
```

说明：

1. `raw_text` 可以来自 `entity`，也可以来自 `value_text`。
2. `raw_kind` 记录来源，例如 `entity` 或 `value_text`。
3. `usage_label` 是整理层根据使用侧写生成的弱标签，不要求来自固定枚举。
4. 原始 `MemoryV2Participants.entity` 不回写。

## 6. 整理动作类型

### 6.1 重复合并

处理同一事实或同一观察被多次写入。

输入示例：

```text
event A: 未來星織让我自己玩。
event B: 未來星織让我自己玩。
```

整理结果：

```text
A, B -> duplicate cluster
cluster canonical_event_id = A
B duplicate_of A
```

注意：

1. 不删除 B。
2. B 可以在召回中降权。
3. A 的 `occurrences` 可以增加，也可以通过 cluster member 体现重复次数。
4. 具体采用哪种计数方式要保持一致。

### 6.2 语义重复合并

处理表达不同但语义基本相同的记忆。

输入示例：

```text
event A: 用户喜欢简洁回答。
event B: 用户偏好直接一点的回复。
```

整理结果：

```text
A, B -> duplicate/semantic_duplicate cluster
可选生成 abstract event:
  用户偏好简洁直接的回答。
```

要求：

1. 需要比较 summary、participants、predicate、status、时间和上下文。
2. 不能只靠 summary embedding。
3. 低置信时只生成 candidate，不自动应用。

### 6.3 语义抽象

从一组细碎事件中沉淀长期记忆。

输入示例：

```text
我在 pixiv 阅读 Sacrai 的漫画《膝枕》。
我觉得这部漫画很可爱。
我认为 Sacrai 擅长日常搞笑又略带色气的作品。
我找不到点赞和收藏按钮。
```

整理结果：

```text
abstract event:
  我喜欢 Sacrai 创作的日常搞笑、反差萌、略带色气的作品风格。

relations:
  原始事件 supports abstract event
```

原则：

1. 抽象事件服务主循环召回。
2. 原始事件作为 evidence 保留。
3. 过程型低价值事件可以降权，但不删除。

### 6.4 认知变化

处理随时间变化的偏好、判断、状态或计划。

输入示例：

```text
event A: 我原本不喜欢 X。
event B: 我后来喜欢 X。
```

整理结果：

```text
evolution event:
  我对 X 的态度从不喜欢转变为喜欢。

A old_state_of evolution_cluster
B new_state_of evolution_cluster
A changed_into B
A, B support evolution event
```

认知变化包括：

1. 偏好变化。
2. 认知修正。
3. 状态变化。
4. 计划变化。
5. 关系变化。

旧事件不失效，只是当前性下降。召回时如果问题关注“现在”，优先新状态；如果问题关注“以前”或“变化过程”，召回演化链。

### 6.5 认知修正

认知修正是认知变化的一种特殊形式。

输入示例：

```text
event A: 我以为图片作者是 A。
event B: 我后来知道图片作者是 B。
```

整理结果：

```text
correction event:
  我曾以为图片作者是 A，后来确认作者是 B。

A corrected_by B
B corrects A
```

要求：

1. 不删除错误认知。
2. 旧认知标记为历史判断。
3. 召回当前事实时优先 B。

### 6.6 冲突保留

处理相似但不兼容的记忆。

输入示例：

```text
event A: 用户喜欢长解释。
event B: 用户不喜欢长解释。
```

整理结果：

```text
A conflicts_with B
可选生成 abstract event:
  用户对回答长度的偏好可能依场景变化。
```

要求：

1. 冲突不等于错误。
2. 不应强行合并。
3. 有上下文差异时，优先生成条件化抽象，而不是二选一。

### 6.7 实体规范化

处理多个原始提及可能指向同一个对象。

输入示例：

```text
Platform:星のgalgame资源社
Group:星のgalgame资源社
Location:星のgalgame资源社
```

整理结果：

```text
canonical entity:
  星のgalgame资源社

maps:
  Platform:星のgalgame资源社 -> canonical
  Group:星のgalgame资源社 -> canonical
  Location:星のgalgame资源社 -> canonical
```

重要限制：

1. 不假设 `Platform`、`Group`、`Location` 是固定枚举。
2. 不直接把这三种前缀规定为可互译。
3. 通过节点使用侧写、上下文、共现、summary 语义和时间分布综合判断。
4. 原始提及保留。

### 6.8 谓词簇

处理开放谓词的相似关系。

输入示例：

```text
like
prefer
enjoy
think highly of
```

整理结果：

```text
predicate cluster:
  preference_positive
members:
  like
  prefer
  enjoy
```

注意：

1. 不回写 `event_type_norm`。
2. 不把开放谓词强制闭集化。
3. 谓词簇只用于召回扩展和整理分析。

### 6.9 低价值过程事件整理

当前数据库中存在大量过程型事件，例如：

```text
open
browse
search
scroll
wait
close
send
say
reply
ask
```

这些事件不一定无用，但不应和长期偏好、事实、关系、状态以同等权重进入主循环召回。

整理层可以将它们归入低价值过程簇：

```text
low_value_trace_cluster
```

处理方式：

1. 不删除。
2. 默认召回降权。
3. 当它们支持某个抽象事件时，作为 evidence 使用。
4. 当用户询问具体历史过程时仍可召回。

## 7. 节点使用侧写

节点使用侧写是实体规范化和开放节点相似性判断的核心。

它不预设节点类型，只统计节点如何被使用。

### 7.1 节点来源

节点可以来自：

1. `MemoryV2Participants.entity`
2. 高频或高价值 `MemoryV2Participants.value_text`
3. `MemoryV2Events.event_type_norm`
4. 可选：summary 中反复出现的短语

### 7.2 侧写内容

一个节点使用侧写可以包含：

```text
raw_text
raw_kind
event_ids
role_distribution
predicate_distribution
status_distribution
context_distribution
neighbor_distribution
value_text_distribution
summary_embedding_center
value_embedding_center
time_distribution
source_distribution
```

说明：

1. `role_distribution`  
   该节点在不同 role 中出现的比例。不要求 role 来自固定集合。

2. `predicate_distribution`  
   该节点参与的事件谓词分布。不要求谓词来自固定集合。

3. `context_distribution`  
   该节点出现的 conv_type、conv_id、conv_name、source 分布。

4. `neighbor_distribution`  
   该节点经常和哪些其他 entity/value_text 一起出现。

5. `summary_embedding_center`  
   该节点关联事件 summary 向量的中心。

6. `time_distribution`  
   该节点出现的时间范围、密度和连续性。

### 7.3 侧写相似度

两个节点的相似度不由单一规则决定，而是由多种弱证据组合。

建议信号：

```text
text_similarity
role_distribution_similarity
predicate_distribution_similarity
context_distribution_similarity
neighbor_distribution_similarity
summary_embedding_similarity
value_text_similarity
temporal_pattern_similarity
conflict_penalty
```

重要原则：

1. 名称相似只是弱证据。
2. 数字片段只是弱证据。
3. 前缀相似或不同只是弱证据。
4. 时间重叠是加分项，不是必要条件。
5. 时间不重叠不代表不相似。
6. 长期并存但使用方式不同，应降低相似度。

### 7.4 泼溅逻辑

整理层中的泼溅逻辑定义为：

```text
从一个节点出发，
收集它连接到的事件、角色、谓词、邻居、上下文和值文本，
形成节点使用侧写；
再比较两个节点侧写的形状是否相似。
```

它不是：

```text
只比较相邻时间窗口
只比较共同事件
只比较固定类型标签
只比较名字
```

时间线性问题的处理：

1. 如果两个节点同时出现且侧写接近，加分。
2. 如果两个节点不同时出现但侧写接近，仍可成为候选。
3. 如果一个节点逐渐替代另一个节点，可能是命名变化。
4. 如果两个节点长期并存且承担不同使用方式，应避免合并。

## 8. 整理流程

### 8.1 sleep 触发

触发条件：

1. 主循环发起 sleep 动作。
2. 新增事件达到一定数量。
3. embedding/backfill 完成后。
4. 数据库重建后首次整理。
5. 手动维护命令触发。

### 8.2 候选生成

候选生成应尽量 deterministic，减少 LLM 成本。

候选来源：

1. 完全相同 summary。
2. 高相似 summary vector。
3. 相同或相似 participant 组合。
4. 相似节点使用侧写。
5. 同一主题时间段内的过程事件。
6. status 或 is_negated 出现变化的相似事件。
7. 高频开放谓词。
8. 高频 entity/value_text。

候选生成只负责发现可能关系，不直接修改召回结构。

### 8.3 候选评分

每个候选应生成证据包：

```text
candidate_type
members
text_features
graph_features
time_features
vector_features
conflict_features
score
```

评分分层：

```text
high confidence:
  可以自动应用软整理

medium confidence:
  进入 LLM 判断或等待更多证据

low confidence:
  忽略或记录为低优先级
```

### 8.4 LLM 判断

LLM 只用于中高价值、中置信、规则无法确定的候选。

LLM 输入应包含：

1. 候选事件摘要。
2. 相关 participants。
3. 时间顺序。
4. status/is_negated。
5. 来源上下文。
6. 节点使用侧写摘要。
7. 候选生成原因。

LLM 输出不直接改库，而是生成 decision：

```text
duplicate
abstract
evolved
corrected
conflict
alias
ignore
defer
```

### 8.5 应用决策

应用决策只做软操作：

1. 创建 cluster。
2. 创建 cluster members。
3. 创建 relation。
4. 创建 canonical entity map。
5. 创建 abstract/evolution/correction event。
6. 标记某些原始事件为低优先级。

不做：

1. 物理删除事件。
2. 改写原始 participants。
3. 覆盖 raw_event_json。
4. 强制改写 event_type_norm。

### 8.6 召回侧消费

召回侧消费整理结果时应遵循：

1. 查询命中 raw entity 时，扩展到 active canonical map 下的同簇 raw mentions。
2. 查询命中 canonical entity 时，扩展到 active raw mentions。
3. duplicate cluster 中优先 canonical event。
4. evolution cluster 中按问题时间语义选择 old/new/current。
5. conflict cluster 中避免只输出单边结论。
6. low_value_trace cluster 默认降权。
7. abstract event 比支持它的过程事件更适合常规主循环召回。

## 9. 可逆性设计

### 9.1 为什么必须可逆

整理层会使用统计、向量和可能的 LLM 判断。任何一种判断都可能出错。

因此整理结果不能破坏原始数据。

### 9.2 撤销单位

撤销单位应是 decision。

撤销一个 decision 时：

1. `MemoryV2ConsolidationDecisions.revoked = 1`
2. 相关 cluster/map/relation 标记 inactive 或 revoked。
3. 由该 decision 创建的 abstract/evolution event 可以标记为整理事件失效。
4. 原始事件不需要恢复，因为从未被删除。

### 9.3 重新整理

撤销后，可以重新运行整理。

新的 decision 不覆盖旧 decision，而是形成决策历史：

```text
decision A: 认为 X 和 Y 是同一实体
decision B: 撤销 A
decision C: 认为 X 和 Y 只是同名不同对象
```

这本身也是认知变化的一部分。

## 10. 与主循环的关系

整理层不直接参与主循环实时思考。

主循环只消费整理后的召回结果：

1. 更少重复。
2. 更少过程噪声。
3. 更好的当前性排序。
4. 能表达认知变化。
5. 能通过 canonical map 找到同一对象的不同提及。

主循环 normal render 仍保持最小输出：

```xml
<memory>
  <mem when="2小时前" confidence="0.90">...</mem>
</memory>
```

整理层内部的 cluster、decision、relation、path、score 不进入 normal render。

## 11. 与抽取侧的关系

抽取侧只负责记录当时模型从上下文中抽取出的事件。

整理层不要求抽取侧：

1. 输出标准实体类型。
2. 输出标准谓词。
3. 输出稳定 role 集合。
4. 识别 canonical entity。
5. 判断哪些事件应该合并。

抽取侧可以保持简单、开放和局部。

整理侧负责长期、全局、慢速的结构修正。

## 12. 推荐实现阶段

### Phase A：只读分析与报告

目标：不写库，只分析当前数据库。

任务：

1. 统计重复 summary。
2. 统计高频 entity/value_text。
3. 构建节点使用侧写。
4. 输出相似实体候选。
5. 输出过程型事件簇候选。

验收：

1. 可以看到候选质量。
2. 不改变数据库。
3. 帮助校准阈值。

### Phase B：候选表落库

目标：把候选写入 `MemoryV2ConsolidationCandidates`。

任务：

1. 建表。
2. 写入 candidate。
3. 保存 evidence_json。
4. 支持重复运行去重。

验收：

1. sleep 可以产生候选。
2. 候选可追踪到原始事件。

### Phase C：高置信软决策

目标：只应用非常保守的整理。

范围：

1. 完全重复 summary。
2. 明确重复的 event。
3. 高置信 entity canonical map。

验收：

1. 不删除原始事件。
2. 可撤销。
3. 召回可消费 canonical map。

### Phase D：LLM 辅助整理

目标：处理语义重复、抽象、变化和冲突。

任务：

1. 设计 consolidation prompt。
2. 限制输入规模。
3. 输出 decision JSON。
4. 只通过 decision 应用。

验收：

1. LLM 不直接改库。
2. 错误 decision 可撤销。
3. 抽象事件有 sources/relations 支撑。

### Phase E：召回整合

目标：召回侧正式消费整理结果。

任务：

1. canonical map 扩展 entity 查询。
2. duplicate cluster 降权成员事件。
3. abstract event 优先于低价值过程事件。
4. evolution cluster 支持当前性排序。
5. conflict cluster 避免单边误导。

验收：

1. 主循环召回更少噪声。
2. 当前事实优先。
3. 历史变化仍可追溯。

## 13. 开放问题

1. 是否需要单独的 `MemoryV2AbstractEvents`，还是直接复用 `MemoryV2Events` 并用 source/relation 标记整理生成？
2. `occurrences` 应由 duplicate cluster 推导，还是继续写入 canonical event？
3. 低价值过程事件是通过 cluster 降权，还是在 event 上增加 internal priority 字段？
4. LLM 整理是否允许创建新的 entity canonical summary？
5. canonical entity 的 key 如何生成，是否只使用内部 ID 而不暴露文本 key？
6. 整理层运行频率如何确定？
7. 当整理结果和后续新记忆冲突时，是否自动生成新的认知变化 decision？

## 14. 核心结论

Memory V2 整理层应迁移“泼溅系统”的方法，而不是迁移它的侧写档案表结构。

适合迁移的是：

```text
通过周围证据形成侧写
周期性比较侧写
先生成候选
再做可撤销决策
保留历史
允许认知演化
```

不适合迁移的是：

```text
固定 Entity/Profile 二分
预设实体类型集合
直接融合档案
删除旧对象
把合并当成绝对真理
```

整理层最终应该让 Memory V2 从“事件堆积”变成“可演进的认知图谱”：

```text
原始事件保留事实痕迹；
事件簇表达主题和变化；
规范实体映射减少碎片；
抽象事件服务主循环；
冲突和修正保留认知历史；
所有整理动作都可撤销。
```
