# 记忆召回层优化方案

## 背景

AIcarus 当前记忆系统已经把长期记忆主要收敛到 `MemoryEvents` + `MemoryRoles`：

- `MemoryEvents` 存事件、事实、状态、偏好等可召回内容。
- `MemoryRoles` 把实体挂到事件上，表达 agent / theme / recipient 等角色。
- 召回侧目前主要依赖实体候选、FTS、近期兜底和一个简化的 spreading activation。

这个方案能用，但召回层仍然偏“候选融合”，不是严格的图搜索。问题主要有三个：

1. 召回路径不够明确，难解释为什么某条记忆被召回。
2. 高频实体容易成为 hub，导致噪声扩散。
3. 多跳关联能力弱，尤其是“用户 -> 事件 -> 相关实体 -> 其他事件”的场景。

旧原型 `entitySystem` 的 BFS / Dijkstra 召回在实测中表现较好，值得借鉴其图搜索思想，但不建议整包移植旧项目。

## 核心结论

应该借鉴 `entitySystem` 的召回层，而不是搬运它的抽取层。

建议把 AIcarus 的召回层升级为：

```text
当前输入
  -> seed entities / seed concepts
  -> FTS + 实体召回得到粗候选
  -> 基于事件图的 Dijkstra / BFS 扩展与重排
  -> 输出带路径解释的 facts / events / evidence
```

抽取层负责产出干净的事实和证据；召回层负责用图路径提高准确率。

## 借鉴点

### 1. 事件作为一等节点

不要只把记忆看作实体之间的二元边，而是把事件作为图节点：

```text
entity:User:qq_123 <-> event:42 <-> entity:Tool:qwen
```

这样可以自然表达多角色事件：

```text
event:42
  agent      User:qq_123
  recipient  self
  theme      "某个项目方案"
  instrument Tool:VSCode
```

好处：

- 支持 N 元关系。
- 避免把复杂事件硬压成三元组。
- 召回路径可以解释为“通过哪个事件连接到哪个实体”。

### 2. 使用带代价的图搜索

推荐从 `entitySystem` 借鉴类似代价函数：

```text
cost = 1.0 / weight
     + degree_penalty
     + time_decay
     + deprecated_penalty
     + context_delta
```

各项含义：

- `weight`：边置信度，来自事件 confidence 或证据 support_confidence。
- `degree_penalty`：节点度数越高，穿过它的成本越高，避免 hub 污染。
- `time_decay`：越旧的边成本越高。
- `deprecated_penalty`：被废弃、被覆盖、被撤销的边增加成本。
- `context_delta`：根据 context / modality 决定是否可通行。

建议初始实现：

```text
degree_penalty = log10(degree(next_node)) * 0.3
time_decay     = min(age_days / 30, 1.0) * 0.2
deprecated     = 2.0
hypothetical   = infinity
evidence       = 0.3 ~ 0.8 extra cost
```

具体权重需要通过生产库离线评估调参。

### 3. 上下文和模态参与路径代价

推荐规则：

```text
episodic / actual      可通行，正常成本
possible               可通行，小幅加成本
hypothetical           默认不可通行，除非用户明确询问假设
deprecated / revoked   可通行但高成本
evidence               可通行但必须按证据渲染，不能当事实
```

这点尤其重要。证据可以帮助召回，但 evidence 不应该混成 fact。

## 证据层关系

抽取层优化后，建议把 evidence 从 `MemoryEvents` 拆出去，形成独立结构：

```text
MemoryFacts / MemoryEvents
  稳定事实、偏好、承诺、项目状态等

MemoryEvidence
  对某个 claim 的支持或反驳证据
```

推荐图结构：

```text
source_actor -> evidence -> claim -> subject_entity
```

召回时可以通过 evidence 走到 claim，但渲染时必须明确：

```xml
<evidence support="0.72">
  某人曾多次提到 X，因此 X 可能成立。
</evidence>
```

不能渲染成：

```xml
<event>X 已经成立</event>
```

## 分阶段实施

### Phase 1：离线评估准备

从生产数据库构建一组测试问题：

- 用户偏好类
- 项目状态类
- 多人群聊实体指代类
- 证据/传闻类
- 假设/反事实类
- 高频实体干扰类

每类至少 5 条，总计 30-50 条即可。

指标：

- recall@5 / recall@10
- 噪声率
- 是否召回到正确主体
- 是否把 evidence 错当 fact
- 单次召回耗时

### Phase 2：构建轻量内存图索引

启动或定期从 SQLite 构建图：

```text
nodes:
  entity:{id}
  event:{event_id}
  evidence:{evidence_id}
  claim:{claim_id}

edges:
  entity <-> event
  event  <-> value/concept
  source_actor <-> evidence
  evidence <-> claim
  claim <-> subject_entity
```

图索引只作为召回层缓存，不替代 SQLite。

### Phase 3：BFS/Dijkstra 作为 reranker

第一版不要直接替换当前召回。

推荐先保留现有候选生成：

```text
entity candidates + FTS candidates + recent fallback
```

然后用图搜索分数重排：

```text
final_score = existing_score - path_cost_bonus
```

这样风险最低，能快速比较前后效果。

### Phase 4：替换简化 spreading activation

等离线评估稳定后，用真正的 Dijkstra 替换当前 `events.py` 里的简化多跳扩展。

目标能力：

- seed entity 到 event 的最短路径。
- event 到相关 entity 的扩展。
- 避免 hub。
- 避免 hypothetical 被误召回。
- 输出 path explanation。

### Phase 5：证据召回

等 evidence 独立表完成后，再把 evidence 加入图搜索：

```text
entity -> evidence -> claim
query  -> claim -> evidence
```

证据召回必须独立渲染，不进入普通 facts 列表。

## 不建议引入的旧复杂度

暂时不要引入：

- TypeDB。
- A* 启发式搜索。
- 完整 revision engine。
- 写入时强制实体合并。
- 大量一阶 `say / ask / answer` 记忆。

这些复杂度可以等召回评估证明需要后再考虑。

## 与抽取层的关系

抽取层优化后的主路径应该是：

```text
原始 chat_messages
  -> consciousness compression
  -> compression_summary
  -> 单次长期记忆抽取
  -> facts / evidence
  -> graph recall
```

也就是说：

- `chat_messages` 仍然保存“谁说了什么”。
- `compression_summary` 负责去噪和压缩上下文。
- 长期记忆抽取只从压缩摘要里抽一次。
- 普通 `say / ask / answer / share` 不再进入长期事实记忆。
- evidence 不等于 fact，后续需要单独表和单独渲染。

## 推荐下一步

1. 先完成抽取层：压缩摘要单次抽取，过滤一阶聊天噪声。
2. 再拆 evidence 表，避免证据混入事实。
3. 然后做离线 recall eval。
4. 最后引入 Dijkstra/BFS reranker。

