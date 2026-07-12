# AIcarus Memory 主循环记忆系统开发计划

## 1. 背景与定位

Memory 的目标不是提供一套面向展示、解释或可视化的记忆系统，而是成为 AIcarus 主循环可稳定依赖的长期记忆基础设施。

本阶段开发只面向主循环逻辑：

```text
压缩后的意识流/对话源
  -> prompt 长期记忆抽取
  -> Memory 写入与去重
  -> embedding/backfill
  -> 图召回与重排
  -> 最小 XML 注入主循环
```

开发默认假设数据库可以删除重建。不考虑旧数据迁移，不为旧 schema 做兼容设计，不为 WebUI 或解释模块保留额外负担。解释、调试、可视化等模块后续自行适配 Memory 的真实数据结构。

## 2. 开发边界

### 2.1 本阶段目标

1. 让 Memory 成为主循环唯一可信赖的长期记忆链路。
2. 固定 prompt 抽取、解析、写入、召回、渲染之间的工程契约。
3. 让记忆系统在失败、重启、空输出、embedding 不可用等情况下保持状态一致。
4. 用可测试、可删库重建的方式推进，不背负旧数据迁移包袱。
5. 清理主循环不再需要的旧记忆逻辑和兼容分支，降低后续维护成本。

### 2.2 明确不做

1. 不做 WebUI、3D 图谱、搜索页面、管理台和可视化体验。
2. 不做旧角色表数据迁移。
3. 不做旧数据库兼容，开发环境默认删库重建。
4. 不为解释模块、可视化模块、调试面板单独设计字段或输出格式。
5. 不把 event id、predicate、participants、recall score、recall path 注入主循环模型上下文。
6. 不引入 TypeDB、外部向量数据库、完整 revision engine 或复杂 ontology。

## 3. 当前基础

当前分支已经具备以下基础能力：

1. `prompt.py` 作为长期记忆抽取 prompt 来源。
2. `parser.py` 解析 `<extract><event>{...}</event></extract>` 输出。
3. `events.py` 包含 Memory schema、事件写入、去重、predicate、participants、sources、relations、vectors、embedding jobs 与召回逻辑。
4. `render.py` 已区分主循环最小渲染和 debug 渲染。
5. `workflow.py` 已有 pending archive job、取消保留、重启续跑等工程化能力。
6. 测试中已有 parser、render、memory 写入等部分覆盖。

这些能力说明下一阶段不是重新设计 Memory，而是把它收敛、稳固、验收，并清除不再服务主循环的旧逻辑。

## 4. 总体架构目标

### 4.1 抽取入口

主循环和意识流压缩侧只向记忆系统提交干净的归档输入：

```text
raw chat / cognition
  -> consciousness compression
  -> compression summary / cognition range
  -> archive task payload
```

长期记忆抽取只从压缩后的来源中进行。普通 `say`、`ask`、`answer`、`share` 等对话动作默认不进入长期事实记忆，除非压缩摘要明确沉淀为事件、状态、偏好、关系或设定。

### 4.2 存储模型

Memory 以事件为一等节点：

1. `MemoryEvents` 保存事件摘要、谓词、状态、置信度、来源、时间、去重签名等核心字段。
2. `MemoryParticipants` 保存事件参与者、实体和值文本。
3. `MemoryPredicates` 保存开放谓词。
4. `MemoryRelations` 保存事件之间的显式关系。
5. `MemoryEventSources` 保存事件与 cognition source 的来源关系。
6. `MemoryVectors` 保存 summary/predicate 向量。
7. `MemoryEmbeddingJobs` 保存 pending/failed/stale/ready 的 embedding 工作状态。

旧表不作为设计约束。必要时可以在开发环境中直接删除数据库并重建。

### 4.3 召回主路径

主循环召回目标是稳定、相关、低噪声，而不是解释展示。

推荐召回流程：

```text
query/context
  -> FTS / entity / summary vector / predicate vector / recent fallback seeds
  -> costed graph expansion
  -> rerank
  -> top-K event summaries
  -> minimal XML render
```

召回层需要具备：

1. hub penalty，避免高频泛实体污染召回。
2. status/context penalty，避免 hypothetical 泄漏到 actual 召回。
3. time decay，减少过旧弱相关事件干扰。
4. deterministic tie-break，同一输入和数据库状态下输出顺序稳定。
5. fallback，在 embedding 不可用时仍可通过 FTS/entity/recent 基础召回工作。

### 4.4 主循环注入格式

正常注入只允许最小内容：

```xml
<memory>
  <mem when="2小时前" confidence="0.80">用户偏好简洁直接的答复。</mem>
</memory>
```

正常上下文中不得包含：

1. event id。
2. event type / predicate。
3. participants。
4. recall score。
5. recall path。
6. source ids。
7. debug reason。

debug 渲染可以保留，但必须和主循环注入路径彻底隔离。

## 5. 阶段计划

### Phase 1：规格对账与主路径冻结

目标：把现有 Memory 代码和设计文档对齐，明确哪些能力已经实现，哪些进入后续阶段。

任务：

1. 梳理 `recall/design.md`，将 checklist 改为真实状态表。
2. 标记已实现、部分实现、废弃、不进入本阶段的条目。
3. 确认主循环实际调用的归档入口、召回入口和渲染入口。
4. 明确 Memory 是唯一新主线，旧记忆路径不再扩展。

验收：

1. 有一份当前实现状态表。
2. 主循环相关入口清晰可追踪。
3. 后续开发不需要在旧 schema 和新 schema 之间猜测。

### Phase 2：归档链路稳固

目标：让归档任务在主循环中失败可控、状态一致、可恢复。

任务：

1. 固定 `prompt.py` 为唯一 archive prompt 来源。
2. 确认 forced-tool archive 旧契约不再进入 记忆主路径。
3. 固定 archive job 生命周期：pending、running、success、failed、cancelled/retry。
4. 空输出、结构错误、LLM 异常不推进 archive signature。
5. shutdown cancel 后保留 pending job，下次启动续跑。
6. 归档写入和来源绑定保持事务一致。

验收：

1. LLM 空输出不会丢弃待归档区间。
2. parser fatal 不会推进签名。
3. 进程中断后 pending job 可恢复。
4. 同一 cognition range 重跑不会重复写入相同事件。

### Phase 3：解析与写入契约完善

目标：让 prompt 输出到数据库写入之间的边界严格、可测、可维护。

任务：

1. parser 严格要求唯一顶层 `<extract>`。
2. 只解析 `<extract>` 内的 `<event>`。
3. 允许空 `<extract></extract>` 表示无长期记忆。
4. 禁止 markdown fence、event 内自由文本、非 JSON 对象。
5. 部分接受批次：有效事件写入，无效事件结构化记录。
6. `summary`、`event_type`、`roles` 为必需字段。
7. `source_id` 与当前 task 输入来源做校验和绑定。
8. 未知字段只进入 raw JSON，不自动扩展表结构。

验收：

1. parser contract 测试覆盖正常、空输出、重复 extract、嵌套 extract、坏 JSON、混合有效/无效事件。
2. 写入测试覆盖默认值、raw JSON 保留、roles 正规化、source link、dedupe。
3. event_type 只做轻量 normalize，不映射到封闭谓词表。

### Phase 4：Embedding 与 backfill 工程化

目标：让向量能力成为增强项，而不是主循环稳定性的单点故障。

任务：

1. 固定 embedding client 接口：输入有序文本 batch，输出同序向量。
2. provider/model/dim/source_hash 写入 vector metadata。
3. summary 和 predicate 至少支持向量生成。
4. embedding 失败不影响事件写入。
5. failed/stale job 可重试。
6. 提供删库重建后的 embedding rebuild/backfill 命令或内部入口。
7. hash fallback 保留为测试和无 provider 环境的稳定降级。

验收：

1. provider 不可用时 archive 写入仍成功。
2. stale vector 可检测并重新排队。
3. backfill 可重复运行且不会产生重复有效向量。
4. 无向量时召回仍能通过 FTS/entity/recent fallback 产出结果。

### Phase 5：召回与重排定版

目标：让主循环召回结果稳定、相关、低噪声，并具备可测试的质量基准。

任务：

1. 固定 seed retrieval：FTS、entity、summary vector、predicate vector、recent fallback。
2. 固定 costed graph expansion：energy budget、depth cap、expanded node cap。
3. 引入或校准 hub penalty、time decay、status traversal penalty。
4. 阻止 hypothetical 默认泄漏到 actual 召回。
5. rerank 结合 seed score、path cost、vector similarity、entity match、freshness、occurrences。
6. 不把 confidence 作为主要排序信号。
7. 输出 top-K 顺序保持 deterministic。
8. 建立小型评估集，覆盖偏好、事实、状态、关系、反事实、重复事件和泛实体噪声。

验收：

1. 精确摘要查询能召回对应事件。
2. 语义相近谓词能在阈值内扩展。
3. 低相似谓词不会误扩展。
4. hub entity 不主导召回。
5. hypothetical 不默认进入 actual 召回。
6. 同一数据库和 query 的 top-K 顺序稳定。
7. normal render 不包含 debug 字段。

### Phase 6：代码稳固与测试门槛

目标：让 Memory 进入主循环前具备足够工程可靠性。

任务：

1. 为 archive、parser、write、embedding、recall、render 建立核心测试矩阵。
2. 增加异常路径测试：空输出、坏输出、provider 失败、数据库重建、pending job 续跑。
3. 统一日志语义，区分 debug、warning、error。
4. 为主循环入口增加最小集成测试。
5. 检查异步任务取消、事务提交和重复执行边界。
6. 固定正常渲染和 debug 渲染的隔离边界。

验收：

1. 关键路径测试可在无外部 provider 环境运行。
2. 失败不会推进不该推进的状态。
3. 取消不会误删 pending job。
4. 主循环注入格式稳定。
5. 删除数据库后首次启动可自动建表并工作。

### Phase 7：无用逻辑清理与工程化收敛

目标：减少旧逻辑包袱，让 Memory 主路径更清晰。

清理原则：

1. 先确认未被主循环调用，再删除。
2. 先补测试或调用点检查，再删除。
3. 不为了清理而改召回行为。
4. 不删除对稳定性有价值的内部诊断和 fallback。

优先清理：

1. forced-tool archive 旧契约残留。
2. legacy event type normalization。
3. 旧角色表召回路径中不再被主循环使用的部分。
4. 为 WebUI/解释模块服务但主循环不需要的字段流转。
5. 重复 facade、历史兼容函数和未调用 helper。
6. 已被 memory repo 取代的旧 memory write/read API。

暂不清理：

1. debug render，因为它有排错价值且不进入主循环。
2. hash embedding fallback，因为它支撑离线测试。
3. pending job 续跑逻辑，因为它属于稳定性能力。
4. raw event JSON，因为它是解析与回溯的重要证据。

验收：

1. 主循环记忆调用路径更短、更明确。
2. 删除旧逻辑后测试通过。
3. 没有为了兼容旧数据库而保留的主动分支。
4. 代码结构能清楚区分 archive、repo、recall、render、embedding。

## 6. 质量门槛

Memory 进入主循环稳定使用前，至少满足以下门槛：

1. 删库后首次启动可自动建表。
2. archive job 可失败、可重试、可中断恢复。
3. parser contract 有完整测试。
4. 写入去重不会因为重复归档产生重复长期事件。
5. embedding 失败不会阻断长期记忆写入。
6. recall 无向量时仍可工作。
7. normal render 不泄露内部字段。
8. 同一输入和数据库状态下 recall top-K 稳定。
9. 旧数据迁移不再作为阻塞项。
10. 主循环只依赖 Memory 的正式入口。

## 7. 风险与应对

### 7.1 召回噪声过高

风险：泛实体、高频谓词或近期 fallback 造成无关记忆进入主循环。

应对：

1. 调整 hub penalty。
2. 限制 recent fallback 触发条件。
3. 加入小型质量评估集。
4. 对过短、上下文依赖强的 summary 加质量惩罚。

### 7.2 embedding provider 不稳定

风险：外部 provider 失败导致 archive 或 recall 不稳定。

应对：

1. embedding 失败只影响向量增强，不影响事件写入。
2. 保留 hash fallback。
3. failed/stale job 可重试。
4. recall 保留 FTS/entity/recent fallback。

### 7.3 prompt 输出漂移

风险：模型输出不符合契约，导致无效事件或污染写入。

应对：

1. parser 严格拒绝坏结构。
2. 部分接受批次，有效和无效分离。
3. raw JSON 保留。
4. fatal 输出不推进 archive signature。

### 7.4 清理误删仍被调用路径

风险：删除旧逻辑影响主循环。

应对：

1. 删除前用搜索和测试确认调用点。
2. 清理批次小步提交。
3. 先做主循环入口测试。
4. 保留 fallback 和诊断逻辑直到新路径稳定。

## 8. 推荐实施顺序

1. 规格对账与主路径冻结。
2. 归档链路稳固。
3. parser/write 契约测试补齐。
4. embedding/backfill 工程化。
5. recall/rerank 质量评估和确定性修正。
6. 主循环注入集成测试。
7. 清理旧逻辑和无用兼容分支。

这个顺序优先保证主循环稳定，再做清理和收敛。避免先大规模删除旧代码，导致问题定位困难。

## 9. 下一步任务清单

第一批建议任务：

1. 更新 `recall/design.md`，标注真实实现状态。
2. 列出主循环当前实际调用的 memory archive/recall/render 入口。
3. 补 parser fatal、empty extract、partial accept 测试。
4. 补 archive 空输出和结构错误不推进 signature 测试。
5. 补 recall deterministic top-K 测试。
6. 梳理旧 memory API 调用点，标记可清理候选。

第一批完成后，再进入 embedding/backfill 和召回质量评估。
