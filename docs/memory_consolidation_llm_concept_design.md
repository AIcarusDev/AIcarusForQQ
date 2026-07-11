# 记忆归档后二步事件整理

## 目标

第一步只提取事实事件。第二步只比较本批新事件与召回的历史原子事件，并完成两件事：

1. 建立新事件到历史事件的一对一连接。
2. 从本批新事件中标注由多个事件组成的 `candidate_storyline`。

第二步不读取或选择现有 storyline 或 summary anchor。

## 模型输入

模型收到一个 JSON 对象：

- `new_events`：本批新事件，每项只有本地 `id`、`summary`、`entities`。
- `existing_events`：召回的历史原子事件，每项同样只有本地 `id`、`summary`、`entities`。

`N*` 和 `H*` 都是单次调用内的临时 ID。数据库事件 ID、事件类型、状态和时间不进入模型输入。

## 模型输出

输出固定为 `<analysis>` 与 `<tidy>`。`<tidy>` 内包含：

- `<link>`：`[{"new_event":"N1","existing_event":"H1"}]`
- `<candidate_storyline>`：`[["N1","N2"]]`

模型不输出关系类型、置信度、证据、标题、revision 或任何数据库 ID。

## 后端落地

- `link` 经本地 ID 校验后写入 `MemoryRelations`，方向为新事件到历史事件，规范关系类型为 `related`。
- `candidate_storyline` 经校验后写入 `MemoryCandidateStorylines`；表内只保留确定性的 `candidate_storyline_id`、事件 ID 集合、状态和系统时间。
- sleep 在允许固化时将 pending candidate storyline 写入 `MemoryStorylines(scope='candidate_storyline')`，并把有效候选标记为 accepted。

未来可为主模型和事件整理模型增加“展开故事线”工具，用于主动查看更多细节；该能力不属于当前实现。
