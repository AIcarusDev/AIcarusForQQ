export const MEMORYQL_META = {
  name: "MemoryQL",
  version: "v1.0",
  status: "服务端只读语法契约",
  nodeBudget: 80,
  edgeBudget: 120,
  depthBudget: 2,
  rowBudget: 100,
};

export const MEMORYQL_QUERY_SHAPE = `MATCH
  <节点声明>
  <关系声明>
WHERE <过滤条件>
EXPAND <起点> DEPTH <1..2>
RETURN <GRAPH | TABLE | RAW>
LIMIT <结果预算>`;

export const MEMORY_QUERY_EXAMPLES = [
  {
    label: "最近事件关系",
    description: "读取少量最近事件关系，让初始子图保持可读并可继续放大探索。",
    query: `MATCH
  $source ISA Event
  $target ISA Event
  ($source)-[RELATES_TO]->($target)
RETURN GRAPH
LIMIT NODES 8 EDGES 4`,
  },
  {
    label: "关于 admin",
    description: "从规范实体命中事件，再向外展开一跳关系。",
    query: `MATCH
  $event ISA Event
  $entity ISA CanonicalEntity
  ($event)-[INVOLVES]->($entity)
WHERE $entity.name ~= "admin"
EXPAND $event DEPTH 1
RETURN GRAPH
LIMIT NODES 36 EDGES 48`,
  },
  {
    label: "故事线与事件",
    description: "查询活跃故事线及其归属事件，返回有限子图。",
    query: `MATCH
  $story ISA Storyline
  $event ISA Event
  ($event)-[PART_OF]->($story)
WHERE $story.status = "active"
RETURN GRAPH
LIMIT NODES 30 EDGES 36`,
  },
  {
    label: "来源追溯",
    description: "按时间筛选事件，并以表格查看来源追溯关系。",
    query: `MATCH
  $event ISA Event
  $source ISA Source
  ($event)-[DERIVED_FROM]->($source)
WHERE $event.occurred_at >= "2026-07-01"
RETURN TABLE
LIMIT ROWS 100`,
  },
];

export const MEMORYQL_GUIDE_SECTIONS = [
  {
    id: "structure",
    index: "01",
    title: "查询基本结构",
    summary: "子句按固定顺序书写；MATCH、RETURN 与 LIMIT 必填，WHERE 和 EXPAND 按需要添加。",
    code: MEMORYQL_QUERY_SHAPE,
    bullets: [
      "每条查询只读取语义层，不提供 CREATE、UPDATE 或 DELETE。",
      "变量以 $ 开头，并在 MATCH 中声明后才能被后续子句引用。",
      "LIMIT 必须显式声明结果预算；服务端硬上限仍会作为第二道保护。",
      "关键字建议使用大写；类型与关系名称必须匹配左侧 Schema。",
    ],
  },
  {
    id: "match",
    index: "02",
    title: "匹配节点与关系",
    summary: "ISA 声明节点类型，带方向的关系模式描述两个变量之间的语义连接。",
    code: `MATCH
  $event ISA Event
  $entity ISA CanonicalEntity
  ($event)-[INVOLVES]->($entity)`,
    bullets: [
      "节点写法：$变量 ISA 节点类型。",
      "关系写法：($起点)-[关系类型]->($终点)。",
      "MemoryQL 1.0 使用显式方向，避免在大图上产生含糊的双向扫描。",
    ],
  },
  {
    id: "filter",
    index: "03",
    title: "过滤条件",
    summary: "WHERE 只作用于 MATCH 已声明的变量；字符串使用双引号，日期使用 ISO 8601 形式。",
    code: `WHERE $entity.name ~= "admin"

WHERE $event.occurred_at >= "2026-07-01"
  AND $event.confidence >= 0.80`,
    bullets: [
      "比较运算：=、!=、>、>=、<、<=。",
      "~= 表示区分当前存储规则的文本包含匹配，不是正则表达式。",
      "可使用 AND / OR 组合条件；复杂表达式应优先拆成可解释的小查询。",
    ],
  },
  {
    id: "expand",
    index: "04",
    title: "有限深度扩展",
    summary: "EXPAND 从已命中的变量继续取邻接关系；它扩展当前结果，而不是加载完整实例图。",
    code: `EXPAND $event DEPTH 1`,
    bullets: [
      "DEPTH 1 适合检查直接关系，DEPTH 2 用于有限上下文。",
      "服务端硬上限为深度 2；超出请求预算的查询会在执行前被拒绝。",
      "不写 EXPAND 时，只返回 MATCH 明确命中的结构。",
    ],
  },
  {
    id: "return",
    index: "05",
    title: "选择返回形式",
    summary: "RETURN 决定结果的首选投影；同一隔离结果仍可在工作台中切换辅助视图。",
    code: `RETURN GRAPH
RETURN TABLE
RETURN RAW`,
    bullets: [
      "GRAPH：用于检查节点、边与局部结构。",
      "TABLE：用于比较字段、排序后的记录或来源清单。",
      "RAW：用于查看结果元数据和调试投影；“解释”是工作台视图，不是 RETURN 类型。",
    ],
  },
  {
    id: "limits",
    index: "06",
    title: "声明结果预算",
    summary: "LIMIT 是查询契约的一部分，不是执行后的视觉裁剪；达到上限时结果必须明确标记截断。",
    code: `LIMIT NODES 80 EDGES 120

LIMIT ROWS 100`,
    bullets: [
      "图结果上限：80 个节点、120 条关系。",
      "表格结果上限：100 行。",
      "深度、节点与关系预算共同约束查询，防止千节点视图重新失效。",
    ],
  },
  {
    id: "safety",
    index: "07",
    title: "安全与实现边界",
    summary: "这份文档对应当前 MemoryQL 1.0 服务端契约；查询通过只读连接与参数化执行计划访问真实记忆。",
    bullets: [
      "只读：语法不包含任何写入或维护命令。",
      "隔离：每次运行生成独立结果集，不会合并进所谓“全局图”。",
      "解耦：语义类型与物理 SQLite 表结构分离，存储变化不应直接破坏查询界面。",
      "版本化：请求必须携带语言版本；不兼容版本会明确拒绝，不会静默改写查询。",
      "有界执行：节点、关系、行数、深度和 500ms 执行时间均有服务端硬上限。",
    ],
  },
];

export const MEMORYQL_CLAUSE_REFERENCE = [
  ["MATCH", "声明要命中的节点与关系", "必填"],
  ["WHERE", "按属性缩小命中范围", "可选"],
  ["EXPAND", "从命中变量有限扩展", "可选，深度 ≤ 2"],
  ["RETURN", "选择 GRAPH / TABLE / RAW 投影", "必填"],
  ["LIMIT", "声明节点、边或行数预算", "必填"],
];
