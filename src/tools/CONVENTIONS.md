# 工具模块约定（Tool Module Conventions）

每个工具必须放在所属 namespace 目录下，并由 root `__init__.py` 按 `namespaces.yaml` 声明的 namespace 自动扫描加载。工具可以是以下两种形式之一：

- **单文件工具**：`src/tools/<namespace>/tool_name.py`
- **文件夹工具**：`src/tools/<namespace>/tool_name/__init__.py`（适合较复杂的工具，可在文件夹内拆分多个辅助模块）

> root `src/tools/` 只放 loader、registry 和共享辅助模块；业务工具不再散放在 root。`not_used/` 不参与 namespace 扫描。

---

## 必须导出

工具可以使用两种合同入口。新工具优先使用 **Python-first contract**；旧工具可以继续使用 legacy `DECLARATION` + `PROMPT_SIGNATURE`。

### 方式 A：Python-first contract（推荐）

工具参数合同定义为 Pydantic model，并通过 `tools.contract.tool(...)` 绑定到 `execute`：

```python
from pydantic import Field
from tools.contract import ToolArgsModel, tool


class GetWeatherArgs(ToolArgsModel):
    city: str = Field(
        min_length=1,
        description="要查询的城市名称，例如「北京」「上海」「Tokyo」等，中英文均可。",
    )


@tool(
    name="get_weather",
    description="查询指定城市的天气情况。",
    args_model=GetWeatherArgs,
)
def execute(args: GetWeatherArgs) -> dict:
    ...
```

这一路径只手写一份 Python 工具合同：

- `args_model` 生成后端 JSON Schema declaration
- `args_model` + tool description 生成模型可见 TypeScript-like signature
- `execute(**arguments)` 入口会先把参数验证为 Pydantic model，再调用工具实现

生成签名应只暴露对模型调用有帮助的约束。低价值约束（例如字符串 `min_length=1`、数组 `minItems=1`）保留在后端校验 schema 中，但不渲染进模型提示；有决策价值的约束（例如 `maximum=15`、枚举、有效范围、非平凡最小长度）应保留。

### 方式 B：Legacy `DECLARATION: dict`

工具的后端校验 schema 声明，包含：

- `name`: 工具名（字符串，唯一）
- `description`: 工具描述（用于 preview/search；loader 会在执行校验用 schema 中剥离该字段）
- `parameters`: JSON Schema 格式的参数定义

如果 schema 需要动态生成（例如包含枚举值，或需要根据当前会话上下文裁剪字段），则导出 `get_declaration(...) -> dict` 函数替代静态 `DECLARATION`。
此时 `DECLARATION` 只需包含 `{"name": "工具名"}` 供框架识别。

`get_declaration` 支持按需声明上下文参数，例如 `session`、`config`；框架会按同名关键字注入。若无需上下文，也可以继续写成无参函数。

### Legacy `PROMPT_SIGNATURE: str` / `get_prompt_signature(...) -> str`

模型可见的 TypeScript-like 工具签名。它只用于 prompt 展示，不参与后端校验。

Legacy 工具必须导出 `PROMPT_SIGNATURE` 或 `get_prompt_signature(...)`；loader 不会把 legacy 本地工具的 JSON Schema 自动转换成模型签名。第一版保持源码可读，不做压缩；使用普通 `//` 注释承载原 description 中真正影响模型调用判断的适用场合、语义和细节引导。

推荐形态：

```python
PROMPT_SIGNATURE = """
// 核心的运行状态管理工具。
runtime_manage(args:
  | {
      action: "wait";
      seconds?: number; // 范围 1~20，单位秒，默认 10。
    }
  | {
      action: "idle";
      minutes?: number; // 范围 1~60，单位分钟，默认 30。
    }
  | {
      action: "sleep";
      minutes?: number; // 范围 30~600，单位分钟，默认 480。
    }
)
"""
```

复杂的 action discriminator 工具应手写 union，保证模型面对约束与后端 JSON Schema 等价：

```python
PROMPT_SIGNATURE = """
goal_manage(args:
  | {
      action: "create";
      goals: { title: string; content: string; reason: string }[];
    }
  | {
      action: "resolve";
      goal_ids: string[];
      resolution: "completed" | "abandoned" | "duplicate" | "superseded" | "mistaken";
    }
)
"""
```

动态工具可导出 `get_prompt_signature(...) -> str`，支持与 `get_declaration(...)` 相同的上下文注入规则。`src/tools/prompt_signatures.py` 中的 schema 转换器只保留给未来外来 MCP/迁移辅助，不作为第一方工具契约路径。

#### Schema 自定义扩展键

框架不再把 JSON Schema 原样传给模型；执行校验用 declaration 会剥离 `description`，但保留本地预处理需要的 `x-` 扩展键。模型可见层只看到手写 `PROMPT_SIGNATURE` 或 `get_prompt_signature(...)` 的返回值。

**`"x-coerce-integer": True`**

适用场景：某 `string` 类型参数在业务上**有且仅有纯数字 ID**（如 QQ 号、群号、消息 ID、记忆事件 ID 等），而个别模型可能将其输出为 JSON 整数。

在对应字段的 schema 中加入：

```json
{
  "type": "string",
  "x-coerce-integer": true,
  "description": "目标 ID"
}
```

作用：参数修复阶段会自动将整数值强制转为字符串，使其通过 schema 校验。**该键在发给模型前会被安全移除**。

### 处理函数（二选一）

**方式 A：无运行时依赖**

```python
def execute(**kwargs) -> dict: ...
```

**方式 B：需要运行时对象（qq_client、session 等）**

```python
REQUIRES_CONTEXT: list[str] = ["qq_client", "session"]

def make_handler(qq_client, session) -> Callable:
    def execute(**kwargs) -> dict: ...
    return execute
```

> `REQUIRES_CONTEXT` 的唯一职责是**依赖注入**——声明 `make_handler` 需要哪些运行时对象。
> 会话类型不在 loader 阶段过滤；不要用 `REQUIRES_CONTEXT` 隐式控制可用范围，目标不匹配由 handler 返回业务错误。

---

## 可选导出

### `EXTERNALLY_PERCEPTIBLE: bool`（默认 `False`）

声明工具成功执行时是否必然对外界产生可被其它客体感知的副作用，例如发送消息、撤回消息、戳一戳或复读消息。

```python
EXTERNALLY_PERCEPTIBLE: bool = True
```

这类工具由执行器优先按模型输出顺序串行执行，避免多个外部动作和其它工具并行交错。它们与焦点切换工具（例如 `enter_qq_session`）同轮调用时会被阻断；当前焦点切换后再发起外部动作，应由下一轮重新决定。启用 `tool_execution_guard` 后，如果工具真正执行前的 `<world>` 相比模型决策帧发生语义变化，这类工具还会先经过一次执行前守门判断；一旦某个外界可感知工具被守门拒绝，本轮后续外界可感知工具也会跳过并要求重新决策，但外界不可感知工具不受影响。

### `condition(config: dict) -> bool`

动态启用/禁用条件，返回 `False` 时工具不出现在任何场景。
用于基于配置或运行时状态的联动（例如：有记忆时才出现"删除记忆"工具）。

```python
def condition(config: dict) -> bool:
    from llm import memory as _memory
    return len(_memory.get_all()) > 0
```

---

## Namespace 渐进式披露

工具是否常驻、折叠或附挂不再由工具模块内的 `ALWAYS_AVAILABLE` 决定，而由 `src/tools/namespaces.yaml` 统一声明。

- `core` namespace 永久打开，不能关闭。
- 其它 namespace 默认折叠，只展示 namespace 名称和 description。
- 模型通过 `namespace_manage.open` 展开 namespace；展开只从下一轮开始生效。
- `namespace_manage.preview` 只返回目标 namespace 内工具的 `name + description`，不返回参数 schema。
- `namespace_manage.search` 只搜索未展开 namespace 内具体工具的 description。
- attach 只在 registry 中声明，工具模块不需要知道自己被哪个 namespace 临时展示。

工具模块不再声明“常驻/潜伏”或单工具作用域；只声明自己的 schema、handler、配置条件和执行元数据。未来如需作用域能力，应基于 namespace 重新设计，而不是给单个函数工具增加独立 scope 元数据。

### `repair_schema_args(args: dict) -> tuple[dict, list[str]]`

可选的 schema 结构修复钩子。
只做“修完后仍需再次通过 JSON Schema 严格校验”的安全修复，例如：

- 整错位字段归位
- 可明确识别的重复字段合并
- 工具专属但可证明安全的结构修正

如果修复需要运行时上下文，也可以导出：

```python
def make_schema_repairer(session, config):
    def repair_schema_args(args: dict) -> tuple[dict, list[str]]:
        ...
    return repair_schema_args
```

### `sanitize_semantic_args(args: dict) -> tuple[dict, list[str], str | None]`

可选的语义清洗/验证钩子。
输入在进入该阶段前，已经是合法 JSON 且通过 schema 校验的参数。

- 返回更新后的 `args`
- 返回变更记录列表
- 如果仍不可接受，返回非空错误信息，框架将阻断执行

如果需要运行时上下文，也可以导出：

```python
def make_semantic_sanitizer(session):
    def sanitize_semantic_args(args: dict) -> tuple[dict, list[str], str | None]:
        ...
    return sanitize_semantic_args
```

---

## ToolCollection 与过滤顺序

`build_tools(config, **context)` 现在返回 namespace-aware `ToolCollection`：

- `active_specs`: 当前 active namespace 中可直接传给 LLM 并执行的 `ToolSpec`
- `latent_specs`: 当前 inactive namespace 中可被发现但本轮不能直接执行的 `ToolSpec`
- `all_specs`: 本轮条件和运行时依赖满足的全部工具

每个 `ToolSpec` 统一承载：

- `declaration`
- `description`
- `prompt_signature`
- `handler`
- `externally_perceptible`
- `schema_repairer`
- `semantic_sanitizer`
- `namespace`

过滤顺序如下：

```
condition(config)
    ↓ False → 跳过
REQUIRES_CONTEXT（依赖对象存在性检查）
    ↓ 缺失 → 跳过
namespace registry
    ↓ 不属于任何 namespace → 跳过
namespace state
    ↓ active namespace / attach → ToolCollection.active_specs
    ↓ inactive namespace → ToolCollection.latent_specs
```

群聊专属工具不通过单工具作用域元数据过滤。它们应在 description 和 handler 返回中明确说明仅群聊可执行。

---

## 示例：完整的群聊专属工具

```python
DECLARATION: dict = {
    "name": "my_group_tool",
    "description": "...",
    "parameters": {...},
}

REQUIRES_CONTEXT: list[str] = ["qq_client", "session"]

def make_handler(qq_client, session):
    def execute(**kwargs) -> dict:
        if session.conv_type != "group":
            return {"error": "my_group_tool 仅能在群聊会话中使用"}
        ...
    return execute
```


