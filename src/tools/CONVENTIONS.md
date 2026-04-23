# 工具模块约定（Tool Module Conventions）

每个工具可以是以下两种形式之一，由 `__init__.py` 在启动时自动扫描加载：

- **单文件工具**：`src/tools/tool_name.py`
- **文件夹工具**：`src/tools/tool_name/__init__.py`（适合较复杂的工具，可在文件夹内拆分多个辅助模块）

> `not_used/` 文件夹及所有 `_` 开头的目录会被自动忽略。

---

## 必须导出

### `DECLARATION: dict`

工具的 schema 声明，包含：

- `name`: 工具名（字符串，唯一）
- `description`: 工具描述（给模型看）
- `parameters`: JSON Schema 格式的参数定义

如果 schema 需要动态生成（例如包含枚举值，或需要根据当前会话上下文裁剪字段），则导出 `get_declaration(...) -> dict` 函数替代静态 `DECLARATION`。
此时 `DECLARATION` 只需包含 `{"name": "工具名"}` 供框架识别。

`get_declaration` 支持按需声明上下文参数，例如 `session`、`config`；框架会按同名关键字注入。若无需上下文，也可以继续写成无参函数。

### 处理函数（二选一）

**方式 A：无运行时依赖**

```python
def execute(**kwargs) -> dict: ...
```

**方式 B：需要运行时对象（napcat_client、session 等）**

```python
REQUIRES_CONTEXT: list[str] = ["napcat_client", "session"]

def make_handler(napcat_client, session) -> Callable:
    def execute(**kwargs) -> dict: ...
    return execute
```

> `REQUIRES_CONTEXT` 的唯一职责是**依赖注入**——声明 `make_handler` 需要哪些运行时对象。
> 会话类型过滤由 `SCOPE` 负责，不要用 `REQUIRES_CONTEXT` 隐式控制可用范围。

---

## 可选导出

### `SCOPE: str`（默认 `"all"`）

声明工具适用的会话类型：

| 值          | 含义                     |
| ----------- | ------------------------ |
| `"all"`     | 群聊和私聊均可用（默认） |
| `"group"`   | 仅群聊可用               |
| `"private"` | 仅私聊可用               |

```python
SCOPE: str = "group"  # 仅群聊
```

### `condition(config: dict) -> bool`

动态启用/禁用条件，返回 `False` 时工具不出现在任何场景。
用于基于配置或运行时状态的联动（例如：有记忆时才出现"删除记忆"工具）。

```python
def condition(config: dict) -> bool:
    from llm import memory as _memory
    return len(_memory.get_all()) > 0
```

---

## 渐进式披露：`ALWAYS_AVAILABLE`（默认 `True`）

声明工具是否在每次 LLM 请求时常驻传入 schema：

| 值      | 含义                                                                   |
| ------- | ---------------------------------------------------------------------- |
| `True`  | 常驻工具，schema 始终传给 LLM（默认）                                  |
| `False` | 潜伏工具，默认不传 schema；模型需先调用 `get_tools` 激活，同轮即可使用 |

```python
ALWAYS_AVAILABLE: bool = False  # 默认不传 schema，需 get_tools 激活
```

潜伏工具的 schema 会出现在 system prompt 的 `<function_tools><hidden>` 中，
模型可以看到工具名并知道需要 `get_tools` 来激活。

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

`build_tools(config, **context)` 现在返回 `ToolCollection`：

- `active_specs`: 当前可直接传给 LLM 并执行的 `ToolSpec`
- `latent_specs`: 潜伏工具 `ToolSpec`，需经 `get_tools` 激活

每个 `ToolSpec` 统一承载：

- `declaration`
- `handler`
- `schema_repairer`
- `semantic_sanitizer`

过滤顺序如下：

```
condition(config)
    ↓ False → 跳过
SCOPE
    ↓ 不符合会话类型 → 跳过
REQUIRES_CONTEXT（依赖对象存在性检查）
    ↓ 缺失 → 跳过
ALWAYS_AVAILABLE
    ↓ True  → 注册到 ToolCollection.active_specs（常驻）
    ↓ False → 注册到 ToolCollection.latent_specs（等待 get_tools 激活）
```

---

## 示例：完整的群聊专属工具

```python
SCOPE: str = "group"

DECLARATION: dict = {
    "name": "my_group_tool",
    "description": "...",
    "parameters": {...},
}

REQUIRES_CONTEXT: list[str] = ["napcat_client", "group_id"]

def make_handler(napcat_client, group_id):
    def execute(**kwargs) -> dict:
        ...
    return execute
```
