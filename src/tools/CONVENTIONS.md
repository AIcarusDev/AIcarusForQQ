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

如果 schema 需要动态生成（例如包含枚举值），则导出 `get_declaration() -> dict` 函数替代静态 `DECLARATION`。
此时 `DECLARATION` 只需包含 `{"name": "工具名"}` 供框架识别。

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

### `WATCHER_ALLOW: bool`（默认 `False`）

是否在**窥屏（watcher）模式**下可用。

窥屏模式下，`SCOPE` 过滤**仍然生效**（根据窥屏目标的会话类型），在此基础上再额外要求 `WATCHER_ALLOW = True`。

```python
SCOPE: str = "group"  # 仅群聊
WATCHER_ALLOW: bool = True  # watcher 模式可用
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

> **注意**：watcher 模式下，`build_tools` 返回的 `latent_registry` 会被忽略（`_`），
> watcher 不支持渐进式披露。

---

## 过滤优先级（build_tools 执行顺序）

```
condition(config)
    ↓ False → 跳过
SCOPE（普通模式和窥屏模式均生效）
    ↓ 不符合会话类型 → 跳过
WATCHER_ALLOW（仅窥屏模式额外检查）
    ↓ False → 跳过
REQUIRES_CONTEXT（依赖对象存在性检查）
    ↓ 缺失 → 跳过
ALWAYS_AVAILABLE
    ↓ True  → 注册到 declarations + registry（常驻）
    ↓ False → 注册到 latent_registry（等待 get_tools 激活）
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
