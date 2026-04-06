# 重构方案：纯函数调用架构 v1.0

> 本文档为最终实施前的设计对齐文档，新窗口开工时以此为准。

---

## 一、重构目标

| 目标       | 说明                                                                                                  |
| ---------- | ----------------------------------------------------------------------------------------------------- |
| 厂商兼容性 | 摈弃 `response_schema` 结构化输出，统一走 function calling，兼容所有支持工具调用的厂商                |
| 架构稳定性 | 消除"结构化输出 + 工具调用混用"引发的模型不稳定                                                       |
| 状态简化   | 摈弃 `previous_cycle` 全局状态，利用 adapter 持久化的 `_contents` 作为 bot 的新意识流                 |
| 工作量降低 | 不再需要手动管理上一轮 JSON 输出，不再需要 `previous_tools_used` 渲染，`activity_log` 不再注入 prompt |

**本次重构范围不包括 Watcher（窥屏）**：Watcher 在重构初期直接等同于 hibernate，后续单独处理。

**Web 界面本次直接架空**：`routes_chat.py` 的 `_run_web_model()` / `commit_bot_messages_web()` 等 Web 端 LLM 调用路径暂不适配新架构，保留文件但功能挂起，后续单独处理。

**mood / intent / expected 字段暂时取消**：重构初期仅保留 `thought` 工具作为内心活动载体，后续再议。

---

## 二、核心架构变化

### 2.1 废弃主模型结构化最终输出

- 删除 `RESPONSE_SCHEMA` 的使用（`response_mime_type: application/json`、`response_json_schema`）
- 删除 `config/schema/main.json` 对应的 Python 侧加载（`schema.py` 的 `RESPONSE_SCHEMA` 常量）
- **保留** `_schema_to_prompt()` 函数 —— IS / Watcher 仍走结构化输出，OpenAI 适配器路径需要此函数注入 schema 约束
- **保留** `json_repair.py` 及 OpenAI 适配器的 `_parse_and_validate_json()` / `_call_json_repair()` —— IS 路径仍需
- `call()` 的 `schema` 参数改为**可选**（默认 `None`），主模型不传，IS/Watcher 传各自 schema
- 主模型的所有行为均通过工具调用表达，**不再存在"最终 JSON 输出"**
- `WATCHER_SCHEMA` 保留，待后续 Watcher 重做时使用

### 2.2 bot 意识流：`adapter._contents`（设计哲学关键点）

> **属于机器人的东西，永远依附于机器人自身。**

`_contents` 是 bot 的新意识流（替代旧的 `_bot_previous_cycle`），挂在 **adapter 实例**上，而非 `ChatSession`。

- `GeminiAdapter._contents: list[types.Content] = []`
- `OpenAICompatAdapter._contents: list[dict] = []`（OpenAI messages 格式）
- Session 本身不持有、不感知 `_contents`

**`_contents[0]`** 永远是当前正在处理的会话的 user message（最新 chat log XML + 未读信息）。每次 activation 开始时刷新它：

```python
fresh_user_parts = build_user_parts(session)
if adapter._contents:
    adapter._contents[0] = types.Content(role="user", parts=fresh_user_parts)
else:
    adapter._contents = [types.Content(role="user", parts=fresh_user_parts)]
```

**Shift 时的 `_contents` 处理**：

- 仅更新 `_contents[0]`，换成新会话的 chat log，**历史工具调用记录保留**
- 机器人明确知道自己为何从哪里来，context 连续

**Idle / 重启时**：

```python
adapter._contents = []
```

**Pruning 策略**：每次将 model response 追加进 `_contents` 前，检查 model+tool 轮数是否超过配置项 `llm_contents_max_rounds`（建议默认 15）。超出则从索引 1 开始裁掉最老的一对（model response + function response）。`_contents[0]` 永远不参与裁剪。

### 2.3 `previous_cycle` 概念彻底废除

**删除清单**：

| 位置                                    | 删除内容                                                                                       |
| --------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `session.py` 全局变量                   | `_bot_previous_cycle`, `_bot_previous_tool_calls`, `_bot_previous_cycle_time`                  |
| `session.py` 函数                       | `get/set_bot_previous_cycle/time/tool_calls` 共6个函数                                         |
| `session.py` 函数                       | `extract_bot_messages()`, `_truncate_tool_calls_for_prompt()`                                  |
| `session.py` ChatSession 字段           | `previous_cycle_json`, `turn_start_seen_ids`, `pending_is_tip`                                 |
| `llm_core.py`                           | `set_bot_previous_cycle*` 的调用、`session.previous_cycle_json` 赋值                           |
| `retry.py`                              | `_snap_cycle` / `_snap_cycle_time` / `_snap_tool_calls` / `_snap_prev_json` 快照和回滚逻辑     |
| `prompt.py` SYSTEM_PROMPT               | 整个 `<previous_cycle>` 块                                                                     |
| `prompt.py` `build_system_prompt()`     | `previous_cycle_json`, `previous_cycle_time`, `previous_tools_used`, `previous_cycle_tip` 参数 |
| `session.py` `build_system_prompt()` 中 | `watcher_nudge` 分支里对 `prev` / `prev_tools` 的写入（watcher 暂停故不需要）                  |
| `lifecycle.py`                          | `load_last_bot_turn()` → `set_bot_previous_cycle()` 的启动恢复代码（注释掉）                   |

### 2.4 `RESULT_MAX_CHARS` / `summarize_result` 语义迁移（字段声明保留）

工具模块上的 `RESULT_MAX_CHARS` / `summarize_result` **保留**，但应用时机从"渲染 `previous_tools_used` 时"变为：

> **在 provider 将 function response 追加进 `_contents` 前**，立即对 result 数据做 summarize 或截断处理。

`_truncate_tool_calls_for_prompt()` 逻辑就地迁移为一个 provider 内部的 `_apply_result_limits(fn_name, result_data) -> Any` 辅助函数。

---

## 三、新工具清单

### 3.1 `thought` — 思考空间

```json
{
  "name": "thought",
  "description": "记录当前的内心想法。在做任何决策前调用，写下自然、真实的思考过程。",
  "parameters": {
    "type": "object",
    "properties": {
      "content": {
        "type": "string",
        "description": "你当前的内心想法，是私密的、自然的心理活动。"
      }
    },
    "required": ["content"]
  }
}
```

- Handler 直接返回 `{"ok": true}`，无副作用
- `RESULT_MAX_CHARS = 0`：result 不写进摘要，但 args（思考内容）在 `_contents` 里自然保留，模型可以回看

**与 native thinking 互斥**：Provider 层在构建 `config` 时，若 `thought` 在工具列表中，则强制 `thinking_config = None`（关闭模型原生思维链），写进 `GeminiAdapter.call()` 入口。

### 3.2 `send_message` — 发送消息（升级版，含副作用 + IS）

**签名**（数组形式，加 `motivation` 字段）：

```json
{
  "name": "send_message",
  "description": "向当前会话发送一条或多条消息。messages 数组中每个元素独立发送，按序执行。",
  "parameters": {
    "type": "object",
    "properties": {
      "motivation": {
        "type": "string",
        "description": "发这些消息的动机或原因。"
      },
      "messages": {
        "type": "array",
        "description": "要发送的消息列表，结构与原 send_messages.json 一致。",
        "items": { ... }
      }
    },
    "required": ["motivation", "messages"]
  }
}
```

（`messages[].items` 结构沿用现有 `send_messages.json` 的 item schema，含 `quote`、`segments`）

- `REQUIRES_CONTEXT: ["session", "napcat_client", "group_id", "user_id"]`（via `make_handler`）
- Handler 内部**串行**发送每条消息，发完每条立即写入 `session.context_messages`，持久化 DB
- **并行调用保护**：`send_message` 必须串行执行。若模型在同一轮并发调用了多个 `send_message`（极端情况），provider 在执行阶段对该工具强制串行（不放进 ThreadPoolExecutor），其他工具不受影响
- **打字延迟 & IS 耗时**：打字延迟由 `napcat/client.py` 的 `send_message()` 内部处理。若触发 IS 检查，IS 耗时可在 handler 内部从后续消息的打字延迟中扣除

**IS 触发条件**（内嵌在 handler 中）：

1. 本次调用包含 **2 条及以上消息**（仅 1 条时不触发 IS）
2. 发送第 `i` 条后，`session.context_messages` 中出现了新的非 bot 消息（与发送前快照对比）
3. 满足以上两个条件时，调用 `check_interruption()`，传入当前 `send_message` 的 `motivation`

**返回值**：

```json
{
  "sent_count": 2,
  "total_count": 4,
  "interrupted": true,
  "interrupt_reason": "用户\"张三\"发来消息"等等别说了"，应中断后续发送",
  "new_messages_count": 1
}
```

- 未中断时 `interrupted: false`，`sent_count == total_count`
- 模型在 function response 中直接读到结果，自行决定后续行为

**删除**：`send_and_commit_bot_messages()`、`_try_is_check()`（整函数删除，逻辑迁移进 handler）

**broadcast 迁移**：原 `send_and_commit_bot_messages()` 中的 `broadcast_chat_event()` 调用迁移到 `send_message` handler 内部，每条消息发送后照常推送到 debug 前端。

### 3.3 `wait` — 等待（合并 `short_wait` 和旧 `loop_control.wait`）

```json
{
  "name": "wait",
  "description": "等待一段时间或等待新消息到达后继续。用于需要等待对方回复、或暂缓决策的场景。",
  "parameters": {
    "type": "object",
    "properties": {
      "timeout": {
        "type": "integer",
        "minimum": 1,
        "maximum": 300,
        "description": "最长等待秒数。"
      },
      "early_trigger": {
        "type": "string",
        "enum": ["new_message", "mentioned"],
        "description": "提前唤醒条件（可选）。"
      },
      "motivation": { "type": "string" }
    },
    "required": ["timeout", "motivation"]
  }
}
```

- Handler 阻塞等待，以**工具被调用那一刻的 `session.context_messages` ID 集合**为基准，等待结束后返回期间新增的消息
- Provider 检测到 `wait` 工具被调用时，**立刻退出工具循环**，将 `loop_action = {"action": "wait", "timeout": ..., "early_trigger": ...}` 返回给上层
- `wait` 的实际异步等待逻辑在 `napcat_handler._run_active_loop()` 中执行（与现在相同），等待结束后重新触发下一轮 activation

旧 `short_wait/` 目录**移至 `not_used/`**，旧 `loop_control.wait` schema 字段**删除**。

### 3.4 `idle` — 进入休眠

```json
{
  "name": "idle",
  "description": "结束当前 activation，进入休眠/窥屏状态。对于简单交互、话题自然结束的情况使用。",
  "parameters": {
    "type": "object",
    "properties": {
      "motivation": {
        "type": "string",
        "description": "为什么要结束当前激活。"
      }
    },
    "required": ["motivation"]
  }
}
```

- Handler 返回 `{"ok": true}`
- Provider 检测到 `idle` 调用后**立刻退出工具循环**，返回 `loop_action = {"action": "idle", "motivation": ...}`
- 退出后 `adapter._contents = []`（意识流清空）

**隐式 idle 兜底**：若模型返回时完全没有工具调用（纯文本输出）：

- 记 `WARNING` 日志
- 系统自动视为 `idle`，`motivation = ""`

### 3.5 `shift` — 切换会话

```json
{
  "name": "shift",
  "description": "切换到另一个会话并立即激活一次循环。目标必须在白名单内。",
  "parameters": {
    "type": "object",
    "properties": {
      "type": { "type": "string", "enum": ["private", "group"] },
      "id": { "type": "string", "description": "目标会话 ID。" },
      "motivation": { "type": "string" }
    },
    "required": ["type", "id", "motivation"]
  }
}
```

- Handler 做白名单校验，失败时直接在返回值中说明原因（不再需要 `pending_error_logger`）
- Provider 检测到 `shift` 调用后**立刻退出工具循环**，返回 `loop_action = {"action": "shift", "type": ..., "id": ..., "motivation": ...}`
- `napcat_handler` 拿到后更新 `adapter._contents[0]`（换新 session 的 chat log），旧 session 无需清空 contents
- **`pending_error_logger` 字段删除**（shift 失败原因已在工具返回值中）

---

## 四、`call()` 接口变更

### 旧签名

```python
def call(self, system_prompt_builder, user_content, gen, schema,
         tool_declarations=None, tool_registry=None, latent_registry=None,
         user_content_refresher=None
) -> tuple[dict | None, dict | None, bool, list[dict], str]
# (result_dict, grounding, repaired, tool_calls_log, system_prompt)
```

### 新签名

```python
def call(self, system_prompt_builder, user_content, gen,
         schema=None,
         tool_declarations=None, tool_registry=None, latent_registry=None,
         user_content_refresher=None
) -> tuple[dict | None, list[dict], str]
# (loop_action, tool_calls_log, system_prompt)
```

- `schema` 参数改为**可选**（默认 `None`）——主模型不传，IS/Watcher 传各自 schema
- 当 `schema is not None` 且 `tool_declarations is None` 时，走结构化 JSON 输出路径（IS/Watcher 遗留行为）
- 当 `schema is None` 时，走纯 function calling 路径（主模型新行为）
- `result_dict` → `loop_action`：`{"action": "idle"|"wait"|"shift", ...params}` 或 `None`（调用彻底失败）
- `repaired` **删除**（主模型路径不再需要，IS 路径内部自行处理不暴露）
- `grounding` **删除**（或合并进 `tool_calls_log` 元信息，后续再议）

### `call_model_with_retry()` 新签名

```python
async def call_model_with_retry(session, conv_key: str) -> tuple[dict | None, list[dict], str, float]
# (loop_action, tool_calls_log, system_prompt, elapsed)
```

retry 逻辑**保留**，条件不变（`unread_count > 0` AND `tool_calls_log == []`）。删除其中 `_snap_cycle` 等所有 `previous_cycle` 快照/回滚相关代码。

---

## 五、Provider 层变更

### 5.1 GeminiAdapter

| 变更项               | 详情                                                                                                       |
| -------------------- | ---------------------------------------------------------------------------------------------------------- |
| 新增实例字段         | `self._contents: list[types.Content] = []`                                                                 |
| 删除配置项           | 主模型路径：`config_kwargs` 中的 `response_mime_type`, `response_json_schema`（`schema is None` 时不设置） |
| 思维链互斥           | 入口检测：若工具列表包含 `thought` → `thinking_config = None`                                              |
| `_contents` 生命周期 | 每次 activation 开始时更新 `_contents[0]`，不再是局部变量                                                  |
| Pruning              | 每轮 `contents.append(candidate_content)` 后检查轮数，超 N 则裁最老一对                                    |
| 终止检测             | 工具调用后检测是否调用了 `idle` / `wait` / `shift` → 立刻退出循环，返回 `loop_action`                      |
| 隐式终止             | 无 `function_calls` 时 → WARNING，返回 `loop_action = {"action": "idle"}`                                  |
| result 截断          | 追加 function response 进 `_contents` 前，调用 `_apply_result_limits()`                                    |

### 5.2 OpenAI 兼容适配器

| 变更项       | 详情                                                                                      |
| ------------ | ----------------------------------------------------------------------------------------- |
| 新增实例字段 | `self._contents: list[dict] = []`（OpenAI messages 格式）                                 |
| 保留         | `_schema_to_prompt()` —— 仅在 `schema is not None` 时使用（IS/Watcher 路径）              |
| 保留         | `_parse_and_validate_json()` / `_call_json_repair()` —— IS 路径仍需                       |
| 删除         | 主模型路径的 `response_format: {"type": "json_object"}` 设置（`schema is None` 时不启用） |
| 其余         | function calling 逻辑基本不变，对齐终止检测逻辑                                           |

---

## 六、`ChatSession` 变更

### 新增字段

无。`_contents` 在 adapter 上，不在 session 上。

### 删除字段

| 字段                   | 原因                                  |
| ---------------------- | ------------------------------------- |
| `previous_cycle_json`  | `previous_cycle` 概念废除             |
| `turn_start_seen_ids`  | `short_wait` 删除，刷新机制替代       |
| `pending_is_tip`       | IS 结果改由 `send_message` 返回值传递 |
| `pending_error_logger` | shift 失败改由工具返回值传递          |

**Watcher 相关字段保留**（`watcher_task`, `watcher_active`, `watcher_nudge` 等），但 `watcher_nudge` 相关的 `build_system_prompt()` 分支先注释掉（watcher 初期等同 hibernate）。

### `build_system_prompt()` 变更

**删除参数**：`previous_cycle_json`, `previous_cycle_time`, `previous_tools_used`, `previous_cycle_tip`

**删除调用**：

- `get_bot_previous_cycle()` / `get_bot_previous_tool_calls()` / `get_bot_previous_cycle_time()`
- `_truncate_tool_calls_for_prompt()`
- `build_activity_log_xml()` 的注入（`activity_log` 代码保留，只是不再传入模板）
- `watcher_nudge` 分支的 `prev` / `prev_tools` 写入

**`SYSTEM_PROMPT` 模板占位符删除**：

- `{previous_cycle_json}`, `{previous_cycle_time}`, `{previous_tools_used}`, `{previous_cycle_tip}` → 删除整个 `<previous_cycle>` 块
- `{activity_log}` → 从模板中移除占位符

---

## 七、System Prompt 变更

### 删除内容

```xml
<!-- 整个块删除 -->
<previous_cycle{previous_cycle_time}>
<output>{previous_cycle_json}</output>
<tools_used>{previous_tools_used}</tools_used>
<tip>{previous_cycle_tip}</tip>
</previous_cycle>
```

以及 dashboard 中的 `{activity_log}` 占位符。

### `<limitation>` 块更新

删除 `previous_cycle` 相关描述，更新为新架构的实际限制。

### `DEFAULT_INSTRUCTIONS` 更新

删除关于 `previous_cycle` 的引用，新增：

- 在做任何决策前先调用 `thought` 进行思考
- 调用 `send_message` 时必须提供 `motivation`
- 每次 activation 最终必须调用 `idle` / `wait` / `shift` 之一来结束

---

## 八、IS（中断哨兵）变更

IS 核心功能**保留**，但接口微调。

### `check_interruption()` 接口调整

**删除参数**：`result: dict`（不再有最终 JSON）

**调整参数**：

- `motivation: str`——从 `send_message` handler 传入当次调用的 `motivation`（替代原来从 `result.decision.motivation` 取）
- `think`（原从 result 取）：**删除**，IS prompt 中该字段移除
- `mood`（原从 result 取）：**删除**，IS prompt 中该字段移除

**IS prompt 微调**：移除 `mood`、`think` 相关段落，保留 `motivation` 传递给 IS 模型。

`is.json` schema 保持不变（`{"continue": bool, "reason": str}`）。

---

## 九、`napcat_handler.py` 变更

### 删除

- `send_and_commit_bot_messages()` 整函数
- `_try_is_check()` 整函数

### `_run_active_loop()` 重构

新的循环主体（概念代码）：

```python
async def _run_active_loop(session, conv_key, group_id, user_id, first_loop_action):
    loop_action = first_loop_action
    while True:
        action = (loop_action or {}).get("action", "idle")

        if action == "idle":
            # adapter._contents 清空，进入 hibernate/watcher（初期直接 hibernate）
            app_state.adapter._contents = []
            ...
            break

        elif action == "wait":
            # 处理 timeout + early_trigger（逻辑与现在基本相同）
            ...

        elif action == "shift":
            # 更新 adapter._contents[0] 为新会话的 chat log
            # 启动新 session 的 activation
            ...
            break

        # 否则 action 为 None（调用失败）也 break
        if loop_action is None:
            break

        # continue：重新 call_model_with_retry
        try:
            loop_action, tool_calls_log, _, elapsed = await call_model_with_retry(session, conv_key)
        except LLMCallFailed as e:
            ...
            break
```

注意：**`continue` 不再是一个工具**，也不再是 `loop_action` 的一个值。每次 activation 内部的工具循环由 provider 自主管理，`loop_action` 只在 activation **结束**时返回一次（值为 `idle` / `wait` / `shift`）。`_run_active_loop` 只负责处理 activation 的间隔调度。

**`activity_log` 调用时机迁移**：

原有的 `activity_log.open_entry()` / `close_current()` 调用从旧 `loop_control` 分支迁移到新 `loop_action` 分支：

- `idle` 分支：`close_current(action="idle", ...)` → `open_entry("hibernate")`（初期 watcher 等同 hibernate）
- `shift` 分支：`close_current(action="shift", ...)` → 新会话 `open_entry("chat", ...)`
- `wait` 分支：不 close（只是暂停等待，同一 activation 延续）

**`pending_early_trigger` 保留**：

`ChatSession.pending_early_trigger` 字段保留不删。新 `wait` 作为终止工具触发 `_run_active_loop` 的 wait 分支时，仍需消费此字段处理"bot 发消息/工具循环期间已收到的触发消息"。

---

## 十、`llm_core.py` 变更

`call_model_and_process()` 新职责：

1. 构建工具集（`build_tools()`，无变化）
2. 构建 `user_content`（chat log XML，与现在相同）
3. 构建 `user_content_refresher`（无变化）
4. 调用 `adapter.call()`（新签名，主模型不传 `schema`）
5. 返回 `(loop_action, tool_calls_log, system_prompt)`

删除：

- `RESPONSE_SCHEMA` import 及传递
- `set_bot_previous_cycle*` 的调用
- `session.previous_cycle_json` 赋值
- `repaired` 相关逻辑
- `commit_bot_messages_web()` —— Web 架空，此函数不再需要

**`save_bot_turn()` 语义变化**：

表结构不改，`result_json` 列改存 `loop_action` dict（如 `{"action": "idle", "motivation": "..."}`），`tool_calls` 列保持存 `tool_calls_log`。`load_last_bot_turn()` 启动恢复逻辑注释掉（不再需要恢复 `previous_cycle`）。

---

## 十一、完整删除清单

| 文件 / 位置                                                                | 操作                                                   |
| -------------------------------------------------------------------------- | ------------------------------------------------------ |
| `config/schema/main.json`                                                  | 删除（或归档，不再加载）                               |
| `src/llm/core/schema.py` — `RESPONSE_SCHEMA`                               | 删除（`WATCHER_SCHEMA` 保留）                          |
| `src/llm/core/provider.py` — `_schema_to_prompt()`                         | **保留**（IS/Watcher 的 OpenAI 路径仍需）              |
| `src/llm/core/json_repair.py`                                              | **保留**（IS 路径仍需）                                |
| `src/tools/short_wait/`                                                    | **移至 `not_used/`**                                   |
| `src/tools/send_short_message/`                                            | **移至 `not_used/`**（与新 `send_message` 功能重叠）   |
| `src/napcat_handler.py` — `send_and_commit_bot_messages()`                 | 删除（逻辑 + broadcast 迁移至 `send_message` handler） |
| `src/napcat_handler.py` — `_try_is_check()`                                | 删除（逻辑迁移至 `send_message` handler）              |
| `src/llm/session.py` — `extract_bot_messages()`                            | 删除                                                   |
| `src/llm/session.py` — `_truncate_tool_calls_for_prompt()`                 | 删除（逻辑迁移至 provider 内）                         |
| `src/llm/session.py` — `_bot_previous_cycle` 等全局变量及 get/set 6 个函数 | 删除                                                   |
| `src/llm/session.py` — `ChatSession.turn_start_seen_ids`                   | 删除                                                   |
| `src/llm/session.py` — `ChatSession.pending_is_tip`                        | 删除                                                   |
| `src/llm/session.py` — `ChatSession.pending_error_logger`                  | 删除                                                   |
| `src/llm/core/llm_core.py` — `commit_bot_messages_web()`                   | 删除（Web 架空）                                       |
| `src/lifecycle.py` — `load_last_bot_turn()` → `set_bot_previous_cycle()`   | 注释掉（不再需要恢复 `previous_cycle`）                |
| `src/llm/core/provider.py` — `ToolRepeatBreaker(name_only_tools=...)`      | 两处硬编码 `{"short_wait"}` 移除，初期用默认值         |
| `src/llm/prompt/prompt.py` — `{activity_log}` 注入                         | 移除占位符（`build_activity_log_xml` 函数本身保留）    |
| `src/llm/prompt/prompt.py` — `<previous_cycle>` 块                         | 删除                                                   |
| `src/llm/prompt/activity_log.py` — import & 调用                           | 从 `build_system_prompt()` 中移除调用                  |

---

## 十二、实施顺序建议

1. **新工具定义**
   - 新建 `thought`、`idle`（`src/tools/` 下新文件）
   - 改写 `wait`（覆盖 `short_wait/`，改名目录或迁移）
   - 迁移 `shift` 从 `napcat_handler` 逻辑 → 独立工具文件
   - 升级 `send_message`（副作用内嵌 + IS 内嵌 + broadcast 迁移，`make_handler` 模式）
   - `send_short_message/` 移至 `not_used/`

2. **Provider 层**
   - `adapter._contents` 新增实例字段
   - `call()` 新签名（`schema` 改为可选，删除结构化输出为主模型的默认行为）
   - 终止检测（`idle` / `wait` / `shift` 调用 → 退出循环）
   - 隐式 idle 兜底
   - `_apply_result_limits()` 辅助函数（替代 `_truncate_tool_calls_for_prompt`）
   - Pruning 实现
   - `thought` 工具 thinking_config 互斥
   - `ToolRepeatBreaker` 移除 `name_only_tools={"short_wait"}` 硬编码

3. **Session / system prompt 层**
   - `ChatSession` 字段删除
   - `build_system_prompt()` 清理（删除 `previous_cycle` 相关）
   - `SYSTEM_PROMPT` 模板更新
   - `DEFAULT_INSTRUCTIONS` 更新

4. **IS 层**
   - `check_interruption()` 接口微调（删除 `result`, `mood`, `think` 参数，传入 `motivation`）
   - IS prompt 文本更新

5. **`llm_core.py` + `retry.py`**
   - 对齐新签名
   - 删除 `previous_cycle` 快照/回滚代码
   - 删除 `commit_bot_messages_web()`（Web 架空）
   - `save_bot_turn()` 改存 `loop_action`

6. **`napcat_handler.py`**
   - 删除 `send_and_commit_bot_messages`
   - 重构 `_run_active_loop()`（含 `activity_log` 调用时机迁移）

7. **`lifecycle.py`**
   - 注释掉 `load_last_bot_turn()` → `set_bot_previous_cycle()` 恢复代码

8. **扫尾 & 清理**
   - 删除 `config/schema/main.json`
   - 删除 `src/llm/core/schema.py` 中 `RESPONSE_SCHEMA`（保留 `WATCHER_SCHEMA`）
   - `short_wait/` 移至 `not_used/`
   - `send_short_message/` 移至 `not_used/`
   - 移除 `ToolRepeatBreaker` 两处 `name_only_tools={"short_wait"}` 硬编码

---

## 附录：关键设计决策速查

| 问题                        | 决策                                                                       |
| --------------------------- | -------------------------------------------------------------------------- |
| 终止方式                    | 模型调用 `idle`/`wait`/`shift` → provider 退出循环；无工具调用 → 隐式 idle |
| `_contents` 挂在哪          | **adapter 实例**（bot 意识流属于 bot 自身）                                |
| shift 是否清空 `_contents`  | 否，仅更新 `_contents[0]`（换聊天记录），历史保留                          |
| idle 是否清空 `_contents`   | 是，`adapter._contents = []`                                               |
| Watcher 初期处理            | 直接等同 hibernate，不做 watcher 模式                                      |
| Web 界面                    | 本次直接架空，`routes_chat.py` LLM 调用路径暂不适配                        |
| mood/intent/expected        | 重构初期暂时取消，仅保留 `thought`                                         |
| native thinking             | 检测到 `thought` 工具 → 强制关闭 `thinking_config`                         |
| IS 触发条件                 | `len(messages) >= 2` AND 发送期间有新用户消息                              |
| IS 结果注入点               | `send_message` 工具的返回值（原生 function response）                      |
| IS 结构化输出               | IS 仍走 `schema` 参数（`call(schema=IS_SCHEMA)`），不受主模型重构影响      |
| `send_message` 并行调用     | 强制串行（不进 ThreadPoolExecutor）                                        |
| `send_short_message`        | 移至 `not_used/`，与新 `send_message` 功能重叠                             |
| `RESULT_MAX_CHARS` 应用时机 | 写入 `_contents` 前（provider 层 `_apply_result_limits()`）                |
| `call_model_with_retry`     | 保留，retry 逻辑不变，删除 `previous_cycle` 快照代码                       |
| `activity_log`              | 代码保留，不再注入 prompt；`open/close` 调用迁移到新 `_run_active_loop`    |
| `save_bot_turn()`           | 表结构不改，`result_json` 列改存 `loop_action` dict                        |
| `pending_early_trigger`     | 保留，新 `wait` 在 `_run_active_loop` 中仍消费此字段                       |
| `_schema_to_prompt()`       | 保留，IS/Watcher 的 OpenAI 路径仍需                                        |
| `json_repair.py`            | 保留，IS 路径仍需                                                          |
| `WATCHER_SCHEMA`            | 保留，待后续 Watcher 重做时使用                                            |
| `ToolRepeatBreaker`         | 移除 `name_only_tools={"short_wait"}` 硬编码，初期用默认值                 |
| broadcast                   | 从 `send_and_commit_bot_messages` 迁移到 `send_message` handler            |
