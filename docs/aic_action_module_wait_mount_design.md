# AIC Action：Module Mount 与等待工具解耦设计

## 1. 背景

当前 `wait` 同时承担了三类职责：

1. core 常驻动作：让 agent 暂时挂起，等待下一轮重新观察。
2. QQ 社交等待：等待当前会话、平台任意会话、mention 等消息事件。
3. browser 等待：等待浏览器 world signature 变化。

这导致 `core` 知道太多具体 world 语义。`wait` 的 schema 里出现 `session`、`platforms`、`browser`、`world`，handler 直接读取 browser 状态，同时 QQ adapter handler 还要理解 `wait_early_trigger` 的内部结构。随着更多平台或工具域接入，这个工具会继续膨胀。

本次升级目标是把“核心等待动作”和“具体模块等待能力”拆开：

- `core` 保留一个极简、泛用、短时间的等待工具。
- QQ、browser、未来平台各自提供细粒度等待工具。
- 这些模块等待工具可以在模块启用时自动挂载到 `core`，模型无需手动打开额外 namespace。
- 工具源码仍归属具体模块，避免 `core` 下出现散装的具体平台等待实现；模型面对层使用 `wait_qq_event`、`wait_browser_event` 这样的短名称。

## 2. 设计原则

1. `core` 只拥有基础动作，不拥有具体 world 的业务语义。
2. module 是文件组织、生命周期和能力挂载单位；namespace 仍是模型可见工具组单位。
3. LLM 可见 namespace 不因文件夹重组而改变。
4. 不可见 namespace 不能被模型打开、关闭、预览或搜索。
5. 自动挂载是显式声明的生命周期行为，不能散落在工具实现里硬编码。
6. 挂载到 `core` 的能力必须是低污染、跨轮控制、环境等待或基础可见性恢复能力；业务动作仍需打开对应 namespace。
7. 等待类工具应通过元数据标记为 passive wait，不再依赖工具名白名单判断重复等待是否安全。

## 3. 概念分层

### Module

module 是代码和运行时归属单位，不直接出现在模型可操作的 namespace 列表中。

示例：

```text
qq
browser
memory
core
```

module 负责：

- 声明内部 namespace。
- 声明运行时条件，例如 QQ adapter 是否启用、browser world 是否存在。
- 声明自动挂载规则。
- 作为文件夹组织边界。

### Visible Namespace

visible namespace 是当前系统已有的模型可见工具组，例如：

```text
core
qq_social
qq_stickers
qq_profile
browser_use
```

这些 namespace 的模型面对语义保持不变：关闭时只展示名称和 description，打开时展示内部 tool schemas。

### Internal Namespace

internal namespace 是严格意义上的 namespace，但模型不可见、不可操作。

它用于承载某个 module 的运行时辅助工具，例如：

```text
qq_runtime
browser_runtime
```

internal namespace 的工具不能因为自身 namespace active 而暴露，只能通过 mount 规则挂到某个 visible namespace。

## 4. 目标目录结构

目标结构支持“功能文件夹 + 内部 namespace”：

```text
src/tools/
  core/
    wait/
    sleep/
    namespace_manage.py

  qq/
    qq_social/
      send_message/
      send_voice.py
      poke.py
    qq_stickers/
      list_stickers.py
      save_sticker.py
    qq_profile/
      get_user_info.py
    qq_runtime/
      wait_qq_event.py

  browser/
    browser_use/
      browser_control/
      browser_locator.py
    browser_runtime/
      wait_browser_event.py
```

对 LLM 来说，`qq` 和 `browser` 文件夹不存在。它仍然只看到 `core`、`qq_social`、`qq_stickers`、`browser_use` 等 namespace。

## 5. Registry 形态

新增 `modules.yaml` 承载 module 生命周期与 mount 规则；`namespaces.yaml` 继续承载 namespace 的模型可见性、路径与工具声明。

`modules.yaml` 示意：

```yaml
modules:
  core:
    path: core
    always_active: true

  qq:
    path: qq
    active_when: qq_adapter_enabled
    namespaces:
      - qq_social
      - qq_stickers
      - qq_profile
      - qq_runtime
    mounts:
      - from: qq_runtime
        to: core
        tools:
          - wait_qq_event
        when: qq_adapter_enabled

  browser:
    path: browser
    active_when: browser_available
    namespaces:
      - browser_use
      - browser_runtime
    mounts:
      - from: browser_runtime
        to: core
        tools:
          - wait_browser_event
        when: browser_world_active
```

`namespaces.yaml` 示意：

```yaml
namespaces:
  core:
    description: ""
    permanent: true
    closeable: false
    visible: true
    path: core
    tools:
      - namespace_manage
      - wait
      - sleep

  qq_social:
    description: "QQ 社交动作：发送文字/表情包/图片/语音消息、撤回消息、发起戳一戳、复读。"
    visible: true
    path: qq_social
    tools:
      - send_message
      - send_voice
      - recall_message
      - poke
      - plus_one

  qq_runtime:
    visible: false
    openable: false
    discoverable: false
    path: qq_runtime
    tools:
      - wait_qq_event

  browser_runtime:
    visible: false
    openable: false
    discoverable: false
    path: browser/browser_runtime
    tools:
      - wait_browser_event
```

`visible: false` 是硬边界：

1. 不渲染到 `<namespaces>`。
2. 不参与 `namespace_manage.open`。
3. 不参与 `namespace_manage.preview`。
4. 不参与 `namespace_manage.search`。
5. 不因为自身 active 而进入 `ToolCollection.active_specs`。
6. 只能通过 mount 进入目标 visible namespace。

## 6. Mount 规则

mount 是比现有 attach 更偏生命周期的机制。

attach 适合表达“打开 A namespace 时顺带暴露 B namespace 的某个辅助工具”，例如 `send_message` 需要 `list_stickers`。

mount 适合表达“某个 module 启用时，它的内部能力自动挂到一个常驻 namespace”，例如 QQ adapter 启用时 `wait_qq_event` 挂到 `core`。

规则：

1. mount 源必须来自 internal namespace。
2. mount 目标必须是 visible namespace，当前主要目标是 `core`。
3. mount 不递归携带源 namespace 的其它工具。
4. mount 不让源 namespace 变成可见或可打开。
5. mount 生命周期由 module/runtime 条件决定，不由模型 open/close 决定。
6. mount 工具调用时刷新目标 namespace 的生命周期；如果目标是 permanent `core`，则无额外影响。
7. mount 工具在 prompt 中显示在目标 namespace 下，不标注来源；本地 `ToolSpec.namespace` 仍保留真实归属和 mounted target，供日志与调试使用。
8. mount 工具在目标 namespace 的工具列表中排在末尾，避免挤占 core 基础动作的优先阅读位置。

建议 `ToolSpec` 增加字段：

```python
namespace: str              # 真实归属，例如 qq_runtime
visible_namespace: str      # 模型面对归属，例如 core
mounted_to: str | None      # 例如 core
mounted_by_module: str | None
visibility: "visible" | "internal"
tool_kind: str | None       # 例如 passive_wait
```

## 7. 等待工具设计

### core.wait

`core.wait` 是泛用短等待，只需要时间参数。

建议 schema：

```json
{
  "name": "wait",
  "description": "短暂等待一段时间，然后重新观察世界。适合不确定是否还会有新变化、需要留出一点反应时间的场景。不会等待某个具体平台或浏览器条件。",
  "parameters": {
    "type": "object",
    "properties": {
      "seconds": {
        "type": "integer",
        "minimum": 1,
        "maximum": 15,
        "description": "等待秒数。"
      }
    },
    "required": ["seconds"]
  }
}
```

语义：

- 不接受 `scope`。
- 不接受 `early_trigger`。
- 不知道 QQ、browser 或其它平台。
- 超时后返回下一轮。
- 可被外部中断只作为进程级取消，不作为业务触发。

### qq_runtime.wait_qq_event

QQ 等待工具归属 `qq_runtime`，在 QQ module active 时挂到 `core`。

第一版只迁移现有 QQ 等待语义，不新增更细的事件模型。也就是说，参数形态暂时沿用当前 `wait` 的明确 QQ 等待语义：`seconds` 加 `early_trigger.scope/condition`。实现上把这些状态从 core wait 拆到 QQ wait request 中，但取消旧 `world` 模糊等待；这类“不确定等什么”的场景由 `core.wait(seconds)` 取代。

示意 schema：

```json
{
  "name": "wait_qq_event",
  "description": "等待 QQ 新消息或被提及事件。适合对话中停顿、等待对方继续说、或结束当前话题后等待其它 QQ 会话新动静。",
  "parameters": {
    "type": "object",
    "properties": {
      "seconds": {
        "type": "integer",
        "minimum": 1,
        "maximum": 60,
        "description": "最长等待秒数。"
      },
      "early_trigger": {
        "type": "object",
        "description": "QQ 等待范围以及提前唤醒条件。第一版沿用去掉 world 后的当前 wait 社交等待语义。",
        "properties": {
          "scope": {
            "type": "string",
            "enum": ["session", "platforms"],
            "description": "session 表示当前 QQ 会话；platforms 表示任意 QQ 会话。"
          },
          "condition": {
            "type": "string",
            "enum": ["any_change", "mentioned"],
            "description": "any_change 表示任意新消息；mentioned 表示私聊、@ 或回复等明确提及。"
          }
        },
        "required": ["scope", "condition"]
      }
    },
    "required": ["seconds", "early_trigger"]
  }
}
```

QQ adapter handler 只需要认识 QQ 自己的 wait request，不再读取 core wait 的 `scope`。这一阶段重点是拆分归属，不重新设计 QQ 等待逻辑；后续再考虑把 `scope/condition` 演进成更贴近 QQ 的 `target/event`。旧 `scope=world` 不迁入 QQ wait；模型需要模糊等待时使用短 `core.wait`。

### browser_runtime.wait_browser_event

browser 等待工具归属 `browser_runtime`，在 browser world active 时挂到 `core`。

第一版只迁移现有 browser 等待语义，不新增 `network_idle`、`dom_ready` 等新事件。也就是说，参数形态暂时沿用当前 `wait` 的明确浏览器等待语义：`seconds` 加 `early_trigger.scope/condition`，其中浏览器只消费 `scope=browser` 且 `condition=any_change`。旧 `scope=world` 不迁入 browser wait，由短 `core.wait` 取代。

示意 schema：

```json
{
  "name": "wait_browser_event",
  "description": "等待浏览器页面出现新变化。适合页面加载、图片生成、异步内容刷新或点击后等待结果。",
  "parameters": {
    "type": "object",
    "properties": {
      "seconds": {
        "type": "integer",
        "minimum": 1,
        "maximum": 60,
        "description": "最长等待秒数。"
      },
      "early_trigger": {
        "type": "object",
        "description": "浏览器等待范围以及提前唤醒条件。第一版沿用去掉 world 后的当前 wait 浏览器等待语义。",
        "properties": {
          "scope": {
            "type": "string",
            "enum": ["browser"],
            "description": "browser 表示浏览器发生语义变化。"
          },
          "condition": {
            "type": "string",
            "enum": ["any_change"],
            "description": "浏览器第一版只支持页面语义变化。"
          }
        },
        "required": ["scope", "condition"]
      }
    },
    "required": ["seconds", "early_trigger"]
  }
}
```

browser 的具体 signature 轮询逻辑从 core wait 迁到 browser module；DOM、network 等更细语义后续再演进。

## 8. Passive Wait 元数据

当前重复等待豁免如果按工具名判断，会在拆出 `wait_qq_event` 后变脆。

建议等待工具导出：

```python
TOOL_KIND = "passive_wait"
```

或：

```python
TOOL_EFFECT = {
    "surface": "core",
    "kind": "passive_wait"
}
```

执行层和 duplicate-response guard 应使用元数据判断：

- `wait`
- `sleep`
- `wait_qq_event`
- `wait_browser_event`

这些工具重复出现通常是正常空转行为，不应被当成重复外部动作。

## 9. Prompt 可见形态

QQ adapter active，browser inactive 时，模型看到：

```xml
<namespace name="core" active="true">[
  {"name":"namespace_manage",...},
  {"name":"wait",...},
  {"name":"sleep",...},
  {"name":"wait_qq_event",...}
]</namespace>
<namespace name="qq_social" description="QQ 社交动作：..." active="false"/>
<namespace name="qq_stickers" description="QQ 表情包收藏管理。" active="false"/>
```

模型看不到：

```xml
<namespace name="qq_runtime" .../>
```

模型也不需要：

```json
{"name":"namespace_manage","arguments":{"open":["qq_runtime"]}}
```

如果同时挂载多个 runtime wait 工具，prompt 中仍然只显示在 `core` 下，并排在 `core` 原生工具之后：

```xml
<namespace name="core" active="true">[
  {"name":"namespace_manage",...},
  {"name":"wait",...},
  {"name":"sleep",...},
  {"name":"wait_qq_event",...},
  {"name":"wait_browser_event",...}
]</namespace>
```

## 10. 构建流程

目标构建流程：

```text
load module registry
  -> discover namespace paths, including nested module paths
  -> import tool modules
  -> evaluate module active_when
  -> evaluate tool condition / REQUIRES_CONTEXT
  -> assign real namespace
  -> compute visible namespace state
  -> apply open / close / ttl for visible namespaces
  -> apply attach rules for visible namespace dependencies
  -> apply module mount rules for internal runtime tools
  -> build active_specs / latent_specs / all_specs
  -> render only visible namespaces
```

`all_specs` 可以包含 internal namespace 工具，便于诊断和执行映射；但 active execution 必须基于 visible availability，而不是 internal namespace 状态。

## 11. 与现有 namespace/attach 的关系

保留现有 namespace 语义：

- `core` permanent。
- visible namespace 默认 closed。
- `namespace_manage.open` 下一轮生效。
- preview/search 不返回 schema。
- TTL 仍按 visible namespace 管理。

新增：

- module 文件夹层。
- nested namespace path。
- internal namespace。
- module mount。
- passive wait 元数据。

attach 和 mount 的差异：

| 机制 | 触发来源 | 源 namespace | 目标 namespace | 模型能否操作源 | 用途 |
| --- | --- | --- | --- | --- | --- |
| attach | host namespace open | visible namespace | visible namespace | 可以 | 参数准备/辅助能力 |
| mount | module runtime active | internal namespace | visible namespace | 不可以 | 常驻运行时能力 |

## 12. 迁移步骤

### Phase 1：文档与 registry 扩展

1. 新增 `modules.yaml`，引入 module-aware registry 数据结构。
2. 支持 namespace `path` 字段。
3. 保持旧 flat `src/tools/<namespace>` 扫描兼容。
4. 增加 `visible/openable/discoverable` 字段，但默认保持当前行为。

### Phase 2：core.wait 缩小

1. 将 `wait` schema 缩小为 `seconds: 1..15`。
2. 移除 `scope` 和 `early_trigger`。
3. `wait` handler 只做短时间睡眠。
4. 保留参数修复兼容一小段时间，但 prompt 不再教学旧字段。

### Phase 3：QQ wait 迁出

1. 新增 `src/platforms/qq/tools/qq_runtime/wait_qq_event.py`。
2. 将 `ConversationSession.wait_event` 相关字段改名或收敛为 QQ wait request 状态，避免看起来属于 core wait。
3. QQ adapter handler 只消费 QQ wait request。
4. 在 QQ module active 时把 `wait_qq_event` mount 到 `core`。
5. 第一版保持当前明确 QQ 等待语义，不新增 `target/event` 语义，也不迁入旧 `world` 模糊等待。

### Phase 4：browser wait 迁出

1. 新增 `src/tools/browser/browser_runtime/wait_browser_event.py`。
2. 将 browser signature polling 从 core wait 移走。
3. 在 browser world active 时把 browser wait mount 到 `core`。
4. 第一版保持当前明确 browser 等待语义，不新增 `network_idle`、`dom_ready` 等事件，也不迁入旧 `world` 模糊等待。

### Phase 5：重复等待与测试

1. 将 duplicate guard 的 passive wait 判断从工具名集合改为工具元数据。
2. 增加 mount 渲染测试。
3. 增加 internal namespace 不可 open/preview/search 的测试。
4. 增加 QQ wait 和 browser wait 的 handler 单测。
5. 增加 prompt snapshot 验证，确保 internal namespace 不进入模型可见列表。

## 13. 决策点

已建议确定：

1. 外层 `qq`、`browser` 称为 module，不称为 namespace。
2. `qq_runtime`、`browser_runtime` 是 internal namespace。
3. internal namespace 对模型不可见、不可打开、不可搜索。
4. module active 时可以 mount internal tools 到 `core`。
5. core `wait` 缩小为短等待，不再承担具体 world 条件。
6. 具体等待语义属于具体 module。
7. module registry 拆出 `modules.yaml`。
8. mount 工具在 prompt 中不标注来源，但排在目标 namespace 工具列表末尾。
9. `wait_qq_event` 第一版暂时保持当前明确 QQ 等待语义，只迁移归属，不新增等待逻辑，不支持 `world`。
10. `wait_browser_event` 第一版暂时保持当前明确 browser 等待语义，只迁移归属，不新增等待逻辑，不支持 `world`。
11. 旧 `world` 模糊等待语义取消，由 `core.wait(seconds)` 的短等待取代。

