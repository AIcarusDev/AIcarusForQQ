# AIC Action System：Namespace 工具重构设计

## 1. 背景

AIC Action System 是项目内的 agent 动作执行层。它不依赖厂商原生 function calling，而是把工具的模型可见签名作为普通上下文发给模型，再从模型文本中解析 `<action><tool_call>...</tool_call></action>`。后端仍保留 JSON Schema 做严格参数验证。

这次重构的核心目标，是把“工具是否展示给模型”的管理单位从单个函数工具提升为 namespace。模型默认只看到极少量常驻能力；当任务需要某一类能力时，再展开对应 namespace 的完整工具签名。

需要解决的问题：

1. 不同模型/厂商对原生函数工具字段支持不一致。
2. 模型在一次回复里既要输出自然语言认知，又要稳定输出工具调用，原生 function calling 不适合承载这一层认知流。
3. 大量工具签名常驻会带来 token 消耗、注意力污染和错误工具选择。
4. 现有 hidden tool group 仍是“隐藏/展开工具集”，还没有 namespace 的寿命、附挂、展开顺序和状态恢复语义。

## 2. 设计原则

1. `core` 是唯一永久常驻 namespace，不能被关闭，并始终作为稳定 prompt 前缀的一部分。
2. 除 `core` 外，其它工具都是平等的“二等公民”：默认只展示 namespace 名称和用途，不展示内部 schema。
3. namespace 管理只负责 prompt 可见性，不改变工具执行安全边界。外界可感知副作用、执行前守门、串行执行、焦点切换工具同轮阻断仍然是工具级元数据和执行器职责。
4. namespace 名称表达领域边界；工具名称表达动作本身。工具名称可以在本次重构中清理，但最终名称仍要短、稳定、可读。
5. 展开 namespace 是状态变化，不是一次性文本替换。它需要明确 open、close、preview、search、TTL、恢复和过期规则。
6. attach 是窄机制，只用于“当前 namespace 的核心动作必须依赖另一个 namespace 的辅助工具准备参数”的场景，不能泛化成跨 namespace 大杂烩。
7. 不再按“群聊工具/私聊工具”拆 namespace，也不在 prompt/build 阶段按会话类型过滤工具。工具可以被模型看到和调用；目标会话不合适时，由工具执行层返回明确业务错误。
8. 对确实只适用于群聊、群公告、群名片等场景的工具，必须在工具 description 中明确写出适用边界，避免模型误以为它是通用动作。
9. namespace 的 open/close 状态是唯一全局状态，不按会话或焦点隔离。模型能在意识流里看到自己刚刚打开过某个 namespace，就不应因为切换会话而被迫重新激活一次。

## 3. 模型面对 AIC Action

目标 prompt 形态：

```md
<tools>

## Examples

Single tool:
<action>
<tool_call>{"namespace":"namespace_name", "name":"tool_name","arguments":{...}}</tool_call>
</action>

Multiple tools:
<action>
<tool_call>{"namespace":"namespace_name", "name":"tool_name","arguments":{...}}</tool_call>
<tool_call>{"namespace":"namespace_name", "name":"tool_name","arguments":{...}}</tool_call>
...additional tools as needed...
</action>

## Rules

- Output one or more `<tool_call>` blocks in `<action>` in the order they are executed.
- Each `<tool_call>` must contain one JSON object.
- Each JSON object must contain `namespace`, short `name`, and `arguments`. Do not put the namespace inside `name`.
- The `arguments` object must conform to the matching tool signature in the active namespace list.
- Tools in inactive namespaces cannot be executed directly. Use `namespace_manage.open` first.

## Namespaces

<namespaces>
  {namespace_blocks}
</namespaces>

</tools>
```

其中 `{namespace_blocks}` 的目标形态：

```xml
<namespace name="core" active="true">
// 核心的运行状态管理工具，实现等待、闲置、休眠。
runtime_manage(args:
  | {
      action: "wait";
      seconds?: number; // 范围 1~180，单位秒，默认 10。
    }
  | {
      action: "idle";
      minutes?: number; // 范围 1~60，单位分钟，默认 5。
    }
  | {
      action: "sleep";
      minutes?: number; // 范围 30~600，单位分钟，默认 480。
    }
)
</namespace>
<namespace name="qq_social" description="QQ 社交动作：发送、撤回、戳一戳、复读。" active="false"/>
<namespace name="qq_stickers" description="QQ 表情包收藏管理。" active="false"/>
<namespace name="browser_use" description="重型浏览器控制和精确 DOM 定位。" active="false"/>
```

渲染规则：

1. `active="true"` 的 namespace 展示完整 TypeScript-like tool signature。
2. `active="false"` 的 namespace 只展示 `name` 和 `description`。
3. 已展开 namespace 不再重复展示 namespace description，因为内部工具 description 已经提供细节。
4. `description` 统一使用完整字段名，不再使用 `des`。
5. `active="true"` 使用正确布尔文本，避免 `ture` 这类拼写进入模型契约。
6. 后端工具 schema 只用于参数校验，不再原样进入 prompt；模型可见签名用 `//` 注释保留必要的适用场合、语义和细节引导。
7. 新工具不默认强制 `additionalProperties: false`。当前主要问题不是模型乱加字段，收益有限；关键结构通过 `required`、`if/then`、参数修复和语义校验控制。

## 4. Namespace 状态机

### 状态

| 状态        | 含义                                                      |
| ----------- | --------------------------------------------------------- |
| `closed`    | namespace 可被发现，但内部工具 schema 不进入 prompt。     |
| `open`      | namespace 已展开，内部工具 schema 进入 prompt，可被调用。 |
| `permanent` | 永久打开且不可关闭。当前只有 `core`。                     |
| `attached`  | 某工具来自其它 namespace，但随当前 namespace 临时展示。   |
| `expired`   | open namespace 超过寿命后自动折叠回 closed。              |

### 转移规则

1. `namespace_manage.open` 将 closed namespace 变为 open。
2. `namespace_manage.close` 将 open namespace 折叠为 closed；`core` 不能被关闭。`close` 是立即生效的顺序动作。
3. `namespace_manage.preview` 和 `namespace_manage.search` 只读，不改变 namespace 状态。
4. 调用某个 open namespace 内的工具，会刷新该 namespace 的寿命。
5. attach 工具被调用时，刷新发起 attach 的 host namespace 寿命。
6. 如果 attach 工具所属 namespace 自身也已 open，则该工具按所属 namespace 计算寿命，不再重复挂在 host namespace 下。
7. `namespace_manage.open` 只影响下一轮及后续 prompt，本轮不动态注入 schema，也不允许同轮继续执行新打开 namespace 内的工具。
8. 模型直接调用 inactive namespace 内的工具，或在同一 `<action>` 中先 open 再调用该 namespace 内工具时，不执行目标工具；系统返回明确回执，说明本轮没有该工具 schema、namespace 已打开、下一轮才可真正调用。
9. 如果模型在同一 `<action>` 中先调用当前已可用的 namespace 工具，再 `close` 该 namespace，则按顺序优雅执行。此时前面的工具调用不再续命，`close` 是本次 namespace 生命周期的休止符。
10. 如果模型先 `close` 某 namespace，再在同一 `<action>` 后续调用该 namespace 内工具，则 `close` 已生效，后续工具被拒绝，并返回“顺序逻辑错误 / namespace 已关闭”的明确回执。
11. schema 校验失败、业务失败、执行前守门拒绝，都仍然算作模型命中了该 namespace，可以刷新寿命；未知工具和 AIC Action 解析错误不刷新寿命。但同轮后续 close 会覆盖这次续命。
12. namespace 状态全局唯一。切换 QQ 会话、焦点或浏览视图不会自动清空 open namespace；只有 TTL、显式 close、运行时重启或配置变化会影响它。
13. 全局 open 不代表当前焦点一定可执行该 namespace 的所有工具。配置关闭、运行时对象不可用时，工具仍可不出现；目标会话不匹配时不做 prompt/build 过滤，而是在执行层返回明确不可用原因。

### 寿命单位

建议使用“轮数”而不是秒数：

- `core.ttl_rounds = null`
- 其它 namespace 可在 namespace registry / yaml 中分别配置 `ttl_rounds`
- 未显式配置时，默认等于用户设置中“意识流喂回模型的最大轮数”
- 每次 open 或命中 namespace 内工具时，记录 `last_active_round`
- 每轮构建 prompt 前折叠 `current_round - last_active_round > ttl_rounds` 的 namespace

这样 namespace 可见寿命与模型仍能看到的工具调用历史保持一致，也便于从 flow 中恢复。

## 5. namespace_manage

`namespace_manage` 是 `tools_manage` 的目标形态。它属于 `core`，永久可用。

`open`、`close`、`preview` 都支持一次传入多个 namespace name。工具回执必须逐项说明实际打开、关闭、已处于目标状态、不可关闭和未找到的项目。

```json
{
  "name": "namespace_manage",
  "description": "管理工具 namespace 的展开、折叠、预览和搜索。只影响工具 schema 是否进入 prompt，不直接执行业务工具。",
  "parameters": {
    "type": "object",
    "properties": {
      "open": {
        "type": "array",
        "items": { "type": "string" },
        "description": "打开一个或多个 namespace，使其内部工具在下一轮可用。"
      },
      "close": {
        "type": "array",
        "items": { "type": "string" },
        "description": "关闭一个或多个 namespace。core 不能关闭。"
      },
      "preview": {
        "type": "array",
        "items": { "type": "string" },
        "description": "预览 namespace 内的工具名称和简短介绍，不展开 schema。"
      },
      "search": {
        "type": "string",
        "description": "用关键词搜索当前未展开 namespace 内部工具的 description。"
      }
    },
    "anyOf": [
      { "required": ["open"] },
      { "required": ["close"] },
      { "required": ["preview"] },
      { "required": ["search"] }
    ]
  }
}
```

回执建议：

```json
{
  "ok": true,
  "opened": ["qq_social"],
  "already_open": [],
  "closed": [],
  "protected": [],
  "not_found": [],
  "active_namespaces": ["core", "qq_social"]
}
```

`preview` 回执只返回目标 namespace 内所有工具的 `name + description`，不返回 `parameters`、`required` 或任何 schema 片段。`search` 只搜索未展开 namespace 内具体函数工具的 description，返回命中的工具名、工具简述和所属 namespace name，不搜索 namespace description / keywords，也不返回完整 schema。

## 6. Attach 机制

namespace 按领域归属划分，但任务链里的参数准备不一定只发生在同一领域内。attach 用来表达这种窄耦合。

例子：`send_message` 属于 `qq_social`，但发送表情包需要 `sticker_id`。`list_stickers` 的真实归属是 `qq_stickers`，却可以作为 `qq_social` 的 attach 工具出现。

规则：

1. attach 工具的真实归属不改变。
2. attach 工具只随 host namespace 展示，不递归携带自己所属 namespace 的其它 attach。
3. host namespace close/expire 时，attach 工具随 host 消失。
4. 如果 attach 工具所属 namespace 自身 open，则该工具回到所属 namespace 下展示，避免重复 schema。
5. attach 的展示寿命与 host namespace 绑定。
6. attach 默认应是只读或参数准备工具。这是文档层面的设计契约，不做框架级硬限制；若确有特殊 side-effect attach，必须在 registry 中写明原因并单独处理。
7. attach 只用于核心动作的必要辅助，不用于“可能顺手会用到”的便利挂载。

示例：

```yaml
qq_social:
  tools:
    - send_message
    - send_voice
    - recall_message
    - poke
    - plus_one
  attach:
    - namespace: qq_stickers
      tool: list_stickers
      reason: "send_message 发送 sticker 需要 sticker_id。"
```

## 7. 目标 Namespace 清单

### core

永久打开，不可关闭，优先缓存。

工具：

- `namespace_manage`（当前 `tools_manage` 的目标名）
- `calculator`
- `runtime_manage`
- `enter_qq_session`
- `think_deeply`
- `recall_memory`
- `goal_manage`（合并当前 `create_goal` + `resolve_goal`，常驻）
- `restart`（当前 `restart_self`，基础自我恢复能力，常驻 core）
- `view_image_by_ref` 或 `examine_image`（二选一；`view_image_by_ref` 当前实现名为 `get_image_by_ref`）
- `web_search`
- `web_extract`
- `get_weather`

说明：

1. 目标管理工具合并为 `goal_manage`，常驻 core，替代当前 `create_goal` 和 `resolve_goal` 两个 public tool。这样不再因为 `resolve_goal` 的 active-goal 条件改变 core 工具 schema。`goal_manage` 使用 `action` discriminator，并用 JSON Schema `if/then` 明确约束：`action=create` 时要求创建目标所需字段，`action=resolve` 时要求 `goal_ids` 和 `resolution`。
2. 图像工具必须二选一：
   - 主模型支持直接看图：使用 `view_image_by_ref`（由当前 `get_image_by_ref` 改名），用于查看 `<world>` 中那些因为上下文预算或注入策略而只展示 image_ref、没有真正注入多模态内容的图片。
   - 主模型不支持直接看图：使用 `examine_image`，通过视觉桥对指定图片做定向或多角度观察。它是给“瞎子模型”补视觉的工具，不受“最大真实多模态信息”配置限制。
   - 两者天然互斥，不能同时出现在同一轮工具 schema 中。
3. `get_self_image` 不进入任何 namespace，归入 `not_used` / 待清理工具，不作为 core 常驻候选。
4. `restart` 是 core 常驻基础能力。它本身只是重启进程，不应在模型面对层被视为高风险工具；真正的安全边界在后端重启实现，必须保证状态落盘、重复触发处理和本轮剩余工具中断语义正确。
5. `web_search`、`web_extract`、`get_weather` 属于轻量外界感知能力，固定放在 core 常驻，不拆成单独 `web_info` namespace。
6. `think_deeply` 保持 core 常驻。它是认知辅助工具，不拆入独立 cognition namespace，也不参与外部动作守门。
7. `recall_memory` 保持 core 常驻。长期记忆检索是基础认知能力，不拆入独立 memory namespace。

### qq_social

QQ 社交动作，主要是外界可感知工具，也是执行前守门重点关注的一组。

工具：

- `send_message`
- `send_voice`（当前 `send_voice_message`）
- `recall_message`
- `poke`
- `plus_one`

attach：

- `qq_stickers.list_stickers`

说明：

1. `send_message`、`send_voice`、`recall_message`、`poke`、`plus_one` 都属于外界可感知动作，namespace 展开不改变守门策略。
2. `plus_one` 当前描述为群聊工具；目标设计中不再按会话类型过滤。若在私聊中可合理复读并发送，则工具应支持；若目标消息或发送目标不满足要求，则执行层返回明确错误。
3. `send_voice` 常驻在 `qq_social` 中展示。即使 TTS 未启用或 worker 不可用，也先由执行层返回明确错误；后续如果真实使用中需要降低噪声，再考虑按配置精细化摘除。

### qq_stickers

QQ 表情包收藏管理。

工具：

- `list_stickers`
- `save_sticker`
- `update_sticker`
- `delete_sticker`

说明：

1. `list_stickers` 可被 `qq_social` attach。
2. 管理类工具不应全部 attach 到社交 namespace；只有 `list_stickers` 是发送 sticker 的必要参数准备工具。

### qq_chat_log_view

QQ 聊天记录窗口浏览和局部历史检索。

工具：

- `scroll_chat_log`
- `search_history`（当前 `search_current_session_chat_history`）

说明：

1. `search_history` 只搜索当前会话历史，不等于跨会话 `search_session`。

### qq_forward_view

QQ 合并转发消息浏览。

工具：

- `browse_forward`（由当前 `open_forward_message` + `browse_forward_view` 合并为单个工具）

说明：

1. `open_forward_message` 和 `browse_forward_view` 必须合并为 `browse_forward`。统一 schema 使用 `action` 区分 `open`、`next_page`、`prev_page`、`back`、`close_all`；只有 `action=open` 时需要 `id`。

### qq_profile

QQ 资料读取和维护。

工具：

- `get_user_info`
- `get_qq_signature`
- `set_qq_signature`（当前 `set_self_qq_signature`）
- `get_avatar`（当前 `get_user_avatar`）

说明：

1. `set_qq_signature` 仅做 public name 改名，参数和行为暂时沿用当前 `set_self_qq_signature`。它是 agent 自身资料维护，不算外界可感知工具，不走聊天世界变化守门；description 仍需明确它会修改 QQ 签名。
2. `get_avatar` 仅做 public name 改名，参数和行为暂时沿用当前 `get_user_avatar`；未来如果需要支持更多目标范围，再单独扩展。

### qq_contacts

QQ 联系人、群聊列表和会话搜索。

工具：

- `list_contact`（当前 `get_contact_list`）
- `search_session`

说明：

1. `search_session` 理论上是 `enter_qq_session` 的参数准备工具，但当前 agent 主要通过 QQ unread、当前消息中的用户/群 ID 或直接目标来触发 `enter_qq_session`，暂时不把它 attach 到 `core.enter_qq_session`，也不为了理论链路提前常驻化。
2. 后续如果真实日志显示模型频繁因为缺少 `search_session` 而无法切换，再考虑把它作为 `core.enter_qq_session` 的 attach。
3. `list_contact` 仅做 public name 改名，参数和行为暂时沿用当前 `get_contact_list`；联系人分页、类型过滤等扩展后续单独设计。
4. `list_contact` 与 `search_session` 的边界暂按现状理解：前者列举，后者搜索并解析可 enter_qq_session 的目标。

### qq_group_info

QQ群信息。

工具：

- `query_group_members`
- `get_group_notice`
- `set_group_card`（当前 `set_self_group_card`）

说明：

1. 当前这些工具有群聊上下文限制。目标设计中不再按会话类型过滤；工具 description 必须标明群聊适用边界，私聊/临时会话等目标不匹配时由执行层返回明确错误。
2. `set_group_card` 仅做 public name 改名，参数和行为暂时沿用当前 `set_self_group_card`。它是 agent 自身群资料维护，不算外界可感知工具，不走聊天世界变化守门；description 仍需明确它会修改当前群名片。

### browser_use

重型浏览器操作。

工具：

- `browser_control`
- `browser_locator`

说明：

1. 当前 `browser_control` 常驻，`browser_locator` 潜伏。目标是二者都归入 `browser_use`，按需展开。
2. `browser_use` 不自动首次打开；模型第一次需要浏览器工具时仍必须显式 `namespace_manage.open(["browser_use"])`。
3. 一旦 `browser_use` 已打开，且 `<world><browser>` 仍有活跃页面，就应自动保持 open，不因普通 TTL 到期而折叠。
4. `browser_control.close_browser` 成功后，应触发 `browser_use` 自动 close。
5. 这类生命周期联动必须通过 namespace registry 中的 lifecycle hook 明确声明，不能在工具实现里散落硬编码，避免 browser 工具和 namespace 管理强耦合。

## 8. 文件与数据结构

推荐把 namespace registry 作为唯一分组来源。工具模块继续负责 declaration、handler、repair、semantic sanitizer、side-effect metadata。

目标结构可以是：

```text
src/tools/
  namespaces.yaml
  core/
    runtime_manage.py
    ...
  qq_social/
    send_message/
    send_voice.py
    ...
  qq_stickers/
    list_stickers.py
    ...
```

也可以先不移动文件，只引入 registry 过渡：

```yaml
namespaces:
  core:
    description: ""
    permanent: true
    ttl_rounds: null
    tools:
      - namespace_manage
      - calculator
      - runtime_manage
      - enter_qq_session
      - think_deeply
      - recall_memory
      - goal_manage
      - restart
      - view_image_by_ref
      - web_search
      - web_extract
      - get_weather

  qq_social:
    description: "QQ 社交动作：发送、撤回、戳一戳、复读。"
    ttl_rounds: null # null 表示使用用户设置中的意识流最大回灌轮数
    tools:
      - send_message
      - send_voice
      - recall_message
      - poke
      - plus_one
    attach:
      - namespace: qq_stickers
        tool: list_stickers
        reason: "发送 sticker 需要 sticker_id。"

  browser_use:
    description: "重型浏览器控制和精确 DOM 定位。"
    ttl_rounds: null
    tools:
      - browser_control
      - browser_locator
    lifecycle:
      keep_open_while: "browser_world_active"
      close_on:
        - tool: browser_control
          action: close_browser
          ok: true
```

最低需要定义：

1. namespace 名称。
2. namespace description。
3. 是否永久开启、是否可关闭。
4. namespace 内工具顺序。
5. attach 工具来源、原因和展示规则。
6. 每个 namespace 的 TTL 配置；未配置时使用用户设置中的意识流最大回灌轮数。
7. lifecycle hook：例如保持 open、工具执行后自动 close 等自定义生命周期联动。
8. 动态 schema 或运行时条件工具的处理约定。
9. 当前工具名到目标工具名的迁移映射。

## 9. 构建与渲染流程

目标流程：

```text
discover tool modules
  -> build ToolSpec
  -> apply condition / runtime context availability
  -> assign ToolSpec to NamespaceSpec
  -> recover namespace state from flow/runtime state
  -> apply namespace open/close/ttl
  -> compute attach tools
  -> render <namespaces>
  -> execute parsed tool calls against active tools only
```

注意：

1. namespace registry 不替代工具模块的 `condition`、上下文依赖和执行校验。
2. active namespace 中的工具仍可能因为当前配置或运行时对象不存在而不出现。
3. 不设计“空 namespace”或 `available=false` 的 prompt 展示。如果某个 namespace 在当前配置/平台/运行时条件下不可用，构建阶段直接摘除整个 namespace。
4. namespace 状态应记录为唯一全局 runtime state，并可从意识流恢复，避免每轮丢失。它不按会话/焦点拆分。
5. 构建工具集合时不按单个函数工具的作用域元数据或当前会话类型过滤。会话类型边界属于工具 description 和执行层业务校验；未来如需作用域能力，应基于 namespace 重新设计。

## 10. Prompt 顺序与稳定前缀

目标顺序：

1. `core` 永远在最前。
2. 后续 namespace 按首次 open 的顺序追加。
3. 已 open namespace 被重复 open 或使用时，不改变顺序，只刷新 TTL。
4. namespace close/expire 后，其 schema block 从 prompt 中消失。
5. 重新 open 已关闭 namespace 时，追加到当前 open namespace 末尾。

稳定前缀注意点：

1. 当前实现不主动向 provider 发起 prompt cache 请求，也不在工具层维护 cache boundary。
2. prompt 层可以对已组装好的稳定前缀（system + tools）做字符串对比和日志诊断，但该逻辑不应依赖函数工具排序表。
3. namespace 内部工具顺序由 `namespaces.yaml` 决定；文件系统发现顺序和诊断日志不应成为模型面对顺序来源。

## 11. 与现有 hidden group 的关系

现有实现中：

- `ALWAYS_AVAILABLE=True` 表示常驻工具。
- `ALWAYS_AVAILABLE=False` 表示潜伏工具。
- `DiscoveryGroup` 把潜伏工具折叠为 `<tool_set>`。
- `tools_manage.get/preview/search` 负责展开、预览、搜索。

namespace 重构后：

1. `ALWAYS_AVAILABLE` 应被 namespace 的 `permanent/default_open` 替代。
2. `DiscoveryGroup` 应被 `NamespaceSpec` 替代。
3. `tools_manage` 必须彻底替换为 `namespace_manage`。模型面对层不保留旧名 alias，旧调用在 AIC Action 下直接视为未知工具。
4. `<tools><activated>/<hidden>` 应替换为 `<tools><namespaces>`。
5. 现有“激活整个 group”的行为可以迁移为“open namespace”。
6. 现有“直接调用隐藏工具时延迟激活、下一轮重试”的行为可以保留，但文案改成 namespace。

## 12. 迁移步骤

1. 建立 namespace registry 和 `NamespaceSpec`，先不移动工具文件。
2. 把当前 hidden groups 映射成目标 namespace，保留现有 ToolSpec 构建逻辑。
3. 引入 `namespace_manage`，替换当前 `tools_manage` 的 preview/search/open 语义；模型面对层不保留 `tools_manage` 兼容名。
4. 将 prompt 渲染从 `<activated>/<hidden>` 改为 `<namespaces>`。
5. 增加全局 namespace state：open 顺序、last active round、TTL、manual close。
6. 实现 attach 计算，先只支持 `qq_social -> qq_stickers.list_stickers`。
7. 更新执行器：只允许 active namespace 内工具执行；inactive 工具调用返回延迟展开回执；会话类型不匹配不在执行器统一拦截，而由工具自身返回业务错误。
8. 从 flow 中恢复 open namespace，而不是只恢复 latent tool name。
9. 更新测试：prompt 渲染、open/close、preview/search、TTL、attach、直接调用 inactive tool、namespace 顺序。
10. 工具 public name 直接切换到目标名，不保留 alias 或运行时兼容。旧名在 AIC Action 下视为未知工具。
11. 清理旧的 hidden group、`ALWAYS_AVAILABLE` 和旧 prompt 文案。

## 13. 已明确的设计决策与观察项

1. namespace manager 已确定彻底改名：模型面对层只允许 `namespace_manage`，不保留 `tools_manage` / `get_tools` 兼容。
2. namespace TTL 已确定：可在 yaml 中按 namespace 单独配置；未配置时等于用户设置中的意识流最大回灌轮数。
3. namespace 状态已确定为唯一全局状态，不按当前会话/焦点单独记录。
4. 会话类型边界已确定：namespace/prompt/build 阶段完全不按群聊/私聊过滤；工具 description 标明适用边界，目标不匹配时由工具执行层返回错误。
5. `search_session` 暂时保持在 `qq_contacts`，不作为 `core.enter_qq_session` attach；后续根据真实失败日志再评估。
6. `get_self_image` 已确定下线：归入 `not_used` / 待清理，不进入 namespace 清单。
7. `restart` 已确定为 core 常驻基础能力；风险边界在后端重启流程，不在 namespace 可见性层。
8. 图像 image_ref 工具最终 public name 已确定为 `view_image_by_ref`，由当前 `get_image_by_ref` 直接改名。
9. `examine_image` 与 `view_image_by_ref` 已确定天然互斥：多模态主模型使用 `view_image_by_ref`；非多模态主模型使用 `examine_image` 通过视觉桥观察，且不受最大真实多模态信息配置限制。
10. `browse_forward` 已确定合并 `open_forward_message` + `browse_forward_view`：用单个 `action` schema 兼容打开、翻页、返回、关闭。
11. `set_qq_signature`、`set_group_card` 已确定不算外界可感知工具，属于 agent 自身资料维护，不走聊天世界变化守门。
12. `browser_use` 生命周期已确定：不自动首次打开；打开后若 browser world 活跃则保持 open；`close_browser` 成功后自动 close。此类联动必须通过 namespace lifecycle hook 明确声明。
13. 空 namespace / unavailable namespace 已确定不展示；条件不满足时直接在构建阶段摘除整个 namespace。未来多平台条件显示也按 namespace 级摘除处理。
14. attach 副作用边界已确定为文档契约：默认只用于只读/参数准备工具，不做框架级硬限制；特殊 side-effect attach 必须在 registry 写明原因并单独处理。
15. namespace search 已确定：只搜索未展开 namespace 内具体函数工具的 description，返回命中工具及所属 namespace name；不搜索 namespace description / keywords。
16. 旧工具名已确定全部不兼容：所有 public tool name 直接改成目标名，不保留模型面对 alias，也不保留运行时 alias。
17. `namespace_manage.preview` 已确定只返回工具 `name + description`，不返回参数或 schema 片段。
18. `namespace_manage.open` 已确定下一轮生效；同轮先 open 再调用新 namespace 工具时拒绝执行，并返回明确原因。
19. `namespace_manage.close` 已确定立即按顺序生效；先工具后 close 可执行但 close 覆盖续命，先 close 后工具则拒绝。
20. `open`、`close`、`preview` 都支持一次传入多个 namespace。
21. `create_goal` / `resolve_goal` 已确定合并为 core 常驻 `goal_manage`，并用 `action` + JSON Schema `if/then` 区分 create / resolve。
22. 新工具不默认加 `additionalProperties: false`。
23. `send_voice` 常驻在 `qq_social`；TTS 不可用时由执行层返回错误，暂不按配置摘除。
24. `web_search`、`web_extract`、`get_weather` 固定放在 core 常驻。
25. `think_deeply` 和 `recall_memory` 固定放在 core 常驻。
26. `get_avatar`、`list_contact`、`set_qq_signature`、`set_group_card` 只做 public name 改名，参数和行为暂时沿用现有工具。

## 14. 工具改名映射草案

| 当前名                                         | 目标名              | namespace       |
| ---------------------------------------------- | ------------------- | --------------- |
| `tools_manage`                                 | `namespace_manage`  | `core`          |
| `create_goal` + `resolve_goal`                 | `goal_manage`       | `core`          |
| `restart_self`                                 | `restart`           | `core`          |
| `get_image_by_ref`                             | `view_image_by_ref` | `core`          |
| `send_voice_message`                           | `send_voice`        | `qq_social`     |
| `open_forward_message` + `browse_forward_view` | `browse_forward`    | `qq_forward_view`  |
| `search_current_session_chat_history`          | `search_history`    | `qq_chat_log_view` |
| `get_user_avatar`                              | `get_avatar`        | `qq_profile`    |
| `set_self_qq_signature`                        | `set_qq_signature`  | `qq_profile`    |
| `get_contact_list`                             | `list_contact`      | `qq_contacts`   |
| `set_self_group_card`                          | `set_group_card`    | `qq_group_info` |

## 15. 最小可落地版本

第一阶段不要一次性完成所有重命名和移动文件。最小版本只需要：

1. 保留当前工具模块位置。
2. 新增 namespace registry。
3. 用 namespace registry 替代 hidden group 渲染。
4. 实现 `namespace_manage.open/preview/search`，`close` 可以先只支持非 core namespace；模型面对层同步移除 `tools_manage`。
5. 只实现一个 attach：`qq_social` 展开时附带 `list_stickers`。
6. TTL 读取 namespace yaml；未配置时使用用户设置中的意识流最大回灌轮数。
7. 所有 public tool name 直接切换为目标名；不保留旧名兼容，不做 alias 过渡。

这样可以先验证 prompt 污染是否下降、模型是否能稳定按 namespace 展开，再进入大规模命名清理。
