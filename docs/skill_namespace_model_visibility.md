# Namespace-bound Skills MVP Model Visibility

## 1. 目标

本文描述最小可行版本的 namespace 绑定 skills 方案，只关注模型眼里是什么样的。

MVP 只做一件事：

> 某个 namespace 打开时，系统顺带把它绑定的主 skill 正文渲染给模型；namespace 关闭或过期时，对应 skill 正文一起消失。

不做：

1. 不把 resource 固化进 prompt。
2. 不做 `skill_manage`。
3. 不做执行期 skill guard。
4. 不在模型可见 namespace 上展示 skill 关联元数据。
5. 不做多个 `<skills>` 容器；多个 skill 统一放进同一个 `<skills>` 容器。

## 2. 设计边界

namespace 仍然只负责工具 schema 可见性。

skill 仍然只负责认知文本可见性。

二者的绑定关系是本地元数据，不直接展示给模型。模型不需要知道“某 namespace 关联了哪个 skill”这件实现事实。模型只会经历：

1. 我打开了一类工具。
2. 下一轮这类工具可用了。
3. 我同时想起了一段相关技能。

## 3. 本地配置语义

配置可以类似这样：

```yaml
qq_social:
  description: "QQ 社交动作：发送文字/表情包/图片/语音消息、撤回消息、发起戳一戳、复读。"
  tools:
    - send_message
    - send_voice
    - recall_message
    - poke
    - plus_one
  skill: "qq-social-style"
```

这只是本地元数据。

模型不会在 `<tools>` 里看到 `skill="qq-social-style"`，也不会看到 `required_skill`、`skill_status`、`mounted_by` 之类字段。

## 4. 模型初始看到什么

当 `qq_social` 未打开时，模型看到的仍然只是 inactive namespace：

```xml
<tools>
  <namespaces>
    <namespace name="core" active="true">[...]</namespace>
    <namespace
      name="qq_social"
      description="QQ 社交动作：发送文字/表情包/图片/语音消息、撤回消息、发起戳一戳、复读。"
      active="false"/>
  </namespaces>
</tools>
```

此时没有 `<skills>` 块。

模型知道有 `qq_social` 这类工具，但看不到内部工具 schema，也看不到 QQ 社交 skill 正文。

## 5. 模型如何主动触发

模型需要使用 QQ 社交动作时，仍然只调用 `namespace_manage.open`：

```xml
<action>
  <tool_call>{"name":"namespace_manage","arguments":{"open":["qq_social"]}}</tool_call>
</action>
```

回执只需要说明 namespace 打开情况：

```json
{
  "ok": true,
  "opened": ["qq_social"],
  "already_open": [],
  "active_namespaces": ["core", "qq_social"]
}
```

本轮不能继续调用 `qq_social` 内部工具。这个规则已经由 namespace 机制决定。

## 6. 下一轮模型看到什么

下一轮，模型看到 `qq_social` 工具 schema：

```xml
<tools>
  <namespaces>
    <namespace name="core" active="true">[...]</namespace>
    <namespace name="qq_social" active="true">[{...send_message schema...},{...send_voice schema...}]</namespace>
  </namespaces>
</tools>
```

同时，user prompt 前部出现一个 skills 容器：

```xml
<skills>
<skill name="qq-social-style">
...qq-social-style 正文...
</skill>
</skills>
```

推荐 prompt 顺序：

```xml
<memory>
...
</memory>

<goals>
...
</goals>

<skills>
<skill name="qq-social-style">
...当前 active namespace 绑定的主 skill 正文...
</skill>
</skills>

<world>
...当前最新外部世界...
</world>
```

每个 `<skill name="...">` 子块只放正文，不放 frontmatter、namespace、opened_by、ttl 等运行元数据。`name` 只用于让模型区分多个 skill。资源正文不放进 `<skill>`，只能通过工具按需读取。

模型视角就是：相关工具可用了，同时它想起了相关使用技巧。

## 7. 被动触发是什么

MVP 没有单独的 skill 被动触发。

只有 namespace 机制本身的被动展开：

如果模型直接调用 inactive namespace 内部工具，例如：

```xml
<action>
  <tool_call>{"name":"send_message","arguments":{"messages":[{"segments":[{"command":"text","content":"确实"}]}]}}</tool_call>
</action>
```

工具天然无法执行，因为 `qq_social` 未打开。执行层按现有 namespace 规则返回：该 namespace 已为下一轮打开，当前工具未执行。

下一轮由于 `qq_social` 已打开，所以：

1. `send_message` schema 出现。
2. `<skills>` 容器和对应 skill 正文出现。

不需要额外判断“skill 是否挂载”。skill 跟 namespace 同生同灭。

## 8. skill 生命周期

MVP 中 skill 没有独立生命周期。

skill 生命周期完全绑定 namespace 生命周期：

1. namespace inactive：不渲染 skill。
2. namespace open：渲染该 namespace 绑定的 skill。
3. namespace close：skill 消失。
4. namespace TTL 过期：skill 消失。
5. namespace 被工具调用续命：skill 随 namespace 继续可见。

也就是说，skill 不单独记录 `last_used_round`，不单独配置 TTL，不单独 close。

如果现有 namespace 状态是全局的，那么 MVP skill 可见性也跟着全局 namespace 状态走。后续如果需要按会话隔离，应先调整 namespace 状态模型，而不是给 skill 单独增加一套 scope。

## 9. resource 生命周期

resource 不进入 prompt 固定前缀。

模型不会在 `<skills>` 里看到资源正文。skill 正文可以用很短的 `References` 提示可读资源 id；当模型确实需要看一眼细节时，调用 `recall_skill_resource` 读取单个资源，结果作为普通工具返回进入 `<action_response>`。

`qq-social-style` 里的 references 继续作为文件组织方式存在。模型可见契约里，只有主 skill 正文会被渲染进 `<skills>` 里的对应 `<skill name="...">` 子块；资源文件只在工具调用返回值中出现。

## 10. 完整模型可见流程

### Round 1：未打开 namespace

模型看到：

```xml
<namespace name="qq_social" description="QQ 社交动作：..." active="false"/>
```

没有 `<skills>`。

模型决定需要 QQ 社交动作：

```xml
<action>
  <tool_call>{"name":"namespace_manage","arguments":{"open":["qq_social"]}}</tool_call>
</action>
```

### Round 2：namespace 打开

模型看到：

```xml
<namespace name="qq_social" active="true">[{...send_message schema...}]</namespace>
```

并看到：

```xml
<skills>
<skill name="qq-social-style">
...qq-social-style 正文...
</skill>
</skills>
```

模型基于 `<world>`、工具 schema 和 `<skills>` 重新判断如何行动。

### Round 3 之后：随 namespace 生命周期变化

如果模型使用 `qq_social` 工具，namespace 续命，对应 `<skill name="qq-social-style">` 继续出现在 `<skills>` 中。

如果模型关闭 `qq_social`：

```xml
<action>
  <tool_call>{"name":"namespace_manage","arguments":{"close":["qq_social"]}}</tool_call>
</action>
```

下一轮 `qq_social` schema 消失，对应 skill 也从 `<skills>` 中消失；如果没有任何 active skill，整个 `<skills>` 容器消失。

如果 namespace TTL 过期，效果相同。

## 11. 不变量

1. 模型看不到 namespace 与 skill 的绑定元数据。
2. 模型最多看到一个 `<skills>` 容器。
3. `<skills>` 内可以有多个 `<skill name="...">` 子块，按 active namespace 顺序去重。
4. 每个 `<skill>` 除 `name` 属性外只包含正文。
5. skill 不拦截工具执行。
6. skill 不单独续命。
7. skill 不单独关闭。
8. namespace open/close/TTL 是 skill 可见性的唯一来源。
9. resource 不进入 MVP。
