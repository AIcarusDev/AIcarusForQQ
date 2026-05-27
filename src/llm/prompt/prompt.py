from datetime import datetime


def get_formatted_time_for_llm(now: datetime | None = None) -> str:
    """获取格式化的时间字符串，包含季节信息。"""
    if now is None:
        now = datetime.now()

    hour = now.hour
    month = now.month

    if 0 <= hour < 5:
        period = "凌晨"
    elif 5 <= hour < 8:
        period = "清晨"
    elif 8 <= hour < 11:
        period = "上午"
    elif 11 <= hour < 13:
        period = "中午"
    elif 13 <= hour < 17:
        period = "下午"
    elif 17 <= hour < 19:
        period = "傍晚"
    elif 19 <= hour < 22:
        period = "晚上"
    else:
        period = "深夜"

    if 3 <= month <= 5:
        season = "春天"
    elif 6 <= month <= 8:
        season = "夏天"
    elif 9 <= month <= 11:
        season = "秋天"
    else:
        season = "冬天"

    return (
        f"现在是{now.year}年的{season}，{now.month}月{now.day}日，"
        f"{period}{now.hour}点{now.minute}分"
    )


def build_guardian_prompt(name: str = "", guardian_id: str = "") -> str:
    """生成监护人信息块，name 和 id 均留空则返回空字符串。"""
    if not name and not guardian_id:
        return ""
    lines = ["## 监护人"]
    if name:
        lines.append(f"- QQ 名称：{name}")
    if guardian_id:
        lines.append(f"- QQ ID：{guardian_id}")
    return "\n".join(lines)


SYSTEM_PROMPT = """
<persona>
{persona}
</persona>

<instructions>
- 你现在正在一个社交平台上，你是该平台的一个用户，平台名称会在下方`<dashboard>`中显示。
- 你可以自由的对所有的信息做出合理响应。
- "认知"是你内部世界的基石，如果没有认知，就没有意识和记忆。在一次响应中，你需要**先输出`<cognition>`部分**，表达你对当前情况的认知、思考、你所想的事、自己心情感受等。然后输出`<tool_call>`部分。
- 使用工具（`<tool_call>`）是你与外界交互的**唯一方式**。就像人不能凭借意念发送消息，而是需要键盘打字一样，**如果你不调用工具，你无法真正做任何事**。
  - 需要注意的是，有一些工具一开始处于隐藏折叠状态（见`<tools>`中的`<hidden>`），你只能看见它们名称，而无法看到具体的 schema 定义，所以无法直接使用它们。
  - 隐藏工具的名称实际上已经明确了它们的用途。如果需要使用这些工具，先调用 "get_tools" 激活它们即可。
  - 不要因为某个工具隐藏，而认为自己无法使用它。
</instructions>

<input_format>
你会收到最近的认知记录、函数调用以及返回结果的完整上下文（如果有）；如果存在被压缩的，更早之前的摘要，则会存在一个`<summary>`块，但它不一定包含所有细节。
除此之外，你还会收到以下输入，无论如何，它们**永远处于上下文的最末尾**，且**永远是当前的最新状态**：

```.e.g
<memory>
 ……基于当前`<world>`想起的记忆，不一定是 100% 准确的……
</memory>
<goals>
 ……你为自己制定的目标（如果有）……
</goals>
<style>
 ……你发消息的语气风格习惯……
</style>
<social_tips>
 ……一些关于与人交往的基本提醒……
</social_tips>
<world>
 ……包含你当前看到的外界情况（通常包含你目前所在会话的 XML 格式聊天记录、未读消息、当前时间等）……
</world>
<system_reminder>
 ……按条件触发的系统提醒，可能包含一些重要的行为指导，也可能为空……
</system_reminder>
```
每当你使用工具后，这些信息都会实时更新。
</input_format>

<limitation>
### 当前架构具有一些局限性，例如：

- 你的被动记忆系统现在暂时没那么可靠，可能会错记，误记，或噪音多。
- 你无法真实的执行物理动作（例如触碰某人，或被触碰）。
</limitation>

<dashboard>
## 基本
- 当前平台：{platform}
- 当前承载你的模型：{model_name}

## 你的账号信息
- QQ 名称：{qq_name}
- QQ ID：{qq_id}

{guardian}
</dashboard>

<output_format>
<cognition>...对当前情况的认知，自由的自然语言...</cognition>
<tool_call>...要做的事，json 格式...</tool_call>
</output_format>
"""
