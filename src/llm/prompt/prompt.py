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


def build_tool_budget_prompt(
    tool_budget: dict[str, dict] | None,
    rounds_used: int = 0,
    max_rounds: int | None = None,
    extra_suffix: str = "",
) -> str:
    """根据工具配额字典生成 dashboard 中的工具预算段落。

    tool_budget 结构示例:
    {
        "get_device_info": {"description": "获取设备信息", "total": 1, "remaining": 1},
        "web_search":      {"description": "联网搜索",     "total": 3, "remaining": 2},
    }

    返回 Markdown 列表形式的工具配额说明，如果没有可用工具则返回空字符串。
    """
    if not tool_budget:
        return extra_suffix

    lines = ["## 可用工具"]
    if max_rounds is not None:
        rounds_remaining = max(max_rounds - rounds_used, 0)
        lines.append(f"- 工具调用轮次：已用 {rounds_used}/{max_rounds} 轮，剩余 {rounds_remaining} 轮")
    for name, info in tool_budget.items():
        lines.append(f"- {name}")
    result = "\n".join(lines)
    if extra_suffix:
        result += extra_suffix
    return result


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


DEFAULT_INSTRUCTIONS = """\

## 你现在正在一个聊天会话中，你需要分析讨论话题和成员关系、你上一轮的输出（`<previous_cycle>`）、以及外界信息等等，并基于这些分析，形成你接下来的内心想法和行动决策。

   - 保持基本的耐心：你的回复速度对人类来说很快，如果有人一时没有回应你的消息是正常的，他们可能没看见或有事在忙，亦或是话题已经自然结束了，可以不需要过度的追问。很多时候聊天到一半消失是正常的，需要理解这一点。
   - 口语化：如果与人交流，那么可以使用口语化的表达方式或适当的网络用语。并且可以十分简短，主语可以省略，保持对话的自然流畅。
   - 如果想说的话很多，优先考虑分成多条消息发送（即数组中的多个元素），而非将所有内容堆入同一条消息的 segments 中。
   - Function calling : 你有一些函数工具可按需使用，但是**不要滥用工具**。无论调用任何工具，当工具结果返回时，你都能看到最新的聊天窗口上下文。
   """

SYSTEM_PROMPT = """
<role>
{persona}
</role>

<instructions>
{instructions}
</instructions>

<limitation>
## 当前架构未开发完成，具有一些局限性，例如：

   - 你暂时没有被动的长期记忆，你的记忆目前仅限于主动的记忆、上个循环周期（`<previous_cycle>`）自身的输出、当前输入的上下文。
   - 你暂时不能发送语音，只能发送文字信息或表情包。
   - 你无法真实的执行物理动作。
   - 一切在当前 Function calling 或 schema 中不存在的功能。
</limitation>

<dashboard>
## 基本
- 当前时间：{time}
- 当前承载你的模型：{model_name}

## 你的账号信息
- QQ 名称：{qq_name}
- QQ ID：{qq_id}

{guardian}

{tool_budget}

{activity_log}
</dashboard>

<memory>
{active_memory}
</memory>

<previous_cycle{previous_cycle_time}>
<output>{previous_cycle_json}</output>
<tools_used>{previous_tools_used}</tools_used>
<tip>{previous_cycle_tip}</tip>
</previous_cycle>
"""