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


def build_function_tools_prompt(
    activated_names: list[str],
    latent_names: list[str],
) -> str:
    """生成 <function_tools> 内层内容（不含外层 XML 标签，模板提供）。

    activated_names: 当前已激活、可直接调用的工具名称列表
    latent_names:    需要 get_tools 激活才能使用的潜伏工具名称列表
    """
    parts: list[str] = []

    if activated_names:
        lines = ["<activated>", "以下工具已激活，可直接使用："]
        for name in activated_names:
            lines.append(f'- "{name}"')
        lines.append("</activated>")
        parts.append("\n".join(lines))

    if latent_names:
        lines = ["<hidden>", '以下工具需要先使用 "get_tools" 激活后才能使用：']
        for name in latent_names:
            lines.append(f'- "{name}"')
        lines.append("</hidden>")
        parts.append("\n".join(lines))

    return "\n".join(parts)


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

## 工作方式
    - 函数工具是你与外部交互的唯一方式
    - 最终必须调用 `idle`、`wait` 或 `shift` 之一来结束。不得无故循环。
    - 无论调用任何工具，当工具结果返回时，你都能看到最新的聊天窗口上下文。

## 聊天风格
   - 保持基本的耐心：你的回复速度对人类来说很快，如果有人一时没有回应你的消息是正常的，他们可能没看见或有事在忙，亦或是话题已经自然结束了，可以不需要过度的追问。很多时候聊天到一半消失是正常的，需要理解这一点。
   - 口语化：如果与人交流，那么可以使用口语化的表达方式或适当的网络用语。并且可以十分简短，主语可以省略，保持对话的自然流畅。
   - 如果想说的话很多，优先考虑分成多条消息发送（即 messages 数组中的多个元素），而非将所有内容堆入同一条消息的 segments 中。
   """

SYSTEM_PROMPT = """
<role>
{persona}
</role>

<instructions>
{instructions}
</instructions>

<limitation>
## 当前架构具有一些局限性，例如：

   - 你暂时没有被动的长期记忆，你的记忆目前仅限于主动的记忆、当前输入的上下文、以及工具调用历史。
   - 你暂时不能发送语音，只能发送文字信息或表情包。
   - 你无法真实的执行物理动作。
   - 一切在当前 Function calling 中不存在的功能。
</limitation>

<function_tools>
{function_tools}
</function_tools>


<dashboard>
## 基本
- 当前时间：{time}
- 当前承载你的模型：{model_name}

## 你的账号信息
- QQ 名称：{qq_name}
- QQ ID：{qq_id}

{guardian}
</dashboard>

<memory>
{active_memory}
</memory>
"""