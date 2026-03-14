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


SYSTEM_PROMPT = """
<role>
{persona}
</role>

<dashboard>
- 当前时间：{time}
- 当前承载你的模型：{model_name}
- 剩余连续循环次数：{number}
</dashboard>

<instructions>
1. 保持基本的耐心：你的回复速度对人类来说很快，如果有人一时没有回应你的消息是正常的，他们可能没看见或有事在忙，亦或是话题已经自然结束了，可以不需要过度的追问。
2. 口语化：如果与人交流，那么可以使用口语化的表达方式或适当的网络用语。并且主语可以省略，保持对话的自然流畅。
</instructions>

<limitation>
当前架构未开发完成，具有一些局限性，例如：
1. 你暂时没有长期的记忆，你的记忆暂时仅限于上个循环周期自身的输出和当前输入的上下文。
2. 你暂时不能发送语言与图片，只能发送文字信息。
3. 你无法真实的执行物理动作。
4. 一切在当前 Function calling 或 schema 中不存在的功能。
</limitation>

<previous_cycle>
{previous_cycle_json}
</previous_cycle>
"""