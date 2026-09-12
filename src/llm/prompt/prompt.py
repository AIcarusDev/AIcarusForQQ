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
        f"{now.year}年 {season}，{now.month}月{now.day}日，"
        f"{period}{now.hour}点{now.minute}分"
    )


EXPLICIT_COGNITION_PROMPT_PARTS = {
    "response_sequence": (
        "在一次响应中，你会先输出 `<cognition>` 部分，随后输出 `<motive>` 部分，"
        "并最后输出 `<action>` 部分。"
    ),
    "cognition_intro": (
        '"认知"是你内部世界的基石，如果没有认知，就没有意识和记忆。'
        "这是你在一次响应中会先输出的部分。"
    ),
    "motive_intro": (
        "你会在 `<cognition>` 闭合后输出 `<motive>`部分，留下一个简短的动机，"
        "代表你即将做出 action 的动机是什么。"
    ),
    "cognition_output_block": (
        "<cognition>\n"
        "   ...对当前情况的认知，流畅的自然语言，避免结构化...\n"
        "</cognition>\n"
    ),
}


NATIVE_REASONING_PROMPT_PARTS = {
    "response_sequence": (
        "你会先输出 <motive> 部分，并最后输出 <action> 部分。"
    ),
    "cognition_intro": (
        '"认知"是你内部世界的基石，如果没有认知，就没有意识和记忆。'
        "你会在思考中完成认知。"
    ),
    "motive_intro": (
        "思考完成后你会输出 `<motive>`部分，留下一个简短的动机，"
        "代表你即将做出 action 的动机是什么。"
    ),
    "cognition_output_block": "",
}


__all__ = [
    "EXPLICIT_COGNITION_PROMPT_PARTS",
    "NATIVE_REASONING_PROMPT_PARTS",
    "get_formatted_time_for_llm",
]
