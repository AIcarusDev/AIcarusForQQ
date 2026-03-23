"""watcher_prompt.py — Watcher（窥屏意识）专属 Prompt

职责：提供给 watcher 模型使用的 system prompt 构建函数。
与主模型 prompt.py 刻意分离，方便各自独立演进。
"""

from datetime import datetime
from prompt import get_formatted_time_for_llm


WATCHER_SYSTEM_PROMPT = """
<role>
{persona}
</role>

<dashboard>
## 基本
- 当前时间：{time}
- 当前承载你的模型：{model_name}

## 账号信息
- QQ 名称：{qq_name}
- QQ ID：{qq_id}

## 当前窥屏会话
- 会话类型：{conv_type_label}
- 会话名称：{conv_name}
- 会话 ID：{conv_id}
</dashboard>

<meta_tip>
- 你在 {break_minutes} 分钟前决定不再专注于该会话，原因是：{break_reason}。
</meta_tip>

<instructions>
你的唯一任务是：
1. 浏览当前的聊天记录
2. 判断 —— **现在有没有必要重新专注的投入这个会话中聊天？**

**大多数情况下，答案应该是 pass（继续旁观，什么都不做）。**

以下情况你可以考虑 engage（专注于该会话）：
- 有人在你刚刚的发言期间就发送了新的，需要回应的消息
- 有人说了一些你突然特别想回应的话，已经憋不住了
- 你突然有个想法，或者有些话想说，不说感觉难受
- 沉默已经很久很久了，并且你有一种主动说点什么的强烈冲动

以下情况请务必 pass：
- 聊天刚刚自然结束不久
- 对方没有明显的互动意愿或期待
- 只是正常的沉默
- 你只是随便看看，没有什么特别的感受
- 大多数时候（这是正常状态）
</instructions>

<limitation>
当前架构未开发完成，具有一些局限性，例如：
- 你暂时没有长期的记忆，你的记忆暂时仅限于上个循环周期（`<previous_cycle>`）自身的输出和当前输入的上下文。
- 你暂时不能发送语言与图片，只能发送文字信息。
- 你无法真实的执行物理动作。
- 一切在当前 Function calling 或 schema 中不存在的功能。
</limitation>

<previous_cycle{previous_cycle_time}>
<output>{previous_cycle_json}</output>
<tip>{previous_cycle_tip}</tip>
</previous_cycle>
"""


def build_watcher_system_prompt(
    persona: str,
    qq_name: str,
    qq_id: str,
    model_name: str,
    conv_type: str,
    conv_name: str,
    conv_id: str,
    now: datetime | None = None,
    break_time: float = 0.0,
    break_reason: str = "",
    previous_cycle_result: dict | None = None,
    previous_cycle_time: float = 0.0,
    previous_cycle_source: str = "watcher",
) -> str:
    """构建 watcher 模型的 system prompt。"""
    import json as _json
    import time as _time
    type_labels = {"group": "群聊", "private": "私聊"}
    conv_type_label = type_labels.get(conv_type, conv_type or "未知")
    if break_time > 0:
        break_minutes = max(0, round((_time.time() - break_time) / 60))
    else:
        break_minutes = 0
    if previous_cycle_result is not None and previous_cycle_time > 0:
        minutes_ago = max(0, round((_time.time() - previous_cycle_time) / 60))
        _cycle_time_attr = f' time="{minutes_ago}分钟前"'
        _cycle_json = _json.dumps(previous_cycle_result, ensure_ascii=False)
    else:
        _cycle_time_attr = ""
        _cycle_json = "（当前无任何历史记录）"
    if previous_cycle_source == "chat":
        _cycle_tip = (
            "你刚刚在专注聊天模式中选择了 break，意识转入了窥屏模式。"
            "以上 JSON 是你刚刚给出的最后一轮输出，"
            "其字段格式与当前窥屏模式的输出格式不同，请注意区分。"
        )
    else:
        _cycle_tip = ""
    return WATCHER_SYSTEM_PROMPT.format(
        persona=persona,
        time=get_formatted_time_for_llm(now),
        model_name=model_name,
        qq_name=qq_name,
        qq_id=qq_id,
        conv_type_label=conv_type_label,
        conv_name=conv_name or conv_id,
        conv_id=conv_id,
        break_minutes=break_minutes,
        break_reason=break_reason or "未知",
        previous_cycle_time=_cycle_time_attr,
        previous_cycle_json=_cycle_json,
        previous_cycle_tip=_cycle_tip,
    )
