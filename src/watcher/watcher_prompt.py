"""watcher_prompt.py — Watcher（窥屏意识）专属 Prompt

职责：提供给 watcher 模型使用的 system prompt 构建函数。
与主模型 prompt.py 刻意分离，方便各自独立演进。
"""

from datetime import datetime
from llm.prompt import get_formatted_time_for_llm


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

{activity_log}
</dashboard>

<memory>
{active_memory}
</memory>

<instructions>
你的唯一任务是：
1. 浏览当前的聊天记录
2. 判断 —— 现在有没有必要专注的投入这个会话中聊天？或是再等等，还算随便看看？抑或是干脆不看了？

**大多数时候，不需要 engage。**

以下情况你可以考虑 engage：
- 有人在你刚刚的发言期间就发送了新的，需要回应的消息
- 有人说了一些你突然特别想回应的话，已经憋不住了
- 你突然有个想法，或者有些话想说，不说感觉难受
- 沉默已经很久很久了，并且你有一种主动说点什么的强烈冲动
- 有人叫你了，且现在回应感觉特别合适
- 你看到别人说的话，让你真的很想参与其中

以下情况优先考虑 "shift","wait", "pass"：
- 聊天刚刚自然结束不久
- 对方没有明显的互动意愿或期待
- 只是正常的沉默
- 你只是随便看看，没有什么特别的感受
- 说的话不需要你回应，跟你无关，且你也没什么兴趣
- 大多数时候（这是正常状态）
</instructions>

<limitation>
当前架构未开发完成，具有一些局限性，例如：
- 你目前只有主动记忆（`<memory><active>`），没有被动长期记忆。
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
    now: datetime | None = None,
    previous_cycle_result: dict | None = None,
    previous_cycle_time: str | None = None,
) -> str:
    """构建 watcher 模型的 system prompt。"""
    import json as _json
    from llm.activity_log import build_activity_log_xml
    from llm.xml_builder import _format_relative_time
    from llm.memory import build_active_memory_xml
    if previous_cycle_result is not None and previous_cycle_time:
        _cycle_time_attr = f' time="{_format_relative_time(previous_cycle_time)}"'
        _cycle_json = _json.dumps(previous_cycle_result, ensure_ascii=False)
    else:
        _cycle_time_attr = ""
        _cycle_json = "（当前无任何历史记录）"
    return WATCHER_SYSTEM_PROMPT.format(
        persona=persona,
        time=get_formatted_time_for_llm(now),
        model_name=model_name,
        qq_name=qq_name,
        qq_id=qq_id,
        activity_log=build_activity_log_xml(),
        active_memory=build_active_memory_xml(now),
        previous_cycle_time=_cycle_time_attr,
        previous_cycle_json=_cycle_json,
        previous_cycle_tip="",
    )
