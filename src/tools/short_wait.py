"""short_wait.py — 短暂挂起，等待用户后续消息

在 LLM 工具调用期间阻塞 N 秒（N 取 3~10），然后返回等待期间
新增的聊天消息（以与聊天记录一致的 XML 格式）。

适用场景：察觉用户话还没说完、需要等待更多上下文再作决策时。
需要运行时上下文：session。
"""

import logging
import time
from typing import Any, Callable

logger = logging.getLogger("AICQ.tools")

# build_tools() 用此字段获取工具名；实际 schema 由 get_declaration() 动态生成
DECLARATION: dict = {
    "name": "short_wait",
}


def get_declaration() -> dict:
    return {
        "max_calls_per_response": 3,
        "name": "short_wait",
        "description": (
            "短暂挂起，等待用户把话说完，然后返回等待期间收到的新消息。"
            "这是一个特殊，但是非常重要和实用的工具。"
            "适用于：察觉用户有可能话还没说完，后面还可能有新内容，或需要倾听时。"
            "如果等待期间有新消息，该工具会返回完整的新消息内容，可据此决定现在要不要发言，如何发言，做什么等等。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "等待秒数。",
                    "minimum": 3,
                    "maximum": 10,
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "挂起等待的原因。"
                    ),
                },
            },
            "required": ["seconds", "reason"],
        },
    }


REQUIRES_CONTEXT: list[str] = ["session"]


def summarize_result(entry: dict):
    """自定义摘要：只保留等待时长和新消息数，消息详情不需要留在 prompt 里。"""
    result = entry.get("result") or {}
    seconds = (entry.get("arguments") or {}).get("seconds", "?")
    count = result.get("new_messages_count")
    if count is not None:
        return f"成功，等待了 {seconds} 秒，期间收到 {count} 条新消息。"
    return f"成功，等待了 {seconds} 秒，期间无新消息。"


def make_handler(session: Any) -> Callable:
    def execute(seconds: int, reason: str, **kwargs) -> dict:
        # 钳位到合法范围
        seconds = max(3, min(10, int(seconds)))

        # 等待前快照
        snapshot = list(session.context_messages)
        snapshot_ids: set[str] = {
            str(m["message_id"])
            for m in snapshot
            if m.get("message_id") is not None
        }
        # 对无 message_id 的条目用 Python 对象 id 作为 fallback 标识
        snapshot_obj_ids: set[int] = {id(m) for m in snapshot}

        logger.info("[tools] short_wait: 开始等待 %ds，原因: %s", seconds, reason)
        for _i in range(seconds):
            time.sleep(1)
            logger.info("[tools] short_wait: 已等待 %d/%ds", _i + 1, seconds)

        # 收集新增消息
        current = list(session.context_messages)
        new_messages: list[dict] = []
        for m in current:
            mid = m.get("message_id")
            if mid is not None:
                if str(mid) not in snapshot_ids:
                    new_messages.append(m)
            else:
                if id(m) not in snapshot_obj_ids:
                    new_messages.append(m)

        if not new_messages:
            logger.info("[tools] short_wait: 等待期间无新消息")
            return {"result": "等待期间无新消息。"}

        # 用与聊天记录一致的 XML 格式渲染新消息
        from llm.xml_builder import (
            _render_message_generic,
            _render_message_group,
            _render_message_private,
            _render_note,
            _resolve_sentinels,
        )

        conv_type = session.conv_type
        conv_meta = session._get_conv_meta()
        # 传入完整当前上下文，供 quote 引用解析
        full_context = current

        lines: list[str] = ["<new_messages>"]
        for msg in new_messages:
            if msg.get("role") == "note":
                lines.extend(_render_note(msg))
            elif conv_type == "group":
                lines.extend(_render_message_group(msg, full_context))
            elif conv_type == "private":
                lines.extend(_render_message_private(msg, conv_meta, full_context))
            else:
                lines.extend(_render_message_generic(msg))
        lines.append("</new_messages>")

        # 收集图片元数据，用于哨兵解析（视觉描述等）
        all_images: dict[str, dict] = {}
        for msg in new_messages:
            all_images.update(msg.get("images") or {})

        xml_str = _resolve_sentinels("\n".join(lines), all_images)

        logger.info("[tools] short_wait: 等待期间收到 %d 条新消息", len(new_messages))
        return {
            "new_messages_count": len(new_messages),
            "new_messages_xml": xml_str,
        }

    return execute
