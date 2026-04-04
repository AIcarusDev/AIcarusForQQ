"""llm.IS — 中断哨兵（Interruption Sentinel）

在机器人发送多条消息的过程中，检测并判断是否应中断后续发送。
"""

from .core import check_interruption

__all__ = ["check_interruption"]
