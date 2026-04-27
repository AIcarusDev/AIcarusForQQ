"""wait/__init__.py — 等待工具

Handler 直接阅吭住 timeout 秒（通过主 loop 上的
``asyncio.Event``），期间 napcat_handler 可根据 early_trigger
条件设置 ``session.wait_event`` 提前唤醒。

返回后由 ``consciousness.main_loop`` 直接进入下一 round，
模型在意识流中看到 ``resumed`` / ``trigger_kind`` / ``elapsed_seconds``。
不再存在 deferred / exit-action 概念。
"""

from .wait import DECLARATION, execute

__all__ = ["DECLARATION", "execute"]
