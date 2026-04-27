"""sleep/__init__.py — 休眠工具

Handler 直接阅吭住 duration 分钟（通过主 loop 上的
``asyncio.Event``），期间 napcat_handler 可以通过设置
``session.sleep_wake_event`` 唤醒它（如被 @ 、被戳、被回复等）。

返回后由 ``consciousness.main_loop`` 直接进入下一 round，
模型能在意识流中看到本次 sleep 的 ``slept_seconds`` / ``woke_up_because``。
不再存在 deferred / exit-action 概念。
"""

from .sleep import DECLARATION, execute

__all__ = ["DECLARATION", "execute"]
