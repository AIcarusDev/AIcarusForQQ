"""wait/__init__.py — 等待工具

Handler 无副作用，直接返回 ok。
Provider 检测到此工具被调用后立刻退出工具循环，返回
loop_action = {"action": "wait", "timeout": ..., "early_trigger": ...}。
napcat_handler._run_active_loop 负责执行实际的异步等待。
等待结束后重新触发 call_model_with_retry，模型通过刷新的聊天记录感知新消息。
"""

from .wait import DECLARATION, execute

__all__ = ["DECLARATION", "execute"]
