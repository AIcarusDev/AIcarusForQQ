"""sleep/__init__.py — 休眠工具

Handler 返回 deferred 标记，实际结果在被唤醒后由 llm_core 补完。
Provider 检测到此工具被调用后立刻退出工具循环，返回
loop_action = {"action": "sleep", "duration": ..., "motivation": ...}。
napcat_handler._run_active_loop 负责关闭当前 activation 并进入休眠。
休眠结束（被唤醒）后，llm_core 基于会话状态补完 deferred result（含 slept_seconds、woke_up_because 等），
再重新触发 call_model_with_retry，模型通过意识流看到休眠结果。
"""

from .sleep import DECLARATION, execute

__all__ = ["DECLARATION", "execute"]
