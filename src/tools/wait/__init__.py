"""wait/__init__.py — 等待工具

Handler 返回 deferred 标记，实际结果在等待结束后由 _run_active_loop 补完。
Provider 检测到此工具被调用后立刻退出工具循环，返回
loop_action = {"action": "wait", "timeout": ..., "early_trigger": ...}。
napcat_handler._run_active_loop 负责执行实际的异步等待。
等待结束后补完 deferred result（含 resumed / trigger_kind / elapsed_seconds），
再重新触发 call_model_with_retry，模型通过意识流看到等待结果。
"""

from .wait import DECLARATION, execute

__all__ = ["DECLARATION", "execute"]
