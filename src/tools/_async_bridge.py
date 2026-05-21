"""Compatibility import for tool modules that wait on async coroutines."""

from runtime.async_bridge import LoopStoppedError, run_coroutine_sync, wait_threadsafe_future_result

__all__ = ["LoopStoppedError", "run_coroutine_sync", "wait_threadsafe_future_result"]

