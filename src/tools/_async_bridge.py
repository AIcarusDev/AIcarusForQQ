"""Shared helpers for sync tool code that waits on async coroutines."""

import asyncio
import concurrent.futures
import time
from typing import Awaitable, TypeVar

T = TypeVar("T")


class LoopStoppedError(RuntimeError):
    """Raised when the target event loop stops before work finishes."""


def _loop_stopped(loop: asyncio.AbstractEventLoop) -> bool:
    return loop.is_closed() or not loop.is_running()


def wait_threadsafe_future_result(
    future: concurrent.futures.Future[T],
    loop: asyncio.AbstractEventLoop,
    *,
    timeout: float | None = None,
    poll_interval: float = 0.5,
) -> T:
    """Wait for a threadsafe future while detecting loop shutdown promptly."""
    if timeout is not None and timeout < 0:
        raise ValueError("timeout must be >= 0 or None")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be > 0")

    deadline = None if timeout is None else time.monotonic() + float(timeout)

    while True:
        wait_timeout = poll_interval
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                wait_timeout = 0
            else:
                wait_timeout = min(poll_interval, remaining)

        try:
            return future.result(timeout=wait_timeout)
        except concurrent.futures.CancelledError as exc:
            if _loop_stopped(loop):
                raise LoopStoppedError("事件循环已停止") from exc
            time.sleep(min(poll_interval, 0.05))
            if _loop_stopped(loop):
                raise LoopStoppedError("事件循环已停止") from exc
            raise
        except concurrent.futures.TimeoutError as exc:
            if _loop_stopped(loop):
                future.cancel()
                raise LoopStoppedError("事件循环已停止") from exc
            if deadline is not None and time.monotonic() >= deadline:
                raise


def run_coroutine_sync(
    coro: Awaitable[T],
    loop: asyncio.AbstractEventLoop,
    *,
    timeout: float | None = None,
    poll_interval: float = 0.5,
) -> T:
    """Run a coroutine on another loop and wait synchronously for its result."""
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return wait_threadsafe_future_result(
        future,
        loop,
        timeout=timeout,
        poll_interval=poll_interval,
    )