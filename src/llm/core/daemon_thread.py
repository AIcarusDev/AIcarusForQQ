"""Helpers for running blocking LLM work without pinning interpreter exit."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


def call_in_daemon_thread(
    fn: Callable[P, T],
    *args: P.args,
    thread_name: str = "daemon-worker",
    **kwargs: P.kwargs,
) -> concurrent.futures.Future[T]:
    """Run a blocking function in a daemon thread and expose a Future."""
    future: concurrent.futures.Future[T] = concurrent.futures.Future()

    def _worker() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - propagate into the Future
            future.set_exception(exc)
        else:
            future.set_result(result)

    threading.Thread(target=_worker, daemon=True, name=thread_name).start()
    return future


async def run_in_daemon_thread(
    fn: Callable[P, T],
    *args: P.args,
    thread_name: str = "daemon-worker",
    **kwargs: P.kwargs,
) -> T:
    """Await blocking work without using asyncio's non-daemon default executor."""
    return await asyncio.wrap_future(
        call_in_daemon_thread(fn, *args, thread_name=thread_name, **kwargs)
    )
