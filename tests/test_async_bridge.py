import asyncio
import concurrent.futures
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tools._async_bridge import LoopStoppedError, run_coroutine_sync


class _LoopRunner:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self._started.set()
        self.loop.run_forever()

        pending = asyncio.all_tasks(self.loop)
        for task in pending:
            task.cancel()
        if pending:
            self.loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self.loop.close()

    def __enter__(self) -> "_LoopRunner":
        self._thread.start()
        if not self._started.wait(timeout=1.0):
            raise RuntimeError("loop thread did not start")
        return self

    def stop(self) -> None:
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=1.0)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


async def _delayed_value(delay: float, value: int) -> int:
    await asyncio.sleep(delay)
    return value


async def _wait_forever() -> None:
    await asyncio.Event().wait()


class AsyncBridgeTests(unittest.TestCase):
    def test_run_coroutine_sync_timeout_none_preserves_long_wait(self) -> None:
        with _LoopRunner() as runner:
            started_at = time.perf_counter()
            result = run_coroutine_sync(
                _delayed_value(0.12, 42),
                runner.loop,
                timeout=None,
                poll_interval=0.02,
            )

        elapsed = time.perf_counter() - started_at
        self.assertEqual(result, 42)
        self.assertGreaterEqual(elapsed, 0.1)

    def test_run_coroutine_sync_preserves_explicit_timeout(self) -> None:
        with _LoopRunner() as runner:
            started_at = time.perf_counter()
            with self.assertRaises(concurrent.futures.TimeoutError):
                run_coroutine_sync(
                    _delayed_value(0.2, 7),
                    runner.loop,
                    timeout=0.05,
                    poll_interval=0.02,
                )

        elapsed = time.perf_counter() - started_at
        self.assertLess(elapsed, 0.15)

    def test_run_coroutine_sync_exits_when_loop_stops(self) -> None:
        with _LoopRunner() as runner:
            stop_timer = threading.Timer(0.05, runner.stop)
            stop_timer.daemon = True
            stop_timer.start()

            started_at = time.perf_counter()
            try:
                with self.assertRaises(LoopStoppedError):
                    run_coroutine_sync(
                        _wait_forever(),
                        runner.loop,
                        timeout=None,
                        poll_interval=0.02,
                    )
            finally:
                stop_timer.cancel()

        elapsed = time.perf_counter() - started_at
        self.assertLess(elapsed, 0.3)


if __name__ == "__main__":
    unittest.main()