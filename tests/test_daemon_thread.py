import asyncio
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm.core.daemon_thread import run_in_daemon_thread


class DaemonThreadTests(unittest.TestCase):
    def test_cancel_wait_does_not_wait_for_blocking_worker(self) -> None:
        release_worker = threading.Event()

        async def _run() -> float:
            task = asyncio.create_task(
                run_in_daemon_thread(
                    release_worker.wait,
                    1.0,
                    thread_name="unit-daemon-worker",
                )
            )
            await asyncio.sleep(0.05)
            started_at = time.perf_counter()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            return time.perf_counter() - started_at

        try:
            elapsed = asyncio.run(_run())
        finally:
            release_worker.set()

        self.assertLess(elapsed, 0.2)


if __name__ == "__main__":
    unittest.main()
