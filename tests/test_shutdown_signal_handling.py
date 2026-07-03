import asyncio
import subprocess
import time

import pytest

import run
from scripts import core_supervisor


@pytest.fixture(autouse=True)
def reset_run_shutdown_state(monkeypatch):
    monkeypatch.setattr(run, "_SHUTDOWN_REQUESTED_AT", None)
    monkeypatch.setattr(run, "_start_browser_cleanup_thread", lambda **_kwargs: None)
    yield
    run._SHUTDOWN_REQUESTED_AT = None


def test_repeated_shutdown_signal_during_grace_does_not_interrupt(monkeypatch):
    monkeypatch.setenv("AICQ_FORCE_SHUTDOWN_AFTER_SECONDS", "60")

    async def _exercise():
        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()

        run._request_shutdown(loop, shutdown_event, None)
        await asyncio.sleep(0)

        assert shutdown_event.is_set()
        run._request_shutdown(loop, shutdown_event, None)

    asyncio.run(_exercise())


def test_repeated_shutdown_signal_after_grace_forces_shutdown(monkeypatch):
    monkeypatch.setenv("AICQ_FORCE_SHUTDOWN_AFTER_SECONDS", "0.1")

    async def _exercise():
        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()
        shutdown_event.set()
        run._SHUTDOWN_REQUESTED_AT = time.monotonic() - 1.0

        with pytest.raises(KeyboardInterrupt):
            run._request_shutdown(loop, shutdown_event, None)

    asyncio.run(_exercise())


class _FakeGracefulProc:
    returncode = None

    def __init__(self):
        self.terminated = False
        self.killed = False

    def poll(self):
        return None

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class _FakeStuckProc(_FakeGracefulProc):
    def wait(self, timeout=None):
        if self.terminated:
            self.returncode = 1
            return 1
        raise subprocess.TimeoutExpired("fake", timeout)


def test_supervisor_waits_for_child_before_terminating_on_ctrl_c():
    proc = _FakeGracefulProc()

    assert core_supervisor._stop_child(proc, graceful_timeout=30) == 0
    assert proc.terminated is False
    assert proc.killed is False


def test_supervisor_terminates_child_after_grace_timeout():
    proc = _FakeStuckProc()

    assert core_supervisor._stop_child(proc, graceful_timeout=0.1) == 1
    assert proc.terminated is True
    assert proc.killed is False
