import asyncio
import threading

from llm.core.daemon_thread import call_in_daemon_thread, run_in_daemon_thread


def _capture_call(*args, marker=None):
    return args, marker, threading.current_thread().name


def test_call_in_daemon_thread_forwards_arguments_without_consuming_thread_name():
    future = call_in_daemon_thread(
        _capture_call,
        "first",
        "second",
        marker="call",
        thread_name="test-daemon-call",
    )

    assert future.result(timeout=2) == (
        ("first", "second"),
        "call",
        "test-daemon-call",
    )


def test_run_in_daemon_thread_forwards_arguments_without_consuming_thread_name():
    result = asyncio.run(
        run_in_daemon_thread(
            _capture_call,
            "first",
            "second",
            marker="run",
            thread_name="test-daemon-run",
        )
    )

    assert result == (("first", "second"), "run", "test-daemon-run")
