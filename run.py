# Copyright (C) 2026  AIcarusDev
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
AIcarusForQQ Launcher
This script sets up the environment and launches the main application.
"""
import os
import sys
import asyncio
import signal
import subprocess
import threading
import time
from collections.abc import Callable


_BROWSER_CLEANUP_LOCK = threading.Lock()
_BROWSER_CLEANUP_THREAD: threading.Thread | None = None
_SHUTDOWN_REQUEST_LOCK = threading.Lock()
_SHUTDOWN_REQUESTED_AT: float | None = None
_DEFAULT_FORCE_SHUTDOWN_AFTER_SECONDS = 30.0


def _iter_shutdown_signals():
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            yield sig


def _close_browser_sessions_best_effort(*, timeout_s: float | None = 8.0) -> None:
    try:
        from browser.session import close_browser_sessions

        closed = close_browser_sessions(timeout_s=timeout_s)
        if closed:
            print(f"[shutdown] closed {closed} browser session(s)")
    except Exception as exc:
        print(f"[shutdown] browser cleanup skipped: {exc}")


def _start_browser_cleanup_thread(*, timeout_s: float | None = 8.0) -> None:
    global _BROWSER_CLEANUP_THREAD

    with _BROWSER_CLEANUP_LOCK:
        if _BROWSER_CLEANUP_THREAD is not None and _BROWSER_CLEANUP_THREAD.is_alive():
            return
        thread = threading.Thread(
            target=_close_browser_sessions_best_effort,
            kwargs={"timeout_s": timeout_s},
            name="browser-shutdown-cleanup",
            daemon=True,
        )
        _BROWSER_CLEANUP_THREAD = thread
        thread.start()


def _force_shutdown_after_seconds() -> float:
    raw = os.environ.get("AICQ_FORCE_SHUTDOWN_AFTER_SECONDS", "")
    if not raw:
        return _DEFAULT_FORCE_SHUTDOWN_AFTER_SECONDS
    try:
        return max(0.1, float(raw))
    except (TypeError, ValueError):
        return _DEFAULT_FORCE_SHUTDOWN_AFTER_SECONDS


def _request_shutdown(
    loop: asyncio.AbstractEventLoop,
    shutdown_event: asyncio.Event,
    signum=None,
) -> None:
    global _SHUTDOWN_REQUESTED_AT

    signame = signal.Signals(signum).name if signum is not None else "signal"
    now = time.monotonic()
    with _SHUTDOWN_REQUEST_LOCK:
        already_requested = shutdown_event.is_set() or _SHUTDOWN_REQUESTED_AT is not None
        if already_requested:
            requested_at = _SHUTDOWN_REQUESTED_AT or now
            elapsed = now - requested_at
            force_after = _force_shutdown_after_seconds()
            _start_browser_cleanup_thread(timeout_s=1.0)
            if elapsed >= force_after:
                print(
                    f"\n🛑 Received {signame} again after {elapsed:.1f}s; forcing shutdown..."
                )
                raise KeyboardInterrupt
            remaining = max(0.0, force_after - elapsed)
            print(
                f"\n🛑 Received {signame} again; shutdown already in progress "
                f"({elapsed:.1f}s). Force available in {remaining:.1f}s."
            )
            return
        _SHUTDOWN_REQUESTED_AT = now
    print(f"\n🛑 Received {signame}; shutting down...")
    loop.call_soon_threadsafe(shutdown_event.set)
    _start_browser_cleanup_thread()


def _install_shutdown_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    shutdown_event: asyncio.Event,
) -> Callable[[], None]:
    """Wire console signals to Hypercorn's shutdown trigger.

    Hypercorn has its own fallback for Windows, but installing our own handler
    here makes the run.py path explicit and lets the signal wake the loop via
    call_soon_threadsafe().
    """
    previous_handlers: dict[int, object] = {}
    loop_handlers: list[int] = []

    def request_shutdown(signum=None, _frame=None):
        _request_shutdown(loop, shutdown_event, signum)

    for sig in _iter_shutdown_signals():
        try:
            loop.add_signal_handler(sig, request_shutdown, sig, None)
            loop_handlers.append(sig)
        except (NotImplementedError, RuntimeError):
            previous_handlers[int(sig)] = signal.getsignal(sig)
            signal.signal(sig, request_shutdown)

    def restore() -> None:
        for sig in loop_handlers:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass
        for sig, previous in previous_handlers.items():
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError):
                pass

    return restore


async def _serve_with_shutdown_trigger(app, hypercorn_config) -> None:
    from hypercorn.asyncio import serve
    import app_state

    shutdown_event = asyncio.Event()
    app_state.server_shutdown_event = shutdown_event
    restore_signal_handlers = _install_shutdown_signal_handlers(
        asyncio.get_running_loop(),
        shutdown_event,
    )
    try:
        await serve(app, hypercorn_config, shutdown_trigger=shutdown_event.wait)
    finally:
        global _SHUTDOWN_REQUESTED_AT
        with _SHUTDOWN_REQUEST_LOCK:
            _SHUTDOWN_REQUESTED_AT = None
        _start_browser_cleanup_thread(timeout_s=1.0)
        if getattr(app_state, "server_shutdown_event", None) is shutdown_event:
            app_state.server_shutdown_event = None
        restore_signal_handlers()

def main():
    # Set the base directory to the location of this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if (
        os.environ.get("AICQ_CORE_SUPERVISED") != "1"
        and os.environ.get("AICQ_DISABLE_CORE_SUPERVISOR") != "1"
    ):
        supervisor = os.path.join(base_dir, "scripts", "core_supervisor.py")
        if os.path.exists(supervisor):
            print("🧭 Starting core supervisor for restart support...")
            try:
                completed = subprocess.run([sys.executable, supervisor], cwd=base_dir)
            except KeyboardInterrupt:
                sys.exit(0)
            sys.exit(int(completed.returncode or 0))

    # Add the src directory to sys.path so modules can be imported
    src_dir = os.path.join(base_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Change working directory to ensure relative paths work as expected
    # (Although we are patching config_loader to be smarter)
    os.chdir(base_dir)

    print(f"🚀 Launching AIcarusForQQ from {base_dir}...")
    
    try:
        from hypercorn.config import Config as HypercornConfig
        from src.main import app
        import app_state
        
        try:
            server_config = app_state.config.get("server", {})
            # Use 5000 as default to be consistent with main.py
            port = server_config.get("port", 5000)
            host = server_config.get("host", "127.0.0.1")
            debug = server_config.get("debug", True)
        except Exception as e:
            print(f"⚠️  Could not load config for port/host: {e}")
            port = 5000
            host = "127.0.0.1"
            debug = True
            
        print(f"🌍 Server starting at http://{host}:{port}")
        hypercorn_config = HypercornConfig()
        hypercorn_config.bind = [f"{host}:{port}"]
        hypercorn_config.use_reloader = False
        asyncio.run(_serve_with_shutdown_trigger(app, hypercorn_config))
        if (
            getattr(app_state, "core_restart_requested", False)
            or getattr(app_state, "launcher_switch_requested", False)
        ):
            exit_code = int(
                getattr(app_state, "core_restart_exit_code", None) or 75
            )
            if getattr(app_state, "core_restart_requested", False):
                print(f"🔁 Core restart requested; exiting with code {exit_code}...")
            else:
                print(f"🔁 Launcher mode switch requested; exiting with code {exit_code}...")
            sys.exit(exit_code)
        print("👋 Good Bye!")
        
    except ImportError as e:
        print(f"❌ Error: Could not import application modules. {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        # 用户手动停止 (Ctrl+C)，允许优雅退出
        _start_browser_cleanup_thread(timeout_s=1.0)
        print("\n👋 Good Bye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
