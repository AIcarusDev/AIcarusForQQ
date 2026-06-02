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
from collections.abc import Callable


def _iter_shutdown_signals():
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            yield sig


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
        if shutdown_event.is_set():
            raise KeyboardInterrupt
        signame = signal.Signals(signum).name if signum is not None else "signal"
        print(f"\n🛑 Received {signame}; shutting down...")
        loop.call_soon_threadsafe(shutdown_event.set)

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
        print("\n👋 Good Bye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
