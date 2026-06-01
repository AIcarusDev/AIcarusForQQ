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

"""launcher.py — AIcarusForQQ 桌面启动器

架构：
  launcher（本进程）
    ├── 子进程 A: run.py AICQ_WEBUI_ONLY=1   ← 仅 Web UI，核心未启动
    │     └── 用户点击"启动核心" → exit code 76
    ├── 子进程 B: run.py（完整模式）           ← Web UI + 核心
    │     └── 用户点击"停止核心" → exit code 77
    ├── PyWebView 窗口（如有 GUI 库）          ← 内嵌系统 WebView，指向 Quart
    └── pystray 托盘图标（如有 GUI 库）

Exit code 约定（与 routes_core.py 保持同步）：
  76 → launcher 以完整模式重启子进程
  77 → launcher 以 webui-only 模式重启子进程

无 GUI 库降级：
  - 缺少 pywebview / pystray / Pillow → 仅启动子进程，打印访问 URL
  - 完全等价于直接运行 run.py，但保留进程切换逻辑
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

# ── 路径常量 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
RUN_PY = BASE_DIR / "run.py"
SRC_DIR = BASE_DIR / "src"

# ── Exit code 约定（与 routes_core.py 同步）──────────────
LAUNCHER_START_CORE_EXIT_CODE = 76   # 子进程请求：以完整模式重启
LAUNCHER_STOP_CORE_EXIT_CODE = 77    # 子进程请求：以 webui-only 模式重启

# ── GUI 依赖探测 ──────────────────────────────────────────
try:
    import webview          # pywebview
    import pystray
    from PIL import Image, ImageDraw
    HAS_GUI = True
except ImportError:
    HAS_GUI = False


# ══════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════

def _get_server_port() -> int:
    """从 config.yaml 读取端口，读不到则返回默认值 5000。"""
    try:
        import yaml  # type: ignore[import]
        cfg_path = BASE_DIR / "data" / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return int(cfg.get("server", {}).get("port", 5000))
    except Exception:
        pass
    return 5000


def _wait_for_server(port: int, proc: "subprocess.Popen | None" = None, timeout: float = 20.0) -> bool:
    """轮询直到 Quart 在给定端口就绪，超时或子进程退出则返回 False。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # 如果子进程已退出，不必再等
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def _build_env(webui_only: bool) -> dict[str, str]:
    """构造子进程的环境变量。"""
    env = os.environ.copy()
    # 防止 run.py 被 core_supervisor 再次包裹
    env["AICQ_DISABLE_CORE_SUPERVISOR"] = "1"
    # 标记由 launcher 管理，启用 /api/core/start|stop 接口
    env["AICQ_LAUNCHER_MODE"] = "1"
    if webui_only:
        env["AICQ_WEBUI_ONLY"] = "1"
    else:
        env.pop("AICQ_WEBUI_ONLY", None)
    return env


def _make_tray_icon_image(size: int = 64) -> "Image.Image":
    """生成一个简单的托盘图标（紫色圆 + 白色猫耳轮廓）。"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # 背景圆
    draw.ellipse([4, 4, size - 4, size - 4], fill=(124, 106, 247, 255))
    # 猫耳（两个小三角，用折线近似）
    ear_w = size // 6
    draw.polygon([
        (size // 4 - ear_w, size // 3),
        (size // 4,          size // 6),
        (size // 4 + ear_w, size // 3),
    ], fill=(255, 255, 255, 200))
    draw.polygon([
        (3 * size // 4 - ear_w, size // 3),
        (3 * size // 4,          size // 6),
        (3 * size // 4 + ear_w, size // 3),
    ], fill=(255, 255, 255, 200))
    return img


# ══════════════════════════════════════════════════════════
#  核心：进程管理循环
# ══════════════════════════════════════════════════════════

class _LauncherState:
    """在线程间共享的可变状态。"""
    def __init__(self, port: int) -> None:
        self.port = port
        self.webui_only = True       # 初始模式：仅 Web UI
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.stop_requested = False  # 用户主动退出（托盘 Quit / 窗口关闭）
        # 供 GUI 线程等待第一次服务就绪
        self.server_ready = threading.Event()
        self.webview_window: "webview.Window | None" = None


def _process_loop(state: _LauncherState) -> None:
    """子进程管理主循环，运行在独立线程（或无 GUI 时的主线程）。"""
    while not state.stop_requested:
        env = _build_env(state.webui_only)
        mode_label = "webui-only" if state.webui_only else "full"
        print(f"[launcher] 启动子进程（模式={mode_label}）...", flush=True)

        proc = subprocess.Popen(
            [sys.executable, str(RUN_PY)],
            env=env,
            cwd=str(BASE_DIR),
        )
        with state.lock:
            state.proc = proc

        # 等待服务就绪
        ready = _wait_for_server(state.port, proc=proc)
        if ready:
            print(f"[launcher] 服务已就绪: http://127.0.0.1:{state.port}", flush=True)
            state.server_ready.set()
            # 通知 PyWebView 刷新页面
            win = state.webview_window
            if win is not None:
                try:
                    win.load_url(f"http://127.0.0.1:{state.port}/")
                except Exception:
                    pass
        else:
            print(f"[launcher] 警告：服务未在预期时间内就绪（端口 {state.port}）", flush=True)
            state.server_ready.set()  # 仍然放行，避免 GUI 永久阻塞

        # 等待子进程结束
        try:
            returncode = proc.wait()
        except KeyboardInterrupt:
            print("[launcher] 收到 Ctrl+C，正在停止...", flush=True)
            _stop_proc(proc)
            break

        with state.lock:
            state.proc = None

        print(f"[launcher] 子进程退出，code={returncode}", flush=True)

        if returncode == LAUNCHER_START_CORE_EXIT_CODE:
            print("[launcher] 切换到完整模式...", flush=True)
            state.webui_only = False
        elif returncode == LAUNCHER_STOP_CORE_EXIT_CODE:
            print("[launcher] 切换到 webui-only 模式...", flush=True)
            state.webui_only = True
        else:
            # 正常退出或错误，终止 launcher
            state.stop_requested = True
            # 如果有 GUI 窗口，销毁它
            win = state.webview_window
            if win is not None:
                try:
                    win.destroy()
                except Exception:
                    pass
            break


def _stop_proc(proc: "subprocess.Popen | None") -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ══════════════════════════════════════════════════════════
#  GUI 模式
# ══════════════════════════════════════════════════════════

def _run_with_gui(state: _LauncherState) -> None:
    """有 GUI 库时：托盘 + PyWebView 主窗口。"""

    # ── 托盘图标 ─────────────────────────────────────────
    tray_icon: "pystray.Icon | None" = None

    def _open_browser(_icon=None, _item=None):
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{state.port}/")

    def _quit_app(_icon=None, _item=None):
        state.stop_requested = True
        with state.lock:
            _stop_proc(state.proc)
        if tray_icon is not None:
            try:
                tray_icon.stop()
            except Exception:
                pass
        win = state.webview_window
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass

    try:
        menu = pystray.Menu(
            pystray.MenuItem("在浏览器中打开", _open_browser),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出 AIcarus Launcher", _quit_app),
        )
        tray_icon = pystray.Icon(
            "AIcarus",
            icon=_make_tray_icon_image(),
            title="AIcarus for QQ",
            menu=menu,
        )
        tray_icon.run_detached()
    except Exception as e:
        print(f"[launcher] 托盘图标创建失败（非致命）: {e}", flush=True)

    # ── 启动进程管理线程 ─────────────────────────────────
    loop_thread = threading.Thread(target=_process_loop, args=(state,), daemon=True)
    loop_thread.start()

    # 等待服务就绪后再显示窗口
    state.server_ready.wait(timeout=30)

    # 如果子进程在服务就绪前就退出了（启动失败），不创建窗口，直接清理
    if state.stop_requested:
        if tray_icon is not None:
            try:
                tray_icon.stop()
            except Exception:
                pass
        loop_thread.join(timeout=5)
        return

    url = f"http://127.0.0.1:{state.port}/"

    # ── PyWebView 主窗口（主线程）────────────────────────
    def _on_closed():
        _quit_app()

    window = webview.create_window(
        title="AIcarus for QQ",
        url=url,
        width=1280,
        height=800,
        min_size=(900, 600),
        background_color="#0d1117",
    )
    state.webview_window = window

    webview.start(
        func=None,
        debug=False,
    )

    # webview.start() 返回说明窗口已关闭
    _quit_app()
    loop_thread.join(timeout=5)


# ══════════════════════════════════════════════════════════
#  无 GUI 降级
# ══════════════════════════════════════════════════════════

def _run_headless(state: _LauncherState) -> None:
    """无 GUI 库时：直接在主线程跑进程管理循环，行为与 run.py 相同。"""
    print(
        "[launcher] 未检测到 GUI 库（pywebview / pystray），以无头模式运行。\n"
        f"  服务启动后请访问 http://127.0.0.1:{state.port}/",
        flush=True,
    )
    _process_loop(state)


# ══════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════

def main() -> None:
    port = _get_server_port()
    state = _LauncherState(port=port)

    print(
        "╔══════════════════════════════════════╗\n"
        "║   AIcarus for QQ  —  Launcher        ║\n"
        "╚══════════════════════════════════════╝",
        flush=True,
    )

    if HAS_GUI:
        _run_with_gui(state)
    else:
        _run_headless(state)


if __name__ == "__main__":
    main()
