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
  75 → run.py 自重启请求，launcher 以相同模式重启子进程
  76 → launcher 以完整模式重启子进程
  77 → launcher 以 webui-only 模式重启子进程

无 GUI 库降级：
  - 缺少 pywebview / pystray / Pillow → 仅启动子进程，打印访问 URL
  - 完全等价于直接运行 run.py，但保留进程切换逻辑
"""

from __future__ import annotations

import os
import hashlib
import html
import json
import signal
import socket
import subprocess
import sys
import threading
import time
import argparse
import urllib.error
import urllib.request
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    import webview
    import pystray

# ── 路径常量 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
RUN_PY = BASE_DIR / "run.py"
SRC_DIR = BASE_DIR / "src"
APP_ICON_PNG = SRC_DIR / "static" / "app-icon-256.png"
APP_ICON_ICO = SRC_DIR / "static" / "app-icon.ico"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from runtime.core_restart import RESTART_EXIT_CODE as CORE_RESTART_EXIT_CODE  # type: ignore[import]
except Exception:
    CORE_RESTART_EXIT_CODE = 75

# ── Exit code 约定（与 routes_core.py 同步）──────────────
LAUNCHER_START_CORE_EXIT_CODE = 76   # 子进程请求：以完整模式重启
LAUNCHER_STOP_CORE_EXIT_CODE = 77    # 子进程请求：以 webui-only 模式重启
USE_NATIVE_WINDOW_FRAME = sys.platform == "win32"

# ── GUI 依赖探测 ──────────────────────────────────────────
GUI_IMPORT_ERROR: ImportError | None = None

try:
    import webview          # pywebview
    import pystray
    from PIL import Image, ImageDraw
    HAS_GUI = True
except ImportError as exc:
    GUI_IMPORT_ERROR = exc
    HAS_GUI = False


def _is_stale_pywebview_return_callback_error(script: str, exc: BaseException) -> bool:
    """识别 pywebview 在页面关闭/重载后回填已失效 Promise 的噪声异常。"""
    if "window.pywebview._returnValuesCallbacks" not in script:
        return False

    payload = exc.args[0] if exc.args else None
    if isinstance(payload, dict):
        name = str(payload.get("name", ""))
        message = str(payload.get("message", ""))
    else:
        name = type(exc).__name__
        message = str(exc)

    return name == "TypeError" and (
        "Cannot read properties of undefined" in message
        or "undefined is not an object" in message
        or "is not a function" in message
    )


def _install_pywebview_return_callback_guard() -> None:
    """
    pywebview 6.x 会无条件向 JS 侧 Promise 回调表回填 API 返回值。
    窗口关闭或页面重载时该表可能已经不存在，导致后台 _call 线程打印未捕获异常。
    """
    if not HAS_GUI:
        return
    try:
        from webview.errors import JavascriptException  # type: ignore[import]
        from webview.window import Window  # type: ignore[import]
    except Exception:
        return

    current = Window.evaluate_js
    if getattr(current, "_aicq_pywebview_callback_guard", False):
        return

    @wraps(current)
    def guarded_evaluate_js(self, script, *args, **kwargs):
        try:
            return current(self, script, *args, **kwargs)
        except JavascriptException as exc:
            if _is_stale_pywebview_return_callback_error(str(script), exc):
                return None
            raise

    guarded_evaluate_js._aicq_pywebview_callback_guard = True  # type: ignore[attr-defined]
    Window.evaluate_js = guarded_evaluate_js  # type: ignore[assignment]


# ══════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════

def _get_server_port() -> int:
    """从用户配置读取端口，读不到则返回默认值 5000。"""
    try:
        import yaml  # type: ignore[import]
        for cfg_path in (
            BASE_DIR / "config" / "config_user.yaml",
            BASE_DIR / "data" / "config.yaml",
        ):
            if not cfg_path.exists():
                continue
            with cfg_path.open(encoding="utf-8") as f:
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


def _webui_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/"


def _http_endpoint_available(url: str, *, timeout: float = 0.8) -> bool:
    """确认 HTTP 层已经能响应，避免 WebView 停在 ERR_CONNECTION_REFUSED。"""
    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        # HTTP 错误说明服务已经能处理请求；交给 WebView 显示真实页面。
        return True
    except Exception:
        return False


def _load_window_url_when_http_ready(
    window: Any,
    state: Any,
    url: str,
    *,
    timeout: float = 20.0,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(state, "stop_requested", False):
            return False
        if _http_endpoint_available(url):
            try:
                window.load_url(url)
                return True
            except Exception as exc:
                print(f"[launcher] WebView 导航失败: {exc}", flush=True)
                return False
        time.sleep(0.25)
    return False


def _launcher_loading_html(port: int, message: str = "正在启动 WebUI...") -> str:
    target_url = _webui_url(port)
    message_html = html.escape(message, quote=False)
    target_url_html = html.escape(target_url, quote=True)
    target_url_js = json.dumps(target_url, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AIcarus for QQ</title>
  <style>
    html, body {{
      height: 100%;
      margin: 0;
      background: #0d1117;
      color: #e6edf3;
      font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    }}
    body {{
      display: grid;
      place-items: center;
    }}
    .boot {{
      width: min(460px, calc(100vw - 48px));
      padding: 28px 30px;
      border: 1px solid #30363d;
      border-radius: 8px;
      background: #161b22;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.32);
    }}
    .title {{
      margin: 0 0 10px;
      font-size: 20px;
      font-weight: 700;
    }}
    .msg {{
      margin: 0;
      color: #8b949e;
      line-height: 1.65;
      font-size: 14px;
    }}
    .dot {{
      display: inline-block;
      width: 8px;
      height: 8px;
      margin-right: 10px;
      border-radius: 50%;
      background: #7c6af7;
      box-shadow: 0 0 0 0 rgba(124, 106, 247, 0.75);
      animation: pulse 1.25s infinite;
    }}
    @keyframes pulse {{
      70% {{ box-shadow: 0 0 0 12px rgba(124, 106, 247, 0); }}
      100% {{ box-shadow: 0 0 0 0 rgba(124, 106, 247, 0); }}
    }}
  </style>
</head>
<body>
  <main class="boot">
    <h1 class="title"><span class="dot"></span>AIcarus for QQ</h1>
    <p class="msg" data-status>{message_html}<br>目标地址：{target_url_html}</p>
  </main>
  <script>
    (function() {{
      const targetUrl = {target_url_js};
      const deadline = Date.now() + 60000;
      const statusEl = document.querySelector('[data-status]');

      async function waitAndNavigate() {{
        let timer = null;
        try {{
          const controller = new AbortController();
          timer = setTimeout(() => controller.abort(), 1500);
          await fetch(targetUrl, {{
            cache: 'no-store',
            mode: 'no-cors',
            signal: controller.signal,
          }});
          clearTimeout(timer);
          window.location.replace(targetUrl);
          return;
        }} catch (e) {{
          if (timer !== null) clearTimeout(timer);
          // Network failure is expected while the child process is still binding.
        }}

        if (Date.now() < deadline) {{
          setTimeout(waitAndNavigate, 750);
          return;
        }}

        if (statusEl) {{
          statusEl.innerHTML = 'WebUI 未在预期时间内响应，请查看控制台日志。<br>目标地址：' + targetUrl;
        }}
      }}

      setTimeout(waitAndNavigate, 250);
    }}());
  </script>
</body>
</html>"""


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


def _make_tray_icon_image(size: int = 64) -> Any:
    """读取应用图标作为托盘图标，缺失时降级到简单占位图。"""
    if APP_ICON_PNG.exists():
        try:
            return Image.open(APP_ICON_PNG).convert("RGBA").resize(  # type: ignore[attr-defined]
                (size, size),
                Image.Resampling.LANCZOS,  # type: ignore[attr-defined]
            )
        except Exception:
            pass

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))  # type: ignore[attr-defined]
    draw = ImageDraw.Draw(img)  # type: ignore[attr-defined]
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


def _webview_icon_path() -> str | None:
    """返回 pywebview 可用的窗口/任务栏图标路径。"""
    for path in (APP_ICON_ICO, APP_ICON_PNG):
        if path.exists():
            return str(path)
    return None


# ══════════════════════════════════════════════════════════
#  关闭偏好持久化
# ══════════════════════════════════════════════════════════

_CLOSE_PREF_PATH = BASE_DIR / "data" / "launcher_close_pref.txt"


def _load_close_pref() -> str | None:
    """读取上次保存的关闭偏好，返回 'quit' / 'tray' / None。"""
    try:
        val = _CLOSE_PREF_PATH.read_text(encoding="utf-8").strip()
        if val in ("quit", "tray"):
            return val
    except OSError:
        pass
    return None


def _save_close_pref(pref: str) -> None:
    try:
        _CLOSE_PREF_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CLOSE_PREF_PATH.write_text(pref, encoding="utf-8")
    except OSError:
        pass


def _ask_close_action(default_pref: str | None) -> tuple[str, bool]:
    """弹出与 WebUI 风格一致的深色对话框，询问关闭窗口时的行为。

    返回 (action, remember)：
      action   : 'quit' | 'tray' | 'cancel'
      remember : 是否记住选择
    """
    import tkinter as tk

    # ── 配色（与 _base.html dark-deep 主题一致）──────────────
    C_SURFACE   = "#161b22"
    C_SURFACE2  = "#1c2128"
    C_BORDER    = "#30363d"
    C_TEXT      = "#e6edf3"
    C_TEXT_SEC  = "#8b949e"
    C_ACCENT    = "#7c6af7"
    C_ACCENT_DIM= "#2d2660"
    C_BTN_HV    = "#9d8fff"
    FONT_UI     = ("Segoe UI", 9)
    FONT_TITLE  = ("Segoe UI", 10, "bold")

    result: dict[str, object] = {"action": "cancel", "remember": False}

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    dlg = tk.Toplevel(root)
    dlg.configure(bg=C_BORDER)   # 外层 1px 边框色
    dlg.resizable(False, False)
    dlg.attributes("-topmost", True)
    dlg.overrideredirect(True)   # 无系统标题栏

    W, H = 320, 210
    sw = dlg.winfo_screenwidth()
    sh = dlg.winfo_screenheight()
    dlg.geometry(f"{W}x{H}+{(sw - W) // 2}+{(sh - H) // 2}")
    dlg.grab_set()

    # ── 自定义标题栏 ─────────────────────────────────────────
    titlebar = tk.Frame(dlg, bg=C_SURFACE2, height=32, cursor="fleur")
    titlebar.pack(fill="x", padx=1, pady=(1, 0))
    titlebar.pack_propagate(False)

    tk.Label(
        titlebar,
        text="关闭 AIcarus",
        font=("Segoe UI", 9),
        bg=C_SURFACE2,
        fg=C_TEXT_SEC,
    ).pack(side="left", padx=12)

    def _cancel_close():
        result["action"] = "cancel"
        dlg.destroy()

    close_btn = tk.Label(
        titlebar,
        text="✕",
        font=("Segoe UI", 10),
        bg=C_SURFACE2,
        fg=C_TEXT_SEC,
        cursor="hand2",
        padx=10,
        pady=4,
    )
    close_btn.pack(side="right")
    close_btn.bind("<Button-1>", lambda e: _cancel_close())
    close_btn.bind("<Enter>", lambda e: close_btn.config(bg="#c0392b", fg="#ffffff"))
    close_btn.bind("<Leave>", lambda e: close_btn.config(bg=C_SURFACE2, fg=C_TEXT_SEC))

    # 拖拽移动对话框
    _drag_state: dict = {}
    def _titlebar_press(e):
        _drag_state["x"] = e.x_root - dlg.winfo_x()
        _drag_state["y"] = e.y_root - dlg.winfo_y()
    def _titlebar_drag(e):
        dlg.geometry(f"+{e.x_root - _drag_state['x']}+{e.y_root - _drag_state['y']}")
    titlebar.bind("<ButtonPress-1>", _titlebar_press)
    titlebar.bind("<B1-Motion>", _titlebar_drag)
    for w in titlebar.winfo_children():
        if str(w) != str(close_btn):
            w.bind("<ButtonPress-1>", _titlebar_press)
            w.bind("<B1-Motion>", _titlebar_drag)

    # ── 内容区 ────────────────────────────────────────────────
    card = tk.Frame(dlg, bg=C_SURFACE, padx=20, pady=16)
    card.pack(fill="both", expand=True, padx=1, pady=(0, 1))

    tk.Label(
        card,
        text="关闭窗口时，你希望：",
        font=FONT_TITLE,
        bg=C_SURFACE,
        fg=C_TEXT,
        anchor="w",
    ).pack(fill="x", pady=(0, 12))

    # 单选按钮（自定义颜色）
    action_var = tk.StringVar(value=default_pref or "tray")

    def _make_radio(parent, text, value):
        rb = tk.Radiobutton(
            parent,
            text=text,
            variable=action_var,
            value=value,
            font=FONT_UI,
            bg=C_SURFACE,
            fg=C_TEXT,
            activebackground=C_SURFACE2,
            activeforeground=C_TEXT,
            selectcolor=C_ACCENT_DIM,
            indicatoron=True,
            anchor="w",
            cursor="hand2",
        )
        rb.pack(fill="x", padx=4, pady=2)

    _make_radio(card, "最小化到托盘", "tray")
    _make_radio(card, "退出 AIcarus Launcher", "quit")

    # 分隔线
    tk.Frame(card, bg=C_BORDER, height=1).pack(fill="x", pady=(12, 8))

    # 底部行：记住选择 + 按钮
    bottom = tk.Frame(card, bg=C_SURFACE)
    bottom.pack(fill="x")

    remember_var = tk.BooleanVar(value=default_pref is not None)
    cb = tk.Checkbutton(
        bottom,
        text="记住我的选择",
        variable=remember_var,
        font=FONT_UI,
        bg=C_SURFACE,
        fg=C_TEXT_SEC,
        activebackground=C_SURFACE,
        activeforeground=C_TEXT,
        selectcolor=C_ACCENT_DIM,
        cursor="hand2",
    )
    cb.pack(side="left")

    btn_frame = tk.Frame(bottom, bg=C_SURFACE)
    btn_frame.pack(side="right")

    def _make_btn(parent, text, command, is_primary=False):
        bg = C_ACCENT if is_primary else C_SURFACE2
        fg = "#ffffff" if is_primary else C_TEXT
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=FONT_UI,
            bg=bg,
            fg=fg,
            activebackground=C_BTN_HV if is_primary else C_BORDER,
            activeforeground="#ffffff" if is_primary else C_TEXT,
            relief="flat",
            padx=14,
            pady=5,
            cursor="hand2",
            bd=0,
        )
        btn.pack(side="left", padx=(6, 0))
        return btn

    def _confirm():
        result["action"] = action_var.get()
        result["remember"] = bool(remember_var.get())
        dlg.destroy()

    dlg.protocol("WM_DELETE_WINDOW", _cancel_close)
    _make_btn(btn_frame, "取消", _cancel_close, is_primary=False)
    _make_btn(btn_frame, "确定", _confirm, is_primary=True)

    dlg.wait_window()
    root.destroy()

    return str(result["action"]), bool(result["remember"])


# ══════════════════════════════════════════════════════════
#  核心：进程管理循环
# ══════════════════════════════════════════════════════════

class _LauncherState:
    """在线程间共享的可变状态。"""
    def __init__(self, port: int, *, webui_only: bool = True) -> None:
        self.port = port
        self.webui_only = webui_only  # 初始模式：仅 Web UI 或完整核心
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.stop_requested = False  # 用户主动退出（托盘 Quit / 窗口关闭）
        # 供 GUI 线程等待第一次服务就绪
        self.server_ready = threading.Event()
        self.webview_window: Optional[Any] = None
        # 关闭行为偏好：'quit' | 'tray' | None（每次询问）
        # 从用户数据目录持久化读写
        self.close_pref: str | None = _load_close_pref()
        # 防止 closing 事件重入（win.destroy() 也会触发 closing）
        self._closing_handled: bool = False
        # frameless 窗口没有系统标题栏，最大化/还原状态由 launcher 同步给前端按钮。
        self.window_maximized: bool = False


def _process_loop(state: _LauncherState) -> None:
    """子进程管理主循环，运行在独立线程（或无 GUI 时的主线程）。"""
    while not state.stop_requested:
        state.server_ready.clear()
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
            url = _webui_url(state.port)
            print(f"[launcher] 服务已就绪: {url}", flush=True)
            state.server_ready.set()
            # TCP ready 早于 HTTP 可稳定访问；导航前再做一次 HTTP 层确认。
            win = state.webview_window
            if win is not None:
                threading.Thread(
                    target=_load_window_url_when_http_ready,
                    args=(win, state, url),
                    daemon=True,
                ).start()
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

        if returncode == CORE_RESTART_EXIT_CODE:
            print("[launcher] core 请求自重启，保持当前模式重启...", flush=True)
            continue
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


def _stop_proc(
    proc: "subprocess.Popen | None",
    *,
    graceful_timeout: float = 0.0,
) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    if graceful_timeout > 0:
        try:
            proc.wait(timeout=graceful_timeout)
            return
        except subprocess.TimeoutExpired:
            pass
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _iter_shutdown_signals():
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            yield sig


def _install_console_shutdown_handlers(
    on_shutdown: Callable[[str], None],
) -> Callable[[], None]:
    """Route console interrupts and close events through launcher shutdown.

    In GUI mode, Ctrl+C can interrupt the pywebview event loop without giving
    the launcher a chance to destroy the window.  Windows console control
    handlers run on a separate system thread, so they can wake the launcher
    even while the main thread is inside native GUI code.
    """
    previous_handlers: dict[int, Union[Callable[[int, FrameType | None], Any], Any]] = {}

    def _handle_signal(signum, _frame) -> None:
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = f"signal {signum}"
        on_shutdown(signame)

    for sig in _iter_shutdown_signals():
        try:
            previous_handlers[int(sig)] = signal.getsignal(sig)
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            pass

    restore_windows_handler: Callable[[], None] | None = None
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            ctrl_names = {
                0: "CTRL_C_EVENT",
                1: "CTRL_BREAK_EVENT",
                2: "CTRL_CLOSE_EVENT",
                5: "CTRL_LOGOFF_EVENT",
                6: "CTRL_SHUTDOWN_EVENT",
            }
            handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

            def _console_ctrl_handler(ctrl_type: int) -> bool:
                on_shutdown(ctrl_names.get(ctrl_type, f"CTRL_EVENT_{ctrl_type}"))
                return True

            handler_ref = handler_type(_console_ctrl_handler)
            if ctypes.windll.kernel32.SetConsoleCtrlHandler(handler_ref, True):
                def _restore_windows_handler() -> None:
                    try:
                        ctypes.windll.kernel32.SetConsoleCtrlHandler(handler_ref, False)
                    except Exception:
                        pass

                restore_windows_handler = _restore_windows_handler
        except Exception:
            restore_windows_handler = None

    def restore() -> None:
        if restore_windows_handler is not None:
            restore_windows_handler()
        for sig, previous in previous_handlers.items():
            try:
                signal.signal(sig, previous)  # type: ignore[arg-type]
            except (ValueError, OSError):
                pass

    return restore


# ══════════════════════════════════════════════════════════
#  GUI 模式
# ══════════════════════════════════════════════════════════

def _run_with_gui(state: _LauncherState) -> None:
    """有 GUI 库时：托盘 + PyWebView 主窗口。"""

    _install_pywebview_return_callback_guard()

    tray_icon: Optional[Any] = None  # type: ignore[assignment]
    shutdown_requested = threading.Event()

    # ── 共享操作原语 ─────────────────────────────────────

    def _stop_tray_icon() -> None:
        if tray_icon is not None:
            try:
                tray_icon.stop()
            except Exception:
                pass

    def _destroy_window() -> None:
        win = state.webview_window
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass

    def _request_quit(
        reason: str,
        *,
        destroy_window: bool,
        graceful_timeout: float = 0.0,
    ) -> None:
        """真正退出：标记停止、关闭 GUI 资源并停止子进程。"""
        first_request = not shutdown_requested.is_set()
        shutdown_requested.set()
        state._closing_handled = True
        state.stop_requested = True
        state.server_ready.set()
        if first_request:
            print(f"[launcher] {reason}，正在退出...", flush=True)
        if destroy_window:
            _destroy_window()
        _stop_tray_icon()
        with state.lock:
            proc = state.proc
        _stop_proc(proc, graceful_timeout=graceful_timeout)

    def _do_quit(
        *,
        destroy_window: bool = False,
        reason: str = "收到退出请求",
        graceful_timeout: float = 0.0,
    ) -> None:
        _request_quit(
            reason,
            destroy_window=destroy_window,
            graceful_timeout=graceful_timeout,
        )

    def _do_hide() -> None:
        """隐藏窗口到托盘。"""
        win = state.webview_window
        if win is not None:
            try:
                win.hide()
            except Exception:
                pass

    def _show_close_dialog_now() -> None:
        if state._closing_handled or shutdown_requested.is_set():
            return
        win = state.webview_window
        if win is not None:
            try:
                win.evaluate_js("window._wcShowCloseDialog && window._wcShowCloseDialog()")
            except Exception:
                # 如果 JS 没准备好，则直接退出以避免卡死
                _do_quit(destroy_window=True, reason="关闭确认界面不可用")

    def _run_later(delay: float, callback: Callable[[], None]) -> None:
        timer = threading.Timer(delay, callback)
        timer.daemon = True
        timer.start()

    def _show_close_dialog() -> None:
        """把页面 JS 延后到 pywebview 的可取消 closing 事件返回后执行。"""
        _run_later(0.05, _show_close_dialog_now)

    restore_shutdown_handlers = _install_console_shutdown_handlers(
        lambda reason: _do_quit(
            destroy_window=True,
            reason=f"收到控制台关闭指令 {reason}",
            graceful_timeout=3.0,
        )
    )

    def _sync_window_maximized(maximized: bool) -> None:
        state.window_maximized = bool(maximized)
        win = state.webview_window
        if win is None:
            return
        js_value = "true" if state.window_maximized else "false"
        try:
            win.evaluate_js(
                "window._wcSetMaximized && window._wcSetMaximized("
                + js_value
                + ")"
            )
        except Exception:
            pass

    # ── 托盘图标 ─────────────────────────────────────────

    def _open_browser(_icon=None, _item=None):
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{state.port}/")

    def _show_window(_icon=None, _item=None):
        win = state.webview_window
        if win is not None:
            try:
                win.show()
            except Exception:
                pass

    def _quit_from_tray(_icon=None, _item=None):
        _do_quit(destroy_window=True, reason="收到托盘退出请求")

    try:
        tray_icon = pystray.Icon(  # type: ignore[attr-defined,union-attr]
            "AIcarus",
            icon=_make_tray_icon_image(),
            title="AIcarus for QQ",
            menu=pystray.Menu(  # type: ignore[attr-defined]
                pystray.MenuItem("显示窗口", _show_window, default=True),  # type: ignore[attr-defined]
                pystray.MenuItem("在浏览器中打开", _open_browser),  # type: ignore[attr-defined]
                pystray.Menu.SEPARATOR,  # type: ignore[attr-defined]
                pystray.MenuItem("退出 AIcarus Launcher", _quit_from_tray),  # type: ignore[attr-defined]
            ),
        )
        tray_icon.run_detached()  # type: ignore[union-attr]
    except Exception as e:
        print(f"[launcher] 托盘图标创建失败（非致命）: {e}", flush=True)

    # ── 启动进程管理线程 ─────────────────────────────────
    loop_thread = threading.Thread(target=_process_loop, args=(state,), daemon=True)
    loop_thread.start()

    if state.stop_requested:
        _stop_tray_icon()
        restore_shutdown_handlers()
        loop_thread.join(timeout=5)
        return

    url = _webui_url(state.port)

    def _load_initial_webui() -> None:
        state.server_ready.wait(timeout=30)
        if shutdown_requested.is_set() or state.stop_requested:
            return
        loaded = _load_window_url_when_http_ready(
            window,
            state,
            url,
            timeout=20.0,
        )
        if loaded or shutdown_requested.is_set() or state.stop_requested:
            return
        try:
            window.load_html(  # type: ignore[union-attr]
                _launcher_loading_html(
                    state.port,
                    "WebUI 未在预期时间内响应，请查看控制台日志。",
                )
            )
        except Exception:
            pass

    # ── closing 事件处理 ─────────────────────────────────
    def _on_closing() -> bool:
        if state._closing_handled:
            return True
        pref = state.close_pref
        if pref == "tray":
            _do_hide()
            return False
        if pref == "quit":
            _do_quit()
            return True
        _show_close_dialog()
        return False

    # ── JS 桥接 API ──────────────────────────────────────
    class _API:
        def minimize(self):
            win = state.webview_window
            if win:
                try:
                    win.minimize()
                except Exception:
                    pass

        def uses_custom_window_controls(self) -> bool:
            return not USE_NATIVE_WINDOW_FRAME

        def toggle_maximize(self):
            win = state.webview_window
            if not win:
                return state.window_maximized
            try:
                if state.window_maximized:
                    win.restore()
                    _sync_window_maximized(False)
                else:
                    win.maximize()
                    _sync_window_maximized(True)
            except Exception:
                pass
            return state.window_maximized

        def move_to(self, x, y):
            win = state.webview_window
            if win:
                try:
                    win.move(int(x), int(y))
                except Exception:
                    pass

        def show_core_switching(self, action: str = "") -> bool:
            action = str(action)
            if action == "start":
                message = "正在启动核心，WebUI 将在服务就绪后自动恢复..."
            elif action == "stop":
                message = "正在停止核心，WebUI 将在服务就绪后自动恢复..."
            else:
                message = "正在切换核心状态，WebUI 将在服务就绪后自动恢复..."

            def _show_transition() -> None:
                win = state.webview_window
                if win is None or shutdown_requested.is_set() or state.stop_requested:
                    return
                try:
                    win.load_html(_launcher_loading_html(state.port, message))
                except Exception:
                    pass

            _run_later(0.05, _show_transition)
            return True

        def request_close(self):
            """JS 关闭按钮调用——若已有偏好直接处理，否则触发页面内确认浮层。"""
            if state._closing_handled:
                return
            pref = state.close_pref
            if pref == "tray":
                _do_hide()
                return
            if pref == "quit":
                _run_later(0.15, lambda: _do_quit(destroy_window=True))
                return
            _show_close_dialog()

        def confirm_close(self, action: str, remember: bool):
            """JS 确认浮层回调，action='tray'|'quit'，remember=True/False。"""
            if state._closing_handled:
                return
            action = str(action)
            if remember and action in ("tray", "quit"):
                state.close_pref = action
                _save_close_pref(action)
            if action == "tray":
                _do_hide()
                return
            if action == "quit":
                _run_later(0.15, lambda: _do_quit(destroy_window=True))
                return

    api = _API()

    # ── 创建窗口 ─────────────────────────────────────────
    window = webview.create_window(  # type: ignore[attr-defined]
        title="AIcarus for QQ",
        html=_launcher_loading_html(state.port),
        width=1280,
        height=800,
        min_size=(900, 600),
        background_color="#0d1117",
        frameless=not USE_NATIVE_WINDOW_FRAME,
        # Dragging is scoped in the template via .pywebview-drag-region when
        # custom frameless chrome is active.
        easy_drag=False,
        js_api=api,
    )
    state.webview_window = window

    try:
        window.events.closing += _on_closing  # type: ignore[union-attr]
        window.events.maximized += lambda: _sync_window_maximized(True)  # type: ignore[union-attr]
        window.events.restored += lambda: _sync_window_maximized(False)  # type: ignore[union-attr]
    except AttributeError:
        pass

    try:
        webview.start(  # type: ignore[attr-defined]
            func=_load_initial_webui,
            debug=False,
            icon=_webview_icon_path(),
        )
    except KeyboardInterrupt:
        _do_quit(
            destroy_window=True,
            reason="收到 Ctrl+C",
            graceful_timeout=3.0,
        )
    finally:
        restore_shutdown_handlers()
        if not state.stop_requested:
            _quit_from_tray()
        else:
            _stop_tray_icon()
        loop_thread.join(timeout=5)


# ══════════════════════════════════════════════════════════
#  无 GUI 降级
# ══════════════════════════════════════════════════════════

def _run_headless(
    state: _LauncherState,
    *,
    reason: str = "以无头模式运行",
    show_gui_error: bool = False,
) -> None:
    """无桌面窗口：直接在主线程跑进程管理循环。"""
    detail = ""
    if show_gui_error and GUI_IMPORT_ERROR is not None:
        detail = f"\n  GUI 依赖导入失败: {GUI_IMPORT_ERROR}"
    mode_label = "WebUI-only" if state.webui_only else "完整核心"
    print(
        f"[launcher] {reason}（初始模式={mode_label}）。\n"
        f"  服务启动后请访问 http://127.0.0.1:{state.port}/{detail}",
        flush=True,
    )
    _process_loop(state)


# ══════════════════════════════════════════════════════════
#  单实例保护
# ══════════════════════════════════════════════════════════

class _LauncherInstanceLock:
    """Process-wide launcher lock.

    Windows users commonly start the app from start.bat, and double-starting
    otherwise creates multiple launchers racing over the same configured port.
    A named mutex keeps the protection outside Python process ancestry.
    """

    def __init__(self, *, acquired: bool, handle: Any = None, kernel32: Any = None) -> None:
        self.acquired = acquired
        self._handle = handle
        self._kernel32 = kernel32

    def release(self) -> None:
        if not self._handle or not self._kernel32:
            return
        try:
            self._kernel32.ReleaseMutex(self._handle)
        finally:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None
            self._kernel32 = None

    def __enter__(self) -> "_LauncherInstanceLock":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def _launcher_instance_mutex_name() -> str:
    digest = hashlib.sha1(str(BASE_DIR).lower().encode("utf-8")).hexdigest()[:16]
    return f"Local\\AIcarusForQQLauncher-{digest}"


def _acquire_launcher_instance_lock() -> _LauncherInstanceLock:
    if sys.platform != "win32":
        return _LauncherInstanceLock(acquired=True)

    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel32.ReleaseMutex.restype = ctypes.c_bool
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_bool

    handle = kernel32.CreateMutexW(None, True, _launcher_instance_mutex_name())
    if not handle:
        raise OSError(ctypes.get_last_error(), "CreateMutexW failed")

    already_exists = ctypes.get_last_error() == 183  # ERROR_ALREADY_EXISTS
    if already_exists:
        kernel32.CloseHandle(handle)
        return _LauncherInstanceLock(acquired=False)
    return _LauncherInstanceLock(acquired=True, handle=handle, kernel32=kernel32)


# ══════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIcarusForQQ desktop/headless launcher")
    surface = parser.add_mutually_exclusive_group()
    surface.add_argument(
        "--gui",
        action="store_true",
        help="force desktop GUI mode; fail if GUI dependencies are unavailable",
    )
    surface.add_argument(
        "--headless",
        action="store_true",
        help="run without a desktop window even when GUI dependencies are installed",
    )
    startup = parser.add_mutually_exclusive_group()
    startup.add_argument(
        "--webui-only",
        action="store_true",
        help="start with WebUI only; the core can be started from the WebUI",
    )
    startup.add_argument(
        "--full",
        action="store_true",
        help="start the full core immediately",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    port = _get_server_port()
    start_webui_only = not args.full
    state = _LauncherState(port=port, webui_only=start_webui_only)

    print(
        "╔══════════════════════════════════════╗\n"
        "║   AIcarus for QQ  —  Launcher        ║\n"
        "╚══════════════════════════════════════╝",
        flush=True,
    )

    try:
        instance_lock = _acquire_launcher_instance_lock()
    except OSError as exc:
        print(f"[launcher] 警告：无法创建单实例锁，将继续启动: {exc}", flush=True)
        instance_lock = _LauncherInstanceLock(acquired=True)

    with instance_lock:
        if not instance_lock.acquired:
            print(
                "[launcher] 已有 AIcarus Launcher 实例在运行，本次启动已退出。\n"
                f"  如需访问当前 WebUI，请打开 http://127.0.0.1:{port}/",
                flush=True,
            )
            return

        if args.gui:
            if not HAS_GUI:
                print(
                    "[launcher] 无法启动 GUI：缺少 pywebview / pystray / Pillow。\n"
                    f"  导入错误: {GUI_IMPORT_ERROR}",
                    flush=True,
                )
                raise SystemExit(1)
            _run_with_gui(state)
        elif args.headless:
            _run_headless(state, reason="按启动参数选择无头模式")
        elif HAS_GUI:
            _run_with_gui(state)
        else:
            if not args.webui_only and not args.full:
                # Preserve the historical direct fallback: without GUI dependencies,
                # python launcher.py behaves like direct run.py plus launcher switching.
                state.webui_only = False
            _run_headless(
                state,
                reason="未检测到 GUI 库，自动降级到无头模式",
                show_gui_error=True,
            )


if __name__ == "__main__":
    main()
