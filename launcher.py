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

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from runtime.core_restart import RESTART_EXIT_CODE as CORE_RESTART_EXIT_CODE
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
    C_BG        = "#0d1117"
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
    def __init__(self, port: int) -> None:
        self.port = port
        self.webui_only = True       # 初始模式：仅 Web UI
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.stop_requested = False  # 用户主动退出（托盘 Quit / 窗口关闭）
        # 供 GUI 线程等待第一次服务就绪
        self.server_ready = threading.Event()
        self.webview_window: "webview.Window | None" = None
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

    tray_icon: "pystray.Icon | None" = None

    # ── 共享操作原语 ─────────────────────────────────────

    def _do_quit() -> None:
        """真正退出：停止子进程、托盘、标记已处理。"""
        state._closing_handled = True
        state.stop_requested = True
        with state.lock:
            _stop_proc(state.proc)
        if tray_icon is not None:
            try:
                tray_icon.stop()
            except Exception:
                pass

    def _do_hide() -> None:
        """隐藏窗口到托盘。"""
        win = state.webview_window
        if win is not None:
            try:
                win.hide()
            except Exception:
                pass

    def _show_close_dialog() -> None:
        win = state.webview_window
        if win is not None:
            try:
                win.evaluate_js("window._wcShowCloseDialog && window._wcShowCloseDialog()")
            except Exception:
                # 如果 JS 没准备好，则直接退出以避免卡死
                _do_quit()

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
        _do_quit()
        win = state.webview_window
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass

    try:
        tray_icon = pystray.Icon(
            "AIcarus",
            icon=_make_tray_icon_image(),
            title="AIcarus for QQ",
            menu=pystray.Menu(
                pystray.MenuItem("显示窗口", _show_window, default=True),
                pystray.MenuItem("在浏览器中打开", _open_browser),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("退出 AIcarus Launcher", _quit_from_tray),
            ),
        )
        tray_icon.run_detached()
    except Exception as e:
        print(f"[launcher] 托盘图标创建失败（非致命）: {e}", flush=True)

    # ── 启动进程管理线程 ─────────────────────────────────
    loop_thread = threading.Thread(target=_process_loop, args=(state,), daemon=True)
    loop_thread.start()

    # 等待服务就绪后再显示窗口
    state.server_ready.wait(timeout=30)

    if state.stop_requested:
        if tray_icon is not None:
            try:
                tray_icon.stop()
            except Exception:
                pass
        loop_thread.join(timeout=5)
        return

    url = f"http://127.0.0.1:{state.port}/"

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

        def request_close(self):
            """JS 关闭按钮调用——若已有偏好直接处理，否则触发页面内确认浮层。"""
            if state._closing_handled:
                return
            pref = state.close_pref
            if pref == "tray":
                _do_hide()
                return
            if pref == "quit":
                win = state.webview_window
                if win is not None:
                    try:
                        win.evaluate_js("window.close()")
                    except Exception:
                        _do_quit()
                else:
                    _do_quit()
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
                state._closing_handled = True
                _do_quit()
                win = state.webview_window
                if win is not None:
                    try:
                        win.evaluate_js("window.close()")
                    except Exception:
                        try:
                            win.destroy()
                        except Exception:
                            pass
                else:
                    _do_quit()

    api = _API()

    # ── 创建窗口 ─────────────────────────────────────────
    window = webview.create_window(
        title="AIcarus for QQ",
        url=url,
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
        window.events.closing += _on_closing
        window.events.maximized += lambda: _sync_window_maximized(True)
        window.events.restored += lambda: _sync_window_maximized(False)
    except AttributeError:
        pass

    webview.start(
        func=None,
        debug=False,
    )

    if not state.stop_requested:
        _quit_from_tray()
    loop_thread.join(timeout=5)


# ══════════════════════════════════════════════════════════
#  无 GUI 降级
# ══════════════════════════════════════════════════════════

def _run_headless(state: _LauncherState) -> None:
    """无 GUI 库时：直接在主线程跑进程管理循环，行为与 run.py 相同。"""
    state.webui_only = False
    detail = f"\n  GUI 依赖导入失败: {GUI_IMPORT_ERROR}" if GUI_IMPORT_ERROR else ""
    print(
        "[launcher] 未检测到 GUI 库（pywebview / pystray），以无头模式运行。\n"
        f"  服务启动后请访问 http://127.0.0.1:{state.port}/{detail}",
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
