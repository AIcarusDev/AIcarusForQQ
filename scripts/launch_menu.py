# Copyright (C) 2026  AIcarusDev
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Interactive startup menu used by start.bat.

start.bat owns environment setup.  This script owns product-level startup
choices, so the batch file can stay small and the GUI/headless contracts remain
testable from Python.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
LAUNCHER_PY = BASE_DIR / "launcher.py"
CORE_SUPERVISOR_PY = BASE_DIR / "scripts" / "core_supervisor.py"
PREF_PATH = BASE_DIR / "data" / "launcher_start_mode.txt"


@dataclass(frozen=True)
class StartupMode:
    key: str
    name: str
    summary: str
    command: tuple[str, ...]


def _launcher_cmd(*args: str) -> tuple[str, ...]:
    return (sys.executable, str(LAUNCHER_PY), *args)


MODES: dict[str, StartupMode] = {
    "gui": StartupMode(
        key="1",
        name="桌面 GUI",
        summary="打开桌面窗口，先进入 WebUI 管理界面，可在窗口内启动核心。",
        command=_launcher_cmd("--gui", "--webui-only"),
    ),
    "webui": StartupMode(
        key="2",
        name="无头 WebUI",
        summary="不打开桌面窗口，只在命令行托管 WebUI，可从浏览器启动/停止核心。",
        command=_launcher_cmd("--headless", "--webui-only"),
    ),
    "console": StartupMode(
        key="3",
        name="纯命令行完整核心",
        summary="沿用旧 start.bat 行为，直接启动完整核心与 WebUI 服务。",
        command=(sys.executable, str(CORE_SUPERVISOR_PY)),
    ),
}

KEY_TO_MODE = {mode.key: mode_name for mode_name, mode in MODES.items()}


def _enable_virtual_terminal() -> bool:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


USE_COLOR = _enable_virtual_terminal()


def _c(code: str) -> str:
    return code if USE_COLOR else ""


C_RESET = _c("\033[0m")
C_DIM = _c("\033[2m")
C_BOLD = _c("\033[1m")
C_ACCENT = _c("\033[38;5;141m")
C_GREEN = _c("\033[38;5;114m")
C_YELLOW = _c("\033[38;5;221m")
C_RED = _c("\033[38;5;203m")
C_CYAN = _c("\033[38;5;117m")


def _clear_screen() -> None:
    if USE_COLOR:
        print("\033[2J\033[H", end="")
    elif os.name == "nt":
        os.system("cls")


def _gui_dependency_status() -> tuple[bool, list[str]]:
    required = (
        ("webview", "pywebview"),
        ("pystray", "pystray"),
        ("PIL", "Pillow"),
    )
    missing = [label for module, label in required if importlib.util.find_spec(module) is None]
    return not missing, missing


def _read_pref() -> str | None:
    try:
        value = PREF_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value if value in MODES else None


def _write_pref(mode_name: str) -> None:
    try:
        PREF_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREF_PATH.write_text(mode_name, encoding="utf-8")
    except OSError:
        pass


def _reset_pref() -> None:
    try:
        PREF_PATH.unlink()
    except FileNotFoundError:
        pass


def _default_mode(gui_available: bool) -> str:
    preferred = _read_pref()
    if preferred == "gui" and not gui_available:
        return "webui"
    if preferred in MODES:
        return preferred
    return "gui" if gui_available else "webui"


def _format_command(command: tuple[str, ...]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return " ".join(command)


def _print_header(gui_available: bool, missing_gui_deps: list[str]) -> None:
    _clear_screen()
    print(f"{C_ACCENT}╔════════════════════════════════════════════════════╗{C_RESET}")
    print(f"{C_ACCENT}║{C_RESET}  {C_BOLD}AIcarus for QQ 启动中心{C_RESET}                         {C_ACCENT}║{C_RESET}")
    print(f"{C_ACCENT}╚════════════════════════════════════════════════════╝{C_RESET}")
    print()
    if gui_available:
        print(f"{C_GREEN}GUI 依赖已就绪{C_RESET}：可以启动桌面窗口。")
    else:
        missing = ", ".join(missing_gui_deps)
        print(f"{C_YELLOW}GUI 依赖不完整{C_RESET}：缺少 {missing}；仍可使用无头模式。")
    print(f"{C_DIM}Python: {sys.executable}{C_RESET}")
    print()


def _print_menu(default_name: str) -> None:
    for mode_name in ("gui", "webui", "console"):
        mode = MODES[mode_name]
        selected = mode_name == default_name
        marker = ">" if selected else " "
        suffix = "  默认" if selected else ""
        print(f"{C_CYAN}{marker} [{mode.key}]{C_RESET} {C_BOLD}{mode.name}{C_RESET}{suffix}")
        print(f"    {C_DIM}{mode.summary}{C_RESET}")
    print()
    print(f"  [0] 退出")
    print()


def _choose_mode() -> str | None:
    gui_available, missing_gui_deps = _gui_dependency_status()
    default_name = _default_mode(gui_available)
    while True:
        _print_header(gui_available, missing_gui_deps)
        _print_menu(default_name)
        prompt = f"请选择启动方式（Enter = {MODES[default_name].name}）："
        choice = input(prompt).strip()
        if not choice:
            return default_name
        if choice == "0":
            return None
        mode_name = KEY_TO_MODE.get(choice) or choice.lower()
        if mode_name in MODES:
            return mode_name
        print(f"{C_RED}无效选择：{choice}{C_RESET}")
        input("按 Enter 重新选择...")


def _run_mode(mode_name: str, *, dry_run: bool, save_choice: bool) -> int:
    mode = MODES[mode_name]
    if save_choice:
        _write_pref(mode_name)
    print()
    print(f"{C_GREEN}启动方式{C_RESET}: {mode.name}")
    print(f"{C_DIM}{_format_command(mode.command)}{C_RESET}")
    print()
    if dry_run:
        return 0
    try:
        completed = subprocess.run(mode.command, cwd=str(BASE_DIR))
    except KeyboardInterrupt:
        print()
        print("启动菜单已收到 Ctrl+C，正在退出。")
        return 0
    return int(completed.returncode or 0)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIcarusForQQ startup menu")
    parser.add_argument(
        "--mode",
        choices=tuple(MODES),
        help="launch a mode directly without showing the interactive menu",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the command that would be launched, then exit",
    )
    parser.add_argument(
        "--no-save-choice",
        action="store_true",
        help="do not remember this selection as the next default",
    )
    parser.add_argument(
        "--reset-choice",
        action="store_true",
        help="forget the saved startup-mode default before continuing",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.reset_choice or args.reset:
        _reset_pref()
    mode_name = args.mode or _choose_mode()
    if mode_name is None:
        print("已取消启动。")
        return 0
    return _run_mode(
        mode_name,
        dry_run=bool(args.dry_run),
        save_choice=not args.no_save_choice,
    )


if __name__ == "__main__":
    raise SystemExit(main())
