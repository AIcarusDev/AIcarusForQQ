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

"""routes_core.py — 核心进程生命周期控制 API

专为 launcher.py 模式设计：WebUI 进程通过特殊 exit code 向外层 launcher
传递"启动核心"或"停止核心"的请求，launcher 再以对应模式重启子进程。

Exit code 约定（与 launcher.py 保持同步）：
  76 (LAUNCHER_START_CORE_EXIT_CODE) — launcher 应以完整模式重启
  77 (LAUNCHER_STOP_CORE_EXIT_CODE)  — launcher 应以 webui-only 模式重启

Endpoints:
  GET  /api/core/status  → {webui_only, launcher_mode}
  POST /api/core/start   → 触发 webui-only 进程以 code 76 退出（仅 webui-only 模式有效）
  POST /api/core/stop    → 触发完整进程以 code 77 退出（仅完整模式有效）
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from quart import Blueprint, jsonify

import app_state

core_bp = Blueprint("core", __name__)
logger = logging.getLogger("AICQ.web.routes_core")

# 与 launcher.py 保持同步的 exit code
LAUNCHER_START_CORE_EXIT_CODE = 76
LAUNCHER_STOP_CORE_EXIT_CODE = 77


def _run_on_main_loop(callback: Callable[[], None]) -> None:
    loop = getattr(app_state, "main_loop", None)
    if loop and loop.is_running():
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            callback()
        else:
            loop.call_soon_threadsafe(callback)
        return
    callback()


def _wake_runtime_shutdown_waiters() -> None:
    """Unblock the core runtime so launcher mode switches can finish promptly."""

    def _wake() -> None:
        shutdown_event = getattr(app_state, "shutdown_event", None)
        if shutdown_event is not None:
            shutdown_event.set()
        first_input_event = getattr(app_state, "first_input_event", None)
        if first_input_event is not None:
            first_input_event.set()
        try:
            from llm.session import sessions
        except Exception:
            logger.debug("无法导入 sessions，跳过 runtime waiter 唤醒", exc_info=True)
            return
        for session in list(sessions.values()):
            sleep_event = getattr(session, "sleep_wake_event", None)
            if sleep_event is not None and not sleep_event.is_set():
                sleep_event.set()
            wait_event = getattr(session, "wait_event", None)
            if wait_event is not None and not wait_event.is_set():
                wait_event.set()

    _run_on_main_loop(_wake)


def _trigger_launcher_switch(exit_code: int) -> None:
    """设置 launcher 切换退出码并向 Hypercorn 发出优雅关机信号。"""
    app_state.launcher_switch_requested = True
    app_state.core_restart_exit_code = exit_code
    _wake_runtime_shutdown_waiters()
    event = getattr(app_state, "server_shutdown_event", None)
    if event is not None:
        loop = getattr(app_state, "main_loop", None)
        if loop and loop.is_running():
            loop.call_soon_threadsafe(event.set)
        else:
            event.set()


@core_bp.route("/api/core/status")
async def api_core_status():
    """返回当前进程的运行模式。

    webui_only: true  → 进程以 AICQ_WEBUI_ONLY=1 启动，核心组件未初始化
    launcher_mode: true → 进程由 launcher.py 管理（AICQ_LAUNCHER_MODE=1）
    """
    return jsonify({
        "webui_only": getattr(app_state, "webui_only", False),
        "launcher_mode": getattr(app_state, "launcher_mode", False),
    })


@core_bp.route("/api/core/start", methods=["POST"])
async def api_core_start():
    """请求 launcher 以完整模式重启（仅 webui-only 模式下有效）。"""
    if not getattr(app_state, "webui_only", False):
        return jsonify({"error": "核心已在运行中，无需重复启动"}), 400
    if not getattr(app_state, "launcher_mode", False):
        return jsonify({"error": "当前不在 launcher 管理模式下，请直接运行 run.py"}), 400
    _trigger_launcher_switch(LAUNCHER_START_CORE_EXIT_CODE)
    return jsonify({"ok": True, "message": "正在启动核心，页面将自动刷新..."})


@core_bp.route("/api/core/stop", methods=["POST"])
async def api_core_stop():
    """请求 launcher 以 webui-only 模式重启（仅完整模式下有效）。"""
    if getattr(app_state, "webui_only", False):
        return jsonify({"error": "核心未运行"}), 400
    if not getattr(app_state, "launcher_mode", False):
        return jsonify({"error": "当前不在 launcher 管理模式下，请直接停止进程"}), 400
    _trigger_launcher_switch(LAUNCHER_STOP_CORE_EXIT_CODE)
    return jsonify({"ok": True, "message": "正在停止核心，页面将自动刷新..."})
