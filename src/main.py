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

"""main.py — 极简主入口

职责仅限于：
  1. 初始化日志、加载配置、填充 app_state
  2. 创建 Quart app，注册蓝图 & 生命周期钩子
  3. 初始化 QQ adapter 客户端（如启用）
  4. 启动服务
"""

import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv
from quart import Quart
from zoneinfo import ZoneInfo

import app_state
from alerting import AlertManager
from email_controller import EmailController
from config_loader import load_config
from web.debug_server import debug_bp, init_debug, broadcast_platform_status
from lifecycle import startup, shutdown
from log_config import setup_logging
from platforms import PlatformRegistry
from platforms.core import CoreRuntime
from platforms.qq import QQRuntime
from platforms.qq.handler import register_qq_platform_handlers
from platforms.qq.supervisor import QQAdapterSupervisor
from tts import TTSServer
from llm.core.provider import (
    create_adapter,
    build_tool_execution_guard_adapter_cfg,
    build_slow_thinking_adapter_cfg,
    build_event_extraction_adapter_cfg,
    build_memory_processing_adapter_cfg,
    build_compression_adapter_cfg,
)
from llm.core.profiles import normalize_profile_config_inplace
from consciousness import ConsciousnessFlow
from llm.core.rate_limiter import MinuteRateLimiter
from web.routes_dashboard import dashboard_bp
from web.routes_agent import agent_bp
from web.routes_maintenance import maintenance_bp
from web.routes_memory import memory_bp
from web.routes_settings import settings_bp
from web.routes_workspace import workspace_bp
from web.routes_tool_stats import tool_stats_bp
from web.routes_token_stats import token_stats_bp
from web.routes_core import core_bp
from web.routes_runtime import runtime_bp
from web.routes_updates import updates_bp
from web.auth import auth_bp, install_auth
from llm.session import init_session_globals
from llm.media.vision_bridge import VisionBridge
from workspace import WorkspaceService, WslWorkspaceBackend

# ── 启动模式标志 ────────────────────────────────────────────
# AICQ_WEBUI_ONLY=1  : 仅启动 Web UI，跳过 ConsciousnessFlow / QQ Adapter 等核心组件
# AICQ_LAUNCHER_MODE=1 : 由 launcher.py 管理，启用 /api/core/start|stop 控制接口
_WEBUI_ONLY = os.environ.get("AICQ_WEBUI_ONLY") == "1"
_LAUNCHER_MODE = os.environ.get("AICQ_LAUNCHER_MODE") == "1"

# ── 环境变量 & 日志 ───────────────────────────────────────
load_dotenv()
setup_logging()

# ── 加载配置 & 填充 app_state ─────────────────────────────
config, prompt_docs = load_config()
normalize_profile_config_inplace(config)
persona = prompt_docs["persona"]

app_state.config = config
app_state.persona = persona
app_state.MODEL = config.get("model", "Pro/zai-org/GLM-5")
app_state.MODEL_NAME = config.get("model_name", app_state.MODEL)
app_state.GEN = config.get("generation", {})
app_state.TIMEZONE = ZoneInfo((config.get("timezone") or "").strip() or "Asia/Shanghai")
app_state.MAX_CALLS_PER_MINUTE = config.get("max_calls_per_minute", 15)
app_state.MAX_CONTEXT = int(config.get("max_context", 10))
app_state.SELF_NAME = config.get("self_name", "小懒猫")
app_state.webui_only = _WEBUI_ONLY
app_state.launcher_mode = _LAUNCHER_MODE
app_state.workspace_service = None

# SiliconFlow 图片兼容补丁开关（默认关闭，仅对绕过其 PIL bug 时启用）
from llm.media.outbound_image import set_siliconflow_compat as _set_sf_compat
_set_sf_compat(bool(app_state.GEN.get("siliconflow_image_compat", False)))

app_state.rate_limiter = MinuteRateLimiter(app_state.MAX_CALLS_PER_MINUTE)
try:
    app_state.adapter = create_adapter(config)
except (ValueError, Exception) as _adapter_err:
    import logging as _log
    _log.getLogger("AICQ").warning(
        "主模型适配器初始化失败（配置未完成？），WebUI 仍可访问以修改配置: %s",
        _adapter_err,
    )
    app_state.adapter = None
if not _WEBUI_ONLY:
    app_state.consciousness_flow = ConsciousnessFlow()
    # computer namespace 保持完全惰性：这里只装配对象，不触碰 WSL。
    app_state.workspace_service = WorkspaceService(WslWorkspaceBackend())
    try:
        app_state.vision_bridge = VisionBridge(config)
    except (ValueError, Exception):
        app_state.vision_bridge = None

    # ── 外界可感知工具执行前守门模型初始化 ─────────────────────────────
    app_state.tool_execution_guard_cfg = config.get("tool_execution_guard", {})
    _guard_cfg = app_state.tool_execution_guard_cfg
    if _guard_cfg.get("enabled", False) and _guard_cfg.get("provider") and _guard_cfg.get("model"):
        try:
            app_state.tool_execution_guard_adapter = create_adapter(
                build_tool_execution_guard_adapter_cfg(config, _guard_cfg)
            )
        except (ValueError, Exception):
            app_state.tool_execution_guard_adapter = None

    # ── 慢思考（think_deeply）子模型初始化 ──────────────────────────
    app_state.slow_thinking_cfg = config.get("slow_thinking", {})
    _st_cfg = app_state.slow_thinking_cfg
    if _st_cfg.get("enabled", True) and _st_cfg.get("provider") and _st_cfg.get("model"):
        app_state.slow_thinking_adapter = create_adapter(
            build_slow_thinking_adapter_cfg(config, _st_cfg)
        )

    # ── 记忆事件提取子模型初始化 ────────────────────────────────────
    app_state.event_extraction_cfg = config.get("memory", {}).get("auto_archive", {})
    _event_extraction_cfg = app_state.event_extraction_cfg
    if (
        _event_extraction_cfg.get("enabled", True)
        and _event_extraction_cfg.get("provider")
        and _event_extraction_cfg.get("model")
    ):
        app_state.event_extraction_adapter = create_adapter(
            build_event_extraction_adapter_cfg(config, _event_extraction_cfg)
        )

    # ── 事件结构化与故事线合成共用子模型初始化 ───────────────────────
    app_state.memory_processing_cfg = config.get("memory", {}).get("processing", {})
    _memory_processing_cfg = app_state.memory_processing_cfg
    if (
        _memory_processing_cfg.get("enabled", False)
        and _memory_processing_cfg.get("provider")
        and _memory_processing_cfg.get("model")
    ):
        try:
            app_state.memory_processing_adapter = create_adapter(
                build_memory_processing_adapter_cfg(config, _memory_processing_cfg)
            )
        except (ValueError, Exception):
            app_state.memory_processing_adapter = None

    # ── 上下文压缩子模型初始化 ─────────────────────────────────────
    app_state.cognition_compression_cfg = config.get("cognition_compression", {})
    _compression_cfg = app_state.cognition_compression_cfg
    if _compression_cfg.get("provider") and _compression_cfg.get("model"):
        try:
            app_state.cognition_compression_adapter = create_adapter(
                build_compression_adapter_cfg(config, _compression_cfg)
            )
        except (ValueError, Exception):
            app_state.cognition_compression_adapter = None

# ── 初始化 Session 子模块 ─────────────────────────────────
init_session_globals(
    max_context=app_state.MAX_CONTEXT,
    timezone=app_state.TIMEZONE,
    persona=persona,
    self_name=app_state.SELF_NAME,
    model_name=app_state.MODEL_NAME,
    guardian_name=config.get("guardian", {}).get("name", ""),
    guardian_id=config.get("guardian", {}).get("id", ""),
)


def _register_core_platform() -> None:
    if app_state.platform_registry is None:
        app_state.platform_registry = PlatformRegistry()
    if app_state.platform_registry.get("core") is None:
        app_state.platform_registry.register(
            CoreRuntime(config.get("platforms", {}).get("core", {}) or {})
        )


app_state.platform_registry = PlatformRegistry()
_register_core_platform()

if not _WEBUI_ONLY:
    # ── 平台运行时（可选）──────────────────────────────────
    _qq_runtime = QQRuntime(config.get("platforms", {}).get("qq", {}) or {})
    _qq_client = _qq_runtime.ensure_client(bot_name=app_state.SELF_NAME)
    app_state.platform_registry.register(_qq_runtime)
    # ── TTS 插件服务端（可选）──────────────────────────
    app_state.tts_cfg = config.get("tts", {}) or {}
    _tts_enabled = app_state.tts_cfg.get("enabled", False)

    def _buffer_tts_audio(task_id: str, pcm: bytes) -> None:
        app_state.tts_audio_buffers.setdefault(task_id, bytearray()).extend(pcm)

    app_state.tts_server = TTSServer(
        host=app_state.tts_cfg.get("host", "127.0.0.1"),
        port=int(app_state.tts_cfg.get("port", 8765)),
        secret_token=app_state.tts_cfg.get("secret_token", ""),
        on_audio_chunk=_buffer_tts_audio,
        max_concurrent_tasks_per_plugin=int(app_state.tts_cfg.get("max_concurrent_tasks_per_plugin", 8)),
    ) if _tts_enabled else None
    # ── 掉线告警（可选）────────────────────────────────
    _alerting_cfg = config.get("alerting", {}) or {}
    app_state.alert_manager = AlertManager(_alerting_cfg)
    if _qq_client and app_state.alert_manager.enabled:
        _qq_client.set_alert_manager(
            app_state.alert_manager,
            heartbeat_timeout=float(_alerting_cfg.get("heartbeat_timeout", 120)),
        )
    # ── QQ adapter 自动重启 监管器（可选）──────────────────
    _qq_runtime.supervisor = QQAdapterSupervisor(
        _qq_runtime.config.get("supervisor", {}) or {},
        client=_qq_client,
        alert=app_state.alert_manager,
    )
    if _qq_client and _qq_runtime.supervisor.is_configured():
        _qq_client.set_supervisor(_qq_runtime.supervisor)
    # ── 邮件远程指令（Phase 3，可选）────────────────────
    app_state.email_controller = EmailController(
        _alerting_cfg,
        supervisor=_qq_runtime.supervisor,
        alert=app_state.alert_manager,
    )
    register_qq_platform_handlers(_qq_runtime)

_qq_runtime_for_debug = app_state.platform_registry.get("qq") if app_state.platform_registry else None
_qq_client_for_debug = getattr(_qq_runtime_for_debug, "client", None)
init_debug(app_state.TIMEZONE, _qq_client_for_debug)
if _qq_client_for_debug:
    _qq_client_for_debug.set_status_change_handler(broadcast_platform_status)

# ── Quart App ─────────────────────────────────────────────
app = Quart(__name__)
app.json.sort_keys = False  # type: ignore[attr-defined]
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
install_auth(app)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _static_asset_exists(filename: str) -> bool:
    safe_name = filename.replace("\\", "/").lstrip("/")
    if "/" in safe_name or safe_name in {"", ".", ".."}:
        return False
    return (_STATIC_DIR / safe_name).is_file()


app.jinja_env.globals["static_asset_exists"] = _static_asset_exists

app.register_blueprint(debug_bp)
app.register_blueprint(dashboard_bp)
app.register_blueprint(agent_bp)
app.register_blueprint(settings_bp)
app.register_blueprint(workspace_bp)
app.register_blueprint(memory_bp)
app.register_blueprint(maintenance_bp)
app.register_blueprint(tool_stats_bp)
app.register_blueprint(token_stats_bp)
app.register_blueprint(core_bp)
app.register_blueprint(runtime_bp)
app.register_blueprint(updates_bp)
app.register_blueprint(auth_bp)

if _WEBUI_ONLY:
    @app.before_serving
    async def _startup_webui_only():
        import asyncio as _asyncio
        import logging as _log
        app_state.main_loop = _asyncio.get_event_loop()
        from database import init_db as _init_db
        await _init_db()
        _log.getLogger("AICQ").info(
            "[startup] WebUI-only 模式已就绪（由 launcher.py 管理）"
        )
    app.after_serving(shutdown)
else:
    app.before_serving(startup)
    app.after_serving(shutdown)

# ══════════════════════════════════════════════════════════
#  启动入口
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Windows 下修复 Ctrl+C 无法终止的问题
    if sys.platform == "win32":
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    srv = config.get("server", {})
    app.run(
        debug=srv.get("debug", True),
        host=srv.get("host", "127.0.0.1"),
        port=srv.get("port", 5000),
        use_reloader=False,
    )


