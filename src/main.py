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
  3. 初始化 NapCat 客户端（如启用）
  4. 启动服务
"""

import signal
import sys

from dotenv import load_dotenv
from quart import Quart
from zoneinfo import ZoneInfo

import app_state
from config_loader import load_config
from debug_server import debug_bp, init_debug
from lifecycle import startup, shutdown
from log_config import setup_logging
from napcat import NapcatClient
from napcat_handler import register_napcat_handlers
from provider import create_adapter
from rate_limiter import MinuteRateLimiter
from routes_chat import chat_bp
from routes_settings import settings_bp
from session import init_session_globals, create_session, sessions
from vision_bridge import VisionBridge

# ── 环境变量 & 日志 ───────────────────────────────────────
load_dotenv()
setup_logging()

# ── 加载配置 & 填充 app_state ─────────────────────────────
config, persona, chat_example = load_config()

app_state.config = config
app_state.persona = persona
app_state.chat_example = chat_example
app_state.MODEL = config.get("model", "gemini-2.0-flash")
app_state.MODEL_NAME = config.get("model_name", app_state.MODEL)
app_state.GEN = config.get("generation", {})
app_state.TIMEZONE = ZoneInfo((config.get("timezone") or "").strip() or "Asia/Shanghai")
app_state.MAX_CALLS_PER_MINUTE = config.get("max_calls_per_minute", 15)
app_state.MAX_CONTEXT = 20
app_state.BOT_NAME = config.get("bot_name", "小懒猫")

app_state.rate_limiter = MinuteRateLimiter(app_state.MAX_CALLS_PER_MINUTE)
app_state.adapter = create_adapter(config)
app_state.vision_bridge = VisionBridge(config.get("vision_bridge", {}))

# ── Watcher 模型（窥屏意识）初始化 ────────────────────────────────
app_state.watcher_cfg = config.get("watcher", {})
if app_state.watcher_cfg.get("enabled", False):
    _watcher_model_cfg = dict(config)
    # provider / base_url：watcher 有自己的则覆盖，否则沿用主模型
    if "provider" in app_state.watcher_cfg:
        _watcher_model_cfg["provider"] = app_state.watcher_cfg["provider"]
    if "base_url" in app_state.watcher_cfg:
        _watcher_model_cfg["base_url"] = app_state.watcher_cfg["base_url"]
    _watcher_model_cfg["model"] = app_state.watcher_cfg.get("model", config.get("model"))
    _watcher_model_cfg["model_name"] = app_state.watcher_cfg.get("model_name", _watcher_model_cfg["model"])
    # generation：watcher 有自己的子块则覆盖，否则沿用主模型
    if "generation" in app_state.watcher_cfg:
        _watcher_model_cfg["generation"] = app_state.watcher_cfg["generation"]
    _watcher_model_cfg.pop("thinking", None)  # watcher 不需要 thinking
    app_state.watcher_adapter = create_adapter(_watcher_model_cfg)

# ── 初始化 Session 子模块 ─────────────────────────────────
init_session_globals(
    max_context=app_state.MAX_CONTEXT,
    timezone=app_state.TIMEZONE,
    persona=persona,
    chat_example=chat_example,
    model_name=app_state.MODEL_NAME,
)
_web_session = create_session()
_web_session.set_conversation_meta("private", "web_user", "网页用户")
sessions["web"] = _web_session

# ── NapCat 客户端（可选）──────────────────────────────────
app_state.napcat_cfg = config.get("napcat", {})
_napcat_enabled = app_state.napcat_cfg.get("enabled", False)
app_state.napcat_client = NapcatClient(bot_name=app_state.BOT_NAME) if _napcat_enabled else None
init_debug(app_state.TIMEZONE, app_state.napcat_client)
register_napcat_handlers()

# ── Quart App ─────────────────────────────────────────────
app = Quart(__name__)
app.json.sort_keys = False  # type: ignore[attr-defined]

app.register_blueprint(debug_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(settings_bp)

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
    app.run(debug=srv.get("debug", True), port=srv.get("port", 5000))
