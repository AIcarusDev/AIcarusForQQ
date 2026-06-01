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

"""app_state.py — 全局共享运行时状态

所有可变的运行时状态集中在此模块。
其他模块通过 ``import app_state`` 来访问和修改共享状态，
避免到处传 global 或循环依赖。
"""

from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo
    from qq_adapter.client import QQAdapterClient
    from llm.core.rate_limiter import MinuteRateLimiter
    from llm.media.vision_bridge import VisionBridge
    from consciousness import ConsciousnessFlow
    from email_controller import EmailController
    from tts.server import TTSServer

# 以下变量由 main.py 初始化阶段赋值，其他模块只读 / 按需写回。

config: dict = {}
persona: str = ""
style_prompt: str = ""
social_tips_private: str = ""
social_tips_group: str = ""
social_tips_temp: str = ""

MODEL: str = ""
MODEL_NAME: str = ""
GEN: dict = {}
TIMEZONE: ZoneInfo = None          # type: ignore[assignment]
MAX_CALLS_PER_MINUTE: int = 15
MAX_CONTEXT: int = 10
BOT_NAME: str = ""

adapter: Any = None      # OpenAICompatAdapter
consciousness_flow: "ConsciousnessFlow" = None  # type: ignore[assignment]
vision_bridge: VisionBridge = None     # type: ignore[assignment]
rate_limiter: MinuteRateLimiter = None  # type: ignore[assignment]

qq_adapter_cfg: dict = {}
qq_adapter_client: QQAdapterClient | None = None

tts_cfg: dict = {}
tts_server: "TTSServer | None" = None
tts_audio_buffers: dict[str, bytearray] = {}

# ── 掉线告警管理器（SMTP）───────────────────────────
alert_manager: Any = None  # alerting.AlertManager

# ── 邮件远程指令控制器（Phase 3）───────────────────
email_controller: "EmailController" = None  # type: ignore[assignment]

# ── QQ adapter 监管器（自动重启 + 二维码邮件）───────────
qq_adapter_supervisor: Any = None  # qq_adapter_supervisor.QQAdapterSupervisor

is_adapter: Any = None   # 中断哨兵（IS）专用适配器
is_cfg: dict = {}

slow_thinking_adapter: Any = None  # 慢思考（think_deeply）专用适配器
slow_thinking_cfg: dict = {}

archiver_adapter: Any = None  # 记忆提取（archiver）专用适配器
archiver_cfg: dict = {}

cognition_compression_adapter: Any = None  # 上下文压缩专用适配器
cognition_compression_cfg: dict = {}

cognition_compression_task: asyncio.Task | None = None
cognition_compression_pending_jobs: list[Any] = []
cognition_compression_inflight_job: Any = None

# ── 主事件循环引用（供 sync→async 的工具调用使用）────────────
main_loop: asyncio.AbstractEventLoop | None = None

# ── Launcher / WebUI-only 模式标志 ───────────────────────────────
# 由 main.py 模块级初始化阶段根据环境变量写入，其他模块只读。
webui_only: bool = False       # 以 AICQ_WEBUI_ONLY=1 启动，核心组件未初始化
launcher_mode: bool = False    # 由 launcher.py 管理（AICQ_LAUNCHER_MODE=1）

# ── Core 自重启协议 ─────────────────────────────────────────
# run.py 注入 Hypercorn 的 shutdown trigger；工具只设置重启请求，
# 主循环在当前 round 持久化后再触发该事件。
server_shutdown_event: asyncio.Event | None = None
core_restart_requested: bool = False
core_restart_exit_code: int | None = None
launcher_switch_requested: bool = False  # launcher.py 模式下请求切换 full/webui-only

# ── LLM 调用互斥锁 ─────────────────────────────────────────────
# 仅用于序列化对 ConsciousnessFlow 与 adapter 的写访问；不承载任何
# "机器人是否在忙" 的语义。Web chat 路径与常驻意识主循环共用此锁。
llm_lock: asyncio.Lock = asyncio.Lock()

# ── 当前意识焦点所在的会话 key（如 "group_123"）。 ────────────
# 主循环每一 round 都从此处取 session；shift 工具直接修改它。
# 启动时从 last_active_session 恢复；为 None 表示数据库为空、等待外部消息。
current_focus: str | None = None
# 上一次主循环 round 处理的会话 key。用于识别"焦点离开过"以重置视口。
last_active_session: str | None = None

# ── 常驻意识主循环 ────────────────────────────────────────────
# 由 lifecycle.startup() 启动；shutdown() 时 cancel。
consciousness_main_task: asyncio.Task | None = None
# 当 current_focus 为 None 时，主循环挂在此 event 上等待"第一条外部消息"。
first_input_event: asyncio.Event = asyncio.Event()
# 触发主循环停止的信号（shutdown 时 set）。
shutdown_event: asyncio.Event = asyncio.Event()

# ── 后台归档任务集合 ──────────────────────────────────────────
# fire-and-forget 的 archive task 在此登记，shutdown 时统一 cancel。
# 任务的实际工作快照已持久化到 pending_archive_jobs 表，下次启动会续跑，
# 因此 cancel 后无需 await 完成（避免 LLM 调用阻塞 Ctrl+C 退出）。
archive_tasks: set[asyncio.Task] = set()
