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
    from napcat.client import NapcatClient
    from llm.core.rate_limiter import MinuteRateLimiter
    from llm.media.vision_bridge import VisionBridge
    from consciousness import ConsciousnessFlow

# 以下变量由 main.py 初始化阶段赋值，其他模块只读 / 按需写回。

config: dict = {}
persona: str = ""

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

napcat_cfg: dict = {}
napcat_client: NapcatClient | None = None

# ── 掉线告警管理器（SMTP）───────────────────────────
alert_manager: Any = None  # alerting.AlertManager

is_adapter: Any = None   # 中断哨兵（IS）专用适配器，None 时回退到主适配器
is_cfg: dict = {}

# ── 主事件循环引用（供 sync→async 的工具调用使用）────────────
main_loop: asyncio.AbstractEventLoop | None = None

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
