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
MAX_CONTEXT: int = 20
BOT_NAME: str = "小懒猫"

adapter: Any = None      # GeminiAdapter | OpenAICompatAdapter
consciousness_flow: "ConsciousnessFlow" = None  # type: ignore[assignment]
vision_bridge: VisionBridge = None     # type: ignore[assignment]
rate_limiter: MinuteRateLimiter = None  # type: ignore[assignment]

napcat_cfg: dict = {}
napcat_client: NapcatClient | None = None

watcher_adapter: Any = None  # 窥屏意识专用适配器（轻量模型）
watcher_cfg: dict = {}

is_adapter: Any = None   # 中断哨兵（IS）专用适配器，None 时回退到主适配器
is_cfg: dict = {}

# ── 主事件循环引用（供 sync→async 的工具调用使用）────────────
main_loop: asyncio.AbstractEventLoop | None = None

# ── 全局意识锁 ──────────────────────────────────────────
# 同一时刻只有一个协程可持有此锁（聊天/窥屏/shift 共用），保证机器人是单一意识流。
consciousness_lock: asyncio.Lock = asyncio.Lock()
# 当前意识焦点所在的会话 key（如 "group_123"），无焦点时为 None。
current_focus: str | None = None
# watcher 是否处于休眠状态（hibernate 决策后到被唤醒 / 自然醒来前为 True）
watcher_hibernating: bool = False
