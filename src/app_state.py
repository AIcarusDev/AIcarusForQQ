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

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo
    from napcat.client import NapcatClient
    from rate_limiter import MinuteRateLimiter
    from vision_bridge import VisionBridge

# 以下变量由 main.py 初始化阶段赋值，其他模块只读 / 按需写回。

config: dict = {}
persona: str = ""
chat_example: str = ""

MODEL: str = ""
MODEL_NAME: str = ""
GEN: dict = {}
TIMEZONE: ZoneInfo = None          # type: ignore[assignment]
MAX_CALLS_PER_MINUTE: int = 15
MAX_CONTEXT: int = 20
BOT_NAME: str = "小懒猫"

adapter: Any = None      # GeminiAdapter | OpenAICompatAdapter
vision_bridge: VisionBridge = None     # type: ignore[assignment]
rate_limiter: MinuteRateLimiter = None  # type: ignore[assignment]

napcat_cfg: dict = {}
napcat_client: NapcatClient | None = None
