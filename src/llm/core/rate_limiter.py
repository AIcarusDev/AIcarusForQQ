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

"""rate_limiter.py — 每分钟 LLM 调用限流器"""

import asyncio
import logging
import time

logger = logging.getLogger("AICQ.llm.rate_limit")


class MinuteRateLimiter:
    """全局每分钟 LLM 调用次数限制器。

    达到上限后挂起，至当前分钟窗口结束后再继续。所有会话共享同一个限制器。
    """

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self._calls: int = 0
        self._window_start: float = time.monotonic()
        self._lock: asyncio.Lock = asyncio.Lock()

    async def acquire(self) -> None:
        """挂起直到允许一次调用。若已达本分钟上限，sleep 至下一分钟开始。"""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._window_start
                if elapsed >= 60.0:
                    self._calls = 0
                    self._window_start = now
                    elapsed = 0.0
                if self._calls < self.max_calls:
                    self._calls += 1
                    return
                wait_secs = 60.0 - elapsed + 0.05
            logger.info(
                "[RateLimit] 已达每分钟上限 %d 次，顺延 %.1f 秒后继续",
                self.max_calls, wait_secs,
            )
            await asyncio.sleep(wait_secs)
