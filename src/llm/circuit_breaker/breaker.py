"""breaker.py — 工具调用重复熔断器

检测 LLM 在工具调用循环中是否对同一工具以完全相同的参数连续调用了过多轮次，
若连续轮次数达到阈值则触发熔断，防止死循环。

判定规则：
- 以 (函数名, 序列化参数) 为 key，独立追踪每个调用签名的连续出现轮次
- 同一轮内的重复调用不累加 streak（仅记录首次）
- 连续 max_streak 轮出现完全相同调用时返回 True（应熔断）
"""

import json

_DEFAULT_MAX_STREAK = 3


class ToolRepeatBreaker:
    """单次 LLM call() 内的工具重复调用熔断器。

    用法::

        breaker = ToolRepeatBreaker()
        # 每轮工具执行前：
        breaker.begin_round(tool_round)
        for each fc:
            if breaker.check_and_record(fn_name, args):
                # 熔断处理...
    """

    def __init__(self, max_streak: int = _DEFAULT_MAX_STREAK):
        self.max_streak = max_streak
        # key -> (last_round_seen, streak_count)
        self._history: dict[str, tuple[int, int]] = {}
        # 同一轮内已首次记录的 key（防止同一轮内重复调用累加 streak）
        self._round_seen: set[str] = set()
        self._current_round: int = 0

    def begin_round(self, round_num: int) -> None:
        """每轮开始时调用，重置当轮首次记录集合。"""
        self._current_round = round_num
        self._round_seen.clear()

    def check_and_record(self, fn_name: str, args: dict) -> bool:
        """检测该调用是否应熔断，并记录本次调用。

        - 连续 max_streak 轮以完全相同参数调用同一工具时，返回 True（应熔断）。
        - 同一轮内的重复调用不累加 streak（只记录首次）。
        """
        key = fn_name + "|" + json.dumps(args, sort_keys=True, ensure_ascii=False)

        if key in self._round_seen:
            # 同一轮内重复调用：不更新 streak，直接用已有计数判断
            _, count = self._history.get(key, (0, 0))
            return count >= self.max_streak

        self._round_seen.add(key)
        last_round, count = self._history.get(key, (-1, 0))
        new_count = count + 1 if self._current_round == last_round + 1 else 1
        self._history[key] = (self._current_round, new_count)
        return new_count >= self.max_streak
