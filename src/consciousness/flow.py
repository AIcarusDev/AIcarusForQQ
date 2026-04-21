"""flow.py — 机器人意识流（Consciousness Flow）。

provider 无关的工具调用历史，记录机器人跨激活、跨 provider 切换的 function calling 状态。
机器人的意识 ≠ 使用的哪个模型；切换 provider 不应清空意识流。

数据模型：
    FlowRound  — 一轮推理循环，包含若干工具调用及对应的执行结果
    ToolCall   — 模型发出的一次工具调用请求（name / args / call_id）
    ToolResponse — 工具返回的结果（name / response / call_id / timestamp）

ConsciousnessFlow 提供：
    - append_round / prune / clear
    - to_openai_messages()      → OpenAI messages 列表
    - dump() / restore()        → JSON 持久化
"""

from __future__ import annotations

import base64
import datetime
import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("AICQ.consciousness")


# ── 数据类 ────────────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """模型发出的一次工具调用请求。"""
    name: str
    args: dict
    call_id: str = ""


@dataclass
class ToolResponse:
    """工具执行结果。"""
    name: str
    response: object        # JSON-serializable
    call_id: str = ""       # 与对应 ToolCall 的 call_id 一致
    # 多模态附件（raw dict 列表，不参与序列化，仅当次激活内有效）
    # 每个 dict 格式：{"mime_type": str, "display_name": str, "data": bytes}
    multimodal_parts: list = field(default_factory=list)


@dataclass
class FlowRound:
    """一轮推理循环：模型请求的 N 个工具调用 + 对应的 N 个结果。"""
    calls: list[ToolCall] = field(default_factory=list)
    responses: list[ToolResponse] = field(default_factory=list)
    timestamp: float | None = None      # 本轮工具执行完成的绝对时间（UNIX 秒）


@dataclass
class RestartPair:
    """进程关闭/重启标记对，在意识流中占 1 个 slot。

    两条 user 消息成对出现，随整体一起被裁剪，不会只剩一半。
    """
    shutdown_time: float
    startup_time: float | None = None   # None = 启动时尚未填入


# ── ConsciousnessFlow ─────────────────────────────────────────────────────────

class ConsciousnessFlow:
    """provider 无关的机器人意识流。

    只存储工具调用历史（calls + responses）。
    用户消息（context_messages）和 system prompt 不属于意识流，
    由各 adapter 在每次调用时单独传入。

    额外提供潜伏工具恢复能力：
    - 若当前保留历史中仍存在 get_tools 的调用与返回，且 activated 包含某潜伏工具
    - 或当前保留历史中仍存在某潜伏工具自身的调用与返回
    则该工具在下一次 activation 中应继续视为可用。
    """

    def __init__(self) -> None:
        self._rounds: list[FlowRound | RestartPair] = []

    # ── 写入 ─────────────────────────────────────────────────────────────────

    def append_round(
        self,
        calls: list[ToolCall],
        responses: list[ToolResponse],
        timestamp: float | None = None,
    ) -> None:
        """追加一轮工具调用记录。"""
        self._rounds.append(FlowRound(
            calls=calls,
            responses=responses,
            timestamp=timestamp if timestamp is not None else time.time(),
        ))

    def prune(self, max_rounds: int) -> None:
        """裁剪至 max_rounds - 1 轮，为即将追加的新一轮腾出空间。"""
        capacity = max_rounds - 1
        if capacity <= 0:
            self._rounds = []
        elif len(self._rounds) > capacity:
            self._rounds = self._rounds[-capacity:]

    def clear(self) -> None:
        """清空所有历史。"""
        self._rounds = []

    def append_shutdown_marker(self) -> None:
        """关闭时调用：将所有 deferred 工具标记为失败，再追加关闭时间戳。"""
        self._complete_all_deferred_as_shutdown()
        self._rounds.append(RestartPair(shutdown_time=time.time()))
        logger.info("[consciousness] 已追加进程关闭标记")

    def complete_startup_marker(self) -> None:
        """重启恢复后调用：在最近一个未配对的 RestartPair 中填入当前启动时间。"""
        for rnd in reversed(self._rounds):
            if isinstance(rnd, RestartPair) and rnd.startup_time is None:
                rnd.startup_time = time.time()
                offline_secs = max(0, round(rnd.startup_time - rnd.shutdown_time))
                logger.info(
                    "[consciousness] 已填入重启时间，共离线 %s",
                    _format_duration(offline_secs),
                )
                return

    def _complete_all_deferred_as_shutdown(self) -> None:
        """将所有仍处于 deferred 状态的工具返回替换为进程关闭中断的失败结果。"""
        count = 0
        for rnd in self._rounds:
            if not isinstance(rnd, FlowRound):
                continue
            for i, tr in enumerate(rnd.responses):
                if isinstance(tr.response, dict) and tr.response.get("deferred"):
                    rnd.responses[i] = ToolResponse(
                        name=tr.name,
                        response={
                            "ok": False,
                            "error": "进程已关闭，工具执行被中断。",
                            "interrupted": True,
                        },
                        call_id=tr.call_id,
                    )
                    count += 1
        if count:
            logger.info("[consciousness] 已将 %d 条 deferred 工具返回标记为进程关闭中断", count)

    @property
    def round_count(self) -> int:
        return len(self._rounds)

    def complete_deferred_response(self, tool_name: str, result: dict) -> bool:
        """将最近一条 deferred 状态的工具返回替换为真实结果。

        从最新一轮往前搜索，找到第一条 name 匹配且 response 含 ``deferred: True``
        的 ToolResponse，用 *result* 原地替换。

        返回是否找到并替换。
        """
        for rnd in reversed(self._rounds):
            if not isinstance(rnd, FlowRound):
                continue
            for i, tr in enumerate(rnd.responses):
                if (
                    tr.name == tool_name
                    and isinstance(tr.response, dict)
                    and tr.response.get("deferred")
                ):
                    rnd.responses[i] = ToolResponse(
                        name=tr.name,
                        response=result,
                        call_id=tr.call_id,
                    )
                    return True
        return False

    def get_deferred_timestamp(self, tool_name: str) -> float | None:
        """返回最近一条 deferred 状态工具返回所在轮次的时间戳，不存在则返回 None。"""
        for rnd in reversed(self._rounds):
            if not isinstance(rnd, FlowRound):
                continue
            for tr in rnd.responses:
                if (
                    tr.name == tool_name
                    and isinstance(tr.response, dict)
                    and tr.response.get("deferred")
                ):
                    return rnd.timestamp
        return None

    def get_recoverable_latent_tool_names(self, latent_names: set[str]) -> set[str]:
        """根据当前保留的意识流历史，推导仍应保持可用的潜伏工具名。

        仅对本轮 build_tools() 产出的 latent_names 生效；调用方需先完成
        condition / SCOPE / REQUIRES_CONTEXT 过滤，避免历史记录绕过当前会话约束。
        """
        if not latent_names:
            return set()

        recoverable: set[str] = set()
        for rnd in self._rounds:
            if not isinstance(rnd, FlowRound):
                continue
            round_call_names = {tc.name for tc in rnd.calls}
            round_responses: dict[str, list[object]] = {}
            for tr in rnd.responses:
                round_responses.setdefault(tr.name, []).append(tr.response)

            # 当前上下文中仍保留着 get_tools 的调用和返回，则其中成功激活过的潜伏工具继续可用。
            if "get_tools" in round_call_names and "get_tools" in round_responses:
                for response in round_responses["get_tools"]:
                    if not isinstance(response, dict):
                        continue
                    activated = response.get("activated")
                    if not isinstance(activated, list):
                        continue
                    for name in activated:
                        if isinstance(name, str) and name in latent_names:
                            recoverable.add(name)

            # 当前上下文中仍保留着某潜伏工具自身的调用和返回，则该工具继续可用。
            for name in latent_names:
                if name in round_call_names and name in round_responses:
                    recoverable.add(name)

            if len(recoverable) == len(latent_names):
                break

        return recoverable

    # ── OpenAI 格式转换 ───────────────────────────────────────────────────────

    def to_openai_messages(self) -> list[dict]:
        """转换为 OpenAI messages 格式（不含 system / 第一条 user）。

        每轮产生：
          {"role": "assistant", "tool_calls": [...]}
          N × {"role": "tool", "content": json_str 或 [{type:text}+{type:image_url}...]}

        当 ToolResponse 含有 multimodal_parts 时，content 使用数组格式，
        供支持原生多模态工具响应的模型消费。
        """
        messages = []
        for rnd in self._rounds:
            if isinstance(rnd, RestartPair):
                messages.append({
                    "role": "user",
                    "content": f"[系统通知] 进程已于 {_format_timestamp(rnd.shutdown_time)} 关闭，所有执行中的工具已中断。",
                })
                if rnd.startup_time is not None:
                    offline_secs = max(0, round(rnd.startup_time - rnd.shutdown_time))
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[系统通知] 进程已于 {_format_timestamp(rnd.startup_time)} 重启，"
                            f"共离线 {_format_duration(offline_secs)}。"
                        ),
                    })
                continue
            if not rnd.calls:
                continue
            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.args, ensure_ascii=False),
                        },
                    }
                    for tc in rnd.calls
                ],
            })
            for tr in rnd.responses:
                text_content = json.dumps(tr.response, ensure_ascii=False)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr.call_id,
                    "content": text_content,
                })
                if tr.multimodal_parts:
                    # tool 消息只能是字符串；图片通过紧随的 user 消息传入，
                    # 供支持原生多模态的模型（LM Studio、gpt-4o 系列等）消费。
                    img_parts: list = [{"type": "text", "text": f"[{tr.name} (收藏的表情包)]"}]
                    for mp in tr.multimodal_parts:
                        data_str: str = (
                            mp["data"] if isinstance(mp["data"], str)
                            else base64.b64encode(mp["data"]).decode()
                        )
                        img_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mp['mime_type']};base64,{data_str}"},
                        })
                    messages.append({
                        "role": "user",
                        "content": img_parts,
                    })
        return messages

    # ── 持久化 ────────────────────────────────────────────────────────────────

    def dump(self) -> tuple[list[dict], list]:
        """序列化为 JSON 兼容格式。multimodal_parts（bytes 数据）不参与序列化。

        返回 (rounds_data, timestamps)，与 database.save_adapter_contents 接口兼容。
        """
        data = []
        timestamps = []
        for rnd in self._rounds:
            if isinstance(rnd, RestartPair):
                data.append({
                    "type": "restart",
                    "shutdown_time": rnd.shutdown_time,
                    "startup_time": rnd.startup_time,
                })
                timestamps.append(None)
            else:
                data.append({
                    "calls": [
                        {
                            "name": tc.name,
                            "args": tc.args,
                            "call_id": tc.call_id,
                        }
                        for tc in rnd.calls
                    ],
                    "responses": [
                        {"name": tr.name, "response": tr.response, "call_id": tr.call_id}
                        for tr in rnd.responses
                    ],
                })
                timestamps.append(rnd.timestamp)
        return data, timestamps

    def restore(self, data: list[dict], timestamps: list) -> None:
        """从序列化数据恢复。"""
        self._rounds = []
        for i, entry in enumerate(data):
            if entry.get("type") == "restart":
                st_raw = entry.get("startup_time")
                self._rounds.append(RestartPair(
                    shutdown_time=float(entry.get("shutdown_time", 0)),
                    startup_time=float(st_raw) if st_raw is not None else None,
                ))
                continue
            calls = [
                ToolCall(
                    name=c.get("name", ""),
                    args=c.get("args", {}),
                    call_id=c.get("call_id", ""),
                )
                for c in entry.get("calls", [])
            ]
            responses = [
                ToolResponse(
                    name=r.get("name", ""),
                    response=r.get("response", {}),
                    call_id=r.get("call_id", ""),
                )
                for r in entry.get("responses", [])
            ]
            ts_raw = timestamps[i] if i < len(timestamps) else None
            ts = float(ts_raw) if ts_raw is not None else None
            if calls or responses:
                self._rounds.append(FlowRound(calls=calls, responses=responses, timestamp=ts))
        logger.info("[consciousness] 已恢复意识流: %d 轮", len(self._rounds))


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _format_relative_time(seconds_ago: float) -> str:
    """将经过秒数转换为中文相对时间描述（如"3分钟前"）。"""
    s = int(abs(seconds_ago))
    if s < 60:
        return f"{s}秒前"
    elif s < 3600:
        return f"{s // 60}分钟前"
    elif s < 86400:
        return f"{s // 3600}小时前"
    else:
        return f"{s // 86400}天前"


def _format_timestamp(ts: float) -> str:
    """将 UNIX 时间戳转为本地时间字符串（精确到分钟）。"""
    dt = datetime.datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M")


def _format_duration(seconds: int) -> str:
    """将秒数转为中文时长描述。"""
    if seconds < 60:
        return f"{seconds}秒"
    elif seconds < 3600:
        return f"{seconds // 60}分钟"
    elif seconds < 86400:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}小时{minutes}分钟" if minutes else f"{hours}小时"
    else:
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        return f"{days}天{hours}小时" if hours else f"{days}天"
