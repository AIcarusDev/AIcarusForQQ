"""flow.py — 机器人意识流（Consciousness Flow）。

provider 无关的工具调用历史，记录机器人跨激活、跨 provider 切换的 function calling 状态。
机器人的意识 ≠ 使用的哪个模型；切换 provider 不应清空意识流。

数据模型：
    FlowRound  — 一轮推理循环，包含若干工具调用及对应的执行结果
    ToolCall   — 模型发出的一次工具调用请求（name / args / call_id）
    ToolResponse — 工具返回的结果（name / response / call_id / timestamp）

ConsciousnessFlow 提供：
    - append_round / prune / clear
    - to_xml_messages()         → XML 文本协议 messages 列表
    - dump() / restore()        → JSON 持久化
"""

from __future__ import annotations

import base64
import copy
import datetime
import json
import logging
import re
import time
from dataclasses import dataclass, field

from llm.core.tool_calling.common import strip_legacy_motivation_fields
from llm.core.tool_calling.xml_protocol import XML_TOOL_CALL_ERROR_NAME
from llm.media.outbound_image import make_data_url

logger = logging.getLogger("AICQ.consciousness")

_LATENT_TOOL_MANAGER_NAMES = frozenset({"tools_manage", "get_tools"})


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
    seq: int = 0
    cognition: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    responses: list[ToolResponse] = field(default_factory=list)
    timestamp: float | None = None  # 本轮工具执行完成的绝对时间（UNIX 秒）
    raw_response: str = ""  # 模型本轮原始输出文本，用于完全重复响应检测。


@dataclass
class RestartPair:
    """进程关闭/重启标记对，在意识流中占 1 个 slot。

    两条 user 消息成对出现，随整体一起被裁剪，不会只剩一半。
    """
    shutdown_time: float
    startup_time: float | None = None   # None = 启动时尚未填入


@dataclass
class CompressionSummary:
    """已注入主上下文的意识流压缩摘要。"""
    text: str
    coverage_end_seq: int = 0
    updated_at: float | None = None


@dataclass
class CompressionJob:
    """一次压缩任务的快照。"""
    task_xml: str
    coverage_end_seq: int
    round_count: int
    base_coverage_end_seq: int = 0
    detected_at: str = ""
    rounds: list[FlowRound] = field(default_factory=list)


# ── ConsciousnessFlow ─────────────────────────────────────────────────────────

class ConsciousnessFlow:
    """provider 无关的机器人意识流。

    只存储工具调用历史（calls + responses）。
    用户消息（context_messages）和 system prompt 不属于意识流，
    由各 adapter 在每次调用时单独传入。

    额外提供潜伏工具恢复能力：
    - 若当前保留历史中仍存在 tools_manage/get_tools 的调用与返回，且 activated 包含某潜伏工具
    - 或当前保留历史中仍存在某潜伏工具自身的调用与返回
    则该工具在下一次 activation 中应继续视为可用。
    """

    def __init__(self) -> None:
        self._rounds: list[FlowRound | RestartPair] = []
        self._compression_summary: CompressionSummary | None = None
        self._ready_compression_summaries: list[CompressionSummary] = []
        self._latent_tool_activity_seq: dict[str, int] = {}
        self._next_seq: int = 1

    # ── 写入 ─────────────────────────────────────────────────────────────────

    def append_round(
        self,
        calls: list[ToolCall],
        responses: list[ToolResponse],
        cognition: str = "",
        timestamp: float | None = None,
        raw_response: str = "",
    ) -> None:
        """追加一轮工具调用记录。"""
        seq = self._next_seq
        cleaned_calls: list[ToolCall] = []
        for call in calls:
            cleaned_args, _changed = strip_legacy_motivation_fields(call.args)
            cleaned_calls.append(ToolCall(name=call.name, args=cleaned_args, call_id=call.call_id))
        self._rounds.append(FlowRound(
            seq=seq,
            cognition=cognition,
            calls=cleaned_calls,
            responses=responses,
            timestamp=timestamp if timestamp is not None else time.time(),
            raw_response=raw_response,
        ))
        self._remember_latent_tool_activity(seq, cleaned_calls, responses)
        self._next_seq += 1

    def prune(self, max_rounds: int) -> None:
        """裁剪至 max_rounds - 1 轮，为即将追加的新一轮腾出空间。"""
        capacity = max_rounds - 1
        self.promote_ready_compression_summary(
            max_rounds,
            incoming_rounds=1,
            required_coverage_end_seq=self._uncovered_flow_seq_that_would_be_dropped(capacity),
        )
        self._drop_covered_rounds()
        if capacity <= 0:
            self._rounds = []
        elif len(self._rounds) > capacity:
            self._rounds = self._rounds[-capacity:]

    def clear(self) -> None:
        """清空所有历史。"""
        self._rounds = []
        self._compression_summary = None
        self._ready_compression_summaries = []
        self._latent_tool_activity_seq = {}
        self._next_seq = 1

    def append_shutdown_marker(self, *, preserve_deferred_tool_names: set[str] | None = None) -> None:
        """关闭时调用：将所有 deferred 工具标记为失败，再追加关闭时间戳。"""
        self._complete_all_deferred_as_shutdown(
            preserve_tool_names=preserve_deferred_tool_names or set()
        )
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

    def _complete_all_deferred_as_shutdown(self, *, preserve_tool_names: set[str]) -> None:
        """将所有仍处于 deferred 状态的工具返回替换为进程关闭中断的失败结果。"""
        count = 0
        for rnd in self._rounds:
            if not isinstance(rnd, FlowRound):
                continue
            for i, tr in enumerate(rnd.responses):
                if tr.name in preserve_tool_names:
                    continue
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

    def recent_rounds(self, limit: int = 5) -> tuple[FlowRound, ...]:
        """Return recent normal flow rounds for read-only policy checks."""
        if limit <= 0:
            return ()
        rounds = [rnd for rnd in self._rounds if isinstance(rnd, FlowRound)]
        return tuple(rounds[-limit:])

    def recent_raw_responses(self, limit: int = 3) -> tuple[str, ...]:
        """Return recent non-empty raw assistant responses for duplicate guards."""
        return tuple(
            rnd.raw_response
            for rnd in self.recent_rounds(limit)
            if getattr(rnd, "raw_response", "")
        )

    @property
    def active_compression_summary(self) -> CompressionSummary | None:
        return self._compression_summary

    @property
    def ready_compression_summaries(self) -> tuple[CompressionSummary, ...]:
        return tuple(self._ready_compression_summaries)

    @property
    def compression_frontier_end_seq(self) -> int:
        """已压缩完成或已注入的最远 coverage，用于后台继续追赶。"""
        end = (
            self._compression_summary.coverage_end_seq
            if self._compression_summary is not None
            else 0
        )
        for summary in self._ready_compression_summaries:
            end = max(end, summary.coverage_end_seq)
        return end

    def build_compression_job(
        self,
        trigger_rounds: int,
        coverage_end: int | None = None,
    ) -> CompressionJob | None:
        """构造一次固定批量的压缩任务输入；不足触发轮数时返回 None。"""
        if coverage_end is None:
            coverage_end = self.compression_frontier_end_seq
        candidates = [
            rnd
            for rnd in self._rounds
            if isinstance(rnd, FlowRound) and rnd.seq > coverage_end
        ]
        if len(candidates) < trigger_rounds:
            return None
        if not candidates:
            return None
        detected_at = _format_os_timestamp()
        rounds_snapshot = copy.deepcopy(candidates[:trigger_rounds])
        task_xml = _format_compression_task_xml(
            current_time=detected_at,
            last_compression=self._summary_text_at_or_before(coverage_end),
            rounds=rounds_snapshot,
        )
        return CompressionJob(
            task_xml=task_xml,
            coverage_end_seq=max(rnd.seq for rnd in rounds_snapshot),
            round_count=len(rounds_snapshot),
            base_coverage_end_seq=coverage_end,
            detected_at=detected_at,
            rounds=rounds_snapshot,
        )

    def render_compression_job(self, job: CompressionJob) -> str:
        """用已生成的前序摘要渲染已冻结的压缩任务快照。"""
        return _format_compression_task_xml(
            current_time=job.detected_at,
            last_compression=self._summary_text_at_or_before(job.base_coverage_end_seq),
            rounds=job.rounds,
        )

    def queue_compression_summary(self, summary_text: str, coverage_end_seq: int) -> bool:
        """保存已完成但尚未注入主上下文的压缩摘要。"""
        text = (summary_text or "").strip()
        if not text:
            return False
        if coverage_end_seq <= self.compression_frontier_end_seq:
            return False
        self._ready_compression_summaries.append(CompressionSummary(
            text=text,
            coverage_end_seq=coverage_end_seq,
            updated_at=time.time(),
        ))
        self._ready_compression_summaries.sort(key=lambda item: item.coverage_end_seq)
        return True

    def apply_compression_summary(self, summary_text: str, coverage_end_seq: int) -> bool:
        """兼容旧调用名：压缩结果先进入 ready 队列，不会立刻注入。"""
        return self.queue_compression_summary(summary_text, coverage_end_seq)

    def promote_ready_compression_summary(
        self,
        max_rounds: int,
        incoming_rounds: int = 0,
        required_coverage_end_seq: int = 0,
    ) -> bool:
        """当 raw 窗口即将超限时，提升最早足够的 ready summary。"""
        if not self._ready_compression_summaries:
            return False
        active_end = (
            self._compression_summary.coverage_end_seq
            if self._compression_summary is not None
            else 0
        )
        projected_raw = self._raw_round_count_after(active_end) + incoming_rounds
        if projected_raw <= max_rounds and required_coverage_end_seq <= active_end:
            return False

        chosen: CompressionSummary | None = None
        for summary in self._ready_compression_summaries:
            if summary.coverage_end_seq <= active_end:
                continue
            raw_after_summary = (
                self._raw_round_count_after(summary.coverage_end_seq)
                + incoming_rounds
            )
            covers_required = summary.coverage_end_seq >= required_coverage_end_seq
            if raw_after_summary <= max_rounds and covers_required:
                chosen = summary
                break
        if chosen is None:
            return False

        self._compression_summary = chosen
        self._ready_compression_summaries = [
            summary
            for summary in self._ready_compression_summaries
            if summary.coverage_end_seq > chosen.coverage_end_seq
        ]
        self._drop_covered_rounds()
        return True

    def _raw_round_count_after(self, coverage_end_seq: int) -> int:
        return sum(
            1
            for rnd in self._rounds
            if isinstance(rnd, FlowRound) and rnd.seq > coverage_end_seq
        )

    def _summary_text_at_or_before(self, coverage_end_seq: int) -> str:
        candidates: list[CompressionSummary] = []
        if (
            self._compression_summary is not None
            and self._compression_summary.coverage_end_seq <= coverage_end_seq
        ):
            candidates.append(self._compression_summary)
        candidates.extend(
            summary
            for summary in self._ready_compression_summaries
            if summary.coverage_end_seq <= coverage_end_seq
        )
        if not candidates:
            return ""
        return max(candidates, key=lambda item: item.coverage_end_seq).text

    def _drop_covered_rounds(self) -> None:
        if self._compression_summary is None:
            return
        covered_seq = self._compression_summary.coverage_end_seq
        self._rounds = [
            rnd
            for rnd in self._rounds
            if not isinstance(rnd, FlowRound) or rnd.seq > covered_seq
        ]

    def _uncovered_flow_seq_that_would_be_dropped(self, capacity: int) -> int:
        if capacity < 0:
            capacity = 0
        drop_count = max(0, len(self._rounds) - capacity)
        if drop_count <= 0:
            return 0
        active_end = (
            self._compression_summary.coverage_end_seq
            if self._compression_summary is not None
            else 0
        )
        seqs = [
            rnd.seq
            for rnd in self._rounds[:drop_count]
            if isinstance(rnd, FlowRound) and rnd.seq > active_end
        ]
        return max(seqs, default=0)

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

    def get_recent_cognitions(self, n: int = 5) -> list[str]:
        """返回最近 n 条非空 cognition 文本（从旧到新），供归档器注入 Track2。"""
        result: list[str] = []
        for rnd in reversed(self._rounds):
            if isinstance(rnd, RestartPair):
                continue
            if rnd.cognition:
                result.append(rnd.cognition)
                if len(result) >= n:
                    break
        return list(reversed(result))

    def _remember_latent_tool_activity(
        self,
        seq: int,
        calls: list[ToolCall],
        responses: list[ToolResponse],
    ) -> None:
        """记录潜伏工具的结构化活跃证据，独立于 raw round 是否被压缩隐藏。"""
        if not calls or not responses:
            return

        response_map: dict[str, list[object]] = {}
        for response in responses:
            response_map.setdefault(response.name, []).append(response.response)

        for manager_name in _LATENT_TOOL_MANAGER_NAMES:
            for response in response_map.get(manager_name, []):
                if not isinstance(response, dict):
                    continue
                activated = response.get("activated")
                if not isinstance(activated, list):
                    continue
                for name in activated:
                    if isinstance(name, str) and name:
                        self._latent_tool_activity_seq[name] = seq

        for call in calls:
            name = call.name
            if not name or name in _LATENT_TOOL_MANAGER_NAMES:
                continue
            if any(
                _tool_response_keeps_latent_active(name, response)
                for response in response_map.get(name, [])
            ):
                self._latent_tool_activity_seq[name] = seq

    def get_recoverable_latent_tool_names(
        self,
        latent_names: set[str],
        max_rounds: int | None = None,
    ) -> set[str]:
        """根据当前保留的意识流历史，推导仍应保持可用的潜伏工具名。

        仅对本轮 build_tools() 产出的 latent_names 生效；调用方需先完成
        condition / SCOPE / REQUIRES_CONTEXT 过滤，避免历史记录绕过当前会话约束。
        """
        if not latent_names:
            return set()

        recoverable: set[str] = set()
        min_seq = self._next_seq - max_rounds if max_rounds is not None else None

        for name, seq in self._latent_tool_activity_seq.items():
            if name not in latent_names:
                continue
            if min_seq is not None and seq < min_seq:
                continue
            recoverable.add(name)

        for rnd in self._rounds:
            if not isinstance(rnd, FlowRound):
                continue
            if min_seq is not None and rnd.seq < min_seq:
                continue
            round_call_names = {tc.name for tc in rnd.calls}
            round_responses: dict[str, list[object]] = {}
            for tr in rnd.responses:
                round_responses.setdefault(tr.name, []).append(tr.response)

            # 当前上下文中仍保留着工具管理调用和返回，则其中成功激活过的潜伏工具继续可用。
            for manager_name in _LATENT_TOOL_MANAGER_NAMES:
                if manager_name not in round_call_names or manager_name not in round_responses:
                    continue
                for response in round_responses[manager_name]:
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
                if (
                    name in round_call_names
                    and any(
                        _tool_response_keeps_latent_active(name, response)
                        for response in round_responses.get(name, [])
                    )
                ):
                    recoverable.add(name)

            if len(recoverable) == len(latent_names):
                break

        return recoverable

    # ── XML 文本协议转换 ───────────────────────────────────────────────────────

    def to_xml_messages(self) -> list[dict]:
        """转换为 XML 文本工具调用协议 messages（不含 system / 当前 user）。

        每轮产生：
          assistant: N 个 <tool_call>{...}</tool_call> 块
          user:      N 个 <tool_response>{...}</tool_response> 块

        当 ToolResponse 含有 multimodal_parts 时，响应 XML 作为 text part，图片紧随其后。
        """
        messages = []
        if self._compression_summary is not None:
            messages.append({
                "role": "user",
                "content": _format_context_summary_xml(self._compression_summary),
            })
        covered_seq = (
            self._compression_summary.coverage_end_seq
            if self._compression_summary is not None
            else 0
        )
        for rnd in self._rounds:
            if isinstance(rnd, RestartPair):
                messages.extend(_restart_pair_messages(rnd))
                continue
            if rnd.seq <= covered_seq:
                continue
            if not rnd.calls:
                for tr in rnd.responses:
                    messages.append({
                        "role": "user",
                        "content": _format_tool_response_xml(tr),
                    })
                continue

            assistant_blocks = []
            if rnd.cognition:
                assistant_blocks.append(_format_cognition_xml(rnd.cognition))
            assistant_blocks.extend(_format_tool_call_xml(tc) for tc in rnd.calls)
            messages.append({
                "role": "assistant",
                "content": "\n".join(assistant_blocks),
            })
            for tr in rnd.responses:
                text_content = _format_tool_response_xml(tr)
                if tr.multimodal_parts:
                    img_parts: list = [{"type": "text", "text": text_content}]
                    for mp in tr.multimodal_parts:
                        data_str: str = (
                            mp["data"] if isinstance(mp["data"], str)
                            else base64.b64encode(mp["data"]).decode()
                        )
                        data_url = make_data_url(data_str, str(mp.get("mime_type") or "image/jpeg"))
                        if not data_url:
                            continue
                        img_parts.append({
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        })
                    messages.append({
                        "role": "user",
                        "content": img_parts,
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": text_content,
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
                    "seq": rnd.seq,
                    "cognition": rnd.cognition,
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
        if self._compression_summary is not None:
            data.insert(0, {
                "type": "compression_summary",
                "text": self._compression_summary.text,
                "coverage_end_seq": self._compression_summary.coverage_end_seq,
                "updated_at": self._compression_summary.updated_at,
            })
            timestamps.insert(0, None)
        for summary in reversed(self._ready_compression_summaries):
            data.insert(1 if self._compression_summary is not None else 0, {
                "type": "compression_ready_summary",
                "text": summary.text,
                "coverage_end_seq": summary.coverage_end_seq,
                "updated_at": summary.updated_at,
            })
            timestamps.insert(1 if self._compression_summary is not None else 0, None)
        if self._latent_tool_activity_seq:
            insert_at = 0
            if self._compression_summary is not None:
                insert_at += 1
            insert_at += len(self._ready_compression_summaries)
            data.insert(insert_at, {
                "type": "latent_tool_activity",
                "activity_seq": dict(sorted(self._latent_tool_activity_seq.items())),
            })
            timestamps.insert(insert_at, None)
        return data, timestamps

    def restore(self, data: list[dict], timestamps: list) -> None:
        """从序列化数据恢复。"""
        self._rounds = []
        self._compression_summary = None
        self._ready_compression_summaries = []
        self._latent_tool_activity_seq = {}
        self._next_seq = 1
        restored_latent_activity: dict[str, int] = {}
        for i, entry in enumerate(data):
            if entry.get("type") == "compression_summary":
                self._compression_summary = CompressionSummary(
                    text=str(entry.get("text") or ""),
                    coverage_end_seq=int(entry.get("coverage_end_seq") or 0),
                    updated_at=(
                        float(entry["updated_at"])
                        if entry.get("updated_at") is not None
                        else None
                    ),
                )
                self._next_seq = max(
                    self._next_seq,
                    self._compression_summary.coverage_end_seq + 1,
                )
                continue
            if entry.get("type") == "compression_ready_summary":
                summary = CompressionSummary(
                    text=str(entry.get("text") or ""),
                    coverage_end_seq=int(entry.get("coverage_end_seq") or 0),
                    updated_at=(
                        float(entry["updated_at"])
                        if entry.get("updated_at") is not None
                        else None
                    ),
                )
                self._ready_compression_summaries.append(summary)
                self._next_seq = max(self._next_seq, summary.coverage_end_seq + 1)
                continue
            if entry.get("type") == "latent_tool_activity":
                raw_activity = entry.get("activity_seq")
                if isinstance(raw_activity, dict):
                    for name, seq in raw_activity.items():
                        if not isinstance(name, str) or not name:
                            continue
                        try:
                            restored_latent_activity[name] = int(seq)
                        except (TypeError, ValueError):
                            continue
                continue
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
                    args=strip_legacy_motivation_fields(c.get("args", {}))[0],
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
                seq = int(entry.get("seq") or self._next_seq)
                self._rounds.append(FlowRound(
                    seq=seq,
                    cognition=str(entry.get("cognition") or ""),
                    calls=calls,
                    responses=responses,
                    timestamp=ts,
                    raw_response=str(entry.get("raw_response") or ""),
                ))
                self._next_seq = max(self._next_seq, seq + 1)
                self._remember_latent_tool_activity(seq, calls, responses)
        for name, seq in restored_latent_activity.items():
            self._latent_tool_activity_seq[name] = max(
                self._latent_tool_activity_seq.get(name, 0),
                seq,
            )
        self._ready_compression_summaries.sort(key=lambda item: item.coverage_end_seq)
        logger.info("[consciousness] 已恢复意识流: %d 轮", len(self._rounds))


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _tool_response_keeps_latent_active(name: str, response: object) -> bool:
    """Return whether a tool response proves the latent tool was actually reachable."""
    if not isinstance(response, dict):
        return True

    activated = response.get("activated")
    if isinstance(activated, list) and name in activated:
        return True

    error = response.get("error")
    if isinstance(error, str) and error.strip() == f"未知工具: {name}":
        return False

    return True


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


def _restart_pair_messages(rnd: RestartPair) -> list[dict]:
    messages = [{
        "role": "user",
        "content": f"[系统通知] 进程已于 {_format_timestamp(rnd.shutdown_time)} 关闭，所有执行中的工具已中断。",
    }]
    if rnd.startup_time is not None:
        offline_secs = max(0, round(rnd.startup_time - rnd.shutdown_time))
        messages.append({
            "role": "user",
            "content": (
                f"[系统通知] 进程已于 {_format_timestamp(rnd.startup_time)} 重启，"
                f"共离线 {_format_duration(offline_secs)}。"
            ),
        })
    return messages


def _format_tool_call_xml(tool_call: ToolCall) -> str:
    payload = {
        "id": tool_call.call_id,
        "name": tool_call.name,
        "arguments": tool_call.args,
    }
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"


def _format_context_summary_xml(summary: CompressionSummary) -> str:
    return (
        "<summary>\n"
        f"{_escape_xml_text(summary.text)}"
        "\n</summary>"
    )


def _format_compression_task_xml(
    current_time: str,
    last_compression: str,
    rounds: list[FlowRound],
) -> str:
    blocks = [
        f"<current_time>{_escape_xml_text(current_time)}</current_time>",
        "<task>",
    ]
    if last_compression.strip():
        blocks.append(
            f"<last_compression>{_escape_xml_text(last_compression.strip())}</last_compression>"
        )
    else:
        blocks.append("<last_compression/>")

    for index, rnd in enumerate(rounds, start=1):
        blocks.append(f'<turn id="{index}">')
        if rnd.cognition:
            blocks.append(_format_cognition_xml(rnd.cognition))
        blocks.extend(_format_tool_call_xml(tc) for tc in rnd.calls)
        blocks.extend(_format_tool_response_xml(tr) for tr in rnd.responses)
        blocks.append("</turn>")

    blocks.append("</task>")
    return "\n".join(blocks)


def _format_os_timestamp(timestamp: float | None = None) -> str:
    dt = (
        datetime.datetime.now().astimezone()
        if timestamp is None
        else datetime.datetime.fromtimestamp(timestamp).astimezone()
    )
    return dt.isoformat(timespec="seconds")


_SUMMARY_BLOCK_RE = re.compile(r"<summary\b[^>]*>(.*?)</summary>", re.DOTALL)


def extract_summary_block(text: str) -> str:
    """从压缩模型输出中提取真正注入上下文的 <summary> 内容。"""
    match = _SUMMARY_BLOCK_RE.search(text or "")
    if not match:
        return (text or "").strip()
    return match.group(1).strip()


def _format_cognition_xml(cognition: str) -> str:
    return f"<cognition>{_escape_xml_text(cognition)}</cognition>"


def _escape_xml_text(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _format_tool_response_xml(tool_response: ToolResponse) -> str:
    if tool_response.name == XML_TOOL_CALL_ERROR_NAME:
        payload = {
            "type": "tool_call_error",
            "response": tool_response.response,
        }
        return f"<tool_feedback>{json.dumps(payload, ensure_ascii=False)}</tool_feedback>"

    payload = {
        "id": tool_response.call_id,
        "name": tool_response.name,
        "response": tool_response.response,
    }
    return f"<tool_response>{json.dumps(payload, ensure_ascii=False)}</tool_response>"


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
