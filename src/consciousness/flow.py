"""flow.py — 机器人意识流（Consciousness Flow）。

provider 无关的工具调用历史，记录机器人跨激活、跨 provider 切换的 function calling 状态。
机器人的意识 ≠ 使用的哪个模型；切换 provider 不应清空意识流。

数据模型：
    FlowRound  — 一轮推理循环，包含若干工具调用及对应的执行结果
    ToolCall   — 模型发出的一次工具调用请求（namespace / name / args / call_id）
    ToolResponse — 工具返回的结果（namespace / name / response / call_id / timestamp）

ConsciousnessFlow 提供：
    - append_round / prune / clear
    - to_xml_messages()         → AIC Action/history messages 列表
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
from llm.core.tool_calling.aic_action import AIC_ACTION_ERROR_NAME, is_aic_action_error_name
from llm.media.outbound_image import make_data_url

logger = logging.getLogger("AICQ.consciousness")

RAW_COGNITION_ROUNDS = 2
MISSING_MOTIVE_TEXT = "有点记不清了"


# ── 数据类 ────────────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """模型发出的一次工具调用请求。"""
    name: str
    args: dict
    call_id: str = ""
    namespace: str = ""


@dataclass
class ToolResponse:
    """工具执行结果。"""
    name: str
    response: object        # JSON-serializable
    call_id: str = ""       # 与对应 ToolCall 的 call_id 一致
    namespace: str = ""
    # 工具可选择将完整 JSON 结果放进 CDATA，避免任意文本干扰外层 XML。
    result_cdata: bool = False
    # 多模态附件（raw dict 列表，不参与序列化，仅当次激活内有效）
    # 每个 dict 格式：{"mime_type": str, "display_name": str, "data": bytes}
    multimodal_parts: list = field(default_factory=list)


def _restore_tool_response(raw: dict) -> ToolResponse:
    """Restore the current JSON result shape and fold old split text into it once."""
    response = raw.get("response", {})
    result_cdata = bool(raw.get("result_cdata", False))
    old_text_payload = raw.get("text_payload")
    if old_text_payload is not None:
        payload = dict(response) if isinstance(response, dict) else {"value": response}
        payload["content"] = str(old_text_payload)
        response = payload
        result_cdata = True
    return ToolResponse(
        namespace=str(raw.get("namespace") or ""),
        name=raw.get("name", ""),
        response=response,
        call_id=raw.get("call_id", ""),
        result_cdata=result_cdata,
    )


@dataclass
class FlowRound:
    """一轮推理循环：模型请求的 N 个工具调用 + 对应的 N 个结果。"""
    seq: int = 0
    cognition: str = ""
    motive: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    responses: list[ToolResponse] = field(default_factory=list)
    request_started_at: float | None = None  # 本轮模型请求开始的绝对时间（UNIX 秒）
    timestamp: float | None = None  # 本轮工具执行完成的绝对时间（UNIX 秒）
    raw_response: str = ""  # 模型本轮原始输出文本，用于完全重复响应检测。
    memory_candidates: list[dict] = field(default_factory=list)


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

    namespace 可见性恢复由 tools.namespaces.recover_namespace_state_from_flow
    基于当前 flow 记录推导，不再在 flow 内维护隐藏工具状态。
    """

    def __init__(self) -> None:
        self._rounds: list[FlowRound | RestartPair] = []
        self._compression_summary: CompressionSummary | None = None
        self._ready_compression_summaries: list[CompressionSummary] = []
        self._next_seq: int = 1

    # ── 写入 ─────────────────────────────────────────────────────────────────

    def append_round(
        self,
        calls: list[ToolCall],
        responses: list[ToolResponse],
        cognition: str = "",
        motive: str = "",
        request_started_at: float | None = None,
        timestamp: float | None = None,
        raw_response: str = "",
        memory_candidates: list[dict] | None = None,
    ) -> None:
        """追加一轮工具调用记录。"""
        seq = self._next_seq
        cleaned_calls: list[ToolCall] = []
        for call in calls:
            cleaned_args, _changed = strip_legacy_motivation_fields(call.args)
            cleaned_calls.append(
                ToolCall(
                    name=call.name,
                    namespace=call.namespace,
                    args=cleaned_args,
                    call_id=call.call_id,
                )
            )
        self._rounds.append(FlowRound(
            seq=seq,
            cognition=cognition,
            motive=motive,
            calls=cleaned_calls,
            responses=responses,
            request_started_at=request_started_at,
            timestamp=timestamp if timestamp is not None else time.time(),
            raw_response=raw_response,
            memory_candidates=copy.deepcopy(memory_candidates or []),
        ))
        self._next_seq += 1

    def attach_memory_candidates_to_latest_round(self, candidates: list[dict]) -> None:
        """Attach the just-used recall candidates to the newest normal round."""

        cleaned = copy.deepcopy([item for item in candidates or [] if isinstance(item, dict)])
        if not cleaned:
            return
        for rnd in reversed(self._rounds):
            if isinstance(rnd, FlowRound):
                rnd.memory_candidates = cleaned
                return

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
                        namespace=tr.namespace,
                        response={
                            "ok": False,
                            "error": "进程已关闭，工具执行被中断。",
                            "interrupted": True,
                        },
                        call_id=tr.call_id,
                        result_cdata=tr.result_cdata,
                    )
                    count += 1
        if count:
            logger.info("[consciousness] 已将 %d 条 deferred 工具返回标记为进程关闭中断", count)

    @property
    def round_count(self) -> int:
        return len(self._rounds)

    @property
    def next_seq(self) -> int:
        return self._next_seq

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
            generated_at=detected_at,
            previous_summary=self._summary_text_at_or_before(coverage_end),
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
            generated_at=job.detected_at,
            previous_summary=self._summary_text_at_or_before(job.base_coverage_end_seq),
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
                        namespace=tr.namespace,
                        response=result,
                        call_id=tr.call_id,
                        result_cdata=tr.result_cdata,
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

    def visible_cognitions(self, limit: int = 8) -> list[str]:
        """Return visible, uncompressed cognition blocks from old to new."""
        if limit <= 0:
            return []
        visible_limit = min(limit, RAW_COGNITION_ROUNDS)
        covered_seq = (
            self._compression_summary.coverage_end_seq
            if self._compression_summary is not None
            else 0
        )
        result: list[str] = []
        for rnd in reversed(self._rounds):
            if isinstance(rnd, RestartPair):
                continue
            if rnd.seq <= covered_seq:
                continue
            if rnd.cognition:
                result.append(rnd.cognition)
                if len(result) >= visible_limit:
                    break
        return list(reversed(result))

    # ── AIC Action 历史转换 ───────────────────────────────────────────────────

    def to_xml_messages(self, reference_time: float | None = None) -> list[dict]:
        """转换为 AIC Action 历史 messages（不含 system / 当前 user）。

        最近 ``RAW_COGNITION_ROUNDS`` 个含 cognition 的轮次保持原始
        assistant/user 形态；更早且未被 summary 覆盖的轮次折叠为 user-role
        ``<old_cycles>``，不再暴露 cognition 原文。

        当 ToolResponse 含有 multimodal_parts 时，响应文本作为 text part，图片紧随其后。
        """
        reference_time = time.time() if reference_time is None else float(reference_time)
        messages: list[dict] = []
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
        raw_cutoff_seq = _raw_cognition_cutoff_seq(self._rounds, covered_seq)
        pending_old_rounds: list[FlowRound] = []

        def flush_old_rounds() -> None:
            if not pending_old_rounds:
                return
            messages.append({
                "role": "user",
                "content": _format_old_cycles_content(
                    pending_old_rounds,
                    reference_time=reference_time,
                ),
            })
            pending_old_rounds.clear()

        for rnd in self._rounds:
            if isinstance(rnd, RestartPair):
                flush_old_rounds()
                messages.extend(_restart_pair_messages(rnd))
                continue
            if rnd.seq <= covered_seq:
                continue
            if raw_cutoff_seq is None or rnd.seq < raw_cutoff_seq:
                pending_old_rounds.append(rnd)
                continue
            flush_old_rounds()
            _append_raw_round_messages(messages, rnd)
        flush_old_rounds()
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
                    "motive": rnd.motive,
                    "request_started_at": rnd.request_started_at,
                    "calls": [
                        {
                            "namespace": tc.namespace,
                            "name": tc.name,
                            "args": tc.args,
                            "call_id": tc.call_id,
                        }
                        for tc in rnd.calls
                    ],
                    "responses": [
                        {
                            "namespace": tr.namespace,
                            "name": tr.name,
                            "response": tr.response,
                            "call_id": tr.call_id,
                            "result_cdata": tr.result_cdata,
                        }
                        for tr in rnd.responses
                    ],
                    "memory_candidates": copy.deepcopy(rnd.memory_candidates),
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
        return data, timestamps

    def restore(self, data: list[dict], timestamps: list) -> None:
        """从序列化数据恢复。"""
        self._rounds = []
        self._compression_summary = None
        self._ready_compression_summaries = []
        self._next_seq = 1
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
            if entry.get("type") == "restart":
                st_raw = entry.get("startup_time")
                self._rounds.append(RestartPair(
                    shutdown_time=float(entry.get("shutdown_time", 0)),
                    startup_time=float(st_raw) if st_raw is not None else None,
                ))
                continue
            calls = [
                ToolCall(
                    namespace=str(c.get("namespace") or ""),
                    name=c.get("name", ""),
                    args=strip_legacy_motivation_fields(c.get("args", {}))[0],
                    call_id=c.get("call_id", ""),
                )
                for c in entry.get("calls", [])
            ]
            responses = [_restore_tool_response(r) for r in entry.get("responses", [])]
            ts_raw = timestamps[i] if i < len(timestamps) else None
            ts = float(ts_raw) if ts_raw is not None else None
            if calls or responses:
                seq = int(entry.get("seq") or self._next_seq)
                self._rounds.append(FlowRound(
                    seq=seq,
                    cognition=str(entry.get("cognition") or ""),
                    motive=str(entry.get("motive") or ""),
                    calls=calls,
                    responses=responses,
                    request_started_at=(
                        float(entry["request_started_at"])
                        if entry.get("request_started_at") is not None
                        else None
                    ),
                    timestamp=ts,
                    raw_response=str(entry.get("raw_response") or ""),
                    memory_candidates=[
                        dict(item)
                        for item in entry.get("memory_candidates", [])
                        if isinstance(item, dict)
                    ],
                ))
                self._next_seq = max(self._next_seq, seq + 1)
        self._ready_compression_summaries.sort(key=lambda item: item.coverage_end_seq)
        logger.info("[consciousness] 已恢复意识流: %d 轮", len(self._rounds))


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _raw_cognition_cutoff_seq(
    rounds: list[FlowRound | RestartPair],
    covered_seq: int,
) -> int | None:
    cognition_rounds = [
        rnd
        for rnd in rounds
        if isinstance(rnd, FlowRound)
        and rnd.seq > covered_seq
        and bool(rnd.cognition)
    ]
    if not cognition_rounds:
        return None
    return cognition_rounds[-RAW_COGNITION_ROUNDS].seq if len(cognition_rounds) >= RAW_COGNITION_ROUNDS else cognition_rounds[0].seq


def _append_raw_round_messages(messages: list[dict], rnd: FlowRound) -> None:
    if not rnd.calls:
        if rnd.responses:
            messages.append({
                "role": "user",
                "content": _format_action_response_content(rnd.responses),
            })
        return

    assistant_blocks: list[str] = []
    if rnd.cognition:
        assistant_blocks.append(_format_cognition_xml(rnd.cognition))
    assistant_blocks.append(_format_motive_xml(rnd.motive))
    assistant_blocks.append(_format_action_xml(rnd.calls))
    messages.append({
        "role": "assistant",
        "content": "\n".join(assistant_blocks),
    })
    if rnd.responses:
        messages.append({
            "role": "user",
            "content": _format_action_response_content(rnd.responses),
        })


def _format_old_cycles_content(
    rounds: list[FlowRound],
    *,
    reference_time: float,
) -> str | list:
    parts: list[dict] = []
    _append_text_content(parts, "<old_cycles>")
    for rnd in rounds:
        start_ago, end_ago = _flow_round_ago(rnd, reference_time=reference_time)
        _append_text_content(
            parts,
            f'\n  <cycle start_ago="{start_ago}" end_ago="{end_ago}">\n',
        )
        _append_text_content(parts, _format_motive_xml(rnd.motive) + "\n")
        _append_text_content(parts, _format_action_xml(rnd.calls) + "\n")
        _append_mixed_content(parts, _format_action_response_content(rnd.responses))
        _append_text_content(parts, "\n  </cycle>")
    _append_text_content(parts, "\n</old_cycles>")
    if all(part.get("type") == "text" for part in parts):
        return "".join(str(part.get("text") or "") for part in parts)
    return parts


def _append_text_content(parts: list[dict], text: str) -> None:
    if not text:
        return
    if parts and parts[-1].get("type") == "text":
        parts[-1]["text"] = str(parts[-1].get("text") or "") + text
        return
    parts.append({"type": "text", "text": text})


def _append_mixed_content(parts: list[dict], content: str | list) -> None:
    if isinstance(content, str):
        _append_text_content(parts, content)
        return
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            _append_text_content(parts, str(part.get("text") or ""))
        elif isinstance(part, dict):
            parts.append(copy.deepcopy(part))


def _flow_round_ago(rnd: FlowRound, *, reference_time: float) -> tuple[str, str]:
    end_at = float(rnd.timestamp) if rnd.timestamp is not None else reference_time
    start_at = (
        float(rnd.request_started_at)
        if rnd.request_started_at is not None
        else end_at
    )
    return (
        _format_compact_duration(reference_time - start_at),
        _format_compact_duration(reference_time - end_at),
    )


def _format_compact_duration(seconds_ago: float) -> str:
    total_seconds = max(0, int(seconds_ago))
    if total_seconds == 0:
        return "0s"

    remaining = total_seconds
    values: list[tuple[int, str]] = []
    for unit_seconds, suffix in ((86400, "d"), (3600, "h"), (60, "m"), (1, "s")):
        value, remaining = divmod(remaining, unit_seconds)
        if value:
            values.append((value, suffix))
    values = values[:2]
    return "".join(
        f"{value}{suffix}" if index == 0 else f"{value:02d}{suffix}"
        for index, (value, suffix) in enumerate(values)
    )

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
    payload = {"id": tool_call.call_id}
    if tool_call.namespace:
        payload["namespace"] = tool_call.namespace
    payload["name"] = tool_call.name
    payload["arguments"] = tool_call.args
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"


def _format_action_xml(tool_calls: list[ToolCall]) -> str:
    blocks = ["<action>"]
    blocks.extend(_format_tool_call_xml(tool_call) for tool_call in tool_calls)
    blocks.append("</action>")
    return "\n".join(blocks)


def _format_context_summary_xml(summary: CompressionSummary) -> str:
    return (
        "<summary>\n"
        f"{_escape_xml_text(summary.text)}"
        "\n</summary>"
    )


def _format_compression_task_xml(
    generated_at: str,
    previous_summary: str,
    rounds: list[FlowRound],
) -> str:
    blocks = [f'<compression_input generated_at="{_escape_xml_text(generated_at)}">']
    if previous_summary.strip():
        blocks.append(
            "<previous_summary>"
            f"{_escape_xml_text(previous_summary.strip())}"
            "</previous_summary>"
        )
    else:
        blocks.append("<previous_summary/>")

    for rnd in rounds:
        start_at, end_at = _compression_cycle_timestamps(
            rnd,
            generated_at=generated_at,
        )
        blocks.append(
            f'<cycle start_at="{_escape_xml_text(start_at)}" '
            f'end_at="{_escape_xml_text(end_at)}">'
        )
        blocks.append(_format_compression_motive_xml(rnd.motive))
        blocks.append(_format_compression_action_xml(rnd.calls))
        blocks.append(_format_compression_action_response_xml(rnd.responses))
        blocks.append("</cycle>")

    blocks.append("</compression_input>")
    return "\n".join(blocks)


def _compression_cycle_timestamps(
    rnd: FlowRound,
    *,
    generated_at: str,
) -> tuple[str, str]:
    if rnd.timestamp is not None:
        end_at = _format_os_timestamp(rnd.timestamp)
    elif rnd.request_started_at is not None:
        end_at = _format_os_timestamp(rnd.request_started_at)
    else:
        end_at = generated_at

    start_at = (
        _format_os_timestamp(rnd.request_started_at)
        if rnd.request_started_at is not None
        else end_at
    )
    return start_at, end_at


def _format_compression_motive_xml(motive: str) -> str:
    text = (motive or "").strip()
    if not text:
        return "<motive/>"
    return f"<motive>{_escape_xml_text(text)}</motive>"


def _format_compression_action_xml(tool_calls: list[ToolCall]) -> str:
    if not tool_calls:
        return "<action/>"
    blocks = ["<action>"]
    for tool_call in tool_calls:
        payload = {"id": tool_call.call_id}
        if tool_call.namespace:
            payload["namespace"] = tool_call.namespace
        payload["name"] = tool_call.name
        payload["arguments"] = tool_call.args
        blocks.append(
            "<tool_call>"
            f"{_escape_xml_text(json.dumps(payload, ensure_ascii=False))}"
            "</tool_call>"
        )
    blocks.append("</action>")
    return "\n".join(blocks)


def _format_compression_action_response_xml(
    tool_responses: list[ToolResponse],
) -> str:
    if not tool_responses:
        return "<action_response/>"
    blocks = ["<action_response>"]
    blocks.extend(_format_action_response_item_xml(response) for response in tool_responses)
    blocks.append("</action_response>")
    return "\n".join(blocks)


def _format_os_timestamp(timestamp: float | None = None) -> str:
    dt = (
        datetime.datetime.now().astimezone()
        if timestamp is None
        else datetime.datetime.fromtimestamp(timestamp).astimezone()
    )
    return dt.isoformat(timespec="seconds")


_SUMMARY_BLOCK_RE = re.compile(r"<summary\b[^>]*>(.*?)</summary>", re.DOTALL)
_COMPRESSION_OUTPUT_RE = re.compile(
    r"\A\s*"
    r"<analysis\b[^>]*>.*?</analysis>"
    r"\s*"
    r"<summary\b[^>]*>(?P<summary>.*?)</summary>"
    r"\s*\Z",
    re.DOTALL,
)


def extract_summary_block(text: str) -> str:
    """从压缩模型输出中提取真正注入上下文的 <summary> 内容。"""
    match = _SUMMARY_BLOCK_RE.search(text or "")
    if not match:
        return (text or "").strip()
    return match.group(1).strip()


def extract_structured_compression_summary(text: str) -> str:
    """严格提取压缩输出：必须是完整 analysis 块后接完整 summary 块。"""
    match = _COMPRESSION_OUTPUT_RE.fullmatch(text or "")
    if not match:
        return ""
    return match.group("summary").strip()


def _format_cognition_xml(cognition: str) -> str:
    return f"<cognition>{_escape_xml_text(cognition)}</cognition>"


def _format_motive_xml(motive: str) -> str:
    visible_motive = (motive or "").strip() or MISSING_MOTIVE_TEXT
    return f"<motive>{_escape_xml_text(visible_motive)}</motive>"


def _escape_xml_text(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _sanitize_xml_text(text: str) -> str:
    """Replace characters forbidden by XML 1.0 while preserving normal text."""
    return "".join(
        char
        if char in {"\t", "\n", "\r"}
        or 0x20 <= ord(char) <= 0xD7FF
        or 0xE000 <= ord(char) <= 0xFFFD
        or 0x10000 <= ord(char) <= 0x10FFFF
        else "\uFFFD"
        for char in str(text)
    )


def _format_cdata(text: str) -> str:
    safe = _sanitize_xml_text(text)
    return "<![CDATA[" + safe.replace("]]>", "]]]]><![CDATA[>") + "]]>"


def _format_action_response_content(tool_responses: list[ToolResponse]) -> str | list:
    if not any(tr.multimodal_parts for tr in tool_responses):
        return _format_action_response_xml(tool_responses)

    parts: list = [{"type": "text", "text": "<action_response>\n"}]
    for index, tool_response in enumerate(tool_responses):
        item_text = _format_action_response_item_xml(tool_response)
        if index > 0:
            item_text = "\n" + item_text
        parts.append({"type": "text", "text": item_text})
        parts.extend(_format_tool_response_image_parts(tool_response))
    parts.append({"type": "text", "text": "\n</action_response>"})
    return parts


def _format_action_response_xml(tool_responses: list[ToolResponse]) -> str:
    blocks = ["<action_response>"]
    blocks.extend(_format_action_response_item_xml(tr) for tr in tool_responses)
    blocks.append("</action_response>")
    return "\n".join(blocks)


def _format_action_response_item_xml(tool_response: ToolResponse) -> str:
    if is_aic_action_error_name(tool_response.name):
        return f"<feedback>{_escape_xml_text(_format_tool_feedback_text(tool_response))}</feedback>"

    payload = {"id": tool_response.call_id}
    if tool_response.namespace:
        payload["namespace"] = tool_response.namespace
    payload["name"] = tool_response.name
    payload["result"] = tool_response.response
    rendered = _sanitize_xml_text(json.dumps(payload, ensure_ascii=False))
    if tool_response.result_cdata:
        cdata = _format_cdata("\n" + rendered + "\n")
        return f"<result>{cdata}</result>"
    return f"<result>{rendered}</result>"


def _format_tool_feedback_text(tool_response: ToolResponse) -> str:
    response = tool_response.response
    detail: str
    if isinstance(response, dict):
        error = response.get("error") or response.get("message")
        if isinstance(error, str) and error.strip():
            detail = error.strip()
        else:
            detail = json.dumps(response, ensure_ascii=False)
    else:
        detail = str(response).strip()
    return f"{AIC_ACTION_ERROR_NAME}: {detail}"


def _format_tool_response_image_parts(tool_response: ToolResponse) -> list[dict]:
    image_parts: list[dict] = []
    for mp in tool_response.multimodal_parts:
        data = mp.get("data") if isinstance(mp, dict) else None
        if data is None:
            continue
        data_str: str = (
            data if isinstance(data, str)
            else base64.b64encode(data).decode()
        )
        data_url = make_data_url(data_str, str(mp.get("mime_type") or "image/jpeg"))
        if not data_url:
            continue
        image_parts.append({
            "type": "image_url",
            "image_url": {"url": data_url},
        })
    return image_parts


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
