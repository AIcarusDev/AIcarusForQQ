"""flow.py — 机器人意识流（Consciousness Flow）

provider 无关的工具调用历史，记录机器人跨激活、跨 provider 切换的 function calling 状态。
机器人的意识 ≠ 使用的哪个模型；切换 provider 不应清空意识流。

数据模型：
  FlowRound  — 一轮推理循环，包含若干工具调用及对应的执行结果
  ToolCall   — 模型发出的一次工具调用请求（name / args / call_id）
  ToolResponse — 工具返回的结果（name / response / call_id / timestamp）

ConsciousnessFlow 提供：
  - append_round / prune / clear
  - to_gemini_contents(now)   → Gemini SDK Content 列表
  - to_openai_messages()      → OpenAI messages 列表
  - dump() / restore()        → JSON 持久化
"""

from __future__ import annotations

import base64
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
    call_id: str = ""       # Gemini fc.id / OpenAI tc.id
    thought_signature: bytes | None = None  # Gemini thinking model 返回的签名，回传历史时必须原样附回


@dataclass
class ToolResponse:
    """工具执行结果（已经过 _apply_result_limits 处理）。"""
    name: str
    response: object        # JSON-serializable，已截断
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


# ── ConsciousnessFlow ─────────────────────────────────────────────────────────

class ConsciousnessFlow:
    """provider 无关的机器人意识流。

    只存储工具调用历史（calls + responses）。
    用户消息（context_messages）和 system prompt 不属于意识流，
    由各 adapter 在每次调用时单独传入。
    """

    def __init__(self) -> None:
        self._rounds: list[FlowRound] = []

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

    @property
    def round_count(self) -> int:
        return len(self._rounds)

    # ── Gemini 格式转换 ───────────────────────────────────────────────────────

    def to_gemini_contents(
        self,
        now: float | None = None,
        native_multimodal_fn_response: bool = True,
    ) -> list:
        """转换为 Gemini SDK Content 列表（不含 user message，由 adapter 拼在最前面）。

        每轮产生两个 Content：
          Content(role="model", parts=[FunctionCall...])
          Content(role="user",  parts=[FunctionResponse...])
        并对工具结果注入 _ago 相对时间字段。

        native_multimodal_fn_response:
          True  （默认，Gemini 3.x）— 图片嵌入 FunctionResponse.parts（FunctionResponseBlob）
          False （Gemini 2.5）      — FunctionResponse 只含 JSON，图片作为独立 inline_data Part
                                     紧随其后追加到同一 Content，避免 2.5 的 400 错误
        """
        from google.genai import types

        if now is None:
            now = time.time()

        # 无 thought_signature 的历史条目（跨 provider 迁移 / 修复前旧数据）
        # 用 Google 官方 bypass 值跳过验证，避免 400 INVALID_ARGUMENT
        _BYPASS_SIGNATURE = b"context_engineering_is_the_way_to_go"

        contents = []
        for rnd in self._rounds:
            # model 侧：function calls
            call_parts = []
            for tc in rnd.calls:
                fc_kwargs: dict = {"name": tc.name, "args": tc.args}
                if tc.call_id:
                    fc_kwargs["id"] = tc.call_id
                call_parts.append(types.Part(
                    function_call=types.FunctionCall(**fc_kwargs),
                    thought_signature=tc.thought_signature if tc.thought_signature is not None else _BYPASS_SIGNATURE,
                ))
            if call_parts:
                contents.append(types.Content(role="model", parts=call_parts))

            # user 侧：function responses
            resp_parts = []
            for tr in rnd.responses:
                resp = tr.response
                if isinstance(resp, dict) and rnd.timestamp is not None:
                    ago_str = _format_relative_time(now - rnd.timestamp)
                    resp = {**resp, "_ago": ago_str}

                if native_multimodal_fn_response and tr.multimodal_parts:
                    # Gemini 3.x：图片嵌入 FunctionResponse.parts
                    multimodal_extras = [
                        types.FunctionResponsePart(
                            inline_data=types.FunctionResponseBlob(
                                mime_type=mp["mime_type"],
                                display_name=mp["display_name"],
                                data=mp["data"],
                            )
                        )
                        for mp in tr.multimodal_parts
                    ]
                    fr_kwargs: dict = {
                        "name": tr.name,
                        "response": resp,
                        "parts": multimodal_extras,
                    }
                    if tr.call_id:
                        fr_kwargs["id"] = tr.call_id
                    resp_parts.append(
                        types.Part(function_response=types.FunctionResponse(**fr_kwargs))
                    )
                else:
                    # Gemini 2.5 / 无多模态附件：FunctionResponse 只含 JSON，
                    # 图片作为独立的 inline_data Part 追加到同一 Content
                    fr_kwargs = {"name": tr.name, "response": {"result": resp}}
                    if tr.call_id:
                        fr_kwargs["id"] = tr.call_id
                    resp_parts.append(
                        types.Part(function_response=types.FunctionResponse(**fr_kwargs))
                    )
                    for mp in tr.multimodal_parts:
                        data = mp["data"]
                        if isinstance(data, str):
                            data = base64.b64decode(data)
                        resp_parts.append(
                            types.Part(inline_data=types.Blob(
                                mime_type=mp["mime_type"],
                                data=data,
                            ))
                        )

            if resp_parts:
                contents.append(types.Content(role="user", parts=resp_parts))

        return contents

    # ── OpenAI 格式转换 ───────────────────────────────────────────────────────

    def to_openai_messages(self) -> list[dict]:
        """转换为 OpenAI messages 格式（不含 system / 第一条 user）。

        每轮产生：
          {"role": "assistant", "tool_calls": [...]}
          N × {"role": "tool", "content": json_str 或 [{type:text}+{type:image_url}...]}

        当 ToolResponse 含有 multimodal_parts 时，content 使用数组格式，
        供支持原生多模态工具响应的模型（如 gpt-4o、Gemini-via-OpenAI-compat）消费。
        """
        messages = []
        for rnd in self._rounds:
            if not rnd.calls:
                continue
            messages.append({
                "role": "assistant",
                "content": None,
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
                if tr.multimodal_parts:
                    # 支持原生多模态工具响应的模型（gpt-4o 系列等）
                    # content 改为数组：先放 JSON 文本，再附图片
                    content: object = [{"type": "text", "text": text_content}]
                    for mp in tr.multimodal_parts:
                        # data 已由工具层 base64 编码（非 Gemini 路径统一 base64）
                        data_str: str = (
                            mp["data"] if isinstance(mp["data"], str)
                            else base64.b64encode(mp["data"]).decode()
                        )
                        content.append({  # type: ignore[union-attr]
                            "type": "image_url",
                            "image_url": {"url": f"data:{mp['mime_type']};base64,{data_str}"},
                        })
                else:
                    content = text_content
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr.call_id,
                    "content": content,
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
            data.append({
                "calls": [
                    {
                        "name": tc.name,
                        "args": tc.args,
                        "call_id": tc.call_id,
                        "thought_signature": base64.b64encode(tc.thought_signature).decode() if tc.thought_signature else None,
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
            calls = [
                ToolCall(
                    name=c.get("name", ""),
                    args=c.get("args", {}),
                    call_id=c.get("call_id", ""),
                    thought_signature=base64.b64decode(c["thought_signature"]) if c.get("thought_signature") else None,
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
