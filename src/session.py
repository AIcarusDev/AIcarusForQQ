"""session.py — 会话管理

ChatSession: 每个会话（Web UI / QQ 群 / QQ 私聊）独立的上下文状态。
包含上下文消息管理、system prompt 构建、LLM 调用封装。
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from xml_builder import build_multimodal_content, format_chat_log_for_display
from prompt import SYSTEM_PROMPT, get_formatted_time_for_llm, build_tool_budget_prompt


@dataclass
class ChatSession:
    """每个会话独立的上下文状态。"""

    context_messages: list[dict] = field(default_factory=list)
    previous_cycle_json: dict | None = None
    remaining_cycles: int = 0

    # 以下字段在 init_session_globals() 时统一注入
    _max_context: int = 20
    _timezone: ZoneInfo | None = None
    _persona: str = ""
    _model_name: str = ""
    _qq_id: str = ""
    _qq_name: str = ""

    def add_to_context(self, entry: dict) -> None:
        self.context_messages.append(entry)
        while len(self.context_messages) > self._max_context:
            self.context_messages.pop(0)

    def build_chat_log_xml(self) -> "str | list":
        return build_multimodal_content(self.context_messages)

    def get_chat_log_display(self) -> str:
        """返回可读的 JSON 格式聊天记录，用于前端/日志展示。"""
        return format_chat_log_for_display(self.context_messages)

    def build_system_prompt(self, tool_budget: dict[str, dict] | None = None) -> str:
        """构建 system prompt，可选传入工具配额信息。

        tool_budget 结构见 prompt.build_tool_budget_prompt() 文档。
        """
        now = datetime.now(self._timezone)
        prev = (
            json.dumps(self.previous_cycle_json, ensure_ascii=False, indent=2)
            if self.previous_cycle_json
            else "null"
        )
        budget_text = build_tool_budget_prompt(tool_budget) if tool_budget else ""
        return SYSTEM_PROMPT.format(
            persona=self._persona,
            time=get_formatted_time_for_llm(now),
            model_name=self._model_name,
            number=self.remaining_cycles,
            previous_cycle_json=prev,
            tool_budget=budget_text,
            qq_id=self._qq_id,
            qq_name=self._qq_name,
        )


# ── 全局默认参数（由 app.py 启动时设置） ─────────────────

_session_defaults: dict = {}


def init_session_globals(
    *,
    max_context: int,
    timezone,
    persona: str,
    model_name: str,
) -> None:
    """由 app.py 在启动时或设置保存后调用，设置所有新/旧 session 的默认参数。"""
    _session_defaults.update(
        max_context=max_context,
        timezone=timezone,
        persona=persona,
        model_name=model_name,
    )
    # 同步更新已存在的所有 session
    for s in sessions.values():
        s._max_context = max_context
        s._timezone = timezone
        s._persona = persona
        s._model_name = model_name


def update_bot_info(qq_id: str, qq_name: str) -> None:
    """NapCat 连接并同步账号信息后调用，将真实 QQ ID 和昵称注入所有会话。"""
    _session_defaults["qq_id"] = qq_id
    _session_defaults["qq_name"] = qq_name
    for s in sessions.values():
        s._qq_id = qq_id
        s._qq_name = qq_name


def update_session_model_name(model_name: str) -> None:
    """切换模型时更新全局默认 model_name，并同步到已存在的 sessions。"""
    _session_defaults["model_name"] = model_name
    for s in sessions.values():
        s._model_name = model_name


def create_session() -> ChatSession:
    """创建新会话，自动应用全局默认参数。"""
    s = ChatSession()
    s._max_context = _session_defaults.get("max_context", 20)
    s._timezone = _session_defaults.get("timezone")
    s._persona = _session_defaults.get("persona", "")
    s._model_name = _session_defaults.get("model_name", "")
    s._qq_id = _session_defaults.get("qq_id", "")
    s._qq_name = _session_defaults.get("qq_name", "")
    return s


# ── 会话存储 ─────────────────────────────────────────────

sessions: dict[str, ChatSession] = {}


def get_or_create_session(key: str) -> ChatSession:
    """获取已有会话，不存在则创建。"""
    if key not in sessions:
        sessions[key] = create_session()
    return sessions[key]


def reset_session(key: str) -> ChatSession:
    """重置指定会话。"""
    sessions[key] = create_session()
    return sessions[key]


# ── 辅助函数 ─────────────────────────────────────────────

def extract_bot_messages(result: dict) -> list[str]:
    """从模型输出中提取每条消息的文本内容。"""
    messages = []
    decision = result.get("decision") or {}
    for msg in decision.get("send_messages") or []:
        parts = []
        for seg in msg.get("segments", []):
            cmd = seg.get("command")
            params = seg.get("params", {})
            if cmd == "text":
                parts.append(params.get("content", ""))
            elif cmd == "at":
                parts.append(f"@{params.get('user_id', '')}")
        text = "".join(parts)
        if text:
            messages.append(text)
    return messages
