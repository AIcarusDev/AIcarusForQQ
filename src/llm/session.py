"""session.py — 会话管理

ChatSession: 每个会话（Web UI / QQ 群 / QQ 私聊）独立的上下文状态。
包含上下文消息管理、system prompt 构建、LLM 调用封装。
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from .prompt.xml_builder import build_multimodal_content, format_chat_log_for_display
from .prompt.prompt import SYSTEM_PROMPT, get_formatted_time_for_llm, build_function_tools_prompt, build_guardian_prompt
from .prompt.goals import build_active_goals_xml

logger = logging.getLogger("AICQ.llm.session")


@dataclass
class ChatSession:
    """每个会话独立的上下文状态。"""

    context_messages: list[dict] = field(default_factory=list)
    # wait 循环状态：由 loop_control.wait 分支设置，用于提前唤醒
    wait_event: asyncio.Event | None = None
    wait_early_trigger: dict | None = None
    # 记录实际触发 early_trigger 的会话 conversation_id（global scope 时为其他会话，session scope 时为 None）
    wait_trigger_from: str | None = None
    # 打字发送期间（lock 占用但 wait_event 尚未创建）到达的消息所能触发的最强 early_trigger 条件
    # 取值：None | "any_message" | "mentioned"，进入 wait 分支时消费后清空
    pending_early_trigger: str | None = None

    # 会话元信息（group/private/web）
    conv_type: str = ""     # "group" | "private" | "" (web)
    conv_id: str = ""       # 群号 或 对方QQ号
    conv_name: str = ""     # 群名 或 对方昵称
    conv_member_count: int = 0  # 群总人数（group 时有效）
    unread_count: int = 0             # 本会话尚未被 bot "看到" 的用户消息计数
    _unread_message_ids: set[str] = field(default_factory=set)

    # 以下字段在 init_session_globals() 时统一注入
    _max_context: int = 10
    _timezone: ZoneInfo | None = None
    _persona: str = ""
    _model_name: str = ""
    _qq_id: str = ""
    _qq_name: str = ""
    _qq_card: str = ""   # Bot 在当前群的群名片（群聊会话专属）
    _guardian_name: str = ""
    _guardian_id: str = ""
    _style_prompt: str = ""
    _social_tips_private: str = ""
    _social_tips_group: str = ""

    # 自然醒事件：sleep 工具持有，被外部 mention/激活 set 后立即返回。
    sleep_wake_event: asyncio.Event | None = None
    # 触发自然醒的来源会话 key（"被 X 群 @ 唤醒" 这类信息）。
    sleep_wake_from: str | None = None
    # sleep handler 启动前若已有 mention 到来，先记在这里，handler 启动时立刻消费。
    sleep_pending_wake: bool = False
    last_wake_reason: str = ""

    # 引用预取缓存：key=message_id, value=简化 entry dict（由 prefetch_quoted_messages 填充）
    quoted_extra: dict = field(default_factory=dict)

    # 聊天窗口视口（scroll_chat_log 工具状态）
    # mode="live"   → 渲染 context_messages（最新窗口，默认）
    # mode="history" → 从数据库按 top_db_id 锚点向上渲染 page_size 条历史消息
    # 视口生命周期与会话窗口同寿：bot 离开本会话（shift 走 / 被其它会话抢焦点）后由 llm_core 自动重置。
    chat_window_view: dict = field(
        default_factory=lambda: {"mode": "live", "top_db_id": None, "page_size": 10}
    )

    def is_browsing_history(self) -> bool:
        """当前是否处于浏览历史聊天记录的状态。"""
        return self.chat_window_view.get("mode") == "history"

    def reset_chat_window_view(self) -> None:
        """将聊天窗口视口重置回 live 模式（最新窗口）。"""
        self.chat_window_view = {"mode": "live", "top_db_id": None, "page_size": 10}

    def mark_unread_message(self, message_id: str | None) -> None:
        """记录一条当前会话尚未被 bot 看到的消息。"""
        mid = str(message_id or "").strip()
        if mid:
            self._unread_message_ids.add(mid)
            self.unread_count = len(self._unread_message_ids)
            return
        self.unread_count += 1

    def clear_unread_messages(self) -> None:
        """当前会话已回到 live 并展示最新窗口，清空未读。"""
        self._unread_message_ids.clear()
        self.unread_count = 0

    def consume_visible_unread_messages(self, visible_messages: list[dict]) -> int:
        """将当前 history 窗口里已经展示给 bot 的未读消息从计数中扣除。"""
        if not self._unread_message_ids:
            return self.unread_count

        visible_ids = {
            str(msg.get("message_id", "")).strip()
            for msg in visible_messages
            if str(msg.get("message_id", "")).strip()
        }
        if not visible_ids:
            return self.unread_count

        self._unread_message_ids.difference_update(visible_ids)
        self.unread_count = len(self._unread_message_ids)
        return self.unread_count

    def set_conversation_meta(self, conv_type: str, conv_id: str, conv_name: str = "", member_count: int = 0) -> None:
        """设置会话元信息（首次消息到达或群名同步时调用）。"""
        self.conv_type = conv_type
        self.conv_id = conv_id
        if conv_name:
            self.conv_name = conv_name
        if member_count:
            self.conv_member_count = member_count

    def add_to_context(self, entry: dict) -> None:
        new_list = self.context_messages.copy()
        new_list.append(entry)
        if len(new_list) > self._max_context:
            new_list = new_list[-self._max_context:]
        # 原子赋值，避免后台 LLM 线程迭代时发生 RuntimeError
        self.context_messages = new_list

    def mark_message_recalled(self, message_id: str, operator_name: str, timestamp: str) -> bool:
        """将指定消息原地替换为撤回通知条目，返回是否找到并修改。"""
        new_list = self.context_messages.copy()
        found = False
        for i, msg in enumerate(new_list):
            if str(msg.get("message_id", "")) == message_id:
                new_list[i] = {
                    "role": "note",
                    "timestamp": timestamp,
                    "content": f"{operator_name}撤回了一条消息",
                    "content_type": "recall",
                    "message_id": message_id,  # 保留 id 供 _build_quote_xml 识别，但不渲染在 XML 里
                }
                found = True
                break
        if found:
            # 原子赋值，避免与后台 LLM 线程的并发迭代冲突
            self.context_messages = new_list
        return found

    def _get_conv_meta(self) -> dict:
        """获取当前会话的元信息字典。"""
        return {
            "type": self.conv_type,
            "id": self.conv_id,
            "name": self.conv_name,
            "member_count": self.conv_member_count,
            "bot_id": self._qq_id,
            "bot_name": self._qq_name,
            "bot_card": self._qq_card,
        }

    def build_chat_log_xml(self) -> "str | list":
        return build_multimodal_content(self.context_messages, self._get_conv_meta(), quoted_extra=self.quoted_extra)

    def get_chat_log_display(self) -> str:
        """返回可读的 XML 格式聊天记录，用于前端/日志展示。"""
        return format_chat_log_for_display(self.context_messages, self._get_conv_meta(), quoted_extra=self.quoted_extra)

    def get_platform_name(self) -> str:
        """返回当前会话所在的平台名称。"""
        if self.conv_type == "" or self.conv_id == "web_user":
            return "Web"
        return "QQ"

    def get_social_tips(self) -> str:
        """按会话类型返回对应的 social tips 文案。"""
        if self.conv_type == "group":
            return self._social_tips_group
        return self._social_tips_private

    @property
    def last_sender_id(self) -> str:
        """最近一条 user 消息的 sender_id（用于记忆 subject 推导）。"""
        for m in reversed(self.context_messages):
            if m.get("role") == "user":
                return str(m.get("sender_id", ""))
        return ""

    def build_system_prompt(
        self,
        activated_names: list[str] | None = None,
        latent_names: list[str] | None = None,
    ) -> str:
        """构建 system prompt，可选传入已激活工具和潜伏工具名称列表。"""
        now = datetime.now(self._timezone)
        budget_text = build_function_tools_prompt(
            activated_names=activated_names or [],
            latent_names=latent_names or [],
        )
        return SYSTEM_PROMPT.format(
            persona=self._persona,
            platform=self.get_platform_name(),
            time=get_formatted_time_for_llm(now),
            model_name=self._model_name,
            function_tools=budget_text,
            qq_name=self._qq_name,
            qq_id=self._qq_id,
            guardian=build_guardian_prompt(self._guardian_name, self._guardian_id),
            goals=build_active_goals_xml(now),
        )


# ── 全局默认参数（由 app.py 启动时设置） ─────────────────

_session_defaults: dict = {}




def init_session_globals(
    *,
    max_context: int,
    timezone,
    persona: str,
    model_name: str,
    guardian_name: str = "",
    guardian_id: str = "",
    style_prompt: str | None = None,
    social_tips_private: str | None = None,
    social_tips_group: str | None = None,
) -> None:
    """由 app.py 在启动时或设置保存后调用，设置所有新/旧 session 的默认参数。"""
    updates = dict(
        max_context=max_context,
        timezone=timezone,
        persona=persona,
        model_name=model_name,
        guardian_name=guardian_name,
        guardian_id=guardian_id,
    )
    if style_prompt is not None:
        updates["style_prompt"] = style_prompt
    if social_tips_private is not None:
        updates["social_tips_private"] = social_tips_private
    if social_tips_group is not None:
        updates["social_tips_group"] = social_tips_group

    _session_defaults.update(updates)

    # 同步更新已存在的所有 session
    for s in sessions.values():
        s._max_context = max_context
        s._timezone = timezone
        s._persona = persona
        s._model_name = model_name
        s._guardian_name = guardian_name
        s._guardian_id = guardian_id
        if style_prompt is not None:
            s._style_prompt = style_prompt
        if social_tips_private is not None:
            s._social_tips_private = social_tips_private
        if social_tips_group is not None:
            s._social_tips_group = social_tips_group


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
    s._max_context = _session_defaults.get("max_context", 10)
    s._timezone = _session_defaults.get("timezone")
    s._persona = _session_defaults.get("persona", "")
    s._model_name = _session_defaults.get("model_name", "")
    s._qq_id = _session_defaults.get("qq_id", "")
    s._qq_name = _session_defaults.get("qq_name", "")
    s._guardian_name = _session_defaults.get("guardian_name", "")
    s._guardian_id = _session_defaults.get("guardian_id", "")
    s._style_prompt = _session_defaults.get("style_prompt", "")
    s._social_tips_private = _session_defaults.get("social_tips_private", "")
    s._social_tips_group = _session_defaults.get("social_tips_group", "")
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


