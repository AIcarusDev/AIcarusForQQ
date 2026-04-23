"""session.py — 会话管理

ChatSession: 每个会话（Web UI / QQ 群 / QQ 私聊）独立的上下文状态。
包含上下文消息管理、system prompt 构建、LLM 调用封装。
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from .prompt.xml_builder import build_multimodal_content, format_chat_log_for_display, _format_relative_time
from .prompt.prompt import SYSTEM_PROMPT, get_formatted_time_for_llm, build_function_tools_prompt, build_guardian_prompt
from .prompt.activity_log import build_activity_log_xml
from .prompt.memory import build_active_memory_xml

logger = logging.getLogger("AICQ.llm")


@dataclass
class ChatSession:
    """每个会话独立的上下文状态。"""

    context_messages: list[dict] = field(default_factory=list)
    # wait 循环状态：由 loop_control.wait 分支设置，用于提前唤醒
    wait_event: asyncio.Event | None = None
    wait_early_trigger: str | None = None
    # 打字发送期间（lock 占用但 wait_event 尚未创建）到达的消息所能触发的最强 early_trigger 类型
    # 取值：None | "new_message" | "mentioned"，进入 wait 分支时消费后清空
    pending_early_trigger: str | None = None

    # 会话元信息（group/private/web）
    conv_type: str = ""     # "group" | "private" | "" (web)
    conv_id: str = ""       # 群号 或 对方QQ号
    conv_name: str = ""     # 群名 或 对方昵称
    conv_member_count: int = 0  # 群总人数（group 时有效）
    pending_error_logger: str = ""  # 下一轮 system prompt 中 error_logger 的内容，消费后清空
    pending_is_tip: str = ""          # IS 中断后注入到下一轮 <tip> 中的提示，消费后清空
    unread_count: int = 0             # 本会话尚未被 bot "看到" 的用户消息计数
    # 本轮 LLM 调用开始时发送给模型的消息 ID 集合（在 prepare_chat_log_with_unread 时设置）
    # short_wait 以此为基准，捕获 LLM 思考期间 + 等待期间所有未见消息
    turn_start_seen_ids: set = field(default_factory=set)

    # 通过 get_tools 激活的潜伏工具名称集合
    # 在 continue 循环间保持：下次 build_tools 后自动预激活其中的工具
    activated_latent_tools: set = field(default_factory=set)

    # 以下字段在 init_session_globals() 时统一注入
    _max_context: int = 20
    _timezone: ZoneInfo | None = None
    _persona: str = ""
    _instructions: str = ""
    _model_name: str = ""
    _qq_id: str = ""
    _qq_name: str = ""
    _qq_card: str = ""   # Bot 在当前群的群名片（群聊会话专属）
    _guardian_name: str = ""
    _guardian_id: str = ""

    # 引用预取缓存：key=message_id, value=简化 entry dict（由 prefetch_quoted_messages 填充）
    quoted_extra: dict = field(default_factory=dict)

    # FTS5 记忆召回：每轮对话前由 prepare_memory_recall() 填充，供 build_system_prompt 使用
    recalled_memories: list = field(default_factory=list)
    # Neo-Davidsonian 事件召回（含角色边）：与 recalled_memories 并行，渲染到 <recent_events>
    recalled_events: list = field(default_factory=list)
    # 本轮事件涉及的 qq_id → nickname 缓存，由 prepare_memory_recall 预取
    _nick_cache: dict = field(default_factory=dict)

    # Watcher（窥屏意识）相关字段
    watcher_task: asyncio.Task | None = None
    watcher_active: bool = False
    watcher_nudge: dict | None = None
    watcher_break_time: float = 0.0
    watcher_break_reason: str = ""
    watcher_last_cycle: dict | None = None
    watcher_last_cycle_time: float = 0.0

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

    @property
    def last_sender_id(self) -> str:
        """最近一条 user 消息的 sender_id（用于记忆 subject 推导）。"""
        for m in reversed(self.context_messages):
            if m.get("role") == "user":
                return str(m.get("sender_id", ""))
        return ""

    async def prepare_memory_recall(self) -> None:
        """执行 FTS5 记忆召回，结果存入 self.recalled_memories。

        在 asyncio.to_thread(call_model_and_process) 之前调用，确保
        build_system_prompt()（同步）能直接读取已计算好的召回结果。
        """
        import app_state
        from .prompt.memory import recall_memories

        last_user_text = ""
        for m in reversed(self.context_messages):
            if m.get("role") == "user":
                last_user_text = str(m.get("content", ""))
                break

        # Phase 2：计算 recall_scope 用于隔离场景记忆
        if self.conv_type == "group":
            context_scope = f"group:qq_{self.conv_id}"
        elif self.conv_type == "private":
            context_scope = f"private:qq_{self.conv_id}"
        else:
            context_scope = ""

        memory_cfg = app_state.config.get("memory", {}) if hasattr(app_state, "config") else {}
        self.recalled_memories = await recall_memories(
            last_user_text,
            sender_id=self.last_sender_id,
            config=memory_cfg,
            context_scope=context_scope,
        )
        # 艾宾浩斯强化：被召回命中的记忆 confidence +0.05（上限由 update_triple_confidence 保证）
        if self.recalled_memories:
            from database import update_triple_confidence
            ids = [r["id"] for r in self.recalled_memories if "id" in r]
            if ids:
                await update_triple_confidence(ids, delta=0.05)

        # ── Neo-Davidsonian 事件召回 ─────────────────────────────────────
        # 不依赖 FTS5；按 sender 实体 + meta + 最近 episodic 取前 N 条
        from database import load_events_for_recall, get_nicknames_by_qq_ids
        events_cfg = (memory_cfg.get("events", {}) or {}) if isinstance(memory_cfg, dict) else {}
        events_limit = int(events_cfg.get("recall_limit", 6))
        sender_entity = f"User:qq_{self.last_sender_id}" if self.last_sender_id else ""
        try:
            self.recalled_events = await load_events_for_recall(
                sender_entity=sender_entity,
                context_scope=context_scope,
                limit=events_limit,
            )
        except Exception:
            # 事件层为新增能力，召回失败不应阻塞主流程
            import logging
            logging.getLogger("AICQ.session").warning(
                "load_events_for_recall 失败，本轮跳过事件召回", exc_info=True,
            )
            self.recalled_events = []

        # 预取本轮所有 User:qq_xxx 的昵称,缓存供 build_system_prompt 同步使用
        qq_ids: set[str] = set()
        for ev in self.recalled_events:
            for r in ev.get("roles") or []:
                ent = r.get("entity") or ""
                if ent.startswith("User:qq_"):
                    qq_ids.add(ent[len("User:qq_"):])
        if self.last_sender_id:
            qq_ids.add(str(self.last_sender_id))
        try:
            self._nick_cache = await get_nicknames_by_qq_ids(list(qq_ids)) if qq_ids else {}
        except Exception:
            self._nick_cache = {}

    def build_chat_log_xml(self) -> "str | list":
        return build_multimodal_content(self.context_messages, self._get_conv_meta(), quoted_extra=self.quoted_extra)

    def get_chat_log_display(self) -> str:
        """返回可读的 XML 格式聊天记录，用于前端/日志展示。"""
        return format_chat_log_for_display(self.context_messages, self._get_conv_meta(), quoted_extra=self.quoted_extra)

    def build_system_prompt(
        self,
        activated_names: list[str] | None = None,
        latent_names: list[str] | None = None,
    ) -> str:
        """构建 system prompt，可选传入已激活工具和潜伏工具名称列表。"""
        now = datetime.now(self._timezone)
        prev = (
            json.dumps(get_bot_previous_cycle(), ensure_ascii=False, indent=2)
            if get_bot_previous_cycle()
            else "null"
        )
        tool_calls = get_bot_previous_tool_calls()
        prev_tools = (
            json.dumps(
                _truncate_tool_calls_for_prompt(tool_calls),
                ensure_ascii=False,
                indent=2,
            )
            if tool_calls
            else "null"
        )
        budget_text = build_function_tools_prompt(
            activated_names=activated_names or [],
            latent_names=latent_names or [],
        )
        cycle_time = get_bot_previous_cycle_time()
        prev_cycle_time_attr = (
            f' time="{_format_relative_time(cycle_time)}"'
            if cycle_time and get_bot_previous_cycle()
            else ""
        )
        _prev_cycle_tip = self.pending_is_tip
        self.pending_is_tip = ""
        if self.watcher_nudge:
            wn = self.watcher_nudge
            self.watcher_nudge = None
            prev = json.dumps(wn["result"], ensure_ascii=False, indent=2)
            prev_cycle_time_attr = f' time="{_format_relative_time(wn["time_iso"])}"'
            prev_tools = "null"  # 窥屏模式无工具调用
        return SYSTEM_PROMPT.format(
            persona=self._persona,
            instructions=self._instructions,
            time=get_formatted_time_for_llm(now),
            model_name=self._model_name,
            previous_cycle_json=prev,
            previous_cycle_time=prev_cycle_time_attr,
            previous_tools_used=prev_tools,
            previous_cycle_tip=_prev_cycle_tip,
            function_tools=budget_text,

            qq_name=self._qq_name,
            qq_id=self._qq_id,
            guardian=build_guardian_prompt(self._guardian_name, self._guardian_id),
            activity_log=build_activity_log_xml(),
            active_memory=build_active_memory_xml(
                now,
                recalled=self.recalled_memories or None,
                recalled_events=self.recalled_events or None,
                sender_entity=(f"User:qq_{self.last_sender_id}" if self.last_sender_id else ""),
                nickname_map=self._build_nickname_map(),
            ),
        )

    def _build_nickname_map(self) -> dict[str, str]:
        """返回 prepare_memory_recall 阶段预取的昵称缓存。

        缓存未命中(如 watcher 或冷启动直接渲染)则返回空 dict,
        渲染端会回退到纯 qq_id 显示,不影响主流程。
        """
        return getattr(self, "_nick_cache", {}) or {}


# ── 全局默认参数（由 app.py 启动时设置） ─────────────────

_session_defaults: dict = {}

# ── bot 意识流全局状态 ──────────────────────────────
# 严格单一意识流：不管哪个会话触发，这里总是保存 bot 最后一次的输出。

_bot_previous_cycle: dict | None = None
_bot_previous_tool_calls: list | None = None
_bot_previous_cycle_time: str | None = None  # ISO 格式 UTC 时间戳

# 各工具 result 在 previous_tools_used 中的最大字符数（超出部分截断）
# DB 中保留完整数据，截断仅在渲染 prompt 时生效
# 具体数值由各工具模块的 RESULT_MAX_CHARS 字段声明，此处仅保留全局默认值。
_DEFAULT_TOOL_RESULT_MAX_CHARS = 2000


def _truncate_tool_calls_for_prompt(tool_calls: list) -> list:
    """按工具模块声明的 RESULT_MAX_CHARS / summarize_result 处理 result。
    仅影响 prompt 渲染，不改原始数据。"""
    # 懒加载，避免循环导入
    from tools import _tool_modules
    _mod_map: dict = {m.DECLARATION.get("name", ""): m for m in _tool_modules}

    out = []
    for entry in tool_calls:
        fn = entry.get("function", "")
        mod = _mod_map.get(fn)

        # 优先 summarize_result 自定义摘要
        summarize_fn = getattr(mod, "summarize_result", None) if mod else None
        if callable(summarize_fn):
            trimmed = dict(entry)
            trimmed["result"] = summarize_fn(entry)
            out.append(trimmed)
            continue

        max_chars: int = (
            getattr(mod, "RESULT_MAX_CHARS", _DEFAULT_TOOL_RESULT_MAX_CHARS)
            if mod else _DEFAULT_TOOL_RESULT_MAX_CHARS
        )

        if max_chars < 0:
            # 整条记录从 prompt 中移除
            continue

        if max_chars == 0:
            # 保留函数名+参数，丢弃 result 字段
            trimmed = dict(entry)
            trimmed.pop("result", None)
            out.append(trimmed)
            continue

        # > 0：保留并按字数截断
        result_str = json.dumps(entry.get("result"), ensure_ascii=False)
        if len(result_str) > max_chars:
            trimmed = dict(entry)
            trimmed["result"] = f"{result_str[:max_chars]}... [后面忘了，原始长度大概 {len(result_str)} 字符这样]"
            out.append(trimmed)
        else:
            out.append(entry)
    return out


def get_bot_previous_cycle() -> dict | None:
    """[全局] 返回 bot 最近一轮输出，重启后由 startup 从 DB 恢复。"""
    return _bot_previous_cycle


def set_bot_previous_cycle(data: dict | None) -> None:
    """[全局] 更新 bot 最近一轮输出。"""
    global _bot_previous_cycle
    _bot_previous_cycle = data


def get_bot_previous_cycle_time() -> str | None:
    """[全局] 返回 bot 最近一轮输出的 ISO 时间戳，重启后由 startup 从 DB 恢复。"""
    return _bot_previous_cycle_time


def set_bot_previous_cycle_time(iso_ts: str | None) -> None:
    """[全局] 更新 bot 最近一轮输出的时间戳。"""
    global _bot_previous_cycle_time
    _bot_previous_cycle_time = iso_ts


def get_bot_previous_tool_calls() -> list | None:
    """[全局] 返回 bot 最近一轮的工具调用记录，重启后由 startup 从 DB 恢复。"""
    return _bot_previous_tool_calls


def set_bot_previous_tool_calls(data: list | None) -> None:
    """[全局] 更新 bot 最近一轮的工具调用记录。"""
    global _bot_previous_tool_calls
    _bot_previous_tool_calls = data




def init_session_globals(
    *,
    max_context: int,
    timezone,
    persona: str,
    instructions: str = "",
    model_name: str,
    guardian_name: str = "",
    guardian_id: str = "",
) -> None:
    """由 app.py 在启动时或设置保存后调用，设置所有新/旧 session 的默认参数。"""
    _session_defaults.update(
        max_context=max_context,
        timezone=timezone,
        persona=persona,
        instructions=instructions,
        model_name=model_name,
        guardian_name=guardian_name,
        guardian_id=guardian_id,
    )
    # 同步更新已存在的所有 session
    for s in sessions.values():
        s._max_context = max_context
        s._timezone = timezone
        s._persona = persona
        s._instructions = instructions
        s._model_name = model_name
        s._guardian_name = guardian_name
        s._guardian_id = guardian_id


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
    s._instructions = _session_defaults.get("instructions", "")
    s._model_name = _session_defaults.get("model_name", "")
    s._qq_id = _session_defaults.get("qq_id", "")
    s._qq_name = _session_defaults.get("qq_name", "")
    s._guardian_name = _session_defaults.get("guardian_name", "")
    s._guardian_id = _session_defaults.get("guardian_id", "")
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

def extract_bot_messages(result: dict) -> list[dict]:
    """从模型输出中提取每条消息的文本内容和结构化内容段。

    返回列表元素格式:
      {"text": "...", "content_segments": [{"type": "mention", ...}, ...]}
    """
    messages = []
    decision = result.get("decision") or {}
    for msg in decision.get("send_messages") or []:
        text_parts = []
        content_segments = []
        for seg in msg.get("segments", []):
            cmd = seg.get("command")
            params = seg.get("params", {})
            if cmd == "text":
                t = params.get("content", "")
                text_parts.append(t)
                if t:
                    content_segments.append({"type": "text", "text": t})
            elif cmd == "at":
                uid = str(params.get("user_id", ""))
                text_parts.append(f"@{uid}")
                content_segments.append({"type": "mention", "uid": uid, "display": f"@{uid}"})
            elif cmd == "sticker":
                sticker_id = params.get("sticker_id", "")
                text_parts.append("[动画表情]")
                content_segments.append({"type": "sticker", "sticker_id": sticker_id})
        if text := "".join(text_parts):
            has_sticker = any(s.get("type") == "sticker" for s in content_segments)
            has_text = any(s.get("type") == "text" for s in content_segments)
            content_type = "sticker" if has_sticker and not has_text else "text"
            messages.append({"text": text, "content_segments": content_segments, "content_type": content_type})
    return messages



