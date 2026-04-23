"""memory_archiver.py — 后台记忆自动归档（Phase 3D）

每轮对话完成后，fire-and-forget 调度一次轻量 one_shot_json 推理，
自动从本轮对话中提取关于用户的记忆三元组并写入 MemoryTriples。
无需模型主动调用 write_memory 工具。

触发位置：napcat_handler._consciousness_task() 在 save_bot_turn 之后
"""

import asyncio
import logging

logger = logging.getLogger("AICQ.archiver")

_SEM = asyncio.Semaphore(2)  # 限制并发 archiver LLM 调用数，防止 token 爆炸和速率超限

_DEFAULT_CONTEXT_TURNS = 5   # 最近几轮（user+bot 各一条算一轮）
_DEFAULT_MAX_PER_TURN   = 3  # 每轮自动写入上限

_EXTRACT_SYSTEM = """你是记忆提取助手。从对话片段中提取两类结构化记忆: events (多角色事件) 与 assertions (静态本体二元事实)。
目标: 让 Bot 在未来对话中能正确召回「谁对谁做了什么」, 以及不变的本体属性。

=== EVENTS (首选) ===
凡是涉及「两方/多方参与者」的事件、Bot 自我认知的承诺、临时状态、随时间会变的事实, 统统用 event。
字段:
  event_type:    简短标签, 如 teaching/correcting/asking/sharing/liking/disliking/promising/experiencing
  summary:       一句话事件摘要 (<=30 字), 用于检索与渲染
  polarity:      positive | negative   (表达否定意图时用 negative, 不要塞进 event_type)
  modality:      actual | hypothetical | possible   (「如果」「可能」用对应值; 事实用 actual)
  confidence:    0.0~1.0, 事实 0.7, 推测 0.4
  context_type:  meta | contract | episodic
    - meta      = Bot 永久自我认知 (如「我是 AI 助手」), 跨所有会话恒激活
    - contract  = 角色扮演 / 临时承诺 (如「现在我扮演 X」), 可被撤销
    - episodic  = 普通对话事件 (默认)
  recall_scope:  global | group:qq_{group_id} | private:qq_{user_id} (依首行 [场景:] 决定)
  roles:         参与者数组, 每项 {role, entity?, value_text?}
    role 取值 (仅这 8 个): agent / patient / theme / recipient / instrument / location / time / attribute
    entity:    'User' -> 自动替换为当前发言用户; 'Bot' -> 自动替换为 Bot:self; 其他保留为外部实体
    value_text: 当承载是一段文本/概念而非已知实体时使用 (如 theme 是被传授的内容)

=== ASSERTIONS (仅限永久本体) ===
只用于「绝对不会随时间改变」的本体属性, 如 'Python isA 编程语言'、'User 职业是 程序员'。
拿不准是否永久 -> 用 event 而非 assertion。
字段: subject, predicate, object_text, recall_scope, confidence
  subject: 'User' (当前用户) / 'Bot' (Bot 自己) / 其他外部实体名

=== 黄金规则 (违反会被静默丢弃) ===
1. 涉及「教/学/告诉/纠正/问/反驳/答应/拒绝」的句子, 必须用 event, 且:
   - agent  = 实施动作的人 (说话者)
   - recipient = 听众 (被告知者)
   - theme   = 内容
   反例 (错): {subject:'User', predicate:'学习到', object:'X'} (搞错主语视角)
   正例 (对): {event_type:'teaching', roles:[
       {role:'agent',entity:'User'},
       {role:'recipient',entity:'Bot'},
       {role:'theme',value_text:'X'}]}
2. 否定不要靠造一个「不喜欢」谓词, 用 polarity='negative'。
3. 假设/反事实用 modality='hypothetical', 不要丢弃也不要当作事实。
4. 会随时间变化的事实 (年龄/状态/今天的天气/正在做某事) 必须是 event, 禁止进 assertions。
5. Bot 在角色扮演中说的话, event 应标 context_type='contract', 不要污染 meta。
6. 无可提取内容时返回 {"events":[],"assertions":[]}。

输出 JSON:
{"events":[...], "assertions":[...]}"""


def _extract_text(content) -> str:
    """将消息 content（str 或 multimodal list）统一转成纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content) if content else ""


async def archive_turn_memories(
    session,
    sender_id: str,
    tool_calls_log: list[dict],
) -> None:
    """后台自动提取本轮对话中的记忆三元组并写入 DB，fire-and-forget 调用。

    Args:
        session: 当前 LLMSession 对象（含 context_messages / conv_* 字段）
        sender_id: 触发本轮对话的用户 QQ 号（字符串）
        tool_calls_log: 本轮工具调用记录列表
    """
    async with _SEM:  # 限流：最多 2 个并发 LLM 调用
        import app_state
        from llm.prompt.memory import add_memory

        # ── 读取运行时配置 ──────────────────────────────────────────────
        cfg = app_state.config.get("memory", {}).get("auto_archive", {})
        if not cfg.get("enabled", True):
            return

        context_turns = int(cfg.get("context_turns", _DEFAULT_CONTEXT_TURNS))
        max_per_turn  = int(cfg.get("max_per_turn",  _DEFAULT_MAX_PER_TURN))

        # ── 本轮 write_memory 工具已写入的三元组（精确去重，避免重复归档）──
        already_written: set[tuple[str, str]] = set()
        for call in tool_calls_log:
            if call.get("function") == "write_memory" and not call.get("circuit_broken"):
                args = call.get("arguments", {})
                pred = args.get("predicate", "")
                obj  = args.get("object_text", "") or args.get("content", "")
                if pred and obj:
                    already_written.add((pred, obj))

        # ── 截取最近 context_turns 轮对话消息 ──────────────────────────
        msgs = session.context_messages[-(context_turns * 2):]
        if not any(m.get("role") == "user" for m in msgs):
            return  # 没有用户发言，无需提取

        lines: list[str] = []
        for m in msgs:
            role    = m.get("role", "")
            content = _extract_text(m.get("content", ""))
            if not content:
                continue
            if role == "user":
                name = m.get("sender_name") or "User"
                lines.append(f"User({name}): {content}")
            elif role == "bot":
                lines.append(f"Bot: {content}")

        if not lines:
            return

        # 在对话片段开头注入会话场景标识，让模型能正确填写 recall_scope
        context_header = f"[场景: {session.conv_type}/{session.conv_id}]\n"
        dialogue = context_header + "\n".join(lines)

        # ── 轻量 LLM 提取 ───────────────────────────────────────────────
        adapter = app_state.adapter
        if adapter is None:
            return

        raw: dict | None = None
        try:
            raw = await asyncio.to_thread(adapter.one_shot_json, _EXTRACT_SYSTEM, dialogue)
        except Exception:
            logger.debug("[archiver] one_shot_json 调用异常", exc_info=True)
            return

        if not raw:
            return

        # 兼容三种返回结构：
        #   新版：{"events":[...], "assertions":[...]}
        #   旧版：{"memories":[...]}
        #   裸数组：[...]（视作 assertions）
        if isinstance(raw, dict):
            events_in: list = raw.get("events", []) if isinstance(raw.get("events"), list) else []
            assertions_in: list = (
                raw.get("assertions")
                or raw.get("memories")
                or []
            )
            if not isinstance(assertions_in, list):
                assertions_in = []
        elif isinstance(raw, list):
            events_in, assertions_in = [], raw
        else:
            return

        if not events_in and not assertions_in:
            return

        written = 0

        # ── 1) 事件层（Neo-Davidsonian）─────────────────────────────────
        from database import write_event as _db_write_event
        from llm.memory_tokenizer import tokenize as _tokenize, register_word as _register

        for ev in events_in:
            if written >= max_per_turn:
                break
            if not isinstance(ev, dict):
                continue

            event_type = str(ev.get("event_type", "")).strip() or "unspecified"
            summary    = str(ev.get("summary", "")).strip()
            if not summary:
                continue

            polarity     = str(ev.get("polarity", "positive")).strip().lower()
            modality     = str(ev.get("modality", "actual")).strip().lower()
            context_type = str(ev.get("context_type", "episodic")).strip().lower()
            recall_scope = str(ev.get("recall_scope") or "global").strip()
            try:
                confidence = float(ev.get("confidence", 0.6))
            except (TypeError, ValueError):
                confidence = 0.6
            confidence = max(0.1, min(1.0, confidence))

            # recall_scope 安全校验
            _valid_prefix = ("group:qq_", "private:qq_")
            if not (recall_scope == "global"
                    or any(recall_scope.startswith(p) for p in _valid_prefix)):
                recall_scope = "global"

            # 角色边规范化：把 "User"/"Bot" sentinel 替换为真实实体 ID
            roles_in = ev.get("roles") or []
            if not isinstance(roles_in, list):
                continue
            normalized_roles: list[dict] = []
            for r in roles_in:
                if not isinstance(r, dict):
                    continue
                role_name = str(r.get("role", "")).strip().lower()
                entity = r.get("entity")
                value_text = r.get("value_text")
                if entity:
                    ent = str(entity).strip()
                    if ent in ("User", "Self"):
                        if not sender_id:
                            continue  # 无法定位 sender，丢弃此角色边
                        ent = f"User:qq_{sender_id}"
                    elif ent == "Bot":
                        ent = "Bot:self"
                    entity = ent
                if value_text is not None:
                    value_text = str(value_text).strip() or None
                if not entity and not value_text:
                    continue
                normalized_roles.append({
                    "role": role_name,
                    "entity": entity,
                    "value_text": value_text,
                    "value_tok": _tokenize(value_text) if value_text else "",
                })

            if not normalized_roles:
                logger.debug("[archiver] event 无有效角色边，跳过：%s", summary)
                continue

            try:
                # summary 也注册到 jieba 词典 + 切词，方便后续召回
                _register(summary)
                summary_tok = _tokenize(summary)
                event_id = await _db_write_event(
                    event_type=event_type,
                    summary=summary,
                    summary_tok=summary_tok,
                    polarity=polarity,
                    modality=modality,
                    confidence=confidence,
                    context_type=context_type,
                    recall_scope=recall_scope,
                    source="自动归档",
                    reason="从对话中自动提取",
                    conv_type=session.conv_type,
                    conv_id=session.conv_id,
                    conv_name=session.conv_name,
                    roles=normalized_roles,
                )
                role_brief = "/".join(
                    f"{r['role']}:{r['entity'] or r['value_text']}"
                    for r in normalized_roles
                )
                logger.info(
                    "[archiver] 写入 event#%d type=%s ctx=%s | %s | %s",
                    event_id, event_type, context_type, summary, role_brief,
                )
                written += 1
            except Exception:
                logger.warning("[archiver] event 写入失败：%s", summary, exc_info=True)

        # ── 2) 静态本体断言（仅永久事实，谨慎使用）──────────────────────
        for item in assertions_in:
            if written >= max_per_turn:
                break
            if not isinstance(item, dict):
                continue

            predicate   = str(item.get("predicate", "")).strip()
            object_text = str(item.get("object_text", "")).strip()
            recall_scope = str(item.get("recall_scope") or "global").strip()
            # 安全校验：recall_scope 只允许合法格式
            _valid_prefix = ("global", "group:qq_", "private:qq_")
            if not any(recall_scope == "global" or recall_scope.startswith(p) for p in _valid_prefix):
                recall_scope = "global"
            # confidence 安全校验：限制在 [0.1, 1.0]，缺省 0.6
            try:
                confidence = float(item.get("confidence", 0.6))
            except (TypeError, ValueError):
                confidence = 0.6
            confidence = max(0.1, min(1.0, confidence))
            if not predicate or not object_text:
                continue

            # subject 路由：
            #   "User" → 用户 QQ 主语（类型 A/C）
            #   "Bot"  → "Bot:self"（bot 自我认知专属命名空间）
            #   "Self" → 旧 sentinel，兼容处理为 "User"
            #   其他   → 直接作为外部实体主语（类型 B/C 涉及人物）
            raw_subject = str(item.get("subject", "User")).strip() or "User"
            if raw_subject in ("User", "Self"):
                if sender_id:
                    subject = f"User:qq_{sender_id}"
                else:
                    logger.warning(
                        "[archiver] sender_id 为空，subject 回退为 UnknownUser；"
                        "同 predicate/object 记忆将被 UNIQUE 索引静默去重"
                    )
                    subject = "UnknownUser"
            elif raw_subject == "Bot":
                subject = "Bot:self"
            else:
                subject = raw_subject

            # 本轮手动记忆工具已写，跳过
            if (predicate, object_text) in already_written:
                logger.debug("[archiver] 跳过（本轮已写）: %s / %s", predicate, object_text)
                continue

            # 重复检查交由数据库层 UNIQUE 索引 + INSERT OR IGNORE 保证，此处无需额外查询
            try:
                await add_memory(
                    content=object_text,
                    predicate=predicate,
                    source="自动归档",
                    reason="从对话中自动提取",
                    conv_type=session.conv_type,
                    conv_id=session.conv_id,
                    conv_name=session.conv_name,
                    subject=subject,
                    origin="passive",
                    recall_scope=recall_scope,
                    confidence=confidence,
                )
                logger.info("[archiver] 写入: [%s] %s → %s", subject, predicate, object_text)
                written += 1
            except Exception:
                logger.warning(
                    "[archiver] 写入失败 %s / %s", predicate, object_text, exc_info=True
                )

        if written:
            logger.info("[archiver] 本轮自动归档 %d 条记忆", written)
