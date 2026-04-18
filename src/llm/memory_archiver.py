"""memory_archiver.py — 后台记忆自动归档（Phase 3D）

每轮对话完成后，fire-and-forget 调度一次轻量 one_shot_json 推理，
自动从本轮对话中提取关于用户的记忆三元组并写入 MemoryTriples。
无需模型主动调用 write_memory 工具。

触发位置：napcat_handler._consciousness_task() 在 save_bot_turn 之后
"""

import asyncio
import logging

import aiosqlite

logger = logging.getLogger("AICQ.archiver")

_DEFAULT_CONTEXT_TURNS = 5   # 最近几轮（user+bot 各一条算一轮）
_DEFAULT_MAX_PER_TURN   = 3  # 每轮自动写入上限

_EXTRACT_SYSTEM = (
    "你是记忆提取助手。从对话片段中提取四类有价值的信息。\n\n"
    "【类型 A — 用户个人信息】\n"
    "- subject 填 \"Self\"，表示主语是发言的用户本人\n"
    "- 提取偏好、习惯、经历、身份、人际关系、观点等个人信息\n"
    "- predicate 要简洁，例如「喜欢」「讨厌」「职业是」「住在」「曾经」「认为」\n"
    "- object_text 不超过 20 字，confidence 建议 0.6\n\n"
    "【类型 B — 用户分享的外部事实】\n"
    "- 仅当用户明确分享了一则新闻 / 公告 / 发布信息时才提取\n"
    "- subject 填写实际主体名称（如 \"Qwen3.6-35B\"、\"OpenAI\"），不含引号\n"
    "- predicate 描述动作，如「发布了」「宣布」「开源了」「更新到」\n"
    "- object_text 不超过 40 字，confidence 建议 0.7\n\n"
    "【类型 C — 用户提到的日常趣事 / 经历 / 故事】\n"
    "- 用户讲述的生活小事、搞笑经历、感受，这类内容虽然平凡，但正是拟人记忆的关键\n"
    "- 可以有两条三元组：涉及人物的事件 + 用户自身的感受/反应\n"
    "  例如：{subject: '用户的同学', predicate: '踩香蕉皮', object_text: '在路上摔倒了', confidence: 0.4}\n"
    "       {subject: 'Self', predicate: '觉得好笑', object_text: '同学踩香蕉皮摔跤的事', confidence: 0.4}\n"
    "- object_text 不超过 25 字，confidence 填 0.4（不确定性较高）\n\n"
    "【类型 D — Bot 自身的陈述】\n"
    "- 来源为 Bot: 行，且 Bot 以第一人称表达了自身的喜好、感受、观点、经历\n"
    "- subject 必须填 \"Bot\"，表示主语是 Bot 自己\n"
    "- 仅提取具体且带有个性色彩的第一人称陈述，跳过回答说明性语句\n"
    "  例如：{subject: 'Bot', predicate: '喜欢', object_text: '苹果的清甜味道', confidence: 0.7}\n"
    "  跳过：\"Qwen 是一个语言模型\" 这类客观事实陈述\n"
    "- object_text 不超过 20 字，confidence 建议 0.7\n\n"
    "通用规则：\n"
    "- recall_scope 只能为三个值之一：global（适合任何场景）、\n"
    "  group:qq_{group_id}（仅关联特定群组）、private:qq_{user_id}（仅对某私聊有效）\n"
    "- 没有值得提取的内容时，返回空数组\n\n"
    '输出严格 JSON，不含任何 Markdown：{"memories": [{"subject": "Self", "predicate": "...", "object_text": "...", "recall_scope": "global", "confidence": 0.6}, ...]}'
)


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
    import app_state
    from database import DB_PATH
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
    dialogue = "\n".join(lines)

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

    # 兼容模型直接返回数组的边界情况
    items = raw.get("memories", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        return

    written = 0

    for item in items:
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
        #   "Self" → 用户 QQ 主语
        #   "Bot"  → "Self"（bot 视角的自我知识，存入同一主语空间）
        #   其他   → 直接作为外部实体主语
        raw_subject = str(item.get("subject", "Self")).strip() or "Self"
        if raw_subject == "Self":
            subject = f"User:qq_{sender_id}" if sender_id else "Self"
        elif raw_subject == "Bot":
            subject = "Self"
        else:
            subject = raw_subject

        # 本轮手动记忆工具已写，跳过
        if (predicate, object_text) in already_written:
            logger.debug("[archiver] 跳过（本轮已写）: %s / %s", predicate, object_text)
            continue

        # DB 精确重复检查，避免跨轮写入相同事实
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("PRAGMA foreign_keys=ON")
                async with db.execute(
                    "SELECT 1 FROM MemoryTriples "
                    "WHERE subject=? AND predicate=? AND object_text=? AND is_deleted=0 LIMIT 1",
                    (subject, predicate, object_text),
                ) as cur:
                    if await cur.fetchone():
                        logger.debug("[archiver] 跳过（DB 已存在）: %s / %s", predicate, object_text)
                        continue
        except Exception:
            logger.debug("[archiver] DB 重复检查异常，继续尝试写入", exc_info=True)

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
