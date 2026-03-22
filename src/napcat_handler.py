# Copyright (C) 2026  AIcarusDev
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""napcat_handler.py — NapCat 消息处理集成

所有 NapCat 相关的处理逻辑：
  - 消息接收、白名单过滤、上下文录入
  - Bot 消息发送 & 入上下文
  - 撤回 / 戳一戳通知
  - 主动循环（continue / wait / shift / break）
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime

import app_state
from database import (
    get_display_name,
    get_group_info,
    get_group_name,
    save_bot_turn,
    save_chat_message,
    upsert_account,
    upsert_chat_session,
    upsert_membership,
)
from debug_server import broadcast_debug_xml
from llm_core import call_model_and_process
from napcat import (
    get_reply_message_id,
    llm_segments_to_napcat,
    napcat_event_to_context,
    napcat_event_to_debug_xml,
    should_respond,
)
from session import (
    extract_bot_messages,
    get_or_create_session,
    sessions,
)
import watcher_core

logger = logging.getLogger("AICQ.app")


# ══════════════════════════════════════════════════════════
#  Bot 消息发送 & 入上下文
# ══════════════════════════════════════════════════════════

async def send_and_commit_bot_messages(
    session,
    result: dict,
    group_id,
    user_id,
    llm_elapsed: float,
    conversation_id: str,
) -> None:
    """发送 bot 消息到 NapCat，拿到真实 QQ ID 后才入上下文并持久化。

    发送失败的消息也入上下文，content_type 标记为 "send_failed"。
    """
    client = app_state.napcat_client
    assert client is not None
    now_ts = datetime.now(app_state.TIMEZONE).isoformat()
    bot_sender_id = session._qq_id or "bot"
    bot_sender_name = session._qq_name or app_state.BOT_NAME
    bot_msgs = extract_bot_messages(result)

    decision = result.get("decision") or {}
    send_messages = decision.get("send_messages") or []
    _first_msg = True
    bot_msg_idx = 0

    for msg in send_messages:
        segments = msg.get("segments", [])
        reply_id = msg.get("quote") or None
        napcat_segs = llm_segments_to_napcat(segments, reply_message_id=reply_id)
        if not napcat_segs:
            continue

        # 发送到 QQ
        send_result = await client.send_message(
            group_id=group_id, user_id=user_id, message=napcat_segs,
            llm_elapsed=llm_elapsed if _first_msg else 0.0,
        )
        _first_msg = False

        # 检查是否有对应的文本上下文条目（与 extract_bot_messages 一一对应）
        if bot_msg_idx >= len(bot_msgs):
            continue
        bot_msg = bot_msgs[bot_msg_idx]
        bot_msg_idx += 1

        if send_result and send_result.get("message_id") is not None:
            real_id = str(send_result["message_id"])
            content_type = "text"
        else:
            real_id = f"failed_{uuid.uuid4().hex[:8]}"
            content_type = "send_failed"
            logger.warning("消息发送失败 conv=%s", conversation_id)

        entry = {
            "role": "bot",
            "message_id": real_id,
            "sender_id": bot_sender_id,
            "sender_name": bot_sender_name,
            "sender_role": "",
            "timestamp": now_ts,
            "content": bot_msg["text"],
            "content_type": content_type,
            "content_segments": bot_msg["content_segments"],
        }
        session.add_to_context(entry)
        try:
            await save_chat_message(conversation_id, entry)
        except Exception:
            logger.warning("[persist] bot消息写入失败 conv=%s", conversation_id, exc_info=True)


# ══════════════════════════════════════════════════════════
#  辅助：会话名解析 & shift 目标校验
# ══════════════════════════════════════════════════════════

async def _resolve_conv_name(conv_type: str, conv_id: str) -> str:
    """查询会话显示名：先查 DB，查不到再问 NapCat，还没有就返回空字符串。"""
    client = app_state.napcat_client
    if conv_type == "private":
        name = await get_display_name("qq", conv_id)
        if name and name != conv_id:
            return name
        if client and client.connected:
            try:
                data = await client.send_api("get_stranger_info", {"user_id": int(conv_id)})
                if isinstance(data, dict):
                    nick = data.get("nickname") or data.get("nick") or ""
                    if nick:
                        return nick
            except Exception as e:
                logger.warning("通过 NapCat API 查询 user_id=%s 的信息失败: %s", conv_id, e)
    elif conv_type == "group":
        name = await get_group_name(conv_id)
        if name:
            return name
        if client and client.connected:
            try:
                data = await client.send_api("get_group_info", {"group_id": int(conv_id)})
                if isinstance(data, dict):
                    gname = data.get("group_name") or ""
                    if gname:
                        return gname
            except Exception as e:
                logger.warning("通过 NapCat API 查询 group_id=%s 的信息失败: %s", conv_id, e)
    return ""


async def _validate_shift_target(target_type: str, target_id: str) -> str | None:
    """检查 shift 目标是否在白名单和 QQ 联系人列表中。

    返回 None 表示合法，返回字符串表示失败原因。
    """
    if target_type not in ("private", "group"):
        return f"未知会话类型 {target_type!r}"

    # 白名单检查
    whitelist_cfg = app_state.napcat_cfg.get("whitelist", {})
    if target_type == "private":
        private_whitelist = [str(u) for u in whitelist_cfg.get("private_users", [])]
        if private_whitelist and target_id not in private_whitelist:
            return f"私聊用户 {target_id} 不在白名单中"
    else:
        group_whitelist = [str(g) for g in whitelist_cfg.get("group_ids", [])]
        if group_whitelist and target_id not in group_whitelist:
            return f"群聊 {target_id} 不在白名单中"

    # QQ 好友 / 已加入群列表检查
    client = app_state.napcat_client
    if client and client.connected:
        if target_type == "private":
            friends = await client.send_api("get_friend_list", {}) or []
            if isinstance(friends, list):
                friend_ids = {str(f.get("user_id", "")) for f in friends if isinstance(f, dict)}
                if target_id not in friend_ids:
                    return f"用户 {target_id} 不在好友列表中"
        else:
            groups = await client.send_api("get_group_list", {}) or []
            if isinstance(groups, list):
                group_ids = {str(g.get("group_id", "")) for g in groups if isinstance(g, dict)}
                if target_id not in group_ids:
                    return f"群 {target_id} 不在已加入的群列表中"

    return None


# ══════════════════════════════════════════════════════════
#  主动循环 & Shift
# ══════════════════════════════════════════════════════════

async def _run_active_loop(
    session,
    conv_key: str,
    group_id,
    user_id,
    result: dict,
) -> None:
    """主动循环公共逻辑：支持 continue / wait / shift / break。"""
    while result:
        lc = result.get("loop_control") or {}
        if not isinstance(lc, dict):
            break

        if "continue" in lc:
            pass
        elif "wait" in lc:
            wait_cfg = lc["wait"]
            timeout = wait_cfg.get("timeout", 60)
            trigger = wait_cfg.get("early_trigger")
            session.wait_early_trigger = trigger
            session.wait_event = asyncio.Event()
            # 消费打字期间积累的 pending trigger：若匹配则直接 pre-fire，跳过实际等待
            _pending = session.pending_early_trigger
            session.pending_early_trigger = None
            if trigger and _pending and (
                trigger == _pending
                or (trigger == "new_message" and _pending == "mentioned")
            ):
                logger.info(
                    "会话 %s 打字发送期间已收到触发消息 (pending=%s)，early_trigger=%s 立即满足",
                    conv_key, _pending, trigger,
                )
                session.wait_event.set()
            logger.info(
                "会话 %s 进入等待 timeout=%ds early_trigger=%s",
                conv_key, timeout, trigger,
            )
            try:
                await asyncio.wait_for(session.wait_event.wait(), timeout=timeout)
                logger.info("会话 %s 等待提前结束 (early_trigger=%s)", conv_key, trigger)
            except asyncio.TimeoutError:
                logger.info("会话 %s 等待超时，继续下一轮循环", conv_key)
            finally:
                session.wait_event = None
                session.wait_early_trigger = None
        elif "shift" in lc:
            shift_cfg = lc["shift"]
            shift_type = shift_cfg.get("type", "")
            shift_id = str(shift_cfg.get("id", ""))
            shift_key = f"{shift_type}_{shift_id}"
            shift_error = await _validate_shift_target(shift_type, shift_id)
            if shift_error:
                session.pending_error_logger = shift_error
                logger.warning("会话 %s shift 验证失败: %s，继续本会话", conv_key, shift_error)
            else:
                next_session = get_or_create_session(shift_key)
                if not next_session.conv_type:
                    next_session.set_conversation_meta(shift_type, shift_id)
                if not next_session.conv_name:
                    next_session.conv_name = await _resolve_conv_name(shift_type, shift_id)
                from_name = session.conv_name or await _resolve_conv_name(session.conv_type, session.conv_id)
                next_session.shift_context = {
                    "from_type": session.conv_type,
                    "from_id": session.conv_id,
                    "from_name": from_name,
                    "motivation": lc.get("motivation", ""),
                }
                asyncio.create_task(
                    _activate_session_shifted(next_session, shift_key, shift_type, shift_id)
                )
                break
        else:
            # loop_control.break：调度 watcher 后台窥屏
            session.watcher_break_time = time.time()
            session.watcher_break_reason = lc.get("motivation", "")
            watcher_core.schedule_watcher(session, conv_key, group_id, user_id)
            break

        session.pending_early_trigger = None  # 新一轮 LLM 决策前清除上一轮遗留的 pending trigger
        _t0 = time.monotonic()
        try:
            await app_state.rate_limiter.acquire()
            result, _, _, _, _, _tool_calls_log = await asyncio.to_thread(call_model_and_process, session)  # type: ignore[assignment]
        except Exception:
            logger.exception("主动循环 LLM 调用失败 (conv=%s)", conv_key)
            break
        _llm_elapsed = time.monotonic() - _t0
        logger.info("LLM 响应耗时（主动循环）%.2fs (conv=%s)", _llm_elapsed, conv_key)

        if result is None:
            break

        await send_and_commit_bot_messages(
            session, result, group_id, user_id, _llm_elapsed, conv_key,
        )
        await save_bot_turn(
            turn_id=uuid.uuid4().hex,
            conv_type=session.conv_type,
            conv_id=session.conv_id,
            result=result,
            tool_calls_log=_tool_calls_log,
        )


async def _activate_session_shifted(
    target_session,
    target_key: str,
    target_type: str,
    target_id: str,
) -> None:
    """shift 切换后在目标会话激活一轮循环，并持续处理后续 loop_control（含递归 shift）。"""
    if app_state.consciousness_lock.locked():
        logger.info("[shift] 意识正忙，本次激活跳过 %s", target_key)
        return

    async with app_state.consciousness_lock:
        app_state.current_focus = target_key
        try:
            try:
                group_id = int(target_id) if target_type == "group" else None
                user_id = int(target_id) if target_type == "private" else None
            except (ValueError, TypeError):
                logger.error("[shift] 目标 ID '%s' 无效，无法转换为整数", target_id)
                return

            _t0 = time.monotonic()
            try:
                await app_state.rate_limiter.acquire()
                result, _, _, _, _, _tool_calls_log = await asyncio.to_thread(call_model_and_process, target_session)  # type: ignore[assignment]
            except Exception:
                logger.exception("[shift] 目标会话 %s LLM 调用失败", target_key)
                return
            _llm_elapsed = time.monotonic() - _t0

            if result is None:
                logger.warning("[shift] 目标会话 %s LLM 返回为空", target_key)
                return

            await send_and_commit_bot_messages(
                target_session, result, group_id, user_id, _llm_elapsed, target_key,
            )
            await save_bot_turn(
                turn_id=uuid.uuid4().hex,
                conv_type=target_session.conv_type,
                conv_id=target_session.conv_id,
                result=result,
                tool_calls_log=_tool_calls_log,
            )

            await _run_active_loop(target_session, target_key, group_id, user_id, result)
        finally:
            app_state.current_focus = None


# ══════════════════════════════════════════════════════════
#  NapCat 事件回调
# ══════════════════════════════════════════════════════════

async def _handle_napcat_message(event: dict, conversation_id: str) -> None:
    """NapCat 消息到达时的处理回调。"""
    client = app_state.napcat_client
    assert client is not None
    bot_id = client.bot_id
    debug_xml = await napcat_event_to_debug_xml(event, bot_id=bot_id, timezone=app_state.TIMEZONE)
    await broadcast_debug_xml(debug_xml, event)

    napcat_cfg = app_state.napcat_cfg
    if napcat_cfg.get("debug_only", False):
        return

    # 白名单过滤：私聊用户 + 群组
    whitelist_cfg = napcat_cfg.get("whitelist", {})
    private_whitelist = [str(u) for u in whitelist_cfg.get("private_users", [])]
    group_whitelist = [str(g) for g in whitelist_cfg.get("group_ids", [])]
    msg_type = event.get("message_type", "")
    sender_id = str(event.get("sender", {}).get("user_id", ""))
    group_id_str = str(event.get("group_id", ""))
    if msg_type == "private":
        if private_whitelist and sender_id not in private_whitelist:
            logger.debug("私聊来自非白名单用户 %s，忽略", sender_id)
            return
    elif msg_type == "group":
        if group_whitelist and group_id_str not in group_whitelist:
            logger.debug("群聊来自非白名单群组 %s，忽略", group_id_str)
            return
    else:
        logger.debug("未知消息类型 %s，忽略 (conv=%s)", msg_type, conversation_id)
        return

    # 纯多模态消息（无文字）暂不处理
    message_segs = event.get("message", [])
    has_real_text = any(
        seg.get("type") == "text" and seg.get("data", {}).get("text", "").strip()
        for seg in message_segs
    )
    has_image = any(seg.get("type") == "image" for seg in message_segs)
    has_unhandled_media_only = (
        not has_real_text
        and not has_image
        and any(seg.get("type") in ("record", "video") for seg in message_segs)
    )
    if has_unhandled_media_only:
        logger.debug("纯语音/视频消息，暂不处理 (conv=%s)", conversation_id)
        return

    session = get_or_create_session(conversation_id)

    need_respond = should_respond(event, client.bot_id, app_state.BOT_NAME)
    if not need_respond and msg_type == "group":
        if _reply_id := get_reply_message_id(message_segs):
            _bot_msg_ids = {
                str(m.get("message_id", ""))
                for m in session.context_messages
                if m.get("role") == "bot"
            }
            if _reply_id in _bot_msg_ids:
                need_respond = True
    if not need_respond:
        logger.debug("NapCat 消息不触发回复，静默记入上下文 (conv=%s)", conversation_id)

    # 设置/更新会话元信息（群名、私聊昵称等）；同步发送者信息到 DB
    msg_type = event.get("message_type", "")
    sender = event.get("sender", {})
    sender_id = str(sender.get("user_id", ""))
    sender_nickname = sender.get("nickname", "")
    if msg_type == "group":
        group_id = str(event.get("group_id", ""))
        sender_card = sender.get("card", "") or sender_nickname
        sender_role = sender.get("role", "member")
        sender_title = sender.get("title", "")
        if not session.conv_type:
            group_name, member_count, bot_card = await get_group_info(group_id)
            session.set_conversation_meta("group", group_id, group_name, member_count)
            session._qq_card = bot_card
        # 懒同步：每次收到消息时更新发送者的账号和群成员关系
        await upsert_membership(
            "qq", sender_id, group_id,
            cardname=sender_card,
            title=sender_title,
            permission_level=sender_role,
        )
    elif msg_type == "private":
        peer_id = str(sender.get("user_id", ""))
        peer_name = sender.get("nickname", "")
        if not session.conv_type:
            session.set_conversation_meta("private", peer_id, peer_name)
        # 懒同步：更新私聊对方的账号信息
        await upsert_account("qq", peer_id, nickname=peer_name)

    bot_display = session._qq_card or session._qq_name or ""
    ctx_entry = await napcat_event_to_context(event, bot_id=client.bot_id, bot_display_name=bot_display, timezone=app_state.TIMEZONE)
    if not ctx_entry:
        return

    # 图片落盘 + pHash 去重 + 视觉描述（有图时在后台线程执行，不阻塞事件循环）
    if ctx_entry.get("images"):
        await asyncio.to_thread(app_state.vision_bridge.process_entry, ctx_entry)

    session.add_to_context(ctx_entry)
    try:
        await save_chat_message(conversation_id, ctx_entry)
        await upsert_chat_session(conversation_id, session.conv_type, session.conv_id, session.conv_name)
    except Exception:
        logger.warning("[persist] 消息写入失败 conv=%s", conversation_id, exc_info=True)

    # new_message early_trigger 语义不依赖 need_respond，必须在过滤之前检查
    _is_focused = app_state.consciousness_lock.locked() and app_state.current_focus == conversation_id
    if _is_focused and session.wait_event is not None and not session.wait_event.is_set():
        if session.wait_early_trigger == "new_message":
            logger.info("会话 %s early_trigger=new_message 条件满足，唤醒等待循环", conversation_id)
            session.wait_event.set()
    # 意识正忙且聚焦于本会话，但 wait_event 尚未创建（仍在发送阶段），提前记录触发强度
    elif _is_focused and session.wait_event is None:
        _is_mention = (
            any(
                seg.get("type") == "at"
                and str(seg.get("data", {}).get("qq", "")) == str(client.bot_id)
                for seg in message_segs
            )
            or get_reply_message_id(message_segs) is not None
        )
        if _is_mention:
            session.pending_early_trigger = "mentioned"
        elif session.pending_early_trigger is None:
            session.pending_early_trigger = "new_message"

    if not need_respond:
        return

    # 严格唯一性：意识同一时刻只能处理一个会话，正忙时其它激活请求一律跳过
    if app_state.consciousness_lock.locked():
        # 只有当前焦点会话的 early_trigger 才有意义
        if app_state.current_focus == conversation_id and session.wait_event is not None and not session.wait_event.is_set():
            trigger = session.wait_early_trigger
            if trigger == "mentioned" and (any(
                seg.get("type") == "at"
                and str(seg.get("data", {}).get("qq", "")) == str(client.bot_id)
                for seg in message_segs
            ) or get_reply_message_id(message_segs) is not None):
                logger.info("会话 %s early_trigger=mentioned 条件满足，唤醒等待循环", conversation_id)
                session.wait_event.set()
        logger.info("意识正忙（焦点=%s），消息已记入 %s 上下文，本次激活跳过", app_state.current_focus, conversation_id)
        return

    async with app_state.consciousness_lock:
        app_state.current_focus = conversation_id
        try:
            # 被动激活主意识：停止所有 watcher（单一意识流不允许同时观察其它会话）
            watcher_core.stop_all_watchers()

            _t0 = time.monotonic()
            try:
                await app_state.rate_limiter.acquire()
                result, _, _, _, _, _tool_calls_log = await asyncio.to_thread(call_model_and_process, session)
            except Exception:
                logger.exception("NapCat LLM 调用失败 (conv=%s)", conversation_id)
                return
            _llm_elapsed = time.monotonic() - _t0
            logger.info("LLM 响应耗时 %.2fs (conv=%s)", _llm_elapsed, conversation_id)

            if result is None:
                logger.warning("NapCat LLM 返回为空 (conv=%s)", conversation_id)
                return

            msg_type = event.get("message_type", "")
            group_id = event.get("group_id") if msg_type == "group" else None
            user_id = event.get("sender", {}).get("user_id") if msg_type == "private" else None

            # 发送消息并以真实 QQ ID 入上下文（发送失败也会入上下文并标记 send_failed）
            await send_and_commit_bot_messages(
                session, result, group_id, user_id, _llm_elapsed, conversation_id,
            )
            await save_bot_turn(
                turn_id=uuid.uuid4().hex,
                conv_type=session.conv_type,
                conv_id=session.conv_id,
                result=result,
                tool_calls_log=_tool_calls_log,
            )

            # 主动循环：支持 continue / wait / break / shift
            await _run_active_loop(session, conversation_id, group_id, user_id, result)
        finally:
            app_state.current_focus = None


async def _handle_napcat_recall(event: dict) -> None:
    """处理群/私聊撤回通知，将对应上下文条目替换为撤回提示。"""
    notice_type = event.get("notice_type", "")
    message_id = str(event.get("message_id", ""))
    if not message_id:
        return

    if notice_type == "group_recall":
        group_id = str(event.get("group_id", ""))
        operator_id = str(event.get("user_id", ""))
        conv_id = f"group_{group_id}"
        operator_name = await get_display_name("qq", operator_id, group_id or None)
    else:  # friend_recall
        peer_id = str(event.get("user_id", ""))
        conv_id = f"private_{peer_id}"
        operator_name = await get_display_name("qq", peer_id, None)

    session = sessions.get(conv_id)
    if not session:
        return

    timestamp = datetime.now(app_state.TIMEZONE).isoformat()
    if session.mark_message_recalled(message_id, operator_name, timestamp):
        logger.debug("撤回通知已处理: conv=%s msg_id=%s operator=%s", conv_id, message_id, operator_name)


async def _handle_napcat_poke(event: dict) -> None:
    """处理戳一戳通知，将动作文本作为 note 记入上下文。"""
    group_id = str(event.get("group_id", ""))
    sender_id = str(event.get("user_id", ""))
    target_id = str(event.get("target_id", ""))
    action = event.get("action") or "戳了戳"
    suffix = event.get("suffix") or ""

    if group_id:
        conv_id = f"group_{group_id}"
        sender_name = await get_display_name("qq", sender_id, group_id)
        target_name = await get_display_name("qq", target_id, group_id)
    else:
        conv_id = f"private_{sender_id}"
        sender_name = await get_display_name("qq", sender_id, None)
        target_name = await get_display_name("qq", target_id, None)

    session = sessions.get(conv_id)
    if not session:
        return

    # 白名单过滤
    whitelist_cfg = app_state.napcat_cfg.get("whitelist", {})
    if group_id:
        group_whitelist = [str(g) for g in whitelist_cfg.get("group_ids", [])]
        if group_whitelist and group_id not in group_whitelist:
            return
    else:
        private_whitelist = [str(u) for u in whitelist_cfg.get("private_users", [])]
        if private_whitelist and sender_id not in private_whitelist:
            return

    poke_text = f"{sender_name} {action} {target_name}{suffix}"
    timestamp = datetime.now(app_state.TIMEZONE).isoformat()
    note_entry = {
        "role": "note",
        "timestamp": timestamp,
        "content": poke_text,
        "content_type": "poke",
    }
    session.add_to_context(note_entry)
    try:
        await save_chat_message(conv_id, note_entry)
        await upsert_chat_session(conv_id, session.conv_type, session.conv_id, session.conv_name)
    except Exception:
        logger.warning("[persist] 戳一戳 note 写入失败 conv=%s", conv_id, exc_info=True)
    logger.debug("戳一戳已记录: conv=%s text=%s", conv_id, poke_text)


# ══════════════════════════════════════════════════════════
#  Watcher 激活专注聊天处理器（由 watcher_core 回调）
# ══════════════════════════════════════════════════════════

async def _handle_watcher_activate(
    session,
    conv_key: str,
    group_id,
    user_id,
) -> None:
    """watcher 决定 activate 后，完整运行一轮专注聊天（含 loop_control）。

    由 watcher_core 通过注入的回调调用；调用者已持有 consciousness_lock。
    """
    _t0 = time.monotonic()
    try:
        await app_state.rate_limiter.acquire()
        result, _, _, _, _, tool_calls_log = await asyncio.to_thread(
            call_model_and_process, session
        )
    except Exception:
        logger.exception("[watcher] 激活专注聊天失败 conv=%s", conv_key)
        return

    logger.info("[watcher] 专注聊天响应耗时 %.2fs conv=%s", time.monotonic() - _t0, conv_key)

    if result is None:
        logger.warning("[watcher] 专注聊天返回为空 conv=%s", conv_key)
        return

    await send_and_commit_bot_messages(
        session, result, group_id, user_id, time.monotonic() - _t0, conv_key,
    )
    await save_bot_turn(
        turn_id=uuid.uuid4().hex,
        conv_type=session.conv_type,
        conv_id=session.conv_id,
        result=result,
        tool_calls_log=tool_calls_log,
    )
    await _run_active_loop(session, conv_key, group_id, user_id, result)


# ══════════════════════════════════════════════════════════
#  注册入口
# ══════════════════════════════════════════════════════════

def register_napcat_handlers() -> None:
    """将消息 / 撤回 / 戳一戳回调注册到 NapCat 客户端。"""
    client = app_state.napcat_client
    if not client:
        return
    client.set_message_handler(_handle_napcat_message)
    client.set_recall_handler(_handle_napcat_recall)
    client.set_poke_handler(_handle_napcat_poke)


# 向 watcher_core 注入激活处理器，消除循环导入
watcher_core.register_activate_handler(_handle_watcher_activate)
