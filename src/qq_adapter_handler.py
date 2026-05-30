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

"""qq_adapter_handler.py — QQ adapter 消息处理集成

新架构下，本模块**只**做事件 → 状态的桥接：
- 消息接收：响应范围过滤、入上下文、广播
- 唤醒信号：mention / poke / 焦点会话新消息 → set 对应 session 的 sleep/wait event
- 撤回 / 戳一戳通知

**不再驱动任何 LLM 调用**。bot 的"思考"由 ``consciousness.main_loop`` 这条
常驻协程独立持有。本模块只负责把外部世界的变化写入共享状态。
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

import app_state
from consciousness import trigger_first_activation
from database import (
    get_display_name,
    get_group_member_display_info,
    get_group_info,
    get_group_name,
    is_bot_chat_message,
    save_chat_message,
    update_chat_message_recalled,
    upsert_account,
    upsert_chat_session,
    upsert_group,
    upsert_membership,
)
from web.debug_server import broadcast_debug_xml, broadcast_chat_event
from qq_adapter import (
    build_group_notice_entry,
    build_recall_notice_entry,
    get_reply_message_id,
    qq_adapter_event_to_context,
    qq_adapter_event_to_debug_xml,
    download_pending_images,
    expand_forward_previews,
    should_respond,
)
from qq_adapter.access_control import whitelist_rejection_reason
from qq_adapter.conversation import (
    event_group_id,
    is_temp_private_event,
    make_temp_session_key,
    set_temp_source,
)
from llm.session import (
    get_or_create_session,
    sessions,
)

logger = logging.getLogger("AICQ.app")


_GROUP_SYSTEM_NOTICE_TYPES = {"group_increase", "group_decrease", "group_ban", "group_admin", "group_card"}
GroupMemberInfo = dict[str, Any]


# ══════════════════════════════════════════════════════════
#  辅助：会话名解析
# ══════════════════════════════════════════════════════════

def _fill_bot_display_info(actor_id: str, info: GroupMemberInfo) -> GroupMemberInfo:
    client = app_state.qq_adapter_client
    bot_id = str(getattr(client, "bot_id", "") or "")
    if actor_id and bot_id and str(actor_id) == bot_id and not (info.get("card") or info.get("nickname")):
        info = dict(info)
        info["nickname"] = app_state.BOT_NAME or bot_id
        info["display"] = info["nickname"]
    return info


async def _fetch_group_member_info_from_qq_adapter(group_id: str, user_id: str) -> GroupMemberInfo | None:
    client = app_state.qq_adapter_client
    if not client or not client.connected:
        return None

    try:
        data = await client.send_api(
            "get_group_member_info",
            {"group_id": str(group_id), "user_id": str(user_id), "no_cache": True},
            timeout=8.0,
        )
    except Exception:
        logger.warning("[QQ adapter] 查询群成员信息失败 group=%s user=%s", group_id, user_id, exc_info=True)
        data = None

    if not data:
        try:
            member_list = await client.send_api(
                "get_group_member_list",
                {"group_id": str(group_id)},
                timeout=12.0,
            )
            if member_list:
                data = next(
                    (m for m in member_list if str(m.get("user_id", "")) == str(user_id)),
                    None,
                )
        except Exception:
            logger.warning("[QQ adapter] 查询群成员列表失败 group=%s user=%s", group_id, user_id, exc_info=True)
            data = None

    if not data:
        return None

    return {
        "id": str(data.get("user_id", user_id) or user_id),
        "card": str(data.get("card", "") or ""),
        "nickname": str(data.get("nickname", "") or ""),
        "permission_level": str(data.get("role", "") or ""),
        "title": str(data.get("title", "") or ""),
        "title_expire_time": int(data.get("title_expire_time", 0) or 0),
        "level": str(data.get("level", "") or ""),
        "display": str(data.get("card", "") or data.get("nickname", "") or user_id),
    }


async def _resolve_group_member_display_info(group_id: str, user_id: str) -> GroupMemberInfo:
    info = await get_group_member_display_info("qq", user_id, group_id)
    info = _fill_bot_display_info(user_id, info)
    if info.get("permission_level") and (info.get("card") or info.get("nickname")):
        return info

    remote_info = await _fetch_group_member_info_from_qq_adapter(group_id, user_id)
    if remote_info:
        nickname = remote_info.get("nickname", "")
        card = remote_info.get("card", "") or nickname
        permission_level = remote_info.get("permission_level", "") or info.get("permission_level", "")
        try:
            await upsert_membership(
                "qq",
                user_id,
                group_id,
                nickname=nickname,
                cardname=card,
                title=remote_info.get("title", ""),
                title_expire_time=int(remote_info.get("title_expire_time", 0) or 0),
                level=remote_info.get("level", ""),
                permission_level=permission_level or "member",
            )
        except Exception:
            logger.warning("[persist] 群成员信息回写失败 group=%s user=%s", group_id, user_id, exc_info=True)
        merged = {
            "id": user_id,
            "card": card or info.get("card", ""),
            "nickname": nickname or info.get("nickname", ""),
            "permission_level": permission_level,
            "title": remote_info.get("title", "") or info.get("title", ""),
            "level": remote_info.get("level", "") or info.get("level", ""),
            "display": card or nickname or info.get("display", "") or user_id,
        }
        return _fill_bot_display_info(user_id, merged)

    return info


async def _resolve_conv_name(conv_type: str, conv_id: str) -> str:
    """查询会话显示名：先查 DB，查不到再问 QQ adapter，还没有就返回空字符串。"""
    client = app_state.qq_adapter_client
    if conv_type in {"private", "temp"}:
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
                logger.warning("通过 QQ adapter API 查询 user_id=%s 的信息失败: %s", conv_id, e)
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
                logger.warning("通过 QQ adapter API 查询 group_id=%s 的信息失败: %s", conv_id, e)
    return ""


def _is_at_bot(message_segs: list, bot_id: str | None) -> bool:
    if bot_id is None:
        return False
    return any(
        seg.get("type") == "at"
        and str(seg.get("data", {}).get("qq", "")) == str(bot_id)
        for seg in message_segs
    )


def _bot_message_ids(session) -> set[str]:
    return {
        str(m.get("message_id", ""))
        for m in session.context_messages
        if m.get("role") == "bot" and str(m.get("message_id", "")).strip()
    }


def _is_reply_to_bot(message_segs: list, session) -> bool:
    reply_id = get_reply_message_id(message_segs)
    return bool(reply_id and reply_id in _bot_message_ids(session))


async def _is_reply_to_bot_message(message_segs: list, session, conversation_id: str) -> bool:
    reply_id = get_reply_message_id(message_segs)
    if not reply_id:
        return False
    if reply_id in _bot_message_ids(session):
        return True
    return await is_bot_chat_message(conversation_id, reply_id)


def _is_mention_level_message(
    event: dict,
    message_segs: list,
    bot_id: str | None,
    *,
    reply_to_bot: bool = False,
) -> bool:
    """判断消息是否应按 mention 强度唤醒。"""
    if event.get("message_type") == "private":
        return True
    return _is_at_bot(message_segs, bot_id) or reply_to_bot


def _build_passive_remark(
    event: dict,
    message_segs: list,
    bot_id: str | None,
    *,
    reply_to_bot: bool = False,
) -> str:
    """根据消息类型生成被动激活的 remark 描述。"""
    if reply_to_bot:
        return "收到回复，被动激活"
    if _is_at_bot(message_segs, bot_id):
        return "被@叫醒了"
    msg_type = event.get("message_type", "")
    if is_temp_private_event(event):
        return "被临时会话消息叫醒了"
    if msg_type == "private":
        return "被私聊消息叫醒了"
    return "被动激活"


# ══════════════════════════════════════════════════════════
#  唤醒信号分发
# ══════════════════════════════════════════════════════════

def _dispatch_wake_signals(
    incoming_session,
    conversation_id: str,
    is_mention: bool,
    wake_remark: str,
) -> None:
    """根据消息归属与 mention 状态，向相关 session 投递唤醒事件。

    - 焦点会话有 ``sleep_wake_event``：mention 命中即唤醒（普通消息不打断 sleep）。
    - 焦点会话有 ``wait_event``：按 early_trigger 决定是否提前唤醒；
      若 wait_event 还未创建（race window），把强度记到 ``pending_early_trigger``。
    - 非焦点会话收到 mention：仍唤醒焦点会话的 sleep（让模型自行 shift）。
    """
    focus_key = app_state.current_focus
    is_focused = (focus_key == conversation_id)

    # ── 焦点会话本身：处理 wait + sleep ──────────────────────────────
    if is_focused:
        sess = incoming_session

        # wait_event：按 early_trigger 判定
        trig = sess.wait_early_trigger
        if sess.wait_event is not None and not sess.wait_event.is_set() and isinstance(trig, dict):
            cond = trig.get("condition")
            if cond == "any_message" or (cond == "mentioned" and is_mention):
                logger.info(
                    "[wake] 焦点 %s 的 wait early_trigger 命中 (cond=%s)",
                    conversation_id, cond,
                )
                sess.wait_event.set()
        elif sess.wait_event is None and trig is None:
            # wait handler 还未创建 event 的 race 窗口：先记下强度
            if is_mention:
                sess.pending_early_trigger = "mentioned"
            elif sess.pending_early_trigger is None:
                sess.pending_early_trigger = "any_message"

        # sleep_wake_event：仅 mention 唤醒（"睡觉时世界还在转，但被叫到名字会醒"）
        if is_mention:
            if sess.sleep_wake_event is not None:
                sess.last_wake_reason = wake_remark
                sess.sleep_wake_from = conversation_id
                sess.sleep_wake_event.set()
                logger.info("[wake] 焦点 %s sleep 被 mention 唤醒", conversation_id)
            elif sess.sleep_arming:
                # sleep handler 启动前的 race 窗口
                sess.sleep_pending_wake = True
                sess.last_wake_reason = wake_remark
                sess.sleep_wake_from = conversation_id

        return

    # ── 非焦点会话被 mention：唤醒焦点会话的 sleep ────────────────────
    if is_mention and focus_key:
        focus_sess = sessions.get(focus_key)
        if focus_sess is None:
            return
        if focus_sess.sleep_wake_event is not None:
            focus_sess.last_wake_reason = wake_remark
            focus_sess.sleep_wake_from = conversation_id
            focus_sess.sleep_wake_event.set()
            logger.info(
                "[wake] 非焦点 %s mention 触发焦点 %s 的 sleep 唤醒",
                conversation_id, focus_key,
            )
        elif focus_sess.sleep_arming:
            focus_sess.sleep_pending_wake = True
            focus_sess.last_wake_reason = wake_remark
            focus_sess.sleep_wake_from = conversation_id

        # global wait：焦点会话的 wait 若 scope=global 也可被任意会话触发
        f_trig = focus_sess.wait_early_trigger
        if (
            focus_sess.wait_event is not None
            and not focus_sess.wait_event.is_set()
            and isinstance(f_trig, dict)
            and f_trig.get("scope") == "global"
        ):
            cond = f_trig.get("condition")
            if cond == "any_message" or (cond == "mentioned" and is_mention):
                focus_sess.wait_trigger_from = conversation_id
                focus_sess.wait_event.set()
                logger.info(
                    "[wake] global wait 被非焦点 %s 命中 (cond=%s)",
                    conversation_id, cond,
                )

    # 注意：非焦点的非 mention 普通消息只入 context，不打断任何 sleep / wait。


# ══════════════════════════════════════════════════════════
#  QQ adapter 事件回调
# ══════════════════════════════════════════════════════════

async def _handle_qq_adapter_message(event: dict, conversation_id: str) -> None:
    """QQ adapter 消息到达时的处理回调。"""
    client = app_state.qq_adapter_client
    assert client is not None
    bot_id = client.bot_id
    debug_xml = await qq_adapter_event_to_debug_xml(event, bot_id=bot_id, timezone=app_state.TIMEZONE)
    await broadcast_debug_xml(debug_xml, event)

    qq_adapter_cfg = app_state.qq_adapter_cfg
    if qq_adapter_cfg.get("debug_only", False):
        return

    # 响应范围过滤：白名单模式只允许白名单；自由模式放开给已接入的 QQ 会话。
    msg_type = event.get("message_type", "")
    sender_id = str(event.get("sender", {}).get("user_id", ""))
    group_id_str = event_group_id(event)
    is_temp = is_temp_private_event(event)
    private_access_type = "temp" if is_temp else "private"
    if msg_type == "private":
        if reason := whitelist_rejection_reason(qq_adapter_cfg, private_access_type, sender_id):
            logger.debug("%s，忽略", reason)
            return
    elif msg_type == "group":
        if reason := whitelist_rejection_reason(qq_adapter_cfg, "group", group_id_str):
            logger.debug("%s，忽略", reason)
            return
    else:
        logger.debug("未知消息类型 %s，忽略 (conv=%s)", msg_type, conversation_id)
        return

    # 纯语音、纯视频均可作为占位进入上下文。
    message_segs = event.get("message", [])

    session = get_or_create_session(conversation_id)
    reply_to_bot = await _is_reply_to_bot_message(message_segs, session, conversation_id)

    need_respond = should_respond(event, client.bot_id, app_state.BOT_NAME)
    if not need_respond and msg_type == "group":
        if reply_to_bot:
            need_respond = True
    if not need_respond:
        logger.debug("QQ adapter 消息不触发回复，静默记入上下文 (conv=%s)", conversation_id)

    # 设置/更新会话元信息（需在构建 ctx_entry 之前完成，以获取正确的 bot_display）
    sender = event.get("sender", {})
    sender_id = str(sender.get("user_id", ""))
    sender_nickname = sender.get("nickname", "")
    if msg_type == "group" and not session.conv_type:
        group_name, member_count, bot_card = await get_group_info(group_id_str)
        session.set_conversation_meta("group", group_id_str, group_name, member_count)
        session._qq_card = bot_card

    elif msg_type == "private" and is_temp:
        peer_name = sender.get("nickname", "")
        source_group_name = ""
        if group_id_str:
            source_group_name = event.get("group_name", "") or await get_group_name(group_id_str)
        if not session.conv_type:
            session.set_conversation_meta(
                "temp",
                str(sender.get("user_id", "")),
                peer_name,
                temp_source_group_id=group_id_str,
                temp_source_group_name=source_group_name,
            )
        else:
            if peer_name and not session.conv_name:
                session.conv_name = peer_name
            set_temp_source(session, group_id_str, source_group_name)

    elif msg_type == "private" and not session.conv_type:
        peer_name = sender.get("nickname", "")
        session.set_conversation_meta("private", str(sender.get("user_id", "")), peer_name)

    bot_display = session._qq_card or session._qq_name or ""
    ctx_entry = await qq_adapter_event_to_context(
        event, bot_id=client.bot_id, bot_display_name=bot_display, timezone=app_state.TIMEZONE,
    )
    if not ctx_entry:
        return

    session.add_to_context(ctx_entry)
    session.mark_unread_message(ctx_entry.get("message_id"))

    # 懒同步：将发送者信息写入 DB
    if msg_type == "group":
        sender_card = sender.get("card", "") or sender_nickname
        sender_role = sender.get("role", "member")
        sender_title = str(sender.get("title", "") or "") if "title" in sender else None
        sender_level = str(sender.get("level", "") or "") if "level" in sender else None
        sender_title_expire_time = int(sender.get("title_expire_time", 0) or 0) if "title_expire_time" in sender else None
        await upsert_membership(
            "qq", sender_id, group_id_str,
            nickname=sender_nickname,
            cardname=sender_card,
            title=sender_title,
            title_expire_time=sender_title_expire_time,
            level=sender_level,
            permission_level=sender_role,
        )
        if sender_title is not None:
            ctx_entry["sender_title"] = sender_title
        if sender_level is not None:
            ctx_entry["sender_level"] = sender_level
    elif msg_type == "private":
        await upsert_account("qq", sender_id, nickname=sender_nickname)

    # 下载 / 展开 / 视觉预处理
    await download_pending_images(ctx_entry)
    await expand_forward_previews(ctx_entry, client)
    if ctx_entry.get("images"):
        await asyncio.to_thread(app_state.vision_bridge.process_entry, ctx_entry)

    # 广播给前端
    _broadcast_entry = {k: v for k, v in ctx_entry.items() if k != "images"}
    await broadcast_chat_event({
        "type": "user_message",
        "conv_id": conversation_id,
        "conv_name": session.conv_name or conversation_id,
        "conv_type": session.conv_type or "unknown",
        "entry": _broadcast_entry,
    })

    try:
        await save_chat_message(conversation_id, ctx_entry)
        await upsert_chat_session(
            conversation_id,
            session.conv_type,
            session.conv_id,
            session.conv_name,
            session.temp_source_group_id,
            session.temp_source_group_name,
        )
    except Exception:
        logger.warning("[persist] 消息写入失败 conv=%s", conversation_id, exc_info=True)

    # ── 唤醒信号分发 ────────────────────────────────────────────────
    is_mention = _is_mention_level_message(
        event,
        message_segs,
        client.bot_id,
        reply_to_bot=reply_to_bot,
    )
    wake_remark = _build_passive_remark(
        event,
        message_segs,
        client.bot_id,
        reply_to_bot=reply_to_bot,
    )
    _dispatch_wake_signals(session, conversation_id, is_mention, wake_remark)

    # ── 触发"首次激活"或唤醒等待中的主循环 ──────────────────────────
    if not need_respond:
        return

    # 若 bot 当前没有焦点（启动后第一条消息），由本消息点燃主循环
    if app_state.current_focus is None:
        trigger_first_activation(initial_focus=conversation_id)
    else:
        # 焦点已有：主循环要么在思考，要么挂在 sleep/wait 等事件上。
        # _dispatch_wake_signals 已经处理了 wake event，主循环会自然下一 round
        # 看到本条消息（每 round 都重建 user prompt）。
        pass


async def _handle_qq_adapter_recall(event: dict) -> None:
    """处理群/私聊撤回通知，将对应 DB/上下文条目替换为撤回提示。"""
    notice_type = event.get("notice_type", "")
    message_id = str(event.get("message_id", ""))
    if not message_id:
        return

    temp_conv_id = ""
    if notice_type == "group_recall":
        group_id = str(event.get("group_id", ""))
        sender_id = str(event.get("user_id", ""))
        operator_id = str(event.get("operator_id", "") or sender_id)
        conv_id = f"group_{group_id}"
        operator_info = await _resolve_group_member_display_info(group_id, operator_id)
        recall_entry = build_recall_notice_entry(
            event,
            operator_name=operator_info.get("nickname", ""),
            operator_card=operator_info.get("card", ""),
            operator_role=operator_info.get("permission_level", ""),
            timezone=app_state.TIMEZONE,
        )
    else:  # friend_recall
        peer_id = str(event.get("user_id", ""))
        conv_id = f"private_{peer_id}"
        temp_conv_id = make_temp_session_key(peer_id)
        peer_name = await get_display_name("qq", peer_id, None)
        recall_entry = build_recall_notice_entry(
            event,
            operator_name=peer_name,
            timezone=app_state.TIMEZONE,
        )

    if not recall_entry:
        return

    try:
        db_updated = await update_chat_message_recalled(
            message_id,
            recall_entry.get("content", ""),
            recall_entry.get("timestamp", ""),
            recall_entry.get("content_segments", []),
            session_key=conv_id,
        )
        if not db_updated and notice_type == "friend_recall":
            # LLBot reports self-sent private recalls with the bot id as
            # user_id, while the stored session is keyed by the peer id.
            for fallback_key in (temp_conv_id, ""):
                db_updated = await update_chat_message_recalled(
                    message_id,
                    recall_entry.get("content", ""),
                    recall_entry.get("timestamp", ""),
                    recall_entry.get("content_segments", []),
                    session_key=fallback_key,
                )
                if db_updated:
                    break
    except Exception:
        db_updated = False
        logger.warning("[persist] 撤回状态写入DB失败 conv=%s msg_id=%s", conv_id, message_id, exc_info=True)

    session = sessions.get(conv_id)
    context_updated = bool(session and session.replace_message_with_note(message_id, recall_entry))
    if not context_updated and notice_type == "friend_recall":
        fallback_keys = [temp_conv_id]
        fallback_keys.extend(
            session_key
            for session_key in list(sessions)
            if session_key != conv_id and (session_key.startswith("private_") or session_key.startswith("temp_"))
        )
        for session_key in fallback_keys:
            if not session_key or session_key == conv_id:
                continue
            candidate = sessions.get(session_key)
            if candidate is None:
                continue
            if candidate.replace_message_with_note(message_id, recall_entry):
                context_updated = True
                break
    if db_updated or context_updated:
        logger.debug("撤回通知已处理: conv=%s msg_id=%s db=%s context=%s", conv_id, message_id, db_updated, context_updated)


async def _handle_qq_adapter_poke(event: dict) -> None:
    """处理戳一戳通知，将动作文本作为 note 记入上下文。"""
    group_id = str(event.get("group_id", ""))
    sender_id = str(event.get("user_id", ""))
    target_id = str(event.get("target_id", ""))
    action = event.get("action") or "戳了戳"
    suffix = event.get("suffix") or ""

    client = app_state.qq_adapter_client
    bot_id = client.bot_id if client else ""

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

    # 响应范围过滤
    if group_id:
        if whitelist_rejection_reason(app_state.qq_adapter_cfg, "group", group_id):
            return
    else:
        if whitelist_rejection_reason(app_state.qq_adapter_cfg, "private", sender_id):
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

    # 戳到 bot 视为 mention 级别的唤醒
    if bot_id and str(target_id) == str(bot_id):
        _dispatch_wake_signals(
            session, conv_id, is_mention=True, wake_remark=f"被 {sender_name} 戳了一下",
        )
        if app_state.current_focus is None:
            trigger_first_activation(initial_focus=conv_id)


async def _handle_qq_adapter_group_notice(event: dict) -> None:
    """处理群成员/权限系统通知，将客观变动写入聊天窗口 note。"""
    notice_type = str(event.get("notice_type", ""))
    if notice_type not in _GROUP_SYSTEM_NOTICE_TYPES:
        return
    if app_state.qq_adapter_cfg.get("debug_only", False):
        return

    group_id = str(event.get("group_id", "") or "")
    if not group_id:
        return

    if whitelist_rejection_reason(app_state.qq_adapter_cfg, "group", group_id):
        return

    conv_id = f"group_{group_id}"
    session = get_or_create_session(conv_id)
    if not session.conv_type:
        group_name, member_count, bot_card = await get_group_info(group_id)
        session.set_conversation_meta("group", group_id, group_name, member_count)
        session._qq_card = bot_card

    if notice_type == "group_card":
        await _handle_group_card_notice(event, group_id)
        return

    operator_id = str(event.get("operator_id", "") or "")
    target_id = str(event.get("user_id", "") or "")
    operator_info = (
        await get_group_member_display_info("qq", operator_id, group_id)
        if operator_id else {"card": "", "nickname": ""}
    )
    target_info = (
        await get_group_member_display_info("qq", target_id, group_id)
        if target_id else {"card": "", "nickname": ""}
    )

    note_entry = build_group_notice_entry(
        event,
        operator_name=operator_info.get("nickname", ""),
        operator_card=operator_info.get("card", ""),
        target_name=target_info.get("nickname", ""),
        target_card=target_info.get("card", ""),
        timezone=app_state.TIMEZONE,
    )
    if not note_entry:
        return

    if target_id and notice_type in ("group_increase", "group_admin"):
        permission = "member"
        if notice_type == "group_admin" and str(event.get("sub_type", "")) == "set":
            permission = "admin"
        await upsert_membership(
            "qq",
            target_id,
            group_id,
            nickname=target_info.get("nickname", ""),
            cardname=target_info.get("card", ""),
            permission_level=permission,
        )

    session.add_to_context(note_entry)
    try:
        await save_chat_message(conv_id, note_entry)
        await upsert_chat_session(conv_id, session.conv_type, session.conv_id, session.conv_name)
    except Exception:
        logger.warning("[persist] 群系统 note 写入失败 conv=%s type=%s", conv_id, notice_type, exc_info=True)

    await broadcast_chat_event({
        "type": "system_notice",
        "conv_id": conv_id,
        "conv_name": session.conv_name or conv_id,
        "conv_type": session.conv_type or "group",
        "entry": note_entry,
    })


async def _handle_group_card_notice(event: dict, group_id: str) -> None:
    """Refresh current membership card state from a group-card notice."""
    target_id = str(event.get("user_id", "") or "").strip()
    if not target_id:
        return

    current_info = await get_group_member_display_info("qq", target_id, group_id)
    remote_info = await _fetch_group_member_info_from_qq_adapter(group_id, target_id)

    event_card = (
        event.get("card_new")
        or event.get("new_card")
        or event.get("card")
        or event.get("cardname")
        or ""
    )
    nickname = (
        (remote_info or {}).get("nickname", "")
        or current_info.get("nickname", "")
    )
    card = (
        (remote_info or {}).get("card", "")
        if remote_info is not None
        else str(event_card or "")
    )
    permission_level = (
        (remote_info or {}).get("permission_level", "")
        or current_info.get("permission_level", "")
        or "member"
    )

    await upsert_membership(
        "qq",
        target_id,
        group_id,
        nickname=nickname,
        cardname=card,
        title=(remote_info or {}).get("title", None),
        title_expire_time=(remote_info or {}).get("title_expire_time", None),
        level=(remote_info or {}).get("level", None),
        permission_level=permission_level,
    )

    client = app_state.qq_adapter_client
    bot_id = str(getattr(client, "bot_id", "") or "")
    if bot_id and target_id == bot_id:
        group_name, member_count, _old_bot_card = await get_group_info(group_id)
        await upsert_group(group_id, group_name, card, member_count)
        for sess in sessions.values():
            if sess.conv_type == "group" and str(sess.conv_id) == str(group_id):
                sess._qq_card = card

    logger.info("[QQ adapter] 群名片已同步 group=%s user=%s card=%r", group_id, target_id, card)


# ══════════════════════════════════════════════════════════
#  注册入口
# ══════════════════════════════════════════════════════════

def register_qq_adapter_handlers() -> None:
    """将消息 / 撤回 / 戳一戳回调注册到 QQ adapter 客户端。"""
    client = app_state.qq_adapter_client
    if not client:
        return
    client.set_message_handler(_handle_qq_adapter_message)
    client.set_recall_handler(_handle_qq_adapter_recall)
    client.set_poke_handler(_handle_qq_adapter_poke)
    client.set_group_notice_handler(_handle_qq_adapter_group_notice)
