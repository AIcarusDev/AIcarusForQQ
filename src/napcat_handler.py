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

新架构下，本模块**只**做事件 → 状态的桥接：
- 消息接收：白名单过滤、入上下文、广播
- 唤醒信号：mention / poke / 焦点会话新消息 → set 对应 session 的 sleep/wait event
- 撤回 / 戳一戳通知

**不再驱动任何 LLM 调用**。bot 的"思考"由 ``consciousness.main_loop`` 这条
常驻协程独立持有。本模块只负责把外部世界的变化写入共享状态。
"""

import asyncio
import logging
from datetime import datetime

import app_state
from consciousness import trigger_first_activation
from database import (
    get_display_name,
    get_group_info,
    get_group_name,
    save_chat_message,
    update_chat_message_recalled,
    upsert_account,
    upsert_chat_session,
    upsert_membership,
)
from web.debug_server import broadcast_debug_xml, broadcast_chat_event
from napcat import (
    get_reply_message_id,
    napcat_event_to_context,
    napcat_event_to_debug_xml,
    download_pending_images,
    expand_forward_previews,
    should_respond,
)
from llm.session import (
    get_or_create_session,
    sessions,
)

logger = logging.getLogger("AICQ.app")


# ══════════════════════════════════════════════════════════
#  辅助：会话名解析
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


def _build_passive_remark(event: dict, message_segs: list, bot_id: str | None) -> str:
    """根据消息类型生成被动激活的 remark 描述。"""
    if get_reply_message_id(message_segs):
        return "收到回复，被动激活"
    is_at = any(
        seg.get("type") == "at"
        and str(seg.get("data", {}).get("qq", "")) == str(bot_id)
        for seg in message_segs
    )
    if is_at:
        return "被@叫醒了"
    msg_type = event.get("message_type", "")
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
            else:
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
        focus_sess.last_wake_reason = wake_remark
        focus_sess.sleep_wake_from = conversation_id
        if focus_sess.sleep_wake_event is not None:
            focus_sess.sleep_wake_event.set()
            logger.info(
                "[wake] 非焦点 %s mention 触发焦点 %s 的 sleep 唤醒",
                conversation_id, focus_key,
            )
        else:
            focus_sess.sleep_pending_wake = True

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

    # 设置/更新会话元信息（需在构建 ctx_entry 之前完成，以获取正确的 bot_display）
    sender = event.get("sender", {})
    sender_id = str(sender.get("user_id", ""))
    sender_nickname = sender.get("nickname", "")
    if msg_type == "group" and not session.conv_type:
        group_name, member_count, bot_card = await get_group_info(group_id_str)
        session.set_conversation_meta("group", group_id_str, group_name, member_count)
        session._qq_card = bot_card

    elif msg_type == "private" and not session.conv_type:
        peer_name = sender.get("nickname", "")
        session.set_conversation_meta("private", str(sender.get("user_id", "")), peer_name)

    bot_display = session._qq_card or session._qq_name or ""
    ctx_entry = await napcat_event_to_context(
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
        sender_title = sender.get("title", "")
        await upsert_membership(
            "qq", sender_id, group_id_str,
            nickname=sender_nickname,
            cardname=sender_card,
            title=sender_title,
            permission_level=sender_role,
        )
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
            conversation_id, session.conv_type, session.conv_id, session.conv_name,
        )
    except Exception:
        logger.warning("[persist] 消息写入失败 conv=%s", conversation_id, exc_info=True)

    # ── 唤醒信号分发 ────────────────────────────────────────────────
    is_mention = (
        any(
            seg.get("type") == "at"
            and str(seg.get("data", {}).get("qq", "")) == str(client.bot_id)
            for seg in message_segs
        )
        or get_reply_message_id(message_segs) is not None
    )
    wake_remark = _build_passive_remark(event, message_segs, client.bot_id)
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
        try:
            await update_chat_message_recalled(message_id, operator_name, timestamp)
        except Exception:
            logger.warning("[persist] 撤回状态写入DB失败 msg_id=%s", message_id, exc_info=True)
        logger.debug("撤回通知已处理: conv=%s msg_id=%s operator=%s", conv_id, message_id, operator_name)


async def _handle_napcat_poke(event: dict) -> None:
    """处理戳一戳通知，将动作文本作为 note 记入上下文。"""
    group_id = str(event.get("group_id", ""))
    sender_id = str(event.get("user_id", ""))
    target_id = str(event.get("target_id", ""))
    action = event.get("action") or "戳了戳"
    suffix = event.get("suffix") or ""

    client = app_state.napcat_client
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

    # 戳到 bot 视为 mention 级别的唤醒
    if bot_id and str(target_id) == str(bot_id):
        _dispatch_wake_signals(
            session, conv_id, is_mention=True, wake_remark=f"被 {sender_name} 戳了一下",
        )
        if app_state.current_focus is None:
            trigger_first_activation(initial_focus=conv_id)


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
