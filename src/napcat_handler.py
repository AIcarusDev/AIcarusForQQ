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
  - 主动循环（continue / wait / shift / sleep）
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
    save_adapter_contents,
    update_chat_message_recalled,
    upsert_account,
    upsert_chat_session,
    upsert_membership,
)
from web.debug_server import broadcast_debug_xml, broadcast_chat_event
from llm.core.retry import call_model_with_retry
from llm.core.provider import LLMCallFailed
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
import llm.prompt.activity_log as activity_log

logger = logging.getLogger("AICQ.app")

# 标记：consciousness_task 已被 create_task 调度但尚未拿到 consciousness_lock 的短暂窗口
# 用于防止在该窗口内有新消息误判为「意识空闲」并重复触发
_spawning_consciousness: bool = False


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
        return "收到@，被动激活"
    msg_type = event.get("message_type", "")
    if msg_type == "private":
        return "收到私聊消息，被动激活"
    return "被动激活"


# ══════════════════════════════════════════════════════════
#  Bot 消息发送 & 入上下文
# ══════════════════════════════════════════════════════════

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
#  自然醒计时器
# ══════════════════════════════════════════════════════════

async def _natural_wake(session, conv_key: str, delay_secs: float) -> None:
    """自然醒计时器：delay_secs 秒后尝试主动激活会话。"""
    try:
        await asyncio.sleep(delay_secs)
    except asyncio.CancelledError:
        return  # 被动唤醒已取消本计时器

    session.sleep_wake_task = None

    global _spawning_consciousness
    if app_state.consciousness_lock.locked() or _spawning_consciousness:
        logger.info("[natural_wake] 意识正忙，自然醒跳过 (conv=%s)", conv_key)
        return

    _spawning_consciousness = True
    try:
        async with app_state.consciousness_lock:
            _spawning_consciousness = False
            app_state.current_focus = conv_key
            try:
                await activity_log.open_entry(
                    "chat",
                    enter_attitude="active",
                    enter_motivation="自然醒",
                    conv_type=session.conv_type,
                    conv_id=session.conv_id,
                    conv_name=session.conv_name or conv_key,
                )
                try:
                    loop_action, tool_calls_log, _, elapsed = await call_model_with_retry(session, conv_key)
                except LLMCallFailed as e:
                    logger.warning("[natural_wake] LLM 调用失败 (conv=%s): %s", conv_key, e)
                    return
                except Exception:
                    logger.exception("[natural_wake] LLM 调用失败 (conv=%s)", conv_key)
                    return
                logger.info("[natural_wake] LLM 响应耗时 %.2fs (conv=%s)", elapsed, conv_key)
                try:
                    group_id = int(session.conv_id) if session.conv_type == "group" else None
                    user_id = int(session.conv_id) if session.conv_type == "private" else None
                except (ValueError, TypeError):
                    group_id = None
                    user_id = None
                await save_bot_turn(
                    turn_id=uuid.uuid4().hex,
                    conv_type=session.conv_type,
                    conv_id=session.conv_id,
                    result=loop_action,
                    tool_calls_log=tool_calls_log,
                )
                await _run_active_loop(session, conv_key, group_id, user_id, loop_action, tool_calls_log)
            finally:
                app_state.current_focus = None
    except Exception:
        _spawning_consciousness = False
        logger.exception("[natural_wake] 意识任务执行异常 (conv=%s)", conv_key)


# ══════════════════════════════════════════════════════════
#  主动循环 & Shift
# ══════════════════════════════════════════════════════════

async def _run_active_loop(
    session,
    conv_key: str,
    group_id,
    user_id,
    first_loop_action: dict | None,
    first_tool_calls_log: list,
) -> None:
    """主动循环公共逻辑：支持 wait / shift / sleep（loop_action 驱动）。"""
    loop_action = first_loop_action
    tool_calls_log = first_tool_calls_log

    while True:
        action = (loop_action or {}).get("action", "sleep")

        if action == "sleep" or loop_action is None:
            _break_motivation = (loop_action or {}).get("motivation", "")
            _duration_min = (loop_action or {}).get("duration", 60)
            # 持久化意识流检查点，确保重启后可恢复
            _c_data, _ts_data = app_state.consciousness_flow.dump()
            asyncio.create_task(save_adapter_contents("flow", _c_data, _ts_data))
            await activity_log.close_current(
                end_attitude="active",
                end_action="sleep",
                end_motivation=_break_motivation,
            )
            # sleep：进入休眠挂起状态，等待被动唤醒或自然醒计时器
            await activity_log.open_entry("hibernate", hibernate_minutes=_duration_min)
            # 调度自然醒计时器
            _wake_task = asyncio.create_task(
                _natural_wake(session, conv_key, _duration_min * 60)
            )
            session.sleep_wake_task = _wake_task
            logger.info("[sleep] 自然醒计时器已启动 duration=%dmin (conv=%s)", _duration_min, conv_key)
            break

        elif action == "wait":
            timeout = loop_action.get("timeout", 60)
            trigger = loop_action.get("early_trigger")
            session.wait_early_trigger = trigger
            session.wait_event = asyncio.Event()
            # 消费打字期间积累的 pending trigger：若匹配则直接 pre-fire，跳过实际等待
            _pending = session.pending_early_trigger
            session.pending_early_trigger = None
            if trigger and _pending and isinstance(trigger, dict):
                _scope = trigger["scope"]
                _cond = trigger["condition"]
                # global scope 不使用 pending（pending 是单会话维度的）
                if _scope == "session" and (
                    _cond == "any_message"
                    or (_cond == "mentioned" and _pending == "mentioned")
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
            _wait_t0 = time.monotonic()
            _resume_reason = "timeout"
            try:
                await asyncio.wait_for(session.wait_event.wait(), timeout=timeout)
                _resume_reason = "triggered"
                logger.info("会话 %s 等待提前结束 (early_trigger=%s)", conv_key, trigger)
            except asyncio.TimeoutError:
                logger.info("会话 %s 等待超时，继续下一轮循环", conv_key)
            finally:
                session.wait_event = None
                session.wait_early_trigger = None
                _trigger_from_key = session.wait_trigger_from
                session.wait_trigger_from = None

            # 补完 wait 的延迟返回，模型下一轮能看到"为什么恢复了"
            _elapsed = round(time.monotonic() - _wait_t0, 1)
            _trigger_from_meta: dict | None = None
            if _resume_reason == "triggered" and _trigger_from_key:
                _src = get_or_create_session(_trigger_from_key)
                if _src.conv_type:
                    _trigger_from_meta = {
                        "type": _src.conv_type,
                        "id": _src.conv_id,
                        "name": _src.conv_name,
                    }
            app_state.consciousness_flow.complete_deferred_response("wait", {
                "ok": True,
                "resumed": _resume_reason,
                "trigger_kind": trigger if _resume_reason == "triggered" else None,
                "trigger_from": _trigger_from_meta,
                "elapsed_seconds": _elapsed,
            })

        elif action == "shift":
            shift_type = loop_action.get("type", "")
            shift_id = str(loop_action.get("id", ""))
            shift_key = f"{shift_type}_{shift_id}"
            shift_motivation = loop_action.get("motivation", "")
            shift_error = await _validate_shift_target(shift_type, shift_id)
            if shift_error:
                logger.warning("会话 %s shift 验证失败: %s，继续本会话", conv_key, shift_error)
                # shift 验证失败：继续当前会话，下一轮循环重新决策
            else:
                next_session = get_or_create_session(shift_key)
                if not next_session.conv_type:
                    next_session.set_conversation_meta(shift_type, shift_id)
                if not next_session.conv_name:
                    next_session.conv_name = await _resolve_conv_name(shift_type, shift_id)
                from_name = session.conv_name or await _resolve_conv_name(session.conv_type, session.conv_id)
                await activity_log.close_current(
                    end_attitude="active",
                    end_action="shift",
                    end_motivation=shift_motivation,
                )
                _from_str = f"{session.conv_type}:{session.conv_id}:{from_name}".rstrip(":")
                asyncio.create_task(
                    _activate_session_shifted(
                        next_session, shift_key, shift_type, shift_id,
                        shift_motivation=shift_motivation,
                        shift_from=_from_str,
                    )
                )
                break

        # 继续：进行下一轮 LLM 决策
        session.pending_early_trigger = None
        try:
            loop_action, tool_calls_log, _, elapsed = await call_model_with_retry(session, conv_key)
        except LLMCallFailed as e:
            logger.warning("主动循环 LLM 调用失败 (conv=%s): %s", conv_key, e)
            break
        except Exception:
            logger.exception("主动循环 LLM 调用失败 (conv=%s)", conv_key)
            break
        logger.info("LLM 响应耗时（主动循环）%.2fs (conv=%s)", elapsed, conv_key)

        await save_bot_turn(
            turn_id=uuid.uuid4().hex,
            conv_type=session.conv_type,
            conv_id=session.conv_id,
            result=loop_action,
            tool_calls_log=tool_calls_log,
        )


async def _activate_session_shifted(
    target_session,
    target_key: str,
    target_type: str,
    target_id: str,
    shift_motivation: str = "",
    shift_from: str = "",
) -> None:
    """shift 切换后在目标会话激活一轮循环，并持续处理后续 loop_action（含递归 shift）。"""
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

            await activity_log.open_entry(
                "chat",
                enter_attitude="active",
                enter_motivation=shift_motivation,
                enter_from=shift_from,
                conv_type=target_session.conv_type,
                conv_id=target_session.conv_id,
                conv_name=target_session.conv_name or target_key,
            )

            try:
                loop_action, tool_calls_log, _, elapsed = await call_model_with_retry(target_session, target_key)
            except LLMCallFailed as e:
                logger.warning("[shift] 目标会话 %s LLM 调用失败: %s", target_key, e)
                return
            except Exception:
                logger.exception("[shift] 目标会话 %s LLM 调用失败", target_key)
                return

            logger.info("[shift] 目标会话 %s LLM 响应耗时 %.2fs", target_key, elapsed)
            await save_bot_turn(
                turn_id=uuid.uuid4().hex,
                conv_type=target_session.conv_type,
                conv_id=target_session.conv_id,
                result=loop_action,
                tool_calls_log=tool_calls_log,
            )
            await _run_active_loop(target_session, target_key, group_id, user_id, loop_action, tool_calls_log)
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

    # 提前构建并入上下文，最小化与发送循环 IS 检测之间的竞态窗口
    bot_display = session._qq_card or session._qq_name or ""
    ctx_entry = await napcat_event_to_context(event, bot_id=client.bot_id, bot_display_name=bot_display, timezone=app_state.TIMEZONE)
    if not ctx_entry:
        return

    session.add_to_context(ctx_entry)
    session.unread_count += 1

    # 懒同步：将发送者信息写入 DB（不影响 IS 时效性，可在入上下文之后执行）
    if msg_type == "group":
        sender_card = sender.get("card", "") or sender_nickname
        sender_role = sender.get("role", "member")
        sender_title = sender.get("title", "")
        await upsert_membership(
            "qq", sender_id, group_id_str,
            cardname=sender_card,
            title=sender_title,
            permission_level=sender_role,
        )
    elif msg_type == "private":
        await upsert_account("qq", sender_id, nickname=sender_nickname)

    # 下载待获取的图片（URL类型），原地更新 entry（引用语义，自动对上下文生效）
    await download_pending_images(ctx_entry)

    # 展开合并转发预览（API 调用填充预览数据，原地修改同步至上下文）
    await expand_forward_previews(ctx_entry, client)

    # 图片落盘 + pHash 去重 + 视觉描述（有图时在后台线程执行，不阻塞事件循环）
    if ctx_entry.get("images"):
        await asyncio.to_thread(app_state.vision_bridge.process_entry, ctx_entry)

    # 广播到日志页面聊天记录（剔除 base64 图片数据）
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
        await upsert_chat_session(conversation_id, session.conv_type, session.conv_id, session.conv_name)
    except Exception:
        logger.warning("[persist] 消息写入失败 conv=%s", conversation_id, exc_info=True)

    # ── early_trigger 检查 ───────────────────────────────────────────────────
    # 计算本条消息是否提及 bot（供多处复用）
    _is_mention = (
        any(
            seg.get("type") == "at"
            and str(seg.get("data", {}).get("qq", "")) == str(client.bot_id)
            for seg in message_segs
        )
        or get_reply_message_id(message_segs) is not None
    )
    _is_focused = app_state.consciousness_lock.locked() and app_state.current_focus == conversation_id
    if _is_focused and session.wait_event is not None and not session.wait_event.is_set():
        # 焦点会话正在等待：按 condition 决定是否唤醒
        _trig = session.wait_early_trigger
        if isinstance(_trig, dict):
            _cond = _trig["condition"]
            if _cond == "any_message" or (_cond == "mentioned" and _is_mention):
                logger.info(
                    "会话 %s early_trigger 条件满足 (condition=%s)，唤醒等待循环",
                    conversation_id, _cond,
                )
                session.wait_event.set()
    elif _is_focused and session.wait_event is None:
        # 意识正忙聚焦于本会话，但 wait_event 尚未创建（仍在发送阶段），提前记录触发强度
        if _is_mention:
            session.pending_early_trigger = "mentioned"
        elif session.pending_early_trigger is None:
            session.pending_early_trigger = "any_message"
    elif app_state.consciousness_lock.locked() and not _is_focused:
        # global scope：意识正忙于其他会话，检查焦点会话是否有全局触发条件
        _fk = app_state.current_focus
        if _fk:
            _fs = get_or_create_session(_fk)
            _ft = _fs.wait_early_trigger
            if (
                isinstance(_ft, dict)
                and _ft.get("scope") == "global"
                and _fs.wait_event is not None
                and not _fs.wait_event.is_set()
            ):
                _cond = _ft["condition"]
                if _cond == "any_message" or (_cond == "mentioned" and _is_mention):
                    logger.info(
                        "global early_trigger 条件满足 (condition=%s, 来自会话 %s)，唤醒焦点会话 %s 等待循环",
                        _cond, conversation_id, _fk,
                    )
                    _fs.wait_trigger_from = conversation_id
                    _fs.wait_event.set()

    if not need_respond:
        return

    # 严格唯一性：意识同一时刻只能处理一个会话，正忙时其它激活请求一律跳过
    global _spawning_consciousness
    if app_state.consciousness_lock.locked() or _spawning_consciousness:
        logger.info("意识正忙（焦点=%s），消息已记入 %s 上下文，本次激活跳过", app_state.current_focus, conversation_id)
        return

    # 标记「即将产生意识任务」，防止在 create_task 到拿锁的短暂空窗内被重复激活
    _spawning_consciousness = True

    async def _consciousness_task() -> None:
        global _spawning_consciousness
        try:
            async with app_state.consciousness_lock:
                _spawning_consciousness = False
                app_state.current_focus = conversation_id
                # 取消自然醒计时器（如有），避免被动唤醒后重复激活
                if session.sleep_wake_task and not session.sleep_wake_task.done():
                    session.sleep_wake_task.cancel()
                    session.sleep_wake_task = None
                try:
                    _remark = _build_passive_remark(event, message_segs, client.bot_id)
                    await activity_log.open_entry(
                        "chat",
                        enter_attitude="passive",
                        enter_remark=_remark,
                        conv_type=session.conv_type,
                        conv_id=session.conv_id,
                        conv_name=session.conv_name or conversation_id,
                    )

                    try:
                        loop_action, tool_calls_log, _, elapsed = await call_model_with_retry(session, conversation_id)
                    except LLMCallFailed as e:
                        logger.warning("NapCat LLM 调用失败 (conv=%s): %s", conversation_id, e)
                        return
                    except Exception:
                        logger.exception("NapCat LLM 调用失败 (conv=%s)", conversation_id)
                        return
                    logger.info("LLM 响应耗时 %.2fs (conv=%s)", elapsed, conversation_id)

                    msg_type = event.get("message_type", "")
                    group_id = event.get("group_id") if msg_type == "group" else None
                    user_id = event.get("sender", {}).get("user_id") if msg_type == "private" else None

                    await save_bot_turn(
                        turn_id=uuid.uuid4().hex,
                        conv_type=session.conv_type,
                        conv_id=session.conv_id,
                        result=loop_action,
                        tool_calls_log=tool_calls_log,
                    )

                    # 主动循环：支持 wait / sleep / shift
                    await _run_active_loop(session, conversation_id, group_id, user_id, loop_action, tool_calls_log)
                finally:
                    app_state.current_focus = None
        except Exception:
            _spawning_consciousness = False
            logger.exception("意识任务执行异常 (conv=%s)", conversation_id)

    # 后台调度：立即释放 _conv_locks[conv_id]，让后续消息能够实时入上下文
    asyncio.create_task(_consciousness_task())


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
