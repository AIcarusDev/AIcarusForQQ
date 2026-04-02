"""watcher_core.py — Watcher（窥屏意识）调度逻辑

在主模型选择 loop_control.idle 后，后台周期性唤醒一个轻量模型，
由它决定要不要重新激活主意识投入对话。

流程：
  idle → schedule_watcher() → 后台 run_watcher_loop() 每隔 interval 秒...
    └→ pass    → 继续睡，等下一轮
    └→ engage  → 把 mood/think/intent 注入 session.watcher_nudge
                  → 调用主模型完整运行一轮（含 loop_control）
"""

import asyncio
import logging
import random
import time
import uuid
from datetime import datetime

from llm.core.schema import WATCHER_SCHEMA
from llm.session import get_bot_previous_cycle, get_bot_previous_cycle_time
import app_state
from .watcher_prompt import build_watcher_system_prompt
import llm.prompt.activity_log as activity_log
from hibernate.hibernate_core import run_hibernate

logger = logging.getLogger("AICQ.watcher")


# ══════════════════════════════════════════════════════════
#  激活回调（由 napcat_handler 注册，避免循环导入）
# ══════════════════════════════════════════════════════════

_engage_session_handler = None  # type: ignore[assignment]


def register_engage_handler(fn) -> None:
    """由 napcat_handler 在模块加载时注入切入回调，避免与 napcat_handler 的循环依赖。"""
    global _engage_session_handler
    _engage_session_handler = fn


# ══════════════════════════════════════════════════════════
#  内部：调用 watcher 模型
# ══════════════════════════════════════════════════════════

def _build_watcher_gen() -> dict:
    """构建 watcher 的生成参数（允许 get_list 工具调用）。"""
    watcher_cfg = app_state.watcher_cfg
    gen = watcher_cfg.get("generation", {})
    return {
        "temperature": gen.get("temperature", 0.7),
        "max_output_tokens": gen.get("max_output_tokens", 800),
    }


def _call_watcher_model(
    session,
) -> dict | None:
    """同步调用 watcher 模型，返回解析后的结果字典，失败返回 None。"""

    adapter = app_state.watcher_adapter
    if adapter is None:
        return None

    watcher_cfg = app_state.watcher_cfg
    model_name = watcher_cfg.get("model_name", watcher_cfg.get("model", "watcher"))

    system_prompt = build_watcher_system_prompt(
        persona=session._persona,
        qq_name=session._qq_name,
        qq_id=session._qq_id,
        model_name=model_name,
        now=datetime.now(session._timezone) if session._timezone else None,
        previous_cycle_result=get_bot_previous_cycle(),
        previous_cycle_time=get_bot_previous_cycle_time(),
    )

    def prompt_builder(activated_names=None, latent_names=None):
        return system_prompt

    from llm.prompt.unread_builder import prepare_chat_log_with_unread
    from llm.prompt.final_reminder import append_final_reminder
    from tools import build_tools
    chat_log = prepare_chat_log_with_unread(session)
    chat_log = append_final_reminder(chat_log, session)
    gen = _build_watcher_gen()

    # 窥屏模式：只收录 WATCHER_ALLOW=True 的工具
    tool_declarations, tool_registry, _ = build_tools(
        app_state.config,
        napcat_client=app_state.napcat_client,
        is_watcher=True,
    )

    result, _, _, _, _ = adapter.call(
        prompt_builder,
        chat_log,
        gen,
        WATCHER_SCHEMA,
        tool_declarations=tool_declarations,
        tool_registry=tool_registry,
    )
    return result


# ══════════════════════════════════════════════════════════
#  辅助：shift 目标校验 & 会话列表
# ══════════════════════════════════════════════════════════

async def _validate_watcher_shift_target(target_type: str, target_id: str) -> str | None:
    """检查 watcher shift 目标是否在白名单和 QQ 联系人列表中。返回 None 表示合法。"""
    if target_type not in ("private", "group"):
        return f"未知会话类型 {target_type!r}"
    whitelist_cfg = app_state.napcat_cfg.get("whitelist", {})
    if target_type == "private":
        private_whitelist = [str(u) for u in whitelist_cfg.get("private_users", [])]
        if private_whitelist and target_id not in private_whitelist:
            return f"私聊用户 {target_id} 不在白名单中"
    else:
        group_whitelist = [str(g) for g in whitelist_cfg.get("group_ids", [])]
        if group_whitelist and target_id not in group_whitelist:
            return f"群聊 {target_id} 不在白名单中"
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




def _pick_random_whitelisted_session():
    """从已知会话中随机选一个在白名单内的，返回 (conv_key, session, group_id, user_id) 或 None。"""
    from llm.session import sessions as _sessions
    whitelist_cfg = app_state.napcat_cfg.get("whitelist", {})
    private_whitelist = {str(u) for u in whitelist_cfg.get("private_users", [])}
    group_whitelist = {str(g) for g in whitelist_cfg.get("group_ids", [])}
    candidates = []
    for key, sess in list(_sessions.items()):
        if not sess.conv_type or not sess.conv_id:
            continue
        conv_id = str(sess.conv_id)
        if sess.conv_type == "private":
            if private_whitelist and conv_id not in private_whitelist:
                continue
            candidates.append((key, sess, None, int(conv_id)))
        elif sess.conv_type == "group":
            if group_whitelist and conv_id not in group_whitelist:
                continue
            candidates.append((key, sess, int(conv_id), None))
    return random.choice(candidates) if candidates else None


# ══════════════════════════════════════════════════════════
#  Watcher 后台循环
# ══════════════════════════════════════════════════════════

async def run_watcher_loop(
    session,
    conv_key: str,
    group_id,
    user_id,
) -> None:
    """Watcher 后台循环：周期性窥屏，支持四种决策。

    - engage：激活主意识切入当前窥屏会话
    - shift：立即切换到另一个会话并启动新 watcher 任务
    - wait：等待指定秒数后再次窥屏同一会话
    - hibernate：休眠指定分钟数后自然醒来继续窥屏
    - pass：等待正常间隔后随机从白名单已知会话中选一个继续窥屏

    退出条件：
    - session.watcher_active 被外部清除（被动消息激活了主意识）
    - 决策 engage 或有效 shift（切入/切换后自行退出）
    - 任务被取消（CancelledError）
    """
    watcher_cfg = app_state.watcher_cfg
    interval: float = float(watcher_cfg.get("interval", 60))
    jitter: float = float(watcher_cfg.get("interval_jitter", 10))

    watch_round = 0

    # 当前窥屏目标（可随 pass/shift 变化）
    watch_session = session
    watch_conv_key = conv_key
    watch_group_id = group_id
    watch_user_id = user_id

    _random_next: bool = False  # pass 后下次清醒时随机切换窥屏目标

    logger.info(
        "[👁] 启动窥屏循环 conv=%s interval=%.0fs",
        conv_key, interval,
    )

    # ── 首轮：消息在 schedule_watcher 调用前已发出，但 consciousness_lock 仍被持有，
    # 等其释放后立即窥屏，模拟"发完消息马上回头看一眼"的人类习惯。
    try:
        while app_state.consciousness_lock.locked():
            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        logger.info("[watcher] 首轮等待 lock 期间被取消 conv=%s", conv_key)
        if session.watcher_task is asyncio.current_task():
            session.watcher_active = False
            session.watcher_task = None
        return
    logger.debug("[watcher] 首轮立即窥屏（lock 已释放）conv=%s", conv_key)

    while session.watcher_active:
        # ── pass 后随机切换窥屏目标 ──
        if _random_next:
            _random_next = False
            picked = _pick_random_whitelisted_session()
            if picked:
                new_key, new_sess, new_gid, new_uid = picked
                if new_key != watch_conv_key:
                    logger.info(
                        "[watcher] pass 随机切换窥屏目标 %s → %s",
                        watch_conv_key, new_key,
                    )
                    watch_session = new_sess
                    watch_conv_key = new_key
                    watch_group_id = new_gid
                    watch_user_id = new_uid

        # 意识正忙（正在处理其它会话），等一个间隔后再试
        if app_state.consciousness_lock.locked():
            logger.info("[watcher] 意识正忙，本轮跳过 conv=%s", watch_conv_key)
            try:
                await asyncio.sleep(max(10.0, interval + random.uniform(-jitter, jitter)))
            except asyncio.CancelledError:
                logger.info("[watcher] 窥屏循环被取消 conv=%s", watch_conv_key)
                break
            continue

        watch_round += 1
        logger.info("[watcher] 第 %d 轮窥屏开始 conv=%s", watch_round, watch_conv_key)

        _t0 = time.monotonic()
        _engaged = False
        _shift_target = None
        _action = "pass"
        _motivation: str = ""
        _wait_secs: float = 0.0
        _hibernate_minutes: int = 0
        _skip: bool = False  # 本轮模型调用失败，跳过决策直接进入正常间隔等待

        async with app_state.consciousness_lock:
            app_state.current_focus = watch_conv_key
            try:
                try:
                    await app_state.rate_limiter.acquire()
                    result = await asyncio.to_thread(
                        _call_watcher_model,
                        watch_session,
                    )
                except Exception as _watcher_exc:
                    from google.genai import errors as _genai_errors
                    _status = getattr(_watcher_exc, 'status_code', None) or getattr(_watcher_exc, 'code', None)
                    _retryable = {429, 500, 502, 503, 504}
                    if isinstance(_watcher_exc, (_genai_errors.ServerError, _genai_errors.ClientError)) and _status in _retryable:
                        logger.warning("[watcher] 模型暂时不可用 (HTTP %s)，跳过本轮 conv=%s", _status, watch_conv_key)
                    else:
                        logger.exception("[watcher] 模型调用失败 conv=%s", watch_conv_key)
                    _skip = True
                    result = None

                if result is None and not _skip:
                    logger.warning("[watcher] 模型返回为空，跳过 conv=%s", watch_conv_key)
                    _skip = True

                if not _skip:
                    assert result is not None
                    logger.info("[watcher] 窥屏耗时 %.2fs conv=%s", time.monotonic() - _t0, watch_conv_key)

                    # 持久化本轮结果，并更新内存中的上轮快照
                    _cycle_id = uuid.uuid4().hex
                    _now_ts = time.time()
                    try:
                        from database import save_watcher_cycle
                        await save_watcher_cycle(
                            cycle_id=_cycle_id,
                            conv_type=watch_session.conv_type,
                            conv_id=watch_session.conv_id,
                            result=result,
                        )
                    except Exception:
                        logger.warning("[watcher] 保存 watcher_cycle 失败 conv=%s", watch_conv_key)
                    watch_session.watcher_last_cycle = result
                    watch_session.watcher_last_cycle_time = _now_ts
                    from llm.session import set_bot_previous_cycle, set_bot_previous_cycle_time
                    set_bot_previous_cycle(result)
                    set_bot_previous_cycle_time(datetime.fromtimestamp(_now_ts, tz=app_state.TIMEZONE).isoformat())

                    decision = result.get("decision") or {}
                    _motivation = decision.get("motivation", "")
                    if "engage" in decision:
                        _action = "engage"
                    elif "shift" in decision:
                        _action = "shift"
                    elif "wait" in decision:
                        _action = "wait"
                    elif "hibernate" in decision:
                        _action = "hibernate"
                    else:
                        _action = "pass"
                    logger.info("[watcher] 决策=%s motivation=%s conv=%s", _action, _motivation, watch_conv_key)

                    if _action == "engage":
                        session.watcher_active = False
                        await activity_log.close_current(
                            end_attitude="active",
                            end_action="engage",
                            end_motivation=_motivation,
                        )
                        watch_session.watcher_nudge = {
                            "result": result,
                            "time_iso": datetime.utcfromtimestamp(_now_ts).isoformat() + "Z",
                        }
                        logger.info("[watcher] 决定开始专注聊天 conv=%s", watch_conv_key)
                        await _engage_from_watcher(watch_session, watch_conv_key, watch_group_id, watch_user_id)
                        _engaged = True

                    elif _action == "shift":
                        shift_cfg = decision["shift"]
                        s_type = shift_cfg.get("type", "")
                        s_id = str(shift_cfg.get("id", ""))
                        s_error = await _validate_watcher_shift_target(s_type, s_id)
                        if s_error:
                            logger.warning("[watcher] shift 目标无效: %s，降级为 pass conv=%s", s_error, watch_conv_key)
                            _action = "pass"
                            _random_next = True
                        else:
                            from llm.session import get_or_create_session as _gcs
                            target_key = f"{s_type}_{s_id}"
                            target_s = _gcs(target_key)
                            if not target_s.conv_type:
                                target_s.set_conversation_meta(s_type, s_id)
                            t_gid = int(s_id) if s_type == "group" else None
                            t_uid = int(s_id) if s_type == "private" else None
                            logger.info(
                                "[watcher] watcher shift %s → %s motivation=%s",
                                watch_conv_key, target_key, _motivation,
                            )
                            await activity_log.close_current(
                                end_attitude="active",
                                end_action="shift",
                                end_motivation=_motivation,
                            )
                            session.watcher_active = False
                            _shift_target = (target_s, target_key, t_gid, t_uid)
                            _engaged = True

                    elif _action == "wait":
                        _wait_secs = float(decision["wait"].get("timeout", 30))
                        logger.info("[watcher] 决定等待 %.0fs 后再看 conv=%s", _wait_secs, watch_conv_key)

                    elif _action == "hibernate":
                        _hibernate_minutes = int(decision["hibernate"].get("minutes", 60))
                        _hibernate_minutes = max(30, min(480, _hibernate_minutes))
                        logger.info("[watcher] 决定休眠 %d 分钟 conv=%s", _hibernate_minutes, watch_conv_key)
                        await activity_log.close_current(
                            end_attitude="active",
                            end_action="hibernate",
                            end_motivation=_motivation,
                        )
                        await activity_log.open_entry(
                            "hibernate",
                            hibernate_minutes=_hibernate_minutes,
                        )
                        app_state.watcher_hibernating = True

                    else:  # pass
                        logger.info("[watcher] 决定 pass，下轮随机漫游 conv=%s", watch_conv_key)
                        _random_next = True

            finally:
                app_state.current_focus = None

        # ── 锁释放后：engage/shift 退出，或执行本轮决策对应的睡眠 ──
        if _engaged:
            if _shift_target:
                target_s, target_key, t_gid, t_uid = _shift_target
                schedule_watcher(target_s, target_key, t_gid, t_uid)
            break

        if _action == "hibernate":
            woke_naturally = await run_hibernate(watch_conv_key, _hibernate_minutes)
            if not woke_naturally:
                break
        elif _action == "wait":
            logger.debug("[watcher] 等待 %.1fs 后窥屏 (conv=%s)", _wait_secs, watch_conv_key)
            try:
                await asyncio.sleep(_wait_secs)
            except asyncio.CancelledError:
                logger.info("[watcher] 窥屏循环被取消 conv=%s", watch_conv_key)
                break
        else:  # pass（含模型调用失败 _skip、shift 降级）
            sleep_time = max(10.0, interval + random.uniform(-jitter, jitter))
            logger.debug("[watcher] 等待 %.1fs 后窥屏 (conv=%s)", sleep_time, watch_conv_key)
            try:
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                logger.info("[watcher] 窥屏循环被取消 conv=%s", watch_conv_key)
                break

    # 只有当前任务仍是注册的 watcher 任务时才清理状态，
    # 防止旧任务被 cancel 后走到这里时误清掉新任务的 watcher_active。
    if session.watcher_task is asyncio.current_task():
        session.watcher_active = False
        session.watcher_task = None
    logger.info("[watcher] 窥屏循环结束 conv=%s (rounds=%d)", conv_key, watch_round)


# ══════════════════════════════════════════════════════════
#  激活主意识
# ══════════════════════════════════════════════════════════

async def _engage_from_watcher(
    session,
    conv_key: str,
    group_id,
    user_id,
) -> None:
    """watcher 决定 engage 后，委托已注入的切入处理器运行一轮主意识。

    调用者需已持有 consciousness_lock 并设置 current_focus。
    """
    if _engage_session_handler is None:
        logger.error("[watcher] 切入处理器未注册，无法切入主意识 conv=%s", conv_key)
        return
    await _engage_session_handler(session, conv_key, group_id, user_id)


# ══════════════════════════════════════════════════════════
#  公共接口
# ══════════════════════════════════════════════════════════

def schedule_watcher(session, conv_key: str, group_id, user_id) -> None:
    """在 loop_control.idle 后调度 watcher 后台任务。"""
    if not app_state.watcher_cfg.get("enabled", False):
        return
    if app_state.watcher_adapter is None:
        return

    # 如果旧 watcher 还在跑，先取消
    if session.watcher_task and not session.watcher_task.done():
        logger.info("[watcher] 取消旧 watcher 任务 conv=%s", conv_key)
        session.watcher_active = False
        session.watcher_task.cancel()

    # 重置上轮 watcher 快照，让新一轮继承本次 break 时的聊天输出而非上次 watcher 输出
    session.watcher_last_cycle = None
    session.watcher_last_cycle_time = 0.0

    session.watcher_active = True
    session.watcher_task = asyncio.create_task(
        run_watcher_loop(session, conv_key, group_id, user_id)
    )
    logger.info("[watcher] 已调度窥屏任务 conv=%s", conv_key)


def stop_watcher(session) -> None:
    """取消会话的 watcher 任务（被动消息激活主意识时调用）。"""
    if session.watcher_task and not session.watcher_task.done():
        session.watcher_active = False
        session.watcher_task.cancel()
        session.watcher_task = None


async def stop_all_watchers(reason: str = "") -> None:
    """停止所有会话的 watcher 任务（意识介入时调用，确保单一意识流）。"""
    from llm.session import sessions
    # 如果当前有运行中的 watcher 条目，标记为被动中断
    await activity_log.close_current(
        end_attitude="passive",
        end_action="interrupted",
        end_remark=reason or "另一会话被动激活，意识切换",
    )
    app_state.watcher_hibernating = False
    for s in sessions.values():
        if s.watcher_task and not s.watcher_task.done():
            s.watcher_active = False
            s.watcher_task.cancel()
            s.watcher_task = None
