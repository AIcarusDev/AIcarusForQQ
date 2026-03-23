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

"""lifecycle.py — Quart 应用生命周期钩子

startup()：初始化数据库、恢复历史会话、清理缓存、启动 NapCat。
shutdown()：停止 NapCat 连接。
"""

import asyncio
import logging

import app_state
from database import (
    init_db,
    get_bot_self,
    upsert_bot_self,
    upsert_group,
    load_chat_sessions,
    load_chat_messages,
    load_last_bot_turn,
    load_activity_log,
    update_activity_entry,
)
from llm.image_cache import evict_cache
from llm.session import (
    get_or_create_session,
    update_bot_info,
    set_bot_previous_cycle,
    set_bot_previous_cycle_time,
    set_bot_previous_tool_calls,
)
import llm.activity_log as _activity_log

logger = logging.getLogger("AICQ.app")


async def startup() -> None:
    """Quart before_serving 钩子。"""
    await init_db()

    # 恢复活动日志（加载最近 N 条，并标记上次进程中断遗留的未关闭条目）
    _log_max = int(app_state.config.get("activity_log", {}).get("max_entries", 10))
    _activity_log.configure(_log_max)
    _activity_rows = await load_activity_log(limit=_log_max)
    _activity_log.restore_from_db(_activity_rows)
    _interrupted = _activity_log.get_current()
    if _interrupted is not None:
        _interrupted = _activity_log.close_current_sync(
            end_attitude="passive",
            end_action="interrupted",
            end_remark="进程中断",
        )
        if _interrupted is not None:
            await update_activity_entry(_interrupted)
        logger.info("[startup] activity_log: 已标记上次中断的未关闭条目")

    # 恢复 bot 上一轮输出（previous_cycle_json）
    _last_turn, _last_tool_calls, _last_turn_time = await load_last_bot_turn()
    if _last_turn:
        set_bot_previous_cycle(_last_turn)
        logger.info("[startup] 已从数据库恢复 previous_cycle_json")
    if _last_turn_time:
        set_bot_previous_cycle_time(_last_turn_time)
    if _last_tool_calls:
        set_bot_previous_tool_calls(_last_tool_calls)
        logger.info("[startup] 已从数据库恢复 previous_tool_calls")

    # 恢复历史 QQ 会话上下文（web 会话每次重启重置，不恢复）
    for _smeta in await load_chat_sessions():
        _key = _smeta["session_key"]
        if _key == "web":
            continue
        _msgs = await load_chat_messages(_key, limit=app_state.MAX_CONTEXT)
        if not _msgs:
            continue
        _s = get_or_create_session(_key)
        _s.set_conversation_meta(_smeta["conv_type"], _smeta["conv_id"], _smeta["conv_name"])
        _s.context_messages = list(_msgs)
        logger.info("[startup] 已恢复会话 %s (%d 条消息)", _key, len(_msgs))

    # 启动时清理过期 / 超量的图片缓存
    _evict_cfg = app_state.config.get("vision_bridge", {}).get("cache_eviction", {})
    try:
        _max_age = int(_evict_cfg.get("max_age_days", 30))
    except (ValueError, TypeError):
        logger.warning("[startup] cache_eviction.max_age_days 配置无效，已回退到默认值 30")
        _max_age = 30
    try:
        _max_size = int(_evict_cfg.get("max_size_mb", 0))
    except (ValueError, TypeError):
        logger.warning("[startup] cache_eviction.max_size_mb 配置无效，已回退到默认值 0")
        _max_size = 0
    if _max_age or _max_size:
        await asyncio.to_thread(evict_cache, max_age_days=_max_age, max_size_mb=_max_size)

    # 启动时从数据库恢复上次同步的 bot 账号信息（NapCat 尚未连接时也能展示）
    saved_qq_id, saved_qq_name = await get_bot_self()
    if saved_qq_id:
        update_bot_info(saved_qq_id, saved_qq_name)

    # NapCat 启动
    client = app_state.napcat_client
    if client:
        napcat_cfg = app_state.napcat_cfg
        host = napcat_cfg.get("host", "127.0.0.1")
        port = napcat_cfg.get("port", 8078)

        async def _sync_bot_profile() -> None:
            """NapCat 连接后同步机器人自身信息。"""
            assert client is not None
            bot_id = client.bot_id
            if not bot_id:
                logger.warning("同步跳过：bot_id 未知")
                return

            login_info = await client.send_api("get_login_info", {})
            if login_info:
                qq_id = str(login_info.get("user_id", bot_id))
                nickname = login_info.get("nickname", "")
                await upsert_bot_self(qq_id, nickname)
                update_bot_info(qq_id, nickname)

            group_list = await client.send_api("get_group_list", {})
            if not group_list:
                logger.warning("获取群列表失败，跳过群名片同步")
                return

            for group in group_list:
                try:
                    group_id = str(group.get("group_id", ""))
                    group_name = group.get("group_name", "")
                    member_count = int(group.get("member_count", 0))
                    if not group_id:
                        continue
                    member_info = await client.send_api(
                        "get_group_member_info",
                        {"group_id": int(group_id), "user_id": int(bot_id)},
                    )
                    bot_card = ""
                    if member_info:
                        bot_card = member_info.get("card") or member_info.get("nickname", "")
                    await upsert_group(group_id, group_name, bot_card, member_count)
                except (ValueError, TypeError) as e:
                    logger.warning("同步群组信息失败 (group=%s): %s", group.get("group_id", "N/A"), e)

            logger.info("机器人自身信息同步完成")

        client.set_connect_handler(_sync_bot_profile)
        await client.start(host=host, port=port)
        # 此处 ws:// 为本地反向 WebSocket 服务端（默认监听 127.0.0.1），流量不经过网络，无需 wss
        logger.info("NapCat 集成已启用，等待连接: ws://%s:%d", host, port)
    else:
        logger.info("NapCat 集成未启用（napcat.enabled = false）")


async def shutdown() -> None:
    """Quart after_serving 钩子。"""
    # 正常关闭时标记当前活动为 interrupted（进程关闭）
    _closed = _activity_log.close_current_sync(
        end_attitude="passive",
        end_action="interrupted",
        end_remark="进程正常关闭",
    )
    if _closed is not None:
        try:
            await update_activity_entry(_closed)
        except Exception:
            logger.warning("[shutdown] activity_log 关闭写入失败", exc_info=True)

    client = app_state.napcat_client
    if client:
        await client.stop()
