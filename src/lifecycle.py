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
    upsert_membership,
    load_chat_sessions,
    load_chat_messages,
    load_activity_log,
    update_activity_entry,
    load_memories,
    load_goals,
    load_adapter_contents,
    save_adapter_contents,
    migrate_bot_memories_to_triples,
)
from memory.repo.triples import load_all_triples
from llm.media.image_cache import evict_cache
from llm.session import (
    get_or_create_session,
    update_bot_info,
)
import llm.prompt.activity_log as _activity_log
import memory as _memory
import llm.prompt.goals as _goals
from memory.tokenizer import (
    load_custom_dict_from_triples,
    tokenize as _tokenize_for_migration,
    configure as _configure_tokenizer,
)

logger = logging.getLogger("AICQ.app")


def _patch_napcat_report_self(config_dir: str, ws_host: str, ws_port: int) -> None:
    """扫描 NapCat 配置目录，将指向本 bot WS 地址的 websocketClient 条目的
    reportSelfMessage 强制设为 true。无匹配或文件不存在时静默跳过。
    """
    import json as _json
    from pathlib import Path as _Path

    cfg_dir = _Path(config_dir)
    if not cfg_dir.is_dir():
        logger.warning("[startup] napcat.config_dir 不存在或不是目录: %s", config_dir)
        return

    target_url_suffixes = (
        f"{ws_host}:{ws_port}",
        f"127.0.0.1:{ws_port}",
        f"localhost:{ws_port}",
    )

    patched_any = False
    for cfg_file in cfg_dir.glob("onebot11_*.json"):
        try:
            data = _json.loads(cfg_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("[startup] 读取 NapCat 配置失败 %s: %s", cfg_file.name, e)
            continue

        changed = False
        for client_entry in data.get("network", {}).get("websocketClients", []):
            url: str = client_entry.get("url", "")
            if any(url.endswith(sfx) or f"/{sfx}" in url for sfx in target_url_suffixes):
                if not client_entry.get("reportSelfMessage", False):
                    client_entry["reportSelfMessage"] = True
                    changed = True
                    logger.info(
                        "[startup] 已自动开启 reportSelfMessage: %s → %s",
                        cfg_file.name, url,
                    )

        if changed:
            try:
                cfg_file.write_text(
                    _json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                patched_any = True
            except Exception as e:
                logger.warning("[startup] 写入 NapCat 配置失败 %s: %s", cfg_file.name, e)

    if not patched_any:
        logger.debug("[startup] NapCat reportSelfMessage 已是 true，无需修改")


async def startup() -> None:
    """Quart before_serving 钩子。"""
    # 记录主事件循环，供 sync 工具通过 run_coroutine_threadsafe 调用 async 函数
    app_state.main_loop = asyncio.get_event_loop()

    await init_db()

    # 恢复活动日志（加载最近 N 条，并标记上次进程中断遗留的未关闭条目）
    _log_max = int(app_state.config.get("activity_log", {}).get("max_entries", 10))
    _activity_log.configure(_log_max)
    _activity_rows = await load_activity_log(limit=_log_max)
    _activity_log.restore_from_db(_activity_rows)
    _interrupted = _activity_log.get_current()
    if _interrupted is not None:
        _activity_log.close_current_sync(
            end_attitude="passive",
            end_action="interrupted",
            end_remark="进程中断",
        )
        await update_activity_entry(_interrupted)
        logger.info("[startup] activity_log: 已标记上次中断的未关闭条目")

    # 恢复长期记忆（Phase 1：从 MemoryTriples 恢复，含 active/passive 拆分 + jieba 词典）
    _mem_cfg = app_state.config.get("memory", {}) or {}
    _max_active  = int(_mem_cfg.get("max_active",  8))
    _max_passive = int(_mem_cfg.get("max_passive", 15))
    _max_self    = int(_mem_cfg.get("max_self",    50))
    _memory.configure(_max_active, _max_passive, _max_self)

    # jieba 可配置参数
    _jieba_cfg = (_mem_cfg.get("jieba", {}) or {}) if isinstance(_mem_cfg, dict) else {}
    try:
        _configure_tokenizer(
            min_token_len=int(_jieba_cfg.get("min_token_len", 2)),
            custom_word_freq=int(_jieba_cfg.get("custom_word_freq", 100)),
        )
    except Exception:
        logger.warning("[startup] jieba tokenizer 配置失败，使用默认参数", exc_info=True)

    # 迁移：bot_memories → MemoryTriples（幂等，仅首次运行时执行）
    try:
        _migrated = await migrate_bot_memories_to_triples(_tokenize_for_migration)
        if _migrated:
            logger.info("[startup] 已将 %d 条旧记忆迁移到 MemoryTriples", _migrated)
    except Exception:
        logger.warning("[startup] bot_memories 迁移失败，跳过", exc_info=True)

    # 加载所有三元组，恢复缓存 + jieba 词典
    try:
        _triple_rows = await load_all_triples()
        _memory.restore(_triple_rows)
        load_custom_dict_from_triples(_triple_rows)
        logger.info("[startup] 已恢复长期记忆: %d 条（jieba 词典已同步）", len(_triple_rows))
    except Exception:
        # 如果 MemoryTriples 表还没有数据或访问失败，回退到 bot_memories
        logger.warning("[startup] 从 MemoryTriples 恢复失败，回退读取 bot_memories", exc_info=True)
        _memory_rows = await load_memories(limit=max(_max_active, _max_passive))
        _memory.restore(_memory_rows)
        logger.info("[startup] 已恢复长期记忆（回退）: %d 条", len(_memory_rows))

    # 恢复活跃目标
    _goal_rows = await load_goals(limit=_goals.get_max_entries())
    _goals.restore(_goal_rows)
    logger.info("[startup] 已恢复活跃目标: %d 条", len(_goal_rows))

    # 恢复意识流（函数调用历史）
    _saved_contents = await load_adapter_contents()
    if _saved_contents:
        _saved_type, _contents_data, _timestamps_data = _saved_contents
        if _saved_type == "flow":
            app_state.consciousness_flow.restore(_contents_data, _timestamps_data)
            app_state.consciousness_flow.complete_startup_marker()
        else:
            logger.info(
                "[startup] 检测到旧格式意识流（type=%s），跳过恢复",
                _saved_type,
            )

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

    # 启动时全面检查表情包收藏（校验文件/SHA-256、纳入孤儿、去重、修复编号空洞）
    from llm.media.sticker_collection import reconcile_stickers
    _rc_stats = await asyncio.to_thread(reconcile_stickers)
    logger.info(
        "[startup] 表情包 reconcile 完成："
        "清除失效=%d 更新哈希=%d 修正改名=%d 纳入孤儿=%d 删除重复=%d",
        _rc_stats["removed_stale"], _rc_stats["updated_hash"],
        _rc_stats["fixed_rename"], _rc_stats["adopted_orphans"], _rc_stats["removed_duplicates"],
    )

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

        # 自动确保 NapCat 开启「上报自身消息」（reportSelfMessage）
        # 若未配置 config_dir 则跳过（不强制要求用户填写）
        _nc_config_dir = napcat_cfg.get("config_dir", "").strip()
        if _nc_config_dir:
            _patch_napcat_report_self(_nc_config_dir, host, port)
        else:
            logger.debug("[startup] napcat.config_dir 未配置，跳过 reportSelfMessage 自动修复")

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
                    # 注意：NapCat 似乎对 get_group_member_info 查 bot 自身有 bug（永远返回"不存在"），
                    # 改用 get_group_member_list 拉全列表，自己从中找 bot 的群名片。
                    bot_card = ""
                    member_list = await client.send_api(
                        "get_group_member_list",
                        {"group_id": int(group_id)},
                    )
                    if member_list:
                        for m in member_list:
                            if str(m.get("user_id", "")) == bot_id:
                                bot_card = m.get("card", "") or m.get("nickname", "")
                                break
                    await upsert_group(group_id, group_name, bot_card, member_count)
                    # bot 自身的群成员关系也需写入，否则 WebUI 中 bot 节点不与群连通
                    if bot_id and member_list:
                        bot_member = next(
                            (m for m in member_list if str(m.get("user_id", "")) == bot_id),
                            None,
                        )
                        bot_role = (bot_member or {}).get("role", "member")
                        await upsert_membership(
                            "qq", bot_id, group_id,
                            cardname=bot_card,
                            permission_level=bot_role,
                        )
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

    # 意识流关闭标记：将 deferred 工具标记为失败，追加关闭时间戳并持久化
    flow = app_state.consciousness_flow
    if flow is not None:
        flow.append_shutdown_marker()
        try:
            _c_data, _ts_data = flow.dump()
            await save_adapter_contents("flow", _c_data, _ts_data)
            logger.info("[shutdown] 意识流关闭标记已写入数据库")
        except Exception:
            logger.warning("[shutdown] 意识流关闭标记写入失败", exc_info=True)

    client = app_state.napcat_client
    if client:
        await client.stop()
