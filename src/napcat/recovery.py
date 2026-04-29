"""napcat/recovery.py — NapCat 历史消息恢复

在 NapCat 重连后后台执行两类同步：
  1. 最近缺失追平：把掉线期间漏掉的最新消息补回本地 DB；
  2. 向旧回填：持续把更早的历史消息分页补回，直到远端没有更早记录。

设计约束：
  - 不阻塞 NapCat ready，恢复始终在后台 task 中执行；
  - 以本地 DB 为唯一真相，幂等写入；
  - 仅“最近缺失追平”会尝试注入 live context；更早回填只写 DB；
  - 浏览历史模式下不改 context_messages，只累计 unread_count。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import app_state
from database import (
    get_chat_message_edge,
    get_existing_chat_message_ids,
    get_group_info,
    load_chat_sessions,
    save_chat_message,
    upsert_chat_session,
)
from llm.session import sessions

from .events import napcat_event_to_context, download_pending_images, expand_forward_previews

logger = logging.getLogger("AICQ.napcat.recovery")

_recovery_task: asyncio.Task | None = None
_recovery_generation = 0


@dataclass(slots=True)
class RecoveryConfig:
    enabled: bool = True
    page_size: int = 50
    max_pages_per_session: int = 0  # 0 = unlimited
    backfill_history: bool = True
    seed_from_whitelist: bool = True


@dataclass(slots=True)
class RecoveryTarget:
    session_key: str
    conv_type: str
    conv_id: str
    conv_name: str = ""


def schedule_history_recovery(client) -> None:
    """在 connect 后台调度历史恢复任务。"""
    cfg = _load_config((app_state.napcat_cfg or {}).get("recovery", {}) or {})
    if not cfg.enabled:
        logger.info("[recovery] 已禁用，跳过历史恢复")
        return

    global _recovery_task, _recovery_generation
    _recovery_generation += 1
    generation = _recovery_generation

    if _recovery_task is not None and not _recovery_task.done():
        _recovery_task.cancel()

    _recovery_task = asyncio.create_task(
        _run_recovery(client, cfg, generation),
        name="napcat_history_recovery",
    )


def _load_config(raw: dict) -> RecoveryConfig:
    def _int(name: str, default: int, minimum: int) -> int:
        try:
            value = int(raw.get(name, default))
        except (TypeError, ValueError):
            return default
        return value if value >= minimum else default

    return RecoveryConfig(
        enabled=bool(raw.get("enabled", True)),
        page_size=_int("page_size", 50, 1),
        max_pages_per_session=_int("max_pages_per_session", 0, 0),
        backfill_history=bool(raw.get("backfill_history", True)),
        seed_from_whitelist=bool(raw.get("seed_from_whitelist", True)),
    )


def _message_id(message: dict) -> str:
    return str(message.get("message_id", "")).strip()


def _dedupe_messages(messages: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen_ids: set[str] = set()
    for message in messages:
        mid = _message_id(message)
        if not mid or mid in seen_ids:
            continue
        seen_ids.add(mid)
        deduped.append(message)
    return deduped


def _can_page(cfg: RecoveryConfig, pages_used: int) -> bool:
    return cfg.max_pages_per_session <= 0 or pages_used < cfg.max_pages_per_session


def _normalize_target(session_key: str, conv_type: str, conv_id: str, conv_name: str = "") -> RecoveryTarget:
    norm_type = (conv_type or "").strip()
    norm_id = (conv_id or "").strip()
    if not norm_type:
        if session_key.startswith("group_"):
            norm_type = "group"
            norm_id = norm_id or session_key[len("group_") :]
        elif session_key.startswith("private_"):
            norm_type = "private"
            norm_id = norm_id or session_key[len("private_") :]
    return RecoveryTarget(
        session_key=session_key,
        conv_type=norm_type,
        conv_id=norm_id,
        conv_name=(conv_name or "").strip(),
    )


async def _build_targets(cfg: RecoveryConfig) -> list[RecoveryTarget]:
    ordered: dict[str, RecoveryTarget] = {}

    def _merge(target: RecoveryTarget) -> None:
        if not target.session_key or target.session_key == "web":
            return
        if target.session_key in ordered:
            existing = ordered[target.session_key]
            if not existing.conv_type and target.conv_type:
                existing.conv_type = target.conv_type
            if not existing.conv_id and target.conv_id:
                existing.conv_id = target.conv_id
            if not existing.conv_name and target.conv_name:
                existing.conv_name = target.conv_name
            return
        ordered[target.session_key] = target

    for meta in await load_chat_sessions():
        _merge(
            _normalize_target(
                meta.get("session_key", ""),
                meta.get("conv_type", ""),
                meta.get("conv_id", ""),
                meta.get("conv_name", ""),
            )
        )

    for session_key, session in sessions.items():
        _merge(
            _normalize_target(
                session_key,
                session.conv_type,
                session.conv_id,
                session.conv_name,
            )
        )

    if cfg.seed_from_whitelist:
        whitelist = (app_state.napcat_cfg or {}).get("whitelist", {}) or {}
        for user_id in whitelist.get("private_users", []) or []:
            user_str = str(user_id).strip()
            if user_str:
                _merge(_normalize_target(f"private_{user_str}", "private", user_str))
        for group_id in whitelist.get("group_ids", []) or []:
            group_str = str(group_id).strip()
            if not group_str:
                continue
            group_name, _, _ = await get_group_info(group_str)
            _merge(_normalize_target(f"group_{group_str}", "group", group_str, group_name))

    return [
        target
        for target in ordered.values()
        if target.conv_type in ("group", "private") and target.conv_id
    ]


async def _run_recovery(client, cfg: RecoveryConfig, generation: int) -> None:
    try:
        targets = await _build_targets(cfg)
        if not targets:
            logger.info("[recovery] 无可恢复会话")
            return

        logger.info("[recovery] 开始历史恢复: 会话=%d page_size=%d", len(targets), cfg.page_size)
        total_recent = 0
        total_older = 0

        for target in targets:
            if generation != _recovery_generation or not getattr(client, "connected", False):
                logger.info("[recovery] 检测到新的连接轮次或连接已断开，中止本轮恢复")
                return
            recent_count, older_count = await _recover_target(client, target, cfg)
            total_recent += recent_count
            total_older += older_count

        logger.info(
            "[recovery] 恢复完成: 会话=%d 最近追平=%d 更早回填=%d",
            len(targets), total_recent, total_older,
        )
    except asyncio.CancelledError:
        logger.info("[recovery] 恢复任务已取消")
        raise
    except Exception:
        logger.exception("[recovery] 恢复任务异常")


async def _recover_target(client, target: RecoveryTarget, cfg: RecoveryConfig) -> tuple[int, int]:
    latest_edge = await get_chat_message_edge(target.session_key, newest=True)
    recent_missing = await _collect_missing_recent_messages(
        client,
        target,
        latest_edge["message_id"] if latest_edge else None,
        cfg,
    )
    recent_count = await _persist_messages(
        client,
        target,
        recent_missing,
        inject_recent=True,
        mark_unread=True,
    )

    older_count = 0
    if cfg.backfill_history:
        earliest_edge = await get_chat_message_edge(target.session_key, newest=False)
        if earliest_edge:
            older_missing = await _collect_older_history(
                client,
                target,
                earliest_edge["message_id"],
                cfg,
            )
            older_count = await _persist_messages(
                client,
                target,
                older_missing,
                inject_recent=False,
                mark_unread=False,
            )

    if recent_count or older_count:
        logger.info(
            "[recovery] 会话 %s 完成: 最近追平=%d 更早回填=%d",
            target.session_key,
            recent_count,
            older_count,
        )
    return recent_count, older_count


async def _fetch_history_messages(
    client,
    target: RecoveryTarget,
    *,
    anchor_message_id: str | None,
    page_size: int,
    reverse_order: bool = False,
) -> list[dict]:
    action = "get_group_msg_history" if target.conv_type == "group" else "get_friend_msg_history"
    peer_key = "group_id" if target.conv_type == "group" else "user_id"
    params: dict[str, object] = {peer_key: int(target.conv_id), "count": page_size}
    if anchor_message_id:
        try:
            params["message_seq"] = int(anchor_message_id)
        except ValueError:
            logger.warning("[recovery] 锚点 message_id 无法转为整数，跳过 message_seq: %s", anchor_message_id)
        else:
            params["reverse_order"] = reverse_order

    resp = await client.send_api_raw(action, params, timeout=20.0)
    if not isinstance(resp, dict):
        return []
    if resp.get("status") != "ok":
        err_msg = str(resp.get("message", ""))
        # 锚点消息在服务端已过期是历史回填的正常边界，避免刷屏
        if "不存在" in err_msg:
            return []
        logger.warning(
            "NapCat API %s 失败: status=%s msg=%s",
            action, resp.get("status"), err_msg,
        )
        return []
    data = resp.get("data") or {}
    messages = data.get("messages", [])
    return messages if isinstance(messages, list) else []


async def _fetch_older_page(
    client,
    target: RecoveryTarget,
    anchor_message_id: str,
    page_size: int,
) -> list[dict]:
    for reverse_order in (False, True):
        batch = _dedupe_messages(
            await _fetch_history_messages(
                client,
                target,
                anchor_message_id=anchor_message_id,
                page_size=page_size,
                reverse_order=reverse_order,
            )
        )
        if not batch:
            return []

        ids = [_message_id(message) for message in batch]
        if anchor_message_id not in ids:
            logger.warning(
                "[recovery] 分页结果未包含锚点，尝试切换方向: session=%s anchor=%s reverse=%s",
                target.session_key,
                anchor_message_id,
                reverse_order,
            )
            continue

        anchor_index = ids.index(anchor_message_id)
        older = batch[:anchor_index]
        newer = batch[anchor_index + 1 :]
        if older:
            return older
        if newer:
            continue
        return []

    logger.warning("[recovery] 无法获得更早分页: session=%s anchor=%s", target.session_key, anchor_message_id)
    return []


async def _collect_missing_recent_messages(
    client,
    target: RecoveryTarget,
    latest_message_id: str | None,
    cfg: RecoveryConfig,
) -> list[dict]:
    latest_batch = _dedupe_messages(
        await _fetch_history_messages(
            client,
            target,
            anchor_message_id=None,
            page_size=cfg.page_size,
        )
    )
    if not latest_batch:
        return []

    if not latest_message_id:
        return latest_batch

    latest_ids = [_message_id(message) for message in latest_batch]
    if latest_message_id in latest_ids:
        return latest_batch[latest_ids.index(latest_message_id) + 1 :]

    collected = latest_batch
    pages_used = 0
    anchor = _message_id(latest_batch[0])
    seen_anchors = {anchor}

    while anchor and _can_page(cfg, pages_used):
        older_batch = _dedupe_messages(await _fetch_older_page(client, target, anchor, cfg.page_size))
        if not older_batch:
            break

        older_ids = [_message_id(message) for message in older_batch]
        if latest_message_id in older_ids:
            return older_batch[older_ids.index(latest_message_id) + 1 :] + collected

        collected = older_batch + collected
        anchor = _message_id(older_batch[0])
        if not anchor or anchor in seen_anchors:
            break
        seen_anchors.add(anchor)
        pages_used += 1

    return collected


async def _collect_older_history(
    client,
    target: RecoveryTarget,
    earliest_message_id: str,
    cfg: RecoveryConfig,
) -> list[dict]:
    collected: list[dict] = []
    anchor = earliest_message_id
    seen_anchors = {anchor}
    pages_used = 0

    while anchor and _can_page(cfg, pages_used):
        older_batch = _dedupe_messages(await _fetch_older_page(client, target, anchor, cfg.page_size))
        if not older_batch:
            break

        collected = older_batch + collected
        anchor = _message_id(older_batch[0])
        if not anchor or anchor in seen_anchors:
            break
        seen_anchors.add(anchor)
        pages_used += 1

    return collected


async def _resolve_conv_name(target: RecoveryTarget, inserted_entries: list[dict]) -> str:
    if target.conv_name:
        return target.conv_name
    session = sessions.get(target.session_key)
    if session and session.conv_name:
        return session.conv_name
    if target.conv_type == "group":
        group_name, _, _ = await get_group_info(target.conv_id)
        return group_name
    for entry in reversed(inserted_entries):
        sender_name = str(entry.get("sender_name", "")).strip()
        if sender_name:
            return sender_name
    return ""


async def _persist_messages(
    client,
    target: RecoveryTarget,
    messages: list[dict],
    *,
    inject_recent: bool,
    mark_unread: bool,
) -> int:
    candidate_ids = [_message_id(message) for message in messages]
    existing_ids = await get_existing_chat_message_ids(target.session_key, candidate_ids)
    session = sessions.get(target.session_key)
    bot_display = ""
    if session is not None:
        bot_display = session._qq_card or session._qq_name or ""
    elif target.conv_type == "group":
        _, _, bot_card = await get_group_info(target.conv_id)
        bot_display = bot_card or app_state.BOT_NAME
    else:
        bot_display = app_state.BOT_NAME

    inserted_entries: list[dict] = []
    inserted_count = 0

    for message in messages:
        message_id = _message_id(message)
        if not message_id or message_id in existing_ids:
            continue

        sender_id = str((message.get("sender") or {}).get("user_id", "")).strip()
        if getattr(client, "bot_id", None) and sender_id == str(client.bot_id):
            continue

        ob11_message = dict(message)
        ob11_message.setdefault("post_type", "message")

        entry = await napcat_event_to_context(
            ob11_message,
            bot_id=getattr(client, "bot_id", None),
            bot_display_name=bot_display,
            timezone=app_state.TIMEZONE,
        )
        if not entry:
            continue

        await download_pending_images(entry)
        await expand_forward_previews(entry, client)

        if (
            inject_recent
            and session is not None
            and entry.get("images")
            and getattr(app_state, "vision_bridge", None) is not None
        ):
            await asyncio.to_thread(app_state.vision_bridge.process_entry, entry)

        await save_chat_message(target.session_key, entry)
        existing_ids.add(message_id)
        inserted_entries.append(entry)
        inserted_count += 1

        if not inject_recent or session is None:
            continue

        if not session.conv_type:
            session.set_conversation_meta(target.conv_type, target.conv_id, target.conv_name)
        if session.is_browsing_history():
            if mark_unread:
                session.mark_unread_message(entry.get("message_id"))
            continue

        session.add_to_context(entry)
        if mark_unread:
            session.mark_unread_message(entry.get("message_id"))

    if inserted_entries:
        conv_name = await _resolve_conv_name(target, inserted_entries)
        await upsert_chat_session(target.session_key, target.conv_type, target.conv_id, conv_name)
        if session is not None and (not session.conv_name) and conv_name:
            session.set_conversation_meta(target.conv_type, target.conv_id, conv_name)

    return inserted_count