"""Durable friend requests and QQ friendship operations."""

from __future__ import annotations

import asyncio
import html
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from platforms.qq.adapter.conversation import format_adapter_error


def run_friend_operation(operation: Callable[[FriendService], Coroutine[Any, Any, dict]]) -> dict:
    import app_state
    from platforms.registry import get_platform
    from tools._async_bridge import run_coroutine_sync

    runtime = get_platform("qq")
    if not runtime or not runtime.connected or runtime.friends is None:
        return {"ok": False, "error": "QQ adapter 未连接"}
    if not runtime.client.bot_id:
        return {"ok": False, "error": "QQ 账号尚未初始化"}
    loop = getattr(app_state, "main_loop", None)
    if loop is None or not loop.is_running():
        return {"ok": False, "error": "主事件循环不可用"}
    try:
        return run_coroutine_sync(operation(runtime.friends), loop, timeout=90)
    except Exception:
        logging.getLogger("AICQ.qq.friends").exception("好友管理操作失败")
        return {"ok": False, "error": "好友管理操作异常，结果未确认，请查询当前状态"}


class FriendService:
    def __init__(self, client: Any):
        self.client = client
        self._lock = asyncio.Lock()
        self._pending: list[dict] = []
        self._account = ""

    @property
    def pending(self) -> list[dict]:
        if self._account != str(self.client.bot_id or ""):
            return []
        return [dict(row) for row in self._pending]

    async def refresh(self) -> list[dict]:
        import aiosqlite
        from database import _connect

        account = str(self.client.bot_id or "")
        async with _connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT flag, user_id, comment, created_at, state FROM qq_friend_requests "
                "WHERE account_id=? AND state='pending' ORDER BY created_at, flag",
                (account,),
            ) as cursor:
                rows = [dict(row) for row in await cursor.fetchall()]
        self._account, self._pending = account, rows
        return self.pending

    async def receive(self, event: dict) -> bool:
        from database import _connect

        if event.get("request_type") != "friend":
            return False
        account = str(event.get("self_id") or self.client.bot_id or "")
        flag = str(event.get("flag") or "").strip()
        user_id = str(event.get("user_id") or "")
        if not account or not flag or not user_id.isdecimal() or int(user_id) <= 0:
            return False
        if account != str(self.client.bot_id or ""):
            return False
        async with self._lock:
            async with _connect() as db:
                cursor = await db.execute(
                    "INSERT OR IGNORE INTO qq_friend_requests "
                    "(account_id, flag, user_id, comment, created_at, state) VALUES (?, ?, ?, ?, ?, 'pending')",
                    (account, flag, user_id, str(event.get("comment") or ""), int(event.get("time") or 0)),
                )
                inserted = cursor.rowcount == 1
                await db.commit()
            await self.refresh()
        return inserted

    async def _call(self, action: str, params: dict) -> dict:
        response = await self.client.send_api_raw(action, params)
        if response is None:
            return {"ok": False, "error": "QQ adapter 未响应，操作结果未知，请查询后再决定是否重试"}
        if response.get("status") != "ok" or response.get("retcode", 0) != 0:
            return {"ok": False, "error": format_adapter_error({**response, "action": action}, "操作失败")}
        return {"ok": True}

    async def decide(self, flag: str, approve: bool, remark: str = "") -> dict:
        from database import _connect

        async with self._lock:
            account = str(self.client.bot_id or "")
            async with _connect() as db:
                async with db.execute(
                    "SELECT user_id, state FROM qq_friend_requests WHERE account_id=? AND flag=?",
                    (account, flag),
                ) as cursor:
                    row = await cursor.fetchone()
            if row is None:
                return {"ok": False, "error": "未找到该好友申请，请先查询收到的申请"}
            if row[1] != "pending":
                return {"ok": False, "error": "该申请已处理", "state": row[1]}
            result = await self._call("set_friend_add_request", {"flag": flag, "approve": approve, "remark": remark})
            if not result["ok"]:
                return result
            state = "approved" if approve else "rejected"
            async with _connect() as db:
                await db.execute(
                    "UPDATE qq_friend_requests SET state=? WHERE account_id=? AND flag=?",
                    (state, account, flag),
                )
                await db.commit()
            await self.refresh()
            return {"ok": True, "user_id": row[0], "state": state}

    async def delete(self, user_id: str) -> dict:
        async with self._lock:
            result = await self._call("delete_friend", {"user_id": int(user_id)})
            return {**result, "user_id": user_id}

    async def list_requests(self) -> dict:
        async with self._lock:
            return {"ok": True, "requests": await self.refresh(), "source": "received_events"}

    def render_pending(self) -> str:
        pending = self.pending
        if not pending:
            return ""
        lines = ['<friend_requests tool="qq_contacts.list_friend_requests">']
        for row in pending[:20]:
            attrs = " ".join(f'{key}="{html.escape(str(row[key]), quote=True)}"' for key in ("flag", "user_id", "created_at"))
            lines.append(f"  <request {attrs}>{html.escape(row['comment'])}</request>")
        lines.append(f"  <total>{len(pending)}</total>")
        lines.append("</friend_requests>")
        return "\n".join(lines)
