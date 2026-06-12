"""Model-visible tool warning factory.

This module is the central place for advisory warnings returned to the model.
Warnings do not block execution and do not change tool success/failure
semantics; they add structured feedback so the model can self-correct.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any


@dataclass(frozen=True)
class ToolWarning:
    """A structured warning that can be embedded in a tool response."""

    code: str
    message: str
    severity: str = "info"
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.details:
            payload.update(self.details)
        return payload


@dataclass(frozen=True)
class DuplicateWarningPolicy:
    mode: str = "exact_args"
    window_seconds: int = 120
    lookback_rounds: int = 5
    warn_after: int = 1
    strong_warn_after: int = 3
    code: str = "DUPLICATE_TOOL_CALL"
    strong_code: str = "REPEATED_DUPLICATE_TOOL_CALL"
    message: str = "同一工具和参数刚刚已经成功执行过，重复执行是否为预期行为。"
    strong_message: str = "同一工具和参数已连续多次成功执行。如果不是刻意重复，请使用已有结果进入下一步。"


class ToolWarningFactory:
    """Builds warning payloads from policy and runtime context."""

    @staticmethod
    def duplicate_call(
        *,
        policy: DuplicateWarningPolicy,
        duplicate_count: int,
        previous_success_age_seconds: float | None,
    ) -> ToolWarning:
        is_strong = duplicate_count >= policy.strong_warn_after
        details: dict[str, Any] = {"duplicate_count": duplicate_count}
        if previous_success_age_seconds is not None:
            details["previous_success_age_seconds"] = round(previous_success_age_seconds, 1)
        return ToolWarning(
            code=policy.strong_code if is_strong else policy.code,
            message=policy.strong_message if is_strong else policy.message,
            severity="warning" if is_strong else "info",
            details=details,
        )

    @staticmethod
    def no_browser_session() -> ToolWarning:
        return ToolWarning(
            code="NO_BROWSER_SESSION",
            message="当前没有可关闭的浏览器会话；无需再次 close。",
            severity="info",
        )


DEFAULT_DUPLICATE_POLICY = DuplicateWarningPolicy()


DUPLICATE_WARNING_POLICIES: dict[str, DuplicateWarningPolicy] = {
    # Repeating waits is normal conversational behavior.
    "wait": DuplicateWarningPolicy(mode="disabled"),
    "sleep": DuplicateWarningPolicy(mode="disabled"),
    # Repeated scrolling is often intentional.
    "scroll_chat_log": DuplicateWarningPolicy(mode="disabled"),
    # Message repetition may be deliberate social behavior, so keep it light.
    "send_message": DuplicateWarningPolicy(
        window_seconds=60,
        lookback_rounds=3,
        strong_warn_after=4,
        code="REPEATED_MESSAGE_CONTENT",
        strong_code="REPEATED_MESSAGE_CONTENT",
        message="这次发送内容与短时间内之前的 send_message 参数相同；如果是刻意复读可忽略。",
        strong_message="这次发送内容已在短时间内多次重复；如果是刻意复读可忽略，否则建议换一种回应方式。",
    ),
    "web_search": DuplicateWarningPolicy(
        code="DUPLICATE_SEARCH_QUERY",
        strong_code="REPEATED_SEARCH_QUERY",
        message="同一搜索关键词刚刚已经成功执行过，重复搜索是否为预期行为。",
        strong_message="同一搜索关键词已连续多次成功执行；如果不是刻意复查，请使用已有搜索结果进入下一步。",
    ),
    "web_extract": DuplicateWarningPolicy(
        code="DUPLICATE_WEB_EXTRACT",
        strong_code="REPEATED_WEB_EXTRACT",
        message="同一网页刚刚已经提取过，重复提取是否为预期行为。",
        strong_message="同一网页已连续多次提取；如果页面没有变化，请使用已有正文结果。",
    ),
    "recall_memory": DuplicateWarningPolicy(
        code="DUPLICATE_MEMORY_RECALL",
        strong_code="REPEATED_MEMORY_RECALL",
        message="同一记忆检索刚刚已经执行过，重复回忆是否为预期行为。",
        strong_message="同一记忆检索已连续多次执行；如果不是刻意复查，请使用已有记忆结果。",
    ),
    "search_session": DuplicateWarningPolicy(
        code="DUPLICATE_SESSION_SEARCH",
        strong_code="REPEATED_SESSION_SEARCH",
        message="同一会话搜索刚刚已经执行过，重复搜索是否为预期行为。",
        strong_message="同一会话搜索已连续多次执行；如果不是刻意复查，请使用已有搜索结果。",
    ),
    "get_contact_list": DuplicateWarningPolicy(
        code="DUPLICATE_CONTACT_LIST_READ",
        strong_code="REPEATED_CONTACT_LIST_READ",
        message="联系人列表刚刚已经读取过，重复读取是否为预期行为。",
        strong_message="联系人列表已连续多次读取；如果列表没有变化，请使用已有结果。",
    ),
    "shift": DuplicateWarningPolicy(
        code="DUPLICATE_SHIFT",
        strong_code="REPEATED_SHIFT",
        message="刚刚已经切换到同一会话，重复 shift 是否为预期行为。",
        strong_message="已连续多次切换到同一会话；如果当前会话没有问题，请直接进行下一步。",
    ),
    "tools_manage": DuplicateWarningPolicy(
        code="DUPLICATE_TOOLS_MANAGE",
        strong_code="REPEATED_TOOLS_MANAGE",
        message="同一工具管理请求刚刚已经执行过，重复执行是否为预期行为。",
        strong_message="同一工具管理请求已连续多次执行；如果工具已经激活或预览，请直接使用已有信息。",
    ),
    "browser_control": DuplicateWarningPolicy(
        mode="browser_control",
        code="DUPLICATE_BROWSER_ACTION",
        strong_code="REPEATED_BROWSER_ACTION",
        message="同一浏览器动作刚刚已经成功执行过，重复执行是否为预期行为。",
        strong_message="同一浏览器动作已连续多次执行；如果页面状态没有变化，请使用当前浏览器结果进入下一步。",
    ),
    "browser_locator": DuplicateWarningPolicy(
        code="DUPLICATE_BROWSER_LOCATOR",
        strong_code="REPEATED_BROWSER_LOCATOR",
        message="同一浏览器定位请求刚刚已经执行过，重复定位是否为预期行为。",
        strong_message="同一浏览器定位请求已连续多次执行；如果页面状态没有变化，请使用已有定位结果。",
    ),
}


def make_warning(code: str, message: str, *, severity: str = "info", **details: Any) -> dict[str, Any]:
    """Public helper for tools that need to construct a standard warning."""
    return ToolWarning(code=code, message=message, severity=severity, details=details or None).to_dict()


def _policy_for(tool_name: str) -> DuplicateWarningPolicy:
    return DUPLICATE_WARNING_POLICIES.get(tool_name, DEFAULT_DUPLICATE_POLICY)


def _normalize_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def _browser_control_fingerprint(args: dict[str, Any]) -> str | None:
    action = str(args.get("action") or "").strip().lower()
    if action == "close":
        return "browser_control:close"
    if action == "open":
        return "browser_control:open:" + _normalize_json({"url": args.get("url")})
    # Scroll/click/back/forward can be meaningful repeats because page state changes.
    return None


def _fingerprint(tool_name: str, args: dict[str, Any], policy: DuplicateWarningPolicy) -> str | None:
    if policy.mode == "disabled":
        return None
    if policy.mode == "browser_control":
        return _browser_control_fingerprint(args)
    return tool_name + ":" + _normalize_json(args)


def _is_successful_response(response: Any) -> bool:
    if not isinstance(response, dict):
        return True
    if response.get("tool_not_executed") is True:
        return False
    if response.get("ok") is False:
        return False
    if response.get("error"):
        return False
    return True


def _append_warning(result: dict[str, Any], warning: ToolWarning | dict[str, Any]) -> None:
    warning_payload = warning.to_dict() if isinstance(warning, ToolWarning) else warning
    warnings = result.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    warnings.append(warning_payload)
    result["warnings"] = warnings
    result.setdefault("warning", warnings[0])


def attach_tool_result_warnings(
    *,
    tool_name: str,
    args: dict[str, Any],
    result: Any,
    flow: Any,
) -> None:
    """Attach advisory warnings to a tool result in-place when appropriate."""
    if not isinstance(result, dict):
        return

    policy = _policy_for(tool_name)
    fingerprint = _fingerprint(tool_name, args, policy)
    if not fingerprint or not _is_successful_response(result):
        return

    recent_rounds = ()
    if flow is not None and hasattr(flow, "recent_rounds"):
        recent_rounds = flow.recent_rounds(policy.lookback_rounds)

    now = time.time()
    duplicate_count = 0
    latest_age: float | None = None

    for rnd in reversed(recent_rounds):
        timestamp = getattr(rnd, "timestamp", None)
        if timestamp is not None and now - float(timestamp) > policy.window_seconds:
            continue
        calls = list(getattr(rnd, "calls", []) or [])
        responses = list(getattr(rnd, "responses", []) or [])
        for idx in range(len(calls) - 1, -1, -1):
            call = calls[idx]
            if getattr(call, "name", "") != tool_name:
                continue
            previous_args = getattr(call, "args", {}) or {}
            if _fingerprint(tool_name, previous_args, policy) != fingerprint:
                continue
            previous_response = responses[idx].response if idx < len(responses) else None
            if not _is_successful_response(previous_response):
                continue
            duplicate_count += 1
            if latest_age is None and timestamp is not None:
                latest_age = max(0.0, now - float(timestamp))

    if duplicate_count < policy.warn_after:
        return

    _append_warning(
        result,
        ToolWarningFactory.duplicate_call(
            policy=policy,
            duplicate_count=duplicate_count + 1,
            previous_success_age_seconds=latest_age,
        ),
    )
