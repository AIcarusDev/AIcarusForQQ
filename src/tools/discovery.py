"""Prompt-facing discovery groups for latent tools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiscoveryGroup:
    name: str
    description: str
    keywords: tuple[str, ...] = ()


TOOL_GROUPS: dict[str, DiscoveryGroup] = {
    "contacts_profile": DiscoveryGroup(
        name="contacts_profile",
        description=(
            "联系人和账号资料能力：搜索好友/群聊列表，查询 QQ 签名、头像，"
            "以及维护自己的 QQ 签名。"
        ),
        keywords=("联系人", "好友", "群聊", "签名", "头像", "资料", "账号"),
    ),
    "group_info": DiscoveryGroup(
        name="group_info",
        description=(
            "群信息能力：查询群成员、群公告列表、群公告详情，"
            "以及维护自己在当前群的群名片。"
        ),
        keywords=("群", "群成员", "成员", "群公告", "公告", "群名片"),
    ),
    "chat_history": DiscoveryGroup(
        name="chat_history",
        description="当前会话历史检索能力：按关键词搜索当前聊天上下文以找回较早消息。",
        keywords=("聊天记录", "历史", "搜索", "消息", "上下文"),
    ),
    "browser_precision": DiscoveryGroup(
        name="browser_precision",
        description=(
            "精确浏览器操作能力：按 CSS、文本、ARIA role 等定位网页元素，"
            "并点击、输入、读取属性或执行细粒度页面操作。"
        ),
        keywords=("浏览器", "网页", "定位", "点击", "输入", "元素", "DOM"),
    ),
    "runtime_control": DiscoveryGroup(
        name="runtime_control",
        description="自身运行时控制能力：在确有必要时安排重启当前程序。",
        keywords=("重启", "运行时", "程序", "恢复"),
    ),
    "misc_hidden": DiscoveryGroup(
        name="misc_hidden",
        description="其它低频隐藏能力。仅在没有更具体工具集匹配时使用。",
        keywords=("其它", "隐藏", "工具"),
    ),
}


TOOL_GROUP_ASSIGNMENTS: dict[str, str] = {
    "get_contact_list": "contacts_profile",
    "get_qq_signature": "contacts_profile",
    "get_user_avatar": "contacts_profile",
    "set_self_qq_signature": "contacts_profile",
    "get_group_members": "group_info",
    "get_group_notice_list": "group_info",
    "get_group_notice_detail": "group_info",
    "set_self_group_card": "group_info",
    "search_current_session_chat_history": "chat_history",
    "browser_locator": "browser_precision",
    "restart_self": "runtime_control",
}


def normalize_group_name(value: object) -> str:
    text = str(value or "").strip()
    return text if text in TOOL_GROUPS else ""


def group_name_for_tool(tool_name: str, override: object = None) -> str:
    return (
        normalize_group_name(override)
        or TOOL_GROUP_ASSIGNMENTS.get(str(tool_name or "").strip(), "")
        or "misc_hidden"
    )


def group_spec_dict(group_name: str) -> dict[str, object]:
    group = TOOL_GROUPS.get(group_name) or TOOL_GROUPS["misc_hidden"]
    return {
        "name": group.name,
        "description": group.description,
        "keywords": group.keywords,
    }
