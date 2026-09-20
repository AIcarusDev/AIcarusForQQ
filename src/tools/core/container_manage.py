"""Manage persistent custom context entries."""

from typing import Annotated, Literal

from pydantic import Field, RootModel, field_validator

from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, tool


class ContainerAddArgs(ToolArgsModel):
    action: Literal["add"]
    content: str = Field(min_length=1, description="要持续保留的任意文本，按原文保存。")
    key: str = Field(default="", description="可选条目标签，允许重复，不用于覆盖条目。")

    @field_validator("content")
    @classmethod
    def nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content 不能为空白")
        return value


class ContainerDeleteArgs(ToolArgsModel):
    action: Literal["delete"]
    item_id: str = Field(min_length=1, description="容器 custom 分节中要删除的条目 ID。")

    @field_validator("item_id")
    @classmethod
    def nonblank_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("item_id 不能为空白")
        return value.strip()


class ContainerKeepArgs(ToolArgsModel):
    action: Literal["keep"]
    item_id: str = Field(min_length=1, description="要续期的活跃 custom 条目 ID。")

    @field_validator("item_id")
    @classmethod
    def nonblank_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("item_id 不能为空白")
        return value.strip()


class ContainerManageArgs(RootModel[Annotated[ContainerAddArgs | ContainerDeleteArgs | ContainerKeepArgs, Field(discriminator="action")]]):
    pass


@tool(
    name="container_manage",
    description=(
        "管理容器 custom 分节中的自定义文本条目。可进行添加、删除、keep 动作。最多直接展示 5 条；每条寿命为 8 个轮次，"
        "每轮结束减少 1，到 0 自动剔除。用 keep 和条目 ID 续期至 8 轮，并提高保留优先级。"
        "新增第 6 条时，最久未添加或 keep 的条目会被推出。被推出或过期的内容不会再出现在上下文中。"
        "add 返回 item_id；delete 和 keep 使用活跃条目 ID。内容按文本保存，不解析为容器结构。"
        "请把它当作一种`工作记忆`来使用。"
    ),
    args_model=ContainerManageArgs,
)
def execute(args: ContainerManageArgs) -> dict:
    import app_state

    loop = app_state.main_loop
    if loop is None or not loop.is_running():
        return {"ok": False, "error": "主事件循环不可用，无法修改容器"}
    try:
        return run_coroutine_sync(_manage(args.root), loop)
    except Exception as exc:
        return {"ok": False, "error": f"修改容器失败: {exc}"}


async def _manage(args: ContainerAddArgs | ContainerDeleteArgs | ContainerKeepArgs) -> dict:
    from llm.prompt import container

    if isinstance(args, ContainerAddArgs):
        entry = await container.add_item("custom", args.content, key=args.key)
        return {"ok": True, "action": "add", "item_id": entry["item_id"]}
    if isinstance(args, ContainerKeepArgs):
        kept = await container.keep_item(args.item_id)
        if not kept:
            return {"ok": False, "error": "未找到指定的活跃 custom 条目", "item_id": args.item_id}
        return {"ok": True, "action": "keep", "item_id": args.item_id,
                "remaining_rounds": container.CUSTOM_LIFETIME_ROUNDS}
    removed = await container.remove_item(args.item_id, section="custom")
    if not removed:
        return {"ok": False, "error": "未找到指定的 custom 条目", "item_id": args.item_id}
    return {"ok": True, "action": "delete", "item_id": args.item_id}


TOOL_CONTRACT = getattr(execute, "__tool_contract__", None)
