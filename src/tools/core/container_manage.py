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


class ContainerManageArgs(RootModel[Annotated[ContainerAddArgs | ContainerDeleteArgs, Field(discriminator="action")]]):
    pass


@tool(
    name="container_manage",
    description=(
        "添加或删除容器 custom 分节中的自定义文本条目（可以作为备忘录，笔记，或任何需要保留的内容）。"
        "持续出现在上下文中，直到主动删除；请及时删除失效内容。"
        "添加返回 item_id，删除使用上下文中的条目 ID。内容按文本保存，不解析为容器结构。"
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


async def _manage(args: ContainerAddArgs | ContainerDeleteArgs) -> dict:
    from llm.prompt import container

    if isinstance(args, ContainerAddArgs):
        entry = await container.add_item("custom", args.content, key=args.key)
        return {"ok": True, "action": "add", "item_id": entry["item_id"]}
    removed = await container.remove_item(args.item_id, section="custom")
    if not removed:
        return {"ok": False, "error": "未找到指定的 custom 条目", "item_id": args.item_id}
    return {"ok": True, "action": "delete", "item_id": args.item_id}


TOOL_CONTRACT = getattr(execute, "__tool_contract__", None)
