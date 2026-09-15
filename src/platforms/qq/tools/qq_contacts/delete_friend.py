"""Remove a QQ friendship."""

from pydantic import Field

from platforms.qq.friends import run_friend_operation
from tools.contract import ToolArgsModel, tool

EXTERNALLY_PERCEPTIBLE = True


class DeleteFriendArgs(ToolArgsModel):
    user_id: str = Field(pattern=r"^[1-9][0-9]*$", description="要删除的好友 QQ 号。", json_schema_extra={"x-coerce-integer": True})


@tool(
    name="delete_friend",
    description="自行删除指定 QQ 好友，无需 Guardian 批准。解除好友关系，不删除本地聊天记录；恢复好友关系需要重新添加。",
    args_model=DeleteFriendArgs,
)
def execute(args: DeleteFriendArgs) -> dict:
    return run_friend_operation(lambda service: service.delete(args.user_id))
