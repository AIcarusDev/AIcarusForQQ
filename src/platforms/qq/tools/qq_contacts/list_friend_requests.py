"""List persisted, unhandled friend requests."""

from platforms.qq.friends import run_friend_operation
from tools.contract import ToolArgsModel, tool


class ListFriendRequestsArgs(ToolArgsModel):
    pass


@tool(
    name="list_friend_requests",
    description="查询当前账号收到且尚未处理的好友申请，返回申请标识 flag、QQ 号和验证留言。只包含系统接入后收到的申请事件。",
    args_model=ListFriendRequestsArgs,
)
def execute(args: ListFriendRequestsArgs) -> dict:
    return run_friend_operation(lambda service: service.list_requests())
