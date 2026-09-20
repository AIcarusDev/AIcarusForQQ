"""Accept or reject one received friend request."""

from pydantic import Field

from platforms.qq.friends import run_friend_operation
from tools.contract import ToolArgsModel, tool

EXTERNALLY_PERCEPTIBLE = True


class HandleFriendRequestArgs(ToolArgsModel):
    flag: str = Field(min_length=1, description="查询好友申请得到的 flag。")
    approve: bool = Field(description="true 接受申请，false 拒绝申请，由你自行决定。")
    remark: str = Field(default="", description="接受时可设置好友备注。")


@tool(
    name="handle_friend_request",
    description="自行接受或拒绝一条好友申请，无需 Guardian 批准。使用申请列表中的 flag，接口成功后返回处理状态。",
    args_model=HandleFriendRequestArgs,
)
def execute(args: HandleFriendRequestArgs) -> dict:
    return run_friend_operation(lambda service: service.decide(args.flag, args.approve, args.remark))
