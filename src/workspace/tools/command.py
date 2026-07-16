"""Run, poll, or stop commands on the Agent's Linux computer."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from pydantic import Field, RootModel

from platforms.focus import current_focus_key
from tools.contract import ToolArgsModel, ToolContract
from workspace.config import COMMAND_OBSERVATION_SECONDS, DEFAULT_AGENT_HOME

from ._common import acknowledge_command, command_text_result, run_on_main_loop


class RunArgs(ToolArgsModel):
    action: Literal["run"] = Field(description="启动命令并观察最多 15 秒。")
    command: str = Field(min_length=1, max_length=65536, description="要交给 Bash 执行的命令。")
    cwd: str = Field(default=DEFAULT_AGENT_HOME, description="命令工作目录，默认是 agent 的主目录 /home/agent。")
    stdin: str = Field(default="", description="可选的标准输入文本。")


class PollArgs(ToolArgsModel):
    action: Literal["poll"] = Field(description="立即读取命令状态和 cursor 后的新增输出。")
    command_id: str = Field(min_length=1, description="run 返回的 command_id。")
    cursor: int = Field(default=0, ge=0, description="从该输出位置继续读取，默认 0。")


class StopArgs(ToolArgsModel):
    action: Literal["stop"] = Field(description="停止仍在运行的命令及其子进程。")
    command_id: str = Field(min_length=1, description="run 返回的 command_id。")


class CommandArgs(RootModel[Annotated[RunArgs | PollArgs | StopArgs, Field(discriminator="action")]]):
    pass


TOOL_CONTRACT = ToolContract(
    name="command",
    description="在 Agent 的 Linux 电脑中运行 Bash 命令；默认用户是 agent，可用 sudo 获得 root 权限。长命令会返回 command_id，可继续轮询或停止。电脑内 localhost 上监听的 TCP Web 服务可由 browser_control 使用相同端口直接打开。",
    args_model=CommandArgs,
)

REQUIRES_CONTEXT = ["workspace_service", "runtime_event_hub", "main_loop"]


def make_handler(workspace_service, runtime_event_hub, main_loop):
    async def execute_async(**kwargs):
        app_state = __import__("app_state")
        action = str(kwargs.get("action") or "")
        if action == "run":
            started = await workspace_service.start_command(
                kwargs["command"], cwd=kwargs.get("cwd", DEFAULT_AGENT_HOME), stdin=kwargs.get("stdin", "")
            )
            target = current_focus_key(getattr(app_state, "current_focus", None)) or ""
            terminal_task = asyncio.create_task(
                workspace_service.wait_for_terminal(
                    started.command_id,
                    timeout=COMMAND_OBSERVATION_SECONDS,
                )
            )
            tasks: set[asyncio.Task] = {terminal_task}
            attention_task = None
            if runtime_event_hub is not None:
                attention_task = asyncio.create_task(
                    runtime_event_hub.wait(
                        timeout=COMMAND_OBSERVATION_SECONDS,
                        target=target,
                        event_types={"attention"},
                    )
                )
                tasks.add(attention_task)
            done, pending = await asyncio.wait(
                tasks,
                timeout=COMMAND_OBSERVATION_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            result = await workspace_service.poll_command(started.command_id, cursor=0)
            if result.terminal:
                await workspace_service.mark_terminal_delivered(result.command_id)
                await acknowledge_command(runtime_event_hub, result.command_id)
            return command_text_result(result)
        if action == "poll":
            result = await workspace_service.poll_command(kwargs["command_id"], cursor=kwargs.get("cursor", 0))
            if result.terminal:
                await workspace_service.mark_terminal_delivered(result.command_id)
                await acknowledge_command(runtime_event_hub, result.command_id)
            return command_text_result(result)
        result = await workspace_service.stop_command(kwargs["command_id"])
        if result.terminal:
            await workspace_service.mark_terminal_delivered(result.command_id)
            await acknowledge_command(runtime_event_hub, result.command_id)
        return command_text_result(result)

    def handler(**kwargs):
        return run_on_main_loop(execute_async(**kwargs), main_loop)

    return handler
