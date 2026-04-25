"""send_message/__init__.py — 发送消息工具（升级版）

包含：
- 逐条发送，写入 session.context_messages，持久化 DB
- 广播到 debug 前端
- IS（中断哨兵）内嵌检测
- send_message 并行调用保护：构建工具时注入的 handler 始终串行执行
"""

from .send_message import (
	DECLARATION,
	REQUIRES_CONTEXT,
	get_declaration,
	make_handler,
	repair_schema_args,
	sanitize_semantic_args,
)

__all__ = [
	"DECLARATION",
	"REQUIRES_CONTEXT",
	"get_declaration",
	"make_handler",
	"repair_schema_args",
	"sanitize_semantic_args",
]
