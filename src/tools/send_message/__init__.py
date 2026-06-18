"""send_message/__init__.py — 发送消息工具（升级版）

包含：
- 单条/数组两种模型可见参数形态，统一写入 session.context_messages，持久化 DB
- 广播到 debug 前端
- send_message 并行调用保护：provider 会优先串行执行外界可感知工具
"""

from .send_message import (
	DECLARATION,
	EXTERNALLY_PERCEPTIBLE,
	REQUIRES_CONTEXT,
	get_declaration,
	make_handler,
	repair_schema_args,
	sanitize_semantic_args,
)

__all__ = [
	"DECLARATION",
	"EXTERNALLY_PERCEPTIBLE",
	"REQUIRES_CONTEXT",
	"get_declaration",
	"make_handler",
	"repair_schema_args",
	"sanitize_semantic_args",
]
