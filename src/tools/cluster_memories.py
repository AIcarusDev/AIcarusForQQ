"""cluster_memories.py — 将语义相关的记忆三元组归并为聚类

聚类让多条记录「互相验证」，search_triples 在命中聚类成员时会通过
activation propagation 自动补充同簇其他成员，提升关联记忆的召回覆盖率。
聚类置信度（label-level）会覆盖成员三元组的 effective_confidence 参与排序。

工具入口：
  create_cluster  — 新建聚类，返回 cluster_id
  assign_cluster  — 将已有三元组 ID 列表分配到某个聚类
  list_clusters   — 列出现有聚类，供模型判断是否已有合适聚类可复用
"""

from typing import Any, Callable

DECLARATION: dict = {
    "name": "cluster_memories",
    "description": (
        "将相关记忆三元组归并为同一聚类，实现置信度交叉验证和关联召回。\n"
        "支持三个操作（通过 action 字段区分）：\n"
        "  create  — 新建聚类，需提供 label；返回 cluster_id\n"
        "  assign  — 将 triple_ids 分配到已有 cluster_id\n"
        "  list    — 列出现有聚类（可在 create/assign 前先 list 确认是否复用）"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "assign", "list"],
                "description": "要执行的操作：create / assign / list",
            },
            "label": {
                "type": "string",
                "description": "（create 时必填）聚类的语义标签，简洁描述这批记忆的共同主题",
            },
            "cluster_id": {
                "type": "integer",
                "description": "（assign 时必填）目标聚类 ID（由 create 或 list 获取）",
            },
            "triple_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "（assign 时必填）要分配的三元组 ID 列表（来自 <memory> 块中的 id 属性）",
            },
            "confidence": {
                "type": "number",
                "description": "（create 时可选）聚类初始置信度，默认 0.6，范围 [0, 1]",
            },
        },
        "required": ["action"],
    },
}

REQUIRES_CONTEXT: list[str] = ["session"]


def make_handler(session: Any) -> Callable:
    import asyncio
    import app_state

    def execute(
        action: str,
        label: str = "",
        cluster_id: int | None = None,
        triple_ids: list[int] | None = None,
        confidence: float = 0.6,
        **kwargs,
    ) -> dict:
        loop: asyncio.AbstractEventLoop | None = app_state.main_loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        action = (action or "").strip().lower()

        if action == "create":
            if not label:
                return {"error": "create 操作需要提供 label"}
            from database import create_cluster
            coro = create_cluster(label=label, confidence=max(0.0, min(1.0, float(confidence))))
            try:
                new_id = asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=10)
            except Exception as e:
                return {"error": f"创建聚类失败: {e}"}
            return {"ok": True, "cluster_id": new_id, "label": label}

        elif action == "assign":
            if cluster_id is None:
                return {"error": "assign 操作需要提供 cluster_id"}
            if not triple_ids:
                return {"error": "assign 操作需要提供至少一个 triple_id"}
            from database import assign_cluster
            coro = assign_cluster(triple_ids=triple_ids, cluster_id=cluster_id)
            try:
                updated = asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=10)
            except Exception as e:
                return {"error": f"分配聚类失败: {e}"}
            return {"ok": True, "cluster_id": cluster_id, "updated": updated}

        elif action == "list":
            from database import list_clusters
            coro = list_clusters(limit=20)
            try:
                clusters = asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=10)
            except Exception as e:
                return {"error": f"列举聚类失败: {e}"}
            return {"ok": True, "clusters": clusters}

        else:
            return {"error": f"未知 action：{action!r}，有效值为 create / assign / list"}

    return execute
