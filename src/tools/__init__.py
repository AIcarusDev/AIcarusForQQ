"""tools/__init__.py — 工具自动发现与注册

扫描本目录下所有非 _ 开头的 .py 文件，收集工具定义，
通过 build_tools(config, **context) 统一构建 (declarations, registry)。

──────────────────────────────────────────────
每个工具模块 **必须** 导出：

    DECLARATION: dict
        工具声明（含 max_calls_per_response、name、description、parameters）

    execute(**kwargs) -> dict
        普通工具处理函数
    ── 或 ──
    REQUIRES_CONTEXT: list[str]
        需要的运行时上下文键名列表
    make_handler(**ctx) -> Callable
        工厂函数，接收上下文关键字参数，返回处理函数

**可选** 导出：

    condition(config: dict) -> bool
        返回 False 时跳过此工具（默认始终启用）
──────────────────────────────────────────────
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Callable, cast

logger = logging.getLogger("AICQ.tools")

# ── 启动时自动发现所有工具模块 ────────────────────────────

_TOOLS_DIR = Path(__file__).parent
_tool_modules: list = []

for _path in sorted(_TOOLS_DIR.glob("*.py")):
    if _path.name.startswith("_"):
        continue
    _mod_name = f"tools.{_path.stem}"
    try:
        _mod = importlib.import_module(_mod_name)
        if hasattr(_mod, "DECLARATION"):
            _tool_modules.append(_mod)
            # logger.debug("[tools] 已加载工具模块: %s", _path.stem)
        else:
            # logger.debug("[tools] 跳过 %s：没有 DECLARATION", _path.name)
            pass
    except Exception as exc:
        logger.warning("[tools] 加载工具模块 %s 失败: %s", _path.name, exc)


# ── 对外接口 ──────────────────────────────────────────────

def build_tools(
    config: dict,
    **context: Any,
) -> tuple[list[dict], dict[str, Callable]]:
    """根据当前配置和运行时上下文，构建工具声明列表和注册表。

    参数
    ----
    config:
        应用配置字典（来自 config.yaml）
    **context:
        运行时上下文，例如 napcat_client=..., group_id=...
        带 REQUIRES_CONTEXT 的工具要求对应键存在且不为 None，
        否则该工具被自动跳过（不添加到声明/注册表中）。

    返回
    ----
    (tool_declarations, tool_registry)
    """
    declarations: list[dict[str, Any]] = []
    registry: dict[str, Callable] = {}

    # 将 config 注入 context，允许工具通过 REQUIRES_CONTEXT 声明后获取
    context["config"] = config

    for mod in _tool_modules:
        name: str = mod.DECLARATION.get("name", "")

        # 1. 检查静态配置条件
        cond = getattr(mod, "condition", None)
        if cond is not None and not cond(config):
            continue

        requires: list[str] | None = getattr(mod, "REQUIRES_CONTEXT", None)

        if requires:
            # 2a. 有运行时上下文依赖 → 检查键均存在且非 None
            if not all(context.get(k) is not None for k in requires):
                continue

            make_handler = getattr(mod, "make_handler", None)
            if make_handler is None:
                logger.warning(
                    "[tools] %s 有 REQUIRES_CONTEXT 但缺少 make_handler，跳过", name
                )
                continue

            ctx_kwargs = {k: context[k] for k in requires}
            handler: Callable = make_handler(**ctx_kwargs)
        else:
            # 2b. 普通工具 → 直接使用 execute
            raw_handler = getattr(mod, "execute", None)
            if not callable(raw_handler):
                logger.warning("[tools] %s 缺少 execute，跳过", name)
                continue
            handler: Callable = raw_handler

        get_decl = getattr(mod, "get_declaration", None)
        decl: dict[str, Any] = cast(dict[str, Any], get_decl() if callable(get_decl) else mod.DECLARATION)
        declarations.append(decl)
        registry[name] = handler

    return declarations, registry
