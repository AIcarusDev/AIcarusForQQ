"""tools/__init__.py — 工具自动发现与注册

扫描本目录下所有非 _ 开头的 .py 文件，收集工具定义，
通过 build_tools(config, **context) 统一构建 ToolCollection。

──────────────────────────────────────────────
每个工具模块 **必须** 导出：

    DECLARATION: dict
        工具声明（含 name、description、parameters）

    execute(**kwargs) -> dict
        普通工具处理函数
    ── 或 ──
    REQUIRES_CONTEXT: list[str]
        需要的运行时上下文键名列表（仅用于依赖注入，不再承担过滤语义）
    make_handler(**ctx) -> Callable
        工厂函数，接收上下文关键字参数，返回处理函数

**可选** 导出：

    SCOPE: str                                  （默认 "all"）
        工具适用的会话类型："group" | "private" | "all"


    condition(config: dict) -> bool
        返回 False 时跳过此工具（默认始终启用）
──────────────────────────────────────────────
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, cast

from .specs import ToolCollection, ToolSpec

logger = logging.getLogger("AICQ.tools")


def _invoke_with_supported_context(func: Callable[..., Any], context: dict[str, Any]) -> Any:
    """按签名过滤上下文后调用工厂函数。"""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func()

    parameters = signature.parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return func(**context)

    accepted_kwargs = {
        name: context[name]
        for name, param in signature.parameters.items()
        if name in context
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return func(**accepted_kwargs)


def _build_declaration(mod: Any, context: dict[str, Any]) -> dict[str, Any]:
    """构建工具 schema，支持 get_declaration 按上下文动态生成。"""
    get_decl = getattr(mod, "get_declaration", None)
    if not callable(get_decl):
        return cast(dict[str, Any], mod.DECLARATION)

    return cast(dict[str, Any], _invoke_with_supported_context(get_decl, context))


def _build_optional_processor(
    mod: Any,
    context: dict[str, Any],
    direct_attr: str,
    factory_attr: str,
) -> Callable | None:
    """构建可选的 schema/semantic 处理钩子。"""
    direct = getattr(mod, direct_attr, None)
    if callable(direct):
        return cast(Callable, direct)

    factory = getattr(mod, factory_attr, None)
    if not callable(factory):
        return None

    built = _invoke_with_supported_context(factory, context)
    return built if callable(built) else None


def _build_handler(mod: Any, context: dict[str, Any], name: str) -> Callable | None:
    """构建工具执行 handler。"""
    requires: list[str] | None = getattr(mod, "REQUIRES_CONTEXT", None)
    if requires:
        if not all(context.get(k) is not None for k in requires):
            return None

        make_handler = getattr(mod, "make_handler", None)
        if make_handler is None:
            logger.warning("[tools] %s 有 REQUIRES_CONTEXT 但缺少 make_handler，跳过", name)
            return None

        ctx_kwargs = {k: context[k] for k in requires}
        return cast(Callable, make_handler(**ctx_kwargs))

    raw_handler = getattr(mod, "execute", None)
    if not callable(raw_handler):
        logger.warning("[tools] %s 缺少 execute，跳过", name)
        return None
    return cast(Callable, raw_handler)

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

# 扫描子目录（文件夹工具），忽略 not_used 和 _ 开头的目录
_IGNORED_DIRS = {"not_used"}
for _dir in sorted(_TOOLS_DIR.iterdir()):
    if not _dir.is_dir():
        continue
    if _dir.name.startswith("_") or _dir.name in _IGNORED_DIRS:
        continue
    if not (_dir / "__init__.py").exists():
        continue
    _mod_name = f"tools.{_dir.name}"
    try:
        _mod = importlib.import_module(_mod_name)
        if hasattr(_mod, "DECLARATION"):
            _tool_modules.append(_mod)
            # logger.debug("[tools] 已加载文件夹工具模块: %s/", _dir.name)
        else:
            # logger.debug("[tools] 跳过 %s/：没有 DECLARATION", _dir.name)
            pass
    except Exception as exc:
        logger.warning("[tools] 加载文件夹工具模块 %s/ 失败: %s", _dir.name, exc)


# ── 对外接口 ──────────────────────────────────────────────

def build_tools(
    config: dict,
    **context: Any,
) -> ToolCollection:
    """根据当前配置和运行时上下文，构建统一工具集合。

    参数
    ----
    config:
        应用配置字典（来自 config.yaml）
    **context:
        运行时上下文，例如 napcat_client=..., session=...
        带 REQUIRES_CONTEXT 的工具要求对应键存在且不为 None，
        否则该工具被自动跳过（不添加到声明/注册表中）。

    返回
    ----
    ToolCollection
    active_specs: 当前可直接传给 LLM 并执行的工具
    latent_specs: 潜伏工具，需经 get_tools 激活后才能使用
    """
    active_specs: dict[str, ToolSpec] = {}
    latent_specs: dict[str, ToolSpec] = {}

    # 将 config 注入 context，允许工具通过 REQUIRES_CONTEXT 声明后获取
    context["config"] = config

    # 提取会话类型用于 SCOPE 过滤
    session = context.get("session")
    conv_type: str | None = getattr(session, "conv_type", None) if session else None

    for mod in _tool_modules:
        name: str = mod.DECLARATION.get("name", "")

        # 1. 检查静态配置条件
        cond = getattr(mod, "condition", None)
        if cond is not None and not cond(config):
            continue

        # 2. SCOPE 过滤
        if conv_type is not None:
            scope: str = getattr(mod, "SCOPE", "all")
            if scope != "all" and scope != conv_type:
                continue

        handler = _build_handler(mod, context, name)
        if handler is None:
            continue

        decl = _build_declaration(mod, context)
        schema_repairer = _build_optional_processor(
            mod,
            context,
            "repair_schema_args",
            "make_schema_repairer",
        )
        semantic_sanitizer = _build_optional_processor(
            mod,
            context,
            "sanitize_semantic_args",
            "make_semantic_sanitizer",
        )
        spec = ToolSpec(
            name=name,
            declaration=decl,
            handler=handler,
            module_name=getattr(mod, "__name__", name),
            always_available=getattr(mod, "ALWAYS_AVAILABLE", True),
            schema_repairer=schema_repairer,
            semantic_sanitizer=semantic_sanitizer,
        )

        # ALWAYS_AVAILABLE=False 的工具进入潜伏注册表，不直接传给 LLM
        if spec.always_available:
            active_specs[name] = spec
        else:
            latent_specs[name] = spec

    return ToolCollection(active_specs=active_specs, latent_specs=latent_specs)


__all__ = ["ToolCollection", "ToolSpec", "build_tools"]
