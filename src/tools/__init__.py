"""tools/__init__.py — 工具自动发现与注册

扫描本目录下所有非 _ 开头的 .py 文件，收集工具定义，
通过 build_tools(config, **context) 统一构建 (declarations, registry)。

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

logger = logging.getLogger("AICQ.tools")


def _make_clamping_wrapper(tool_name: str, decl: dict, handler: Callable) -> Callable:
    """对整数参数做 min/max 钳位的包装器。

    若 LLM 输出的整数参数超出 DECLARATION 中声明的 minimum/maximum，
    则静默将其修正到最近合法值，并打印一条警告日志，不向 LLM 暴露任何错误。
    """
    props: dict = decl.get("parameters", {}).get("properties", {})
    clamp_rules: dict[str, tuple] = {}
    for param_name, schema in props.items():
        if schema.get("type") != "integer":
            continue
        lo = schema.get("minimum")
        hi = schema.get("maximum")
        if lo is not None or hi is not None:
            clamp_rules[param_name] = (lo, hi)

    if not clamp_rules:
        return handler

    def _wrapper(**kwargs: Any) -> Any:
        for param, (lo, hi) in clamp_rules.items():
            if param not in kwargs:
                continue
            val = kwargs[param]
            if not isinstance(val, int):
                continue
            clamped = val
            if lo is not None:
                clamped = max(clamped, lo)
            if hi is not None:
                clamped = min(clamped, hi)
            if clamped != val:
                lo_str = str(lo) if lo is not None else "-∞"
                hi_str = str(hi) if hi is not None else "+∞"
                logger.warning(
                    "[tools] 参数越界自动修正: 工具=%s 参数=%s 原值=%d → 修正为=%d（允许范围 [%s, %s]）",
                    tool_name, param, val, clamped, lo_str, hi_str,
                )
                kwargs[param] = clamped
        return handler(**kwargs)

    return _wrapper


def _build_declaration(mod: Any, context: dict[str, Any]) -> dict[str, Any]:
    """构建工具 schema，支持 get_declaration 按上下文动态生成。"""
    get_decl = getattr(mod, "get_declaration", None)
    if not callable(get_decl):
        return cast(dict[str, Any], mod.DECLARATION)

    try:
        signature = inspect.signature(get_decl)
    except (TypeError, ValueError):
        return cast(dict[str, Any], get_decl())

    parameters = signature.parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return cast(dict[str, Any], get_decl(**context))

    accepted_kwargs = {
        name: context[name]
        for name, param in signature.parameters.items()
        if name in context
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return cast(dict[str, Any], get_decl(**accepted_kwargs))

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
) -> tuple[list[dict], dict[str, Callable], dict[str, tuple[dict, Callable]]]:
    """根据当前配置和运行时上下文，构建工具声明列表和注册表。

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
    (tool_declarations, tool_registry, latent_registry)
    tool_declarations/tool_registry: 常驻工具（ALWAYS_AVAILABLE=True，默认值）
    latent_registry: 潜伏工具 {name: (declaration, handler)}，需经 get_tools 激活
    """
    declarations: list[dict[str, Any]] = []
    registry: dict[str, Callable] = {}
    latent_registry: dict[str, tuple[dict, Callable]] = {}

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

        # 3. REQUIRES_CONTEXT：依赖注入（检查键均存在且非 None）
        requires: list[str] | None = getattr(mod, "REQUIRES_CONTEXT", None)

        if requires:
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
            # 普通工具 → 直接使用 execute
            raw_handler = getattr(mod, "execute", None)
            if not callable(raw_handler):
                logger.warning("[tools] %s 缺少 execute，跳过", name)
                continue
            handler: Callable = raw_handler

        decl = _build_declaration(mod, context)
        handler = _make_clamping_wrapper(name, decl, handler)

        # ALWAYS_AVAILABLE=False 的工具进入潜伏注册表，不直接传给 LLM
        always_available: bool = getattr(mod, "ALWAYS_AVAILABLE", True)
        if always_available:
            declarations.append(decl)
            registry[name] = handler
        else:
            latent_registry[name] = (decl, handler)

    return declarations, registry, latent_registry
