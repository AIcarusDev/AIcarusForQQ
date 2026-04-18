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

    WATCHER_ALLOW: bool                         （默认 False）
        为 True 时，该工具在窥屏（watcher）模式下可用

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
    is_watcher: bool（通过 **context 传入，默认 False）
        为 True 时进入窥屏模式：只收录 WATCHER_ALLOW=True 的工具，跳过 SCOPE 检查。
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

    # 提取控制标志（不污染 context）
    is_watcher: bool = bool(context.pop("is_watcher", False))

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

        # 2. SCOPE 过滤（普通模式和窥屏模式均生效）
        if conv_type is not None:
            scope: str = getattr(mod, "SCOPE", "all")
            if scope != "all" and scope != conv_type:
                continue

        # 3. WATCHER_ALLOW 过滤（仅窥屏模式额外检查）
        if is_watcher and not getattr(mod, "WATCHER_ALLOW", False):
            continue

        # 4. REQUIRES_CONTEXT：依赖注入（检查键均存在且非 None）
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

        get_decl = getattr(mod, "get_declaration", None)
        decl: dict[str, Any] = cast(dict[str, Any], get_decl() if callable(get_decl) else mod.DECLARATION)

        # ALWAYS_AVAILABLE=False 的工具进入潜伏注册表，不直接传给 LLM
        always_available: bool = getattr(mod, "ALWAYS_AVAILABLE", True)
        if always_available:
            declarations.append(decl)
            registry[name] = handler
        else:
            latent_registry[name] = (decl, handler)

    return declarations, registry, latent_registry
