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
        工具适用的会话类型说明："group" | "private" | "all"。
        loader 不再用它做 prompt/build 阶段过滤。

    EXTERNALLY_PERCEPTIBLE: bool                （默认 False）
        工具成功执行时必然产生可被外部客体感知的副作用。
        这类工具由执行器优先串行执行，且与 shift 同轮调用时会被阻断。
        若其中一个被执行前守门拒绝，本轮后续同类工具也会跳过。

    condition(config: dict) -> bool
        返回 False 时跳过此工具（默认始终启用）
──────────────────────────────────────────────
"""

import importlib
import inspect
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, cast

from llm.compression.config import normalize_generation_config

from .namespaces import (
    CORE_NAMESPACE,
    NamespaceRegistry,
    NamespaceRuntimeState,
    load_namespace_registry,
    recover_namespace_state_from_flow,
)
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
    factory = getattr(mod, factory_attr, None)
    if callable(factory):
        built = _invoke_with_supported_context(factory, context)
        if callable(built):
            return built

    direct = getattr(mod, direct_attr, None)
    if callable(direct):
        return cast(Callable, direct)
    return None


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
    *,
    namespace_state: NamespaceRuntimeState | None = None,
    current_round: int = 0,
    default_ttl_rounds: int | None = None,
    flow: Any = None,
    **context: Any,
) -> ToolCollection:
    """根据当前配置和运行时上下文，构建统一工具集合。

    参数
    ----
    config:
        应用配置字典（来自 config.yaml）
    **context:
        运行时上下文，例如 qq_adapter_client=..., session=...
        带 REQUIRES_CONTEXT 的工具要求对应键存在且不为 None，
        否则该工具被自动跳过（不添加到声明/注册表中）。

    返回
    ----
    ToolCollection
    active_specs: 当前 active namespace 中可直接传给 LLM 并执行的工具
    latent_specs: inactive namespace 中可被发现但本轮不能直接执行的工具
    """
    registry = load_namespace_registry()
    all_specs: dict[str, ToolSpec] = {}
    # 将 config 注入 context，允许工具通过 REQUIRES_CONTEXT 声明后获取
    context["config"] = config

    for mod in _tool_modules:
        name: str = mod.DECLARATION.get("name", "")

        # 1. 检查静态配置条件
        cond = getattr(mod, "condition", None)
        if cond is not None and not cond(config):
            continue

        handler = _build_handler(mod, context, name)
        if handler is None:
            continue

        decl = _build_declaration(mod, context)
        name = str(decl.get("name") or name).strip()
        namespace = registry.namespace_for_tool(name)
        if not namespace:
            continue

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
            externally_perceptible=bool(getattr(mod, "EXTERNALLY_PERCEPTIBLE", False)),
            always_available=(namespace == CORE_NAMESPACE),
            schema_repairer=schema_repairer,
            semantic_sanitizer=semantic_sanitizer,
            namespace=namespace,
        )
        all_specs[name] = spec

    if namespace_state is None:
        namespace_state = NamespaceRuntimeState()
    if not namespace_state.recovered_from_flow and flow is not None:
        recovered = recover_namespace_state_from_flow(
            flow,
            registry,
            max_rounds=default_ttl_rounds,
            current_round=current_round,
        )
        namespace_state.replace_with(recovered)
    namespace_state.recovered_from_flow = True

    if default_ttl_rounds is None:
        default_ttl_rounds = normalize_generation_config(
            (config or {}).get("generation")
        )["llm_contents_max_rounds"]
    namespace_state.apply_ttl(
        registry,
        current_round=current_round,
        default_ttl_rounds=default_ttl_rounds,
    )

    active_specs, latent_specs, active_namespace_order = _partition_namespace_specs(
        all_specs,
        registry,
        namespace_state,
    )
    namespace_specs = {
        name: spec
        for name, spec in registry.namespaces.items()
        if any(tool in all_specs for tool in spec.tools)
    }

    return ToolCollection(
        active_specs=active_specs,
        latent_specs=latent_specs,
        all_specs=all_specs,
        namespace_specs=namespace_specs,
        namespace_registry=registry,
        namespace_state=namespace_state,
        active_namespace_order=active_namespace_order,
        round_index=current_round,
    )


def _partition_namespace_specs(
    all_specs: dict[str, ToolSpec],
    registry: NamespaceRegistry,
    namespace_state: NamespaceRuntimeState,
) -> tuple[dict[str, ToolSpec], dict[str, ToolSpec], list[str]]:
    active_namespace_order = [
        namespace
        for namespace in namespace_state.active_namespaces(registry)
        if namespace in registry.namespaces
        and any(tool in all_specs for tool in registry.namespaces[namespace].tools)
    ]
    active_namespaces = set(active_namespace_order)
    active_specs: dict[str, ToolSpec] = {}

    for namespace in active_namespace_order:
        ns_spec = registry.namespaces[namespace]
        for tool_name in ns_spec.tools:
            spec = all_specs.get(tool_name)
            if spec is not None:
                active_specs[tool_name] = replace(spec, attached_to="")

    for namespace in active_namespace_order:
        ns_spec = registry.namespaces[namespace]
        for attach in ns_spec.attach:
            if attach.namespace in active_namespaces:
                continue
            attached_spec = all_specs.get(attach.tool)
            if attached_spec is not None:
                active_specs.setdefault(attach.tool, replace(attached_spec, attached_to=namespace))

    latent_specs = {
        name: spec
        for name, spec in all_specs.items()
        if name not in active_specs
    }
    return active_specs, latent_specs, active_namespace_order


__all__ = ["ToolCollection", "ToolSpec", "build_tools"]
