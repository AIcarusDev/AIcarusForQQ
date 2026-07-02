"""tools/__init__.py — 工具自动发现与注册

按 namespaces.yaml 中声明的 namespace 目录扫描工具定义，
通过 build_tools(config, **context) 统一构建 ToolCollection。

──────────────────────────────────────────────
每个工具模块 **必须** 导出：

    DECLARATION: dict
        执行校验 schema（含 name、description、parameters）

    PROMPT_SIGNATURE: str
        模型可见 TypeScript-like 函数签名

    execute(**kwargs) -> dict
        普通工具处理函数
    ── 或 ──
    REQUIRES_CONTEXT: list[str]
        需要的运行时上下文键名列表（仅用于依赖注入，不再承担过滤语义）
    make_handler(**ctx) -> Callable
        工厂函数，接收上下文关键字参数，返回处理函数

**可选** 导出：

    EXTERNALLY_PERCEPTIBLE: bool                （默认 False）
        工具成功执行时必然产生可被外部客体感知的副作用。
        这类工具由执行器优先串行执行，且与焦点切换工具同轮调用时会被阻断。
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
    ModuleRegistry,
    NamespaceRegistry,
    NamespaceRuntimeState,
    load_module_registry,
    load_namespace_registry,
    recover_namespace_state_from_flow,
)
from .contract import get_contract_from_module
from .prompt_signatures import normalize_prompt_signature, strip_schema_descriptions
from .specs import ToolCollection, ToolEffect, ToolSpec

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
        contract = get_contract_from_module(mod)
        if contract is not None:
            return contract.declaration()
        return cast(dict[str, Any], mod.DECLARATION)

    return cast(dict[str, Any], _invoke_with_supported_context(get_decl, context))


def _build_prompt_signature(
    mod: Any,
    declaration: dict[str, Any],
    context: dict[str, Any],
) -> str:
    """Build the model-facing TypeScript-like signature for a tool."""
    get_signature = getattr(mod, "get_prompt_signature", None)
    if callable(get_signature):
        return normalize_prompt_signature(_invoke_with_supported_context(get_signature, context))

    signature = getattr(mod, "PROMPT_SIGNATURE", None)
    if isinstance(signature, str):
        return normalize_prompt_signature(signature)

    contract = get_contract_from_module(mod)
    if contract is not None:
        return normalize_prompt_signature(contract.prompt_signature())

    name = str(declaration.get("name") or getattr(mod, "__name__", "")).strip()
    raise RuntimeError(
        f"tool {name!r} must export PROMPT_SIGNATURE or get_prompt_signature; "
        "first-party tools must not fall back to generated JSON-Schema-derived signatures"
    )


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


def _build_tool_effect(value: Any) -> ToolEffect | None:
    if isinstance(value, ToolEffect):
        return value
    if not isinstance(value, dict):
        return None
    surface = str(value.get("surface") or "").strip()
    kind = str(value.get("kind") or "").strip()
    if not surface or not kind:
        return None
    return ToolEffect(surface=surface, kind=kind)


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

# ── 启动时按 namespace 目录自动发现工具模块 ─────────────────

_TOOLS_DIR = Path(__file__).parent
_tool_modules: list = []


def _import_tool_module(module_name: str, display_name: str) -> None:
    try:
        _mod = importlib.import_module(module_name)
        if hasattr(_mod, "DECLARATION") or get_contract_from_module(_mod) is not None:
            _tool_modules.append(_mod)
            # logger.debug("[tools] 已加载工具模块: %s", display_name)
        else:
            # logger.debug("[tools] 跳过 %s：没有 DECLARATION", display_name)
            pass
    except Exception as exc:
        logger.warning("[tools] 加载工具模块 %s 失败: %s", display_name, exc)


def _discover_tool_modules() -> None:
    try:
        registry = load_namespace_registry()
    except Exception as exc:
        logger.warning("[tools] 读取 namespace registry 失败: %s", exc)
        return

    for namespace in registry.order:
        ns_spec = registry.namespaces.get(namespace)
        if ns_spec is None:
            continue
        namespace_path = ns_spec.path or namespace
        namespace_dir = _TOOLS_DIR / Path(namespace_path)
        if not namespace_dir.is_dir():
            continue
        module_prefix = "tools." + ".".join(Path(namespace_path).parts)
        for path in sorted(namespace_dir.glob("*.py")):
            if path.name.startswith("_") or path.name == "__init__.py":
                continue
            _import_tool_module(
                f"{module_prefix}.{path.stem}",
                f"{namespace_path}/{path.name}",
            )
        for path in sorted(namespace_dir.iterdir()):
            if not path.is_dir():
                continue
            if path.name.startswith("_") or not (path / "__init__.py").exists():
                continue
            _import_tool_module(
                f"{module_prefix}.{path.name}",
                f"{namespace_path}/{path.name}/",
            )


def _discovered_tool_names() -> set[str]:
    names: set[str] = set()
    for mod in _tool_modules:
        contract = get_contract_from_module(mod)
        if contract is not None and contract.name:
            names.add(contract.name)
            continue
        declaration = getattr(mod, "DECLARATION", None)
        if not isinstance(declaration, dict):
            continue
        name = str(declaration.get("name") or "").strip()
        if name:
            names.add(name)
    return names


def _module_tool_name(mod: Any) -> str:
    contract = get_contract_from_module(mod)
    if contract is not None:
        return contract.name
    declaration = getattr(mod, "DECLARATION", None)
    if isinstance(declaration, dict):
        return str(declaration.get("name") or "").strip()
    return ""


def _warn_missing_registry_tools(registry: NamespaceRegistry) -> None:
    discovered = _discovered_tool_names()
    for namespace in registry.order:
        spec = registry.namespaces.get(namespace)
        if spec is None:
            continue
        missing = [tool for tool in spec.tools if tool not in discovered]
        if not missing:
            continue
        logger.warning(
            "[tools] namespace %s 中声明了未发现的工具模块: %s",
            namespace,
            ", ".join(missing),
        )


def _condition_enabled(name: str, config: dict, context: dict[str, Any]) -> bool:
    if not name:
        return True
    if name == "qq_adapter_enabled":
        client = context.get("qq_adapter_client")
        return bool(client and getattr(client, "connected", True))
    if name == "browser_available":
        return True
    if name == "browser_world_active":
        try:
            from browser.session import browser_world_view_state

            state = browser_world_view_state()
            return bool(state.get("active"))
        except Exception:
            logger.debug("[tools] browser active check failed", exc_info=True)
            return False
    logger.warning("[tools] unknown module condition %s; treating as disabled", name)
    return False


def _module_active(module_name: str, modules: ModuleRegistry, config: dict, context: dict[str, Any]) -> bool:
    module = modules.modules.get(module_name)
    if module is None:
        return False
    if module.always_active:
        return True
    return _condition_enabled(module.active_when, config, context)


def _active_modules(modules: ModuleRegistry, config: dict, context: dict[str, Any]) -> set[str]:
    return {
        name
        for name in modules.order
        if _module_active(name, modules, config, context)
    }


def _module_for_namespace(namespace: str, modules: ModuleRegistry) -> str:
    for module_name in modules.order:
        module = modules.modules.get(module_name)
        if module is not None and namespace in module.namespaces:
            return module_name
    return ""


_discover_tool_modules()
_REGISTRY_WARNED = False


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
    module_registry = load_module_registry()
    global _REGISTRY_WARNED
    if not _REGISTRY_WARNED:
        _warn_missing_registry_tools(registry)
        _REGISTRY_WARNED = True

    all_specs: dict[str, ToolSpec] = {}
    # 将 config 注入 context，允许工具通过 REQUIRES_CONTEXT 声明后获取
    context["config"] = config

    for mod in _tool_modules:
        name = _module_tool_name(mod)

        # 1. 检查静态配置条件
        cond = getattr(mod, "condition", None)
        if cond is not None and not cond(config):
            continue

        handler = _build_handler(mod, context, name)
        if handler is None:
            continue

        raw_decl = _build_declaration(mod, context)
        name = str(raw_decl.get("name") or name).strip()
        description = str(raw_decl.get("description") or "").strip()
        prompt_signature = _build_prompt_signature(mod, raw_decl, context)
        decl = cast(dict[str, Any], strip_schema_descriptions(raw_decl))
        namespace = registry.namespace_for_tool(name)
        if not namespace:
            continue
        namespace_spec = registry.get(namespace)
        if namespace_spec is None:
            continue
        module_name = _module_for_namespace(namespace, module_registry)
        if module_name and not _module_active(module_name, module_registry, config, context):
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
            description=description,
            prompt_signature=prompt_signature,
            handler=handler,
            module_name=getattr(mod, "__name__", name),
            externally_perceptible=bool(getattr(mod, "EXTERNALLY_PERCEPTIBLE", False)),
            always_available=(namespace == CORE_NAMESPACE),
            schema_repairer=schema_repairer,
            semantic_sanitizer=semantic_sanitizer,
            namespace=namespace,
            visible_namespace=namespace if namespace_spec.visible else "",
            visibility="visible" if namespace_spec.visible else "internal",
            tool_kind=str(getattr(mod, "TOOL_KIND", "") or "").strip(),
            effect=_build_tool_effect(getattr(mod, "TOOL_EFFECT", None)),
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
        module_registry,
        namespace_state,
        config,
        context,
    )
    namespace_specs = {
        name: spec
        for name, spec in registry.namespaces.items()
        if spec.visible and any(tool in all_specs for tool in spec.tools)
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
    module_registry: ModuleRegistry,
    namespace_state: NamespaceRuntimeState,
    config: dict,
    context: dict[str, Any],
) -> tuple[dict[str, ToolSpec], dict[str, ToolSpec], list[str]]:
    active_namespace_order = [
        namespace
        for namespace in namespace_state.active_namespaces(registry)
        if namespace in registry.namespaces
        and registry.namespaces[namespace].visible
        and any(tool in all_specs for tool in registry.namespaces[namespace].tools)
    ]
    active_namespaces = set(active_namespace_order)
    active_specs: dict[str, ToolSpec] = {}

    for namespace in active_namespace_order:
        ns_spec = registry.namespaces[namespace]
        for tool_name in ns_spec.tools:
            spec = all_specs.get(tool_name)
            if spec is not None:
                active_specs[tool_name] = replace(
                    spec,
                    attached_to="",
                    mounted_to="",
                    visible_namespace=namespace,
                )

    for namespace in active_namespace_order:
        ns_spec = registry.namespaces[namespace]
        for attach in ns_spec.attach:
            if attach.namespace in active_namespaces:
                continue
            attached_spec = all_specs.get(attach.tool)
            if attached_spec is not None:
                active_specs.setdefault(
                    attach.tool,
                    replace(attached_spec, attached_to=namespace, visible_namespace=namespace),
                )

    active_modules = _active_modules(module_registry, config, context)
    for module_name in module_registry.order:
        if module_name not in active_modules:
            continue
        module = module_registry.modules[module_name]
        for mount in module.mounts:
            target = mount.target_namespace
            source = mount.source_namespace
            target_spec = registry.get(target)
            source_spec = registry.get(source)
            if target not in active_namespaces:
                continue
            if target_spec is None or source_spec is None:
                continue
            if not target_spec.visible or source_spec.visible:
                continue
            if not _condition_enabled(mount.when, config, context):
                continue
            for tool_name in mount.tools:
                mounted_spec = all_specs.get(tool_name)
                if mounted_spec is None or mounted_spec.namespace != source:
                    continue
                active_specs.setdefault(
                    tool_name,
                    replace(
                        mounted_spec,
                        visible_namespace=target,
                        mounted_to=target,
                        mounted_by_module=module_name,
                    ),
                )

    latent_specs = {
        name: spec
        for name, spec in all_specs.items()
        if name not in active_specs
        and spec.visibility != "internal"
    }
    return active_specs, latent_specs, active_namespace_order


__all__ = ["ToolCollection", "ToolEffect", "ToolSpec", "build_tools"]
