"""Provider entrypoint.

The concrete responsibilities live in:
- transport.py: OpenAI-compatible SDK transport and provider generation args
- round_runner.py: one-round LLM orchestration
- tool_executor.py: local XML tool execution
"""

from __future__ import annotations

from .round_runner import LLMCallFailed, LLMRoundRunner as _LLMRoundRunner, RoundResult


def create_adapter(cfg: dict) -> _LLMRoundRunner:
    """根据 config 中的 OpenAI 兼容模型供应商创建一轮执行器。"""
    return _LLMRoundRunner(cfg)


def _clean_model_text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _build_explicit_adapter_cfg(main_cfg: dict, model_cfg: dict, label: str) -> dict:
    provider = _clean_model_text(model_cfg.get("provider"))
    model = _clean_model_text(model_cfg.get("model"))
    if not provider or not model:
        raise ValueError(f"{label} 必须显式配置 provider 和 model")

    cfg = dict(main_cfg)
    cfg.pop("model_name", None)
    cfg.pop("profile", None)
    cfg.pop("base_url", None)
    cfg.pop("api_key_env", None)
    cfg["provider"] = provider
    cfg["model"] = model
    if "generation" in model_cfg:
        cfg["generation"] = model_cfg["generation"]
    if "vision" in model_cfg:
        cfg["vision"] = model_cfg["vision"]
    return cfg


def build_is_adapter_cfg(main_cfg: dict, is_cfg: dict) -> dict:
    """构建 IS（中断哨兵）专用的 adapter 配置。"""
    return _build_explicit_adapter_cfg(main_cfg, is_cfg, "IS 中断哨兵")


def build_slow_thinking_adapter_cfg(main_cfg: dict, st_cfg: dict) -> dict:
    """构建 slow_thinking 专用的 adapter 配置。"""
    return _build_explicit_adapter_cfg(main_cfg, st_cfg, "慢思考模型")


def build_archiver_adapter_cfg(main_cfg: dict, archiver_cfg: dict) -> dict:
    """构建记忆提取（archiver）专用的 adapter 配置。"""
    return _build_explicit_adapter_cfg(main_cfg, archiver_cfg, "记忆归档模型")


def build_compression_adapter_cfg(main_cfg: dict, compression_cfg: dict) -> dict:
    """构建上下文压缩专用的 adapter 配置。"""
    return _build_explicit_adapter_cfg(main_cfg, compression_cfg, "上下文压缩模型")


__all__ = [
    "LLMCallFailed",
    "RoundResult",
    "build_archiver_adapter_cfg",
    "build_compression_adapter_cfg",
    "build_is_adapter_cfg",
    "build_slow_thinking_adapter_cfg",
    "create_adapter",
]
