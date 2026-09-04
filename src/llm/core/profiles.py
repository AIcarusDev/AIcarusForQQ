"""profiles.py — OpenAI 兼容模型供应商定义与解析。"""

from fnmatch import fnmatchcase
import re
from urllib.parse import urlparse


_VALID_THINKING_CONTROLS = {
    "enable_thinking",
    "thinking",
    "reasoning_effort",
    "reasoning_effort_none",
    "none",
}

# OpenCode Go aggregates heterogeneous upstreams behind one provider URL. These
# controls were verified against the live API on 2026-09-02. Keep unknown or
# explicitly forced-thinking models on the provider fallback (``none``).
_OPENCODE_GO_MODEL_THINKING_CONTROLS = {
    "kimi-k2.6": "thinking",
    "kimi-k3": "reasoning_effort_none",
    "deepseek-v4-*": "thinking",
    "qwen3.*": "enable_thinking",
    "minimax-m3": "thinking",
    "mimo-v2.5*": "thinking",
    "glm-5.2": "reasoning_effort_none",
}


def _clean_text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _slug_env_part(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()
    return slug or "CUSTOM"


def _default_api_key_env(provider_id: str) -> str:
    return f"MODEL_PROVIDER_{_slug_env_part(provider_id)}_API_KEY"


def _looks_like_gemini_provider(provider_id: str, provider: dict) -> bool:
    base_url = _clean_text(provider.get("base_url")).lower()
    display_name = _clean_text(provider.get("name")).lower()
    provider_key = _clean_text(provider_id).lower()
    return (
        "generativelanguage.googleapis.com" in base_url
        or display_name == "gemini"
        or provider_key == "gemini"
    )


def _looks_like_deepseek_provider(provider: dict) -> bool:
    base_url = _clean_text(provider.get("base_url")).lower()
    if not base_url:
        return False
    parsed = urlparse(base_url if "://" in base_url else f"//{base_url}")
    return (parsed.hostname or "").lower() == "api.deepseek.com"


def _looks_like_opencode_console_go_provider(provider: dict) -> bool:
    """Return whether this is OpenCode's strict Console Go compatibility API."""
    base_url = _clean_text(provider.get("base_url")).lower()
    if not base_url:
        return False
    parsed = urlparse(base_url if "://" in base_url else f"//{base_url}")
    path = (parsed.path or "").rstrip("/")
    return (
        (parsed.hostname or "").lower() == "opencode.ai"
        and (path == "/zen/go" or path.startswith("/zen/go/"))
    )


def _normalize_thinking_control(value) -> str | None:
    normalized = _clean_text(value).lower()
    return normalized if normalized in _VALID_THINKING_CONTROLS else None


def _normalize_model_thinking_controls(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    controls: dict[str, str] = {}
    for raw_pattern, raw_control in value.items():
        pattern = _clean_text(raw_pattern).lower()
        control = _normalize_thinking_control(raw_control)
        if pattern and control is not None:
            controls[pattern] = control
    return controls


def resolve_model_thinking_control(provider: dict, model: str) -> str:
    """Resolve one model's wire-level thinking protocol for a provider."""
    fallback = _normalize_thinking_control(provider.get("thinking_control")) or "none"
    controls = _normalize_model_thinking_controls(
        provider.get("model_thinking_controls")
    )
    normalized_model = _clean_text(model).lower()
    if not normalized_model:
        return fallback

    exact = controls.get(normalized_model)
    if exact is not None:
        return exact
    for pattern, control in controls.items():
        if any(token in pattern for token in "*?[") and fnmatchcase(
            normalized_model, pattern
        ):
            return control
    return fallback


def get_selected_provider_name(cfg: dict) -> str:
    return _clean_text(cfg.get("provider"))


def _dedupe_provider_name(name: str, used_names: set[str]) -> str:
    if not name:
        return name
    if name not in used_names:
        used_names.add(name)
        return name

    suffix = 1
    while True:
        candidate = f"{name}({suffix})"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        suffix += 1


def _normalize_provider_entry(name: str, raw: dict) -> dict:
    merged = dict(raw)

    if "name" in merged:
        merged["name"] = _clean_text(merged.get("name"))
    else:
        merged["name"] = name
    _base_url = _clean_text(merged.get("base_url"))
    # 兼容用户误填完整端点 URL（如 .../v1/chat/completions）
    for _suffix in ("/chat/completions", "/completions"):
        if _base_url.endswith(_suffix):
            _base_url = _base_url[: -len(_suffix)]
            break
    merged["base_url"] = _base_url.rstrip("/")
    if "api_key_env" in merged:
        merged["api_key_env"] = _clean_text(merged.get("api_key_env"))
    else:
        merged["api_key_env"] = _default_api_key_env(name)
    merged["requires_api_key"] = bool(merged.get("requires_api_key", True))
    merged["supports_response_format"] = bool(merged.get("supports_response_format", True))
    looks_like_gemini = _looks_like_gemini_provider(name, merged)
    looks_like_deepseek = _looks_like_deepseek_provider(merged)
    looks_like_opencode_console_go = _looks_like_opencode_console_go_provider(merged)
    if looks_like_gemini:
        thinking_control = "reasoning_effort"
    elif (
        looks_like_deepseek
        and _normalize_thinking_control(merged.get("thinking_control")) != "none"
    ):
        # DeepSeek V4 uses ``thinking.type``. Override the legacy
        # ``enable_thinking`` value that older settings saves may contain.
        thinking_control = "thinking"
    elif looks_like_opencode_console_go:
        # Console Go rejects this non-standard field for its routed models.
        # Enforce the endpoint capability even if an older UI save says otherwise.
        thinking_control = "none"
    else:
        thinking_control = _normalize_thinking_control(merged.get("thinking_control"))
        if thinking_control is None:
            thinking_control = (
                "enable_thinking"
                if bool(merged.get("supports_enable_thinking", True))
                else "none"
            )
    merged["thinking_control"] = thinking_control
    merged["supports_enable_thinking"] = thinking_control == "enable_thinking"
    model_thinking_controls = (
        dict(_OPENCODE_GO_MODEL_THINKING_CONTROLS)
        if looks_like_opencode_console_go
        else {}
    )
    model_thinking_controls.update(
        _normalize_model_thinking_controls(merged.get("model_thinking_controls"))
    )
    if model_thinking_controls:
        merged["model_thinking_controls"] = model_thinking_controls
    else:
        merged.pop("model_thinking_controls", None)
    merged["supports_assistant_prefill"] = (
        False
        if looks_like_gemini
        else bool(merged.get("supports_assistant_prefill", True))
    )
    return merged


def _normalize_model_binding(section: dict) -> None:
    provider = _clean_text(section.get("provider"))
    model = _clean_text(section.get("model"))
    if provider:
        section["provider"] = provider
    else:
        section.pop("provider", None)
    if model:
        section["model"] = model
    else:
        section.pop("model", None)
    section.pop("profile", None)
    section.pop("base_url", None)
    section.pop("api_key_env", None)
    section.pop("model_name", None)
    section.pop("prompt_file", None)


def normalize_profile_config_inplace(cfg: dict) -> dict:
    provider = _clean_text(cfg.get("provider"))
    model = _clean_text(cfg.get("model"))
    if provider:
        cfg["provider"] = provider
    else:
        cfg.pop("provider", None)
    if model:
        cfg["model"] = model
    else:
        cfg.pop("model", None)
    cfg.pop("profile", None)
    cfg.pop("base_url", None)
    cfg.pop("api_key_env", None)
    cfg.pop("profiles", None)
    cfg.pop("openai_profiles", None)

    main_model = model
    if _clean_text(cfg.get("model_name")) == "":
        cfg["model_name"] = main_model

    for section_name in ("is", "slow_thinking", "cognition_compression"):
        section = cfg.get(section_name)
        if isinstance(section, dict):
            _normalize_model_binding(section)

    memory = cfg.get("memory")
    if isinstance(memory, dict):
        auto_archive = memory.get("auto_archive")
        if isinstance(auto_archive, dict):
            _normalize_model_binding(auto_archive)

    vision_bridge = cfg.get("vision_bridge")
    if isinstance(vision_bridge, dict):
        _normalize_model_binding(vision_bridge)

    return cfg


def sanitize_model_providers(
    raw_providers: dict | None,
    *,
    dedupe_display_names: bool = False,
) -> dict[str, dict]:
    if not isinstance(raw_providers, dict):
        return {}

    merged: dict[str, dict] = {}
    used_names: set[str] = set()
    for name, raw in raw_providers.items():
        provider_id = _clean_text(name)
        if not provider_id:
            continue

        provider = _normalize_provider_entry(provider_id, raw if isinstance(raw, dict) else {})
        if dedupe_display_names and provider["name"]:
            provider["name"] = _dedupe_provider_name(provider["name"], used_names)
        elif provider["name"]:
            used_names.add(provider["name"])
        merged[provider_id] = provider
    return merged


def get_model_providers(cfg: dict | None = None) -> dict[str, dict]:
    return sanitize_model_providers((cfg or {}).get("model_providers"))


def resolve_model_provider(cfg: dict) -> tuple[str, dict, dict[str, dict]]:
    providers = get_model_providers(cfg)
    provider_name = get_selected_provider_name(cfg)
    provider = providers.get(provider_name)
    if provider is None:
        raise ValueError(
            f"未知的模型供应商: {provider_name!r}，"
            f"可选值: {' / '.join(sorted(providers))}"
        )
    return provider_name, dict(provider), providers

def get_configured_api_key_names(cfg: dict | None = None) -> tuple[str, ...]:
    names: set[str] = {
        provider.get("api_key_env", "")
        for provider in get_model_providers(cfg).values()
        if isinstance(provider, dict)
    }
    return tuple(sorted(name for name in names if name))
