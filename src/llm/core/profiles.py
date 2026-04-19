"""profiles.py — OpenAI 兼容 profile 定义与解析。"""

from copy import deepcopy

DEFAULT_PROFILE_NAME = "siliconflow"

DEFAULT_OPENAI_PROFILES: dict[str, dict] = {
    "dashscope": {
        "label": "阿里云百炼 (DashScope)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "default_model": "qwen3.5-flash",
        "requires_api_key": True,
        "supports_response_format": True,
    },
    "siliconflow": {
        "label": "硅基流动 (SiliconFlow)",
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key_env": "SILICONFLOW_API_KEY",
        "default_model": "Pro/zai-org/GLM-5",
        "requires_api_key": True,
        "supports_response_format": True,
    },
    "bigmodel": {
        "label": "智谱 (BigModel)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "BIGMODEL_API_KEY",
        "default_model": "glm-5",
        "requires_api_key": True,
        "supports_response_format": True,
    },
    "lmstudio": {
        "label": "LM Studio（本地）",
        "base_url": "http://localhost:1234/v1",
        "api_key_env": "",
        "default_model": "local-model",
        "requires_api_key": False,
        "supports_response_format": False,
    },
}


def _clean_text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def get_selected_profile_name(cfg: dict, default: str = DEFAULT_PROFILE_NAME) -> str:
    return _clean_text(cfg.get("profile") or cfg.get("provider")) or default


def normalize_profile_config_inplace(cfg: dict) -> dict:
    profile = _clean_text(cfg.get("profile"))
    legacy_provider = _clean_text(cfg.get("provider"))
    if not profile and legacy_provider:
        cfg["profile"] = legacy_provider
    cfg.pop("provider", None)

    legacy_profiles = cfg.pop("profiles", None)
    if "openai_profiles" not in cfg and isinstance(legacy_profiles, dict):
        cfg["openai_profiles"] = legacy_profiles

    for section_name in ("is",):
        section = cfg.get(section_name)
        if not isinstance(section, dict):
            continue
        section_profile = _clean_text(section.get("profile"))
        section_provider = _clean_text(section.get("provider"))
        if not section_profile and section_provider:
            section["profile"] = section_provider
        section.pop("provider", None)

    return cfg


def _normalize_profile_entry(name: str, raw: dict, base: dict | None = None) -> dict:
    merged = dict(base or {})
    merged.update(raw)

    merged["label"] = _clean_text(merged.get("label")) or merged.get("label") or name
    merged["base_url"] = _clean_text(merged.get("base_url"))
    merged["api_key_env"] = _clean_text(merged.get("api_key_env"))
    merged["default_model"] = _clean_text(merged.get("default_model"))
    merged["requires_api_key"] = bool(merged.get("requires_api_key", True))
    merged["supports_response_format"] = bool(merged.get("supports_response_format", True))
    return merged


def get_openai_profiles(cfg: dict | None = None) -> dict[str, dict]:
    merged = {
        name: _normalize_profile_entry(name, deepcopy(profile))
        for name, profile in DEFAULT_OPENAI_PROFILES.items()
    }

    raw_profiles = (cfg or {}).get("openai_profiles")
    if isinstance(raw_profiles, dict):
        for name, raw in raw_profiles.items():
            if not isinstance(raw, dict):
                continue
            profile_name = _clean_text(name)
            if not profile_name:
                continue
            merged[profile_name] = _normalize_profile_entry(
                profile_name,
                raw,
                merged.get(profile_name),
            )
    return merged


def resolve_openai_profile(cfg: dict) -> tuple[str, dict, dict[str, dict]]:
    profiles = get_openai_profiles(cfg)
    profile_name = get_selected_profile_name(cfg)
    profile = profiles.get(profile_name)
    if profile is None:
        raise ValueError(
            f"未知的 profile: {profile_name!r}，"
            f"可选值: {' / '.join(sorted(profiles))}"
        )

    resolved = dict(profile)
    base_url = _clean_text(cfg.get("base_url"))
    if base_url:
        resolved["base_url"] = base_url

    api_key_env = _clean_text(cfg.get("api_key_env"))
    if api_key_env:
        resolved["api_key_env"] = api_key_env

    return profile_name, resolved, profiles


def get_configured_api_key_names(cfg: dict | None = None) -> tuple[str, ...]:
    names: set[str] = {
        profile.get("api_key_env", "")
        for profile in get_openai_profiles(cfg).values()
        if isinstance(profile, dict)
    }

    vision_bridge = (cfg or {}).get("vision_bridge")
    if isinstance(vision_bridge, dict):
        names.add(_clean_text(vision_bridge.get("api_key_env")) or "VISION_BRIDGE_API_KEY")
    else:
        names.add("VISION_BRIDGE_API_KEY")

    return tuple(sorted(name for name in names if name))