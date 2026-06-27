from __future__ import annotations

from browser.config import (
    browser_multimodal_image_limit,
    browser_profile_dir,
    browser_screenshot_annotations_enabled,
    normalize_browser_control_config,
)
from llm.compression.config import normalize_generation_config, normalize_world_multimodal_image_limit
from llm.core.profiles import get_configured_api_key_names, sanitize_model_providers
from llm.core.transport import add_enabled_sampling_kwargs, normalize_generation_for_provider
from qq_adapter.access_control import is_session_allowed_by_config, whitelist_rejection_reason
from qq_adapter.config import normalize_qq_adapter_config


def test_browser_control_config_normalizes_public_settings():
    cfg = normalize_browser_control_config(
        {"profile_dir": "  cache/browser_profile/test  ", "multimodal_image_limit": "-1", "annotate_screenshots": "yes"}
    )

    assert cfg == {
        "profile_dir": "cache/browser_profile/test",
        "multimodal_image_limit": -1,
        "annotate_screenshots": True,
    }
    assert browser_profile_dir({"browser_control": cfg}) == "cache/browser_profile/test"
    assert browser_multimodal_image_limit({"vision": False, "browser_control": cfg}) == 0
    assert browser_screenshot_annotations_enabled({"browser_control": cfg}) is True


def test_generation_config_bounds_context_and_image_limits():
    assert normalize_world_multimodal_image_limit("-1") == -1
    assert normalize_world_multimodal_image_limit("-2") == 5

    cfg = normalize_generation_config(
        {
            "llm_contents_max_rounds": 4,
            "cognition_compression_trigger_rounds": 99,
            "world_multimodal_image_limit": 0,
        }
    )

    assert cfg["llm_contents_max_rounds"] == 6
    assert cfg["cognition_compression_trigger_rounds"] == 5
    assert cfg["world_multimodal_image_limit"] == 0


def test_qq_adapter_config_and_access_control_are_normalized_in_place():
    app_cfg = {
        "qq_adapter": {
            "adapter": "LLoneBot",
            "port": "70000",
            "whitelist": {"enabled": True, "private_users": ["u_alice", ""], "group_ids": ["g_dev"]},
        }
    }

    normalized = normalize_qq_adapter_config(app_cfg)

    assert app_cfg["qq_adapter"] is normalized
    assert normalized["adapter"] == "llonebot"
    assert normalized["name"] == "LLoneBot"
    assert normalized["port"] == 65535
    assert is_session_allowed_by_config(normalized, "private", "u_alice") is True
    assert is_session_allowed_by_config(normalized, "group", "g_other") is False
    assert whitelist_rejection_reason(normalized, "group", "g_other")


def test_provider_normalization_derives_env_names_and_gemini_thinking_control():
    providers = sanitize_model_providers(
        {
            "local test": {"base_url": "http://localhost/v1/chat/completions", "requires_api_key": False},
            "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai"},
        }
    )

    assert providers["local test"]["base_url"] == "http://localhost/v1"
    assert providers["local test"]["api_key_env"] == "MODEL_PROVIDER_LOCAL_TEST_API_KEY"
    assert providers["local test"]["requires_api_key"] is False
    assert providers["gemini"]["thinking_control"] == "reasoning_effort"

    key_names = get_configured_api_key_names({"model_providers": providers})
    assert key_names == ("MODEL_PROVIDER_GEMINI_API_KEY", "MODEL_PROVIDER_LOCAL_TEST_API_KEY")


def test_generation_transport_maps_thinking_flags_by_provider():
    gen = normalize_generation_for_provider(
        {"enable_thinking": False},
        thinking_control="reasoning_effort",
        model="gemini-2.5-flash",
    )

    assert gen["reasoning_effort"] == "none"
    assert "extra_body" not in gen

    gen = normalize_generation_for_provider(
        {},
        thinking_control="enable_thinking",
        model="plain-model",
    )

    assert gen["extra_body"] == {"enable_thinking": True}


def test_advanced_sampling_only_sends_enabled_parameters():
    create_kwargs = {"model": "test-model", "temperature": 0.7}
    add_enabled_sampling_kwargs(
        create_kwargs,
        {
            "advanced_sampling": {
                "top_p": {"enabled": True, "value": 0.8},
                "top_k": {"enabled": True, "value": 20},
                "min_p": {"enabled": True, "value": 0.0},
                "presence_penalty": {"enabled": True, "value": 1.5},
                "frequency_penalty": {"enabled": False, "value": 0.2},
                "repeat_penalty": {"enabled": True, "value": 1.0},
            }
        },
    )

    assert create_kwargs["top_p"] == 0.8
    assert create_kwargs["presence_penalty"] == 1.5
    assert "frequency_penalty" not in create_kwargs
    assert create_kwargs["extra_body"] == {
        "top_k": 20,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
    }

    create_kwargs = {"model": "test-model", "temperature": 0.7}
    add_enabled_sampling_kwargs(
        create_kwargs,
        {
            "advanced_sampling": {
                "top_p": {"enabled": False, "value": 0.8},
                "top_k": {"enabled": False, "value": 20},
                "presence_penalty": {"enabled": False, "value": 1.5},
            }
        },
    )

    assert create_kwargs == {"model": "test-model", "temperature": 0.7}
