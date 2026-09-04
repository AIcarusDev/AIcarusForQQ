from __future__ import annotations

import yaml

from browser.config import (
    browser_image_send_confirmation,
    browser_image_source_url_mode,
    browser_multimodal_image_limit,
    browser_profile_dir,
    browser_screenshot_annotations_enabled,
    normalize_browser_control_config,
)
from config_loader import (
    normalize_guardian_config_inplace,
    normalize_guardian_info,
    save_config,
)
from llm.compression.config import normalize_generation_config, normalize_world_multimodal_image_limit
from llm.core.profiles import (
    get_configured_api_key_names,
    resolve_model_thinking_control,
    sanitize_model_providers,
)
from llm.core.transport import (
    OpenAICompatClient,
    add_enabled_sampling_kwargs,
    add_extra_generation_kwargs,
    normalize_generation_for_provider,
)
from platforms.qq.adapter.access_control import is_session_allowed_by_config, whitelist_rejection_reason
from platforms.qq.adapter.config import normalize_qq_platform_config, runtime_adapter_config


def test_guardian_info_normalizes_nullable_text_and_migrates_legacy_mapping():
    assert normalize_guardian_info(None) is None
    assert normalize_guardian_info("  \n  ") is None
    assert normalize_guardian_info("  第一行\n第二行  ") == "第一行\n第二行"
    normalized = normalize_guardian_info({"name": "智慧米塔", "id": "123456"})
    assert normalized is not None
    assert "智慧米塔" in normalized
    assert "123456" in normalized
    assert normalize_guardian_info({"name": "", "id": ""}) is None

    config = {"guardian": {"name": "智慧米塔", "id": ""}}
    normalize_guardian_config_inplace(config)
    assert isinstance(config["guardian"], str)
    assert "智慧米塔" in config["guardian"]


def test_general_config_save_preserves_latest_guardian(tmp_path):
    path = tmp_path / "config.yaml"
    save_config(
        {"guardian": "磁盘中的最新介绍", "model": "old"},
        str(path),
        preserve_latest_workspace=False,
        preserve_latest_guardian=False,
    )

    stale_page_snapshot = {"guardian": None, "model": "new"}
    save_config(
        stale_page_snapshot,
        str(path),
        preserve_latest_workspace=False,
    )

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded == {"guardian": "磁盘中的最新介绍", "model": "new"}
    assert stale_page_snapshot["guardian"] == "磁盘中的最新介绍"


def test_browser_control_config_normalizes_public_settings():
    cfg = normalize_browser_control_config(
        {
            "profile_dir": "  cache/browser_profile/test  ",
            "multimodal_image_limit": "-1",
            "annotate_screenshots": "yes",
            "image_source_url": "sanitized",
            "image_send_confirmation": "HIGH-RISK",
        }
    )

    assert cfg == {
        "profile_dir": "cache/browser_profile/test",
        "multimodal_image_limit": -1,
        "annotate_screenshots": True,
        "image_source_url": "sanitized",
        "image_send_confirmation": "high_risk",
    }
    assert browser_profile_dir({"browser_control": cfg}) == "cache/browser_profile/test"
    assert browser_multimodal_image_limit({"vision": False, "browser_control": cfg}) == 0
    assert browser_screenshot_annotations_enabled({"browser_control": cfg}) is True
    assert browser_image_source_url_mode({"browser_control": cfg}) == "sanitized"
    assert browser_image_send_confirmation({"browser_control": cfg}) == "high_risk"


def test_browser_image_confirmation_defaults_off() -> None:
    cfg = normalize_browser_control_config({})
    assert cfg["image_send_confirmation"] == "off"
    assert cfg["image_source_url"] == "full"


def test_generation_config_bounds_context_and_image_limits():
    assert normalize_world_multimodal_image_limit("-1") == -1
    assert normalize_world_multimodal_image_limit("-2") == 5

    cfg = normalize_generation_config(
        {
            "json_self_repair_retries": 1,
            "llm_contents_max_rounds": 4,
            "cognition_compression_trigger_rounds": 99,
            "world_multimodal_image_limit": 0,
        }
    )

    assert cfg["llm_contents_max_rounds"] == 6
    assert cfg["cognition_compression_trigger_rounds"] == 5
    assert cfg["world_multimodal_image_limit"] == 0
    assert cfg["native_reasoning_as_cognition"] is False
    assert "json_self_repair_retries" not in cfg
    assert normalize_generation_config(
        {"native_reasoning_as_cognition": 1}
    )["native_reasoning_as_cognition"] is True


def test_qq_platform_config_and_access_control_are_normalized_in_place():
    app_cfg = {
        "qq_adapter": {
            "adapter": "LLoneBot",
            "port": "70000",
            "whitelist": {"enabled": True, "private_users": ["u_alice", ""], "group_ids": ["g_dev"]},
        },
        "platforms": {
            "qq": {
                "adapter": {
                    "file_transfer": {
                        "host_directory": r"C:\AICQ\transfer",
                        "adapter_directory": "/app/napcat/transfer",
                    }
                }
            }
        },
    }

    normalized = normalize_qq_platform_config(app_cfg, remove_legacy=True)
    runtime_cfg = runtime_adapter_config(normalized)

    assert app_cfg["platforms"]["qq"] is normalized
    assert "qq_adapter" not in app_cfg
    assert normalized["adapter"]["type"] == "llonebot"
    assert normalized["adapter"]["name"] == "LLoneBot"
    assert normalized["adapter"]["reverse_ws"]["port"] == 65535
    assert runtime_cfg["file_transfer"] == {
        "host_directory": r"C:\AICQ\transfer",
        "adapter_directory": "/app/napcat/transfer",
    }
    assert normalized["attention"]["respond_to_self_name"] is True
    assert is_session_allowed_by_config(runtime_cfg, "private", "u_alice") is True
    assert is_session_allowed_by_config(runtime_cfg, "group", "g_other") is False
    assert whitelist_rejection_reason(runtime_cfg, "group", "g_other")


def test_qq_platform_self_name_response_can_be_disabled():
    app_cfg = {"qq_adapter": {"respond_to_self_name": False}}

    normalized = normalize_qq_platform_config(app_cfg, remove_legacy=True)

    assert normalized["attention"]["respond_to_self_name"] is False


def test_provider_normalization_derives_env_names_and_gemini_thinking_control():
    providers = sanitize_model_providers(
        {
            "local test": {"base_url": "http://localhost/v1/chat/completions", "requires_api_key": False},
            "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai"},
            "deepseek": {
                "base_url": "https://api.deepseek.com",
                "thinking_control": "enable_thinking",
            },
        }
    )

    assert providers["local test"]["base_url"] == "http://localhost/v1"
    assert providers["local test"]["api_key_env"] == "MODEL_PROVIDER_LOCAL_TEST_API_KEY"
    assert providers["local test"]["requires_api_key"] is False
    assert providers["local test"]["supports_assistant_prefill"] is True
    assert providers["gemini"]["thinking_control"] == "reasoning_effort"
    assert providers["gemini"]["supports_assistant_prefill"] is False
    assert providers["deepseek"]["thinking_control"] == "thinking"
    assert providers["deepseek"]["supports_enable_thinking"] is False

    key_names = get_configured_api_key_names({"model_providers": providers})
    assert key_names == (
        "MODEL_PROVIDER_DEEPSEEK_API_KEY",
        "MODEL_PROVIDER_GEMINI_API_KEY",
        "MODEL_PROVIDER_LOCAL_TEST_API_KEY",
    )


def test_provider_can_disable_assistant_prefill_explicitly():
    providers = sanitize_model_providers(
        {
            "local": {
                "base_url": "http://localhost/v1",
                "requires_api_key": False,
                "supports_assistant_prefill": False,
            },
        }
    )

    assert providers["local"]["supports_assistant_prefill"] is False


def test_deepseek_thinking_detection_is_endpoint_scoped_and_honors_none():
    providers = sanitize_model_providers(
        {
            "deepseek_disabled": {
                "base_url": "https://api.deepseek.com/v1",
                "thinking_control": "none",
            },
            "hosted_deepseek": {
                "name": "DeepSeek via SiliconFlow",
                "base_url": "https://api.siliconflow.cn/v1",
            },
        }
    )

    assert providers["deepseek_disabled"]["thinking_control"] == "none"
    assert providers["hosted_deepseek"]["thinking_control"] == "enable_thinking"


def test_opencode_console_go_resolves_thinking_control_per_model():
    providers = sanitize_model_providers(
        {
            "console_go": {
                "base_url": "https://opencode.ai/zen/go/v1",
                "thinking_control": "enable_thinking",
                "supports_enable_thinking": True,
            },
            "other_opencode": {
                "base_url": "https://opencode.ai/other/v1",
                "thinking_control": "enable_thinking",
            },
        }
    )

    assert providers["console_go"]["thinking_control"] == "none"
    assert providers["console_go"]["supports_enable_thinking"] is False
    assert resolve_model_thinking_control(
        providers["console_go"], "kimi-k2.7-code"
    ) == "none"
    assert resolve_model_thinking_control(
        providers["console_go"], "kimi-k2.6"
    ) == "thinking"
    assert resolve_model_thinking_control(
        providers["console_go"], "kimi-k3"
    ) == "reasoning_effort_none"
    assert resolve_model_thinking_control(
        providers["console_go"], "deepseek-v4-pro"
    ) == "thinking"
    assert resolve_model_thinking_control(
        providers["console_go"], "qwen3.8-flash"
    ) == "enable_thinking"
    assert resolve_model_thinking_control(
        providers["console_go"], "minimax-m3"
    ) == "thinking"
    assert resolve_model_thinking_control(
        providers["console_go"], "minimax-m2.7"
    ) == "none"
    assert resolve_model_thinking_control(
        providers["console_go"], "mimo-v2.5-pro"
    ) == "thinking"
    assert resolve_model_thinking_control(
        providers["console_go"], "glm-5.2"
    ) == "reasoning_effort_none"
    assert resolve_model_thinking_control(
        providers["console_go"], "glm-5.3-flash"
    ) == "none"
    assert providers["other_opencode"]["thinking_control"] == "enable_thinking"


def test_provider_model_thinking_control_override_wins_over_builtin_pattern():
    provider = sanitize_model_providers(
        {
            "console_go": {
                "base_url": "https://opencode.ai/zen/go/v1",
                "model_thinking_controls": {
                    "qwen3.8-flash": "none",
                },
            }
        }
    )["console_go"]

    assert resolve_model_thinking_control(provider, "qwen3.8-flash") == "none"
    assert resolve_model_thinking_control(provider, "qwen3.7-plus") == "enable_thinking"


def test_opencode_client_emits_each_models_supported_thinking_protocol():
    provider = sanitize_model_providers(
        {"console_go": {"base_url": "https://opencode.ai/zen/go/v1"}}
    )["console_go"]
    client = object.__new__(OpenAICompatClient)
    client._thinking_control = provider["thinking_control"]
    client._model_thinking_controls = provider["model_thinking_controls"]

    expected_by_model = {
        "kimi-k2.7-code": {},
        "glm-5.3-flash": {},
        "qwen3.8-flash": {"extra_body": {"enable_thinking": False}},
        "deepseek-v4-flash": {
            "extra_body": {"thinking": {"type": "disabled"}}
        },
        "kimi-k3": {"reasoning_effort": "none"},
    }
    for model, expected in expected_by_model.items():
        client.model = model
        generation = client.normalize_generation({"enable_thinking": False})
        request_kwargs = {}
        add_extra_generation_kwargs(request_kwargs, generation)
        if extra_body := generation.get("extra_body"):
            request_kwargs["extra_body"] = extra_body
        assert request_kwargs == expected


def test_memory_processing_adapter_uses_explicit_model_binding():
    from llm.core.provider import build_memory_processing_adapter_cfg

    cfg = build_memory_processing_adapter_cfg(
        {
            "provider": "main",
            "model": "main-model",
            "model_name": "display name",
            "base_url": "https://legacy.example/v1",
            "api_key_env": "LEGACY_KEY",
            "model_providers": {
                "memory": {"base_url": "https://memory.example/v1", "api_key_env": "MEMORY_KEY"},
            },
            "generation": {"temperature": 1.0, "max_output_tokens": 10000},
        },
        {
            "provider": "memory",
            "model": "memory-model",
            "generation": {"temperature": 0.2, "max_output_tokens": 4000, "enable_thinking": False},
        },
    )

    assert cfg["provider"] == "memory"
    assert cfg["model"] == "memory-model"
    assert cfg["generation"] == {"temperature": 0.2, "max_output_tokens": 4000, "enable_thinking": False}
    assert "model_name" not in cfg
    assert "base_url" not in cfg
    assert "api_key_env" not in cfg


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

    gen = normalize_generation_for_provider(
        {
            "enable_thinking": False,
            "extra_body": {"enable_thinking": True},
        },
        thinking_control="thinking",
        model="deepseek-v4-flash",
    )

    assert gen["extra_body"] == {"thinking": {"type": "disabled"}}

    gen = normalize_generation_for_provider(
        {"extra_body": {"thinking": {"type": "enabled"}}},
        thinking_control="thinking",
        model="deepseek-v4-pro",
    )

    assert gen["extra_body"] == {"thinking": {"type": "enabled"}}

    gen = normalize_generation_for_provider(
        {
            "enable_thinking": False,
            "extra_body": {"thinking": {"type": "enabled"}},
        },
        thinking_control="enable_thinking",
        model="legacy-compatible-model",
    )

    assert gen["extra_body"] == {"enable_thinking": False}

    gen = normalize_generation_for_provider(
        {
            "enable_thinking": False,
            "extra_body": {"thinking": {"type": "enabled"}},
        },
        thinking_control="reasoning_effort_none",
        model="kimi-k3",
    )

    assert gen["reasoning_effort"] == "none"
    assert "extra_body" not in gen

    gen = normalize_generation_for_provider(
        {
            "enable_thinking": False,
            "reasoning_effort": "none",
        },
        thinking_control="none",
        model="kimi-k2.7-code",
    )

    assert "reasoning_effort" not in gen
    assert "extra_body" not in gen


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

