"""config_loader.py — 配置加载与运行时覆盖

负责：
    1. 从 config.yaml 读取配置
    2. 从多份 Markdown 文档读取 prompt 相关文本
    3. 加载 .model_override.json（运行时模型覆盖）
    4. 提供 save_model_override() 持久化切换
"""

import json
import logging
import os
import re

import yaml

from llm.core.profiles import (
    get_configured_api_key_names,
    normalize_profile_config_inplace,
)

logger = logging.getLogger("AICQ.config")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_DIR = os.path.join(_BASE_DIR, "config")
_DATA_DIR = os.path.join(_BASE_DIR, "data")

_RUNTIME_OVERRIDE_FILE = os.path.join(_BASE_DIR, ".model_override.json")
_USER_CONFIG_PATH = os.path.join(_BASE_DIR, "config_user.yaml")  # 用户副本
_DEFAULT_CONFIG_PATH = os.path.join(_CONFIG_DIR, "config.yaml")

_PROMPT_DOC_DEFAULTS: dict[str, tuple[str, str]] = {
    "persona": (
        os.path.join("data", "persona.md"),
        "你是一个乐于助人的 AI 助手。",
    ),
    "style": (
        os.path.join("data", "style.md"),
        "在这里填写 bot 的语气、措辞、句长、表达偏好等风格约束。",
    ),
    "social_tips_private": (
        os.path.join("data", "social_tips", "private.md"),
        "在这里填写私聊场景下的社交提醒。",
    ),
    "social_tips_group": (
        os.path.join("data", "social_tips", "group.md"),
        "在这里填写群聊场景下的社交提醒。",
    ),
}


def _resolve_project_path(path: str) -> str:
    """将配置中的路径解析为项目内绝对路径。"""
    if os.path.isabs(path):
        return path
    return os.path.join(_BASE_DIR, path)


def _read_or_create_text_file(path: str, default_text: str, label: str) -> str:
    """读取文本文件；不存在时创建默认文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(default_text)
        logger.warning("%s file not found, created default at %s", label, path)

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_prompt_docs(
    config: dict,
    persona_path: str | None = None,
) -> dict[str, str]:
    """读取 prompt 相关 Markdown 文档，返回文案字典。"""
    prompt_files = config.get("prompt_files", {})
    docs: dict[str, str] = {}

    for key, (default_rel_path, default_text) in _PROMPT_DOC_DEFAULTS.items():
        configured_path = persona_path if key == "persona" and persona_path is not None else prompt_files.get(key, default_rel_path)
        abs_path = _resolve_project_path(configured_path)
        docs[key] = _read_or_create_text_file(abs_path, default_text, key)

    return docs


def load_config(
    config_path: str | None = None,
    persona_path: str | None = None,
) -> tuple[dict, dict[str, str]]:
    """加载配置文件和 prompt 文档。

    优先加载用户副本 config_user.yaml，否则使用母版 config/config.yaml。
    Returns: (config_dict, prompt_docs)
    """
    if config_path is None:
        if os.path.exists(_USER_CONFIG_PATH):
            actual_config_path = _USER_CONFIG_PATH
        else:
            actual_config_path = _DEFAULT_CONFIG_PATH
    else:
        actual_config_path = config_path

    with open(actual_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    normalize_profile_config_inplace(config)

    prompt_docs = load_prompt_docs(config, persona_path=persona_path)

    # 运行时覆盖
    try:
        with open(_RUNTIME_OVERRIDE_FILE, "r", encoding="utf-8") as f:
            ov = json.load(f)
        profile = ov.get("profile") or ov.get("provider")
        if profile:
            config["profile"] = profile
        config["model"] = ov["model"]
        config["model_name"] = ov.get("model_name", ov["model"])
        if ov.get("base_url"):
            config["base_url"] = ov["base_url"]
        elif "base_url" in config:
            del config["base_url"]
        logger.info(
            "已应用运行时覆盖: profile=%s model=%s",
            config.get("profile", ""),
            config["model"],
        )
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("运行时覆盖文件无效，已忽略: %s", e)

    return config, prompt_docs


def save_model_override(
    profile: str,
    model: str,
    model_name: str,
    base_url: str | None = None,
) -> None:
    """持久化模型切换到 .model_override.json。"""
    try:
        with open(_RUNTIME_OVERRIDE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "profile": profile,
                    "model": model,
                    "model_name": model_name,
                    "base_url": base_url or None,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        logger.warning("写入运行时覆盖文件失败: %s", e)


def save_config(config_dict: dict, config_path: str = _USER_CONFIG_PATH) -> None:
    """将配置字典写入用户副本 config_user.yaml（不覆盖母版 config.yaml）。"""
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def save_persona(text: str, persona_path: str | None = None) -> None:
    """将 persona 文本写回 persona.md。"""
    if persona_path is None:
        persona_path = os.path.join(_DATA_DIR, "persona.md")
    with open(persona_path, "w", encoding="utf-8") as f:
        f.write(text)


_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def read_env_keys(
    key_names: "list[str] | tuple[str, ...] | set[str] | None" = None,
    env_path: str = ".env",
) -> dict[str, str]:
    """读取 .env 中的 API Key，返回掩码版本（后4位可见）。"""
    names = tuple(key_names or get_configured_api_key_names())
    result = {k: "" for k in names}
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip()
                    if key in result:
                        result[key] = _mask_key(val)
    except FileNotFoundError:
        pass
    return result


def save_env_key(key_name: str, value: str, env_path: str = ".env") -> None:
    """更新 .env 中某个 Key 的值。若 value 全为 * 则跳过（掩码占位，不实际写入）。"""
    if not _ENV_NAME_RE.fullmatch(key_name):
        raise ValueError(f"不支持的 key: {key_name}")
    if set(value) <= {"*"}:
        return  # 用户没有修改，跳过

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    found = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key_name}=") or stripped == key_name:
            new_lines.append(f"{key_name}={value}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"{key_name}={value}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def _mask_key(val: str) -> str:
    """将 API Key 掩码，仅保留后4位可见。"""
    if not val:
        return ""
    if len(val) <= 4:
        return "*" * len(val)
    return "*" * (len(val) - 4) + val[-4:]


_ENV_PROXY_NAMES = ("OPENAI_PROXY", "TAVILY_PROXY")


def read_env_proxies(env_path: str = ".env") -> dict[str, str]:
    """读取 .env 中的代理配置，返回掩码版本（为了安全性）。"""
    result = {"OPENAI_PROXY": "", "TAVILY_PROXY": ""}
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip()
                    if key in _ENV_PROXY_NAMES:
                        result[key] = _mask_key(val) if val else ""
    except FileNotFoundError:
        pass
    return result


def save_env_proxy(proxy_name: str, value: str, env_path: str = ".env") -> None:
    """更新 .env 中某个代理的值。若 value 全为 * 则跳过（掩码占位，不实际写入）。"""
    if proxy_name not in _ENV_PROXY_NAMES:
        raise ValueError(f"不支持的代理: {proxy_name}")
    if value and set(value) <= {"*"}:
        return  # 用户没有修改，跳过

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    found = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{proxy_name}=") or stripped == proxy_name:
            if value:
                new_lines.append(f"{proxy_name}={value}\n")
            # 如果 value 为空则删除此行（不添加）
            found = True
        else:
            new_lines.append(line)

    if not found and value:
        new_lines.append(f"{proxy_name}={value}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
