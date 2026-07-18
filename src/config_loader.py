"""config_loader.py — 配置加载与运行时覆盖

负责：
    1. 从 config.yaml 读取配置
    2. 从 Markdown 文档读取 persona 文本
    3. 加载 .model_override.json（运行时模型覆盖）
    4. 提供 save_model_override() 持久化切换
"""

import json
import logging
import os
import re
import tempfile
import threading
from copy import deepcopy

import yaml

from llm.core.profiles import (
    get_configured_api_key_names,
    normalize_profile_config_inplace,
)
from llm.compression.config import normalize_generation_config
from platforms.qq.adapter.config import normalize_qq_platform_config
from workspace.config import normalize_workspace_config_inplace

logger = logging.getLogger("AICQ.config")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_DIR = os.path.join(_BASE_DIR, "config")
_DATA_DIR = os.path.join(_BASE_DIR, "data")

_RUNTIME_OVERRIDE_FILE = os.path.join(_BASE_DIR, ".model_override.json")
_USER_CONFIG_PATH = os.path.join(_CONFIG_DIR, "config_user.yaml")  # 用户副本
_TEMPLATE_CONFIG_PATH = os.path.join(_BASE_DIR, "templates", "config.yaml.template")
_CONFIG_WRITE_LOCK = threading.RLock()

_PROMPT_DOC_DEFAULTS: dict[str, tuple[str, str]] = {
    "persona": (
        os.path.join("config", "persona.md"),
        "你是一个乐于助人的 AI 助手。",
    ),
}


def normalize_guardian_info(value: object) -> str | None:
    """Normalize guardian config to the canonical nullable free-form text."""
    if isinstance(value, dict):
        name = str(value.get("name") or "").strip()
        guardian_id = str(value.get("id") or "").strip()
        lines: list[str] = []
        if name:
            lines.append(f"- QQ 名称：{name}")
        if guardian_id:
            lines.append(f"- QQ ID：{guardian_id}")
        return "\n".join(lines) or None
    if isinstance(value, str):
        return value.strip() or None
    return None


def normalize_guardian_config_inplace(config: dict) -> str | None:
    """Migrate the legacy guardian ``{name, id}`` mapping in memory."""
    original = config.get("guardian")
    normalized = normalize_guardian_info(original)
    config["guardian"] = normalized
    if isinstance(original, dict):
        logger.info("已将旧 guardian name/id 配置迁移为自由文本介绍")
    return normalized


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

    优先加载 config/config_user.yaml；不存在时从 templates/config.yaml.template
    自动生成一份，让首次启动也能进入 WebUI 完成配置。
    Returns: (config_dict, prompt_docs)
    """
    if config_path is None:
        if not os.path.exists(_USER_CONFIG_PATH):
            if os.path.exists(_TEMPLATE_CONFIG_PATH):
                try:
                    import shutil
                    os.makedirs(os.path.dirname(_USER_CONFIG_PATH), exist_ok=True)
                    shutil.copyfile(_TEMPLATE_CONFIG_PATH, _USER_CONFIG_PATH)
                    logger.warning(
                        "未检测到配置文件，已从模板自动生成 %s，请进入 WebUI 完成配置。",
                        _USER_CONFIG_PATH,
                    )
                except Exception as e:
                    logger.error("从模板生成 config_user.yaml 失败: %s", e)
            else:
                logger.error(
                    "未找到配置文件且模板缺失: %s",
                    _TEMPLATE_CONFIG_PATH,
                )
        actual_config_path = _USER_CONFIG_PATH
    else:
        actual_config_path = config_path

    with open(actual_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if "self_name" not in config and "bot_name" in config:
        config["self_name"] = config.get("bot_name", "")
    config.pop("bot_name", None)
    normalize_guardian_config_inplace(config)

    normalize_profile_config_inplace(config)
    config["generation"] = normalize_generation_config(config.get("generation"))
    normalize_qq_platform_config(config, remove_legacy=True)
    normalize_workspace_config_inplace(config, project_root=_BASE_DIR)

    prompt_docs = load_prompt_docs(config, persona_path=persona_path)

    # 运行时覆盖
    try:
        with open(_RUNTIME_OVERRIDE_FILE, "r", encoding="utf-8") as f:
            ov = json.load(f)
        provider = ov.get("provider")
        if provider:
            config["provider"] = provider
        config["model"] = ov["model"]
        config["model_name"] = ov.get("model_name", ov["model"])
        logger.info(
            "已应用运行时覆盖: provider=%s model=%s",
            config.get("provider", ""),
            config["model"],
        )
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("运行时覆盖文件无效，已忽略: %s", e)

    return config, prompt_docs


def save_model_override(
    provider: str,
    model: str,
    model_name: str,
) -> None:
    """持久化模型切换到 .model_override.json。"""
    try:
        with open(_RUNTIME_OVERRIDE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "provider": provider,
                    "model": model,
                    "model_name": model_name,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        logger.warning("写入运行时覆盖文件失败: %s", e)


def _atomic_write_config_unlocked(config_dict: dict, target: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".config-", suffix=".yaml.tmp", dir=os.path.dirname(target))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            yaml.dump(
                config_dict,
                handle,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _read_config_mapping_unlocked(target: str) -> dict:
    try:
        with open(target, "r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}
    return value if isinstance(value, dict) else {}


def save_config(
    config_dict: dict,
    config_path: str = _USER_CONFIG_PATH,
    *,
    preserve_latest_workspace: bool = True,
    preserve_latest_guardian: bool = True,
) -> None:
    """Atomically save config while preserving independently owned sections."""
    target = os.path.abspath(config_path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with _CONFIG_WRITE_LOCK:
        if preserve_latest_workspace or preserve_latest_guardian:
            latest = _read_config_mapping_unlocked(target)
        else:
            latest = {}
        if preserve_latest_workspace:
            if isinstance(latest.get("workspace"), dict):
                normalize_workspace_config_inplace(latest, project_root=_BASE_DIR)
                config_dict["workspace"] = deepcopy(latest["workspace"])
        if preserve_latest_guardian and "guardian" in latest:
            config_dict["guardian"] = normalize_guardian_info(latest.get("guardian"))
        _atomic_write_config_unlocked(config_dict, target)


def save_workspace_config(
    workspace_dict: dict,
    *,
    base_config: dict | None = None,
    config_path: str = _USER_CONFIG_PATH,
) -> dict:
    """Atomically replace only workspace config and return the merged file."""

    target = os.path.abspath(config_path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with _CONFIG_WRITE_LOCK:
        merged = _read_config_mapping_unlocked(target)
        if not merged:
            merged = deepcopy(base_config or {})
        merged["workspace"] = deepcopy(workspace_dict)
        _atomic_write_config_unlocked(merged, target)
        return merged


def save_persona(text: str, persona_path: str | None = None) -> None:
    """将 persona 文本写回 persona.md。"""
    if persona_path is None:
        persona_path = os.path.join(_CONFIG_DIR, "persona.md")
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


def read_env_values(
    key_names: "list[str] | tuple[str, ...] | set[str] | None" = None,
    env_path: str = ".env",
) -> dict[str, str]:
    """读取 .env 中指定键的原始值。"""
    names = tuple(key_names or ())
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
                        result[key] = val
    except FileNotFoundError:
        pass
    return result


def save_env_key(key_name: str, value: str, env_path: str = ".env") -> None:
    """更新 .env 中某个 Key 的值。若 value 包含 * 则跳过（掩码占位，不实际写入）。"""
    if not _ENV_NAME_RE.fullmatch(key_name):
        raise ValueError(f"不支持的 key: {key_name}")
    if "*" in value:
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


def save_env_value(key_name: str, value: str, env_path: str = ".env") -> None:
    """更新 .env 中某个普通文本值；空字符串表示删除该键。"""
    if not _ENV_NAME_RE.fullmatch(key_name):
        raise ValueError(f"不支持的 key: {key_name}")
    if value and "*" in value:
        return

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
            if value:
                new_lines.append(f"{key_name}={value}\n")
            found = True
        else:
            new_lines.append(line)

    if value and not found:
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
    """更新 .env 中某个代理的值。若 value 包含 * 则跳过（掩码占位，不实际写入）。"""
    if proxy_name not in _ENV_PROXY_NAMES:
        raise ValueError(f"不支持的代理: {proxy_name}")
    if value and "*" in value:
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


# ── SMTP 凭据（掉线告警）────────────────────────────────────────
# 仅 PASSWORD 字段视为机密、读取时掩码；其它字段直接回显。
_ENV_SMTP_NAMES = (
    "AICQ_SMTP_HOST",
    "AICQ_SMTP_PORT",
    "AICQ_SMTP_USE_SSL",
    "AICQ_SMTP_USER",
    "AICQ_SMTP_PASSWORD",
    "AICQ_SMTP_SENDER",
    "AICQ_SMTP_RECIPIENTS",
)
_ENV_SMTP_SECRET_NAMES = ("AICQ_SMTP_PASSWORD",)


def read_env_smtp(env_path: str = ".env") -> dict[str, str]:
    """读取 .env 中的 SMTP 配置。密码字段掩码，其它原样返回。"""
    result = {k: "" for k in _ENV_SMTP_NAMES}
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key in result:
                    result[key] = _mask_key(val) if key in _ENV_SMTP_SECRET_NAMES else val
    except FileNotFoundError:
        pass
    return result


def save_env_smtp(values: dict, env_path: str = ".env") -> None:
    """批量更新 .env 中的 SMTP 配置。

    机密字段全为 * 时视为"未修改"跳过；空字符串则删除该行。
    """
    cleaned: dict[str, str | None] = {}  # None 表示删除
    for name in _ENV_SMTP_NAMES:
        if name not in values:
            continue
        raw = values.get(name)
        if raw is None:
            continue
        sval = str(raw).strip()
        if name in _ENV_SMTP_SECRET_NAMES and sval and set(sval) <= {"*"}:
            # 用户没改密码，跳过
            continue
        cleaned[name] = sval if sval else None

    if not cleaned:
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    seen: set[str] = set()
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        matched: str | None = None
        for name in cleaned:
            if stripped.startswith(f"{name}=") or stripped == name:
                matched = name
                break
        if matched is not None:
            seen.add(matched)
            new_val = cleaned[matched]
            if new_val:
                new_lines.append(f"{matched}={new_val}\n")
            # 空字符串 → 删除（不写）
        else:
            new_lines.append(line)

    # 追加未出现过的新键
    for name, val in cleaned.items():
        if name not in seen and val:
            new_lines.append(f"{name}={val}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


# ── IMAP 凭据（远程邮件指令，Phase 3）──────────────────────────
_ENV_IMAP_NAMES = (
    "AICQ_IMAP_HOST",
    "AICQ_IMAP_PORT",
    "AICQ_IMAP_USE_SSL",
    "AICQ_IMAP_USER",
    "AICQ_IMAP_PASSWORD",
)
_ENV_IMAP_SECRET_NAMES = ("AICQ_IMAP_PASSWORD",)


def read_env_imap(env_path: str = ".env") -> dict[str, str]:
    """读取 .env 中的 IMAP 配置。密码字段掩码，其它原样返回。"""
    result = {k: "" for k in _ENV_IMAP_NAMES}
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key in result:
                    result[key] = _mask_key(val) if key in _ENV_IMAP_SECRET_NAMES else val
    except FileNotFoundError:
        pass
    return result


def save_env_imap(values: dict, env_path: str = ".env") -> None:
    """批量更新 .env 中的 IMAP 配置。语义同 save_env_smtp。"""
    cleaned: dict[str, str | None] = {}
    for name in _ENV_IMAP_NAMES:
        if name not in values:
            continue
        raw = values.get(name)
        if raw is None:
            continue
        sval = str(raw).strip()
        if name in _ENV_IMAP_SECRET_NAMES and sval and set(sval) <= {"*"}:
            continue
        cleaned[name] = sval if sval else None

    if not cleaned:
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    seen: set[str] = set()
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        matched: str | None = None
        for name in cleaned:
            if stripped.startswith(f"{name}=") or stripped == name:
                matched = name
                break
        if matched is not None:
            seen.add(matched)
            new_val = cleaned[matched]
            if new_val:
                new_lines.append(f"{matched}={new_val}\n")
        else:
            new_lines.append(line)

    for name, val in cleaned.items():
        if name not in seen and val:
            new_lines.append(f"{name}={val}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

