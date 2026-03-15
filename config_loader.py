"""config_loader.py — 配置加载与运行时覆盖

负责：
  1. 从 config.yaml 读取配置
  2. 从 persona.md 读取角色设定
  3. 加载 .model_override.json（运行时模型覆盖）
  4. 提供 save_model_override() 持久化切换
"""

import json
import logging

import yaml

logger = logging.getLogger("AICQ.config")

_RUNTIME_OVERRIDE_FILE = ".model_override.json"


def load_config(
    config_path: str = "config.yaml",
    persona_path: str = "persona.md",
) -> tuple[dict, str]:
    """加载配置文件和角色设定。

    Returns: (config_dict, persona_text)
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(persona_path, "r", encoding="utf-8") as f:
        persona = f.read()

    # 运行时覆盖
    try:
        with open(_RUNTIME_OVERRIDE_FILE, "r", encoding="utf-8") as f:
            ov = json.load(f)
        config["provider"] = ov["provider"]
        config["model"] = ov["model"]
        config["model_name"] = ov.get("model_name", ov["model"])
        if ov.get("base_url"):
            config["base_url"] = ov["base_url"]
        elif "base_url" in config:
            del config["base_url"]
        logger.info(
            "已应用运行时覆盖: provider=%s model=%s",
            config["provider"],
            config["model"],
        )
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("运行时覆盖文件无效，已忽略: %s", e)

    return config, persona


def save_model_override(
    provider: str,
    model: str,
    model_name: str,
    base_url: str | None = None,
) -> None:
    """持久化模型切换到 .model_override.json。"""
    try:
        with open(_RUNTIME_OVERRIDE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "provider": provider,
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
