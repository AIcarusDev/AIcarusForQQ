"""schema.py — LLM 结构化输出 JSON Schema 定义（从 schema/*.json 加载）"""

import json
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
_WATCHER_SCHEMA_PATH = _BASE_DIR / "config" / "schema" / "watcher.json"

with open(_WATCHER_SCHEMA_PATH, encoding="utf-8") as _f:
    WATCHER_SCHEMA = json.load(_f)
