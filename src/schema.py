"""schema.py — LLM 结构化输出 JSON Schema 定义（从 schema/main.json 加载）"""

import json
from pathlib import Path

# schema/main.json is now in ../config/schema/main.json relative to this file
_BASE_DIR = Path(__file__).parent.parent
_SCHEMA_PATH = _BASE_DIR / "config" / "schema" / "main.json"

with open(_SCHEMA_PATH, encoding="utf-8") as _f:
    RESPONSE_SCHEMA = json.load(_f)
