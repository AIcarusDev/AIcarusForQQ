"""schema.py — LLM 结构化输出 JSON Schema 定义（从 schema/main.json 加载）"""

import json
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "schema" / "main.json"
with open(_SCHEMA_PATH, encoding="utf-8") as _f:
    RESPONSE_SCHEMA = json.load(_f)
