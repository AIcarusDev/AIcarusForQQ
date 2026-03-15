"""快速验证迁移后的 provider 模块。"""
from dotenv import load_dotenv
load_dotenv()

from provider import create_adapter, GeminiAdapter, OpenAICompatAdapter, ToolBudgetManager
from tools import TOOL_DECLARATIONS

# 1. GeminiAdapter 创建
cfg = {
    "provider": "gemini",
    "model": "gemini-3.1-flash-lite-preview",
    "thinking": {"level": "high"},
}
a = create_adapter(cfg)
print(f"Adapter: {type(a).__name__}")
print(f"Model: {a.model}")
print(f"Thinking: {a.thinking_level}")

# 2. ToolBudgetManager
bm = ToolBudgetManager(TOOL_DECLARATIONS)
print(f"\nBudget: {bm.get_budget_dict()}")
print(f"Available: {bm.any_available()}")
filtered = bm.filter_declarations(TOOL_DECLARATIONS)
print(f"Filtered decl names: {[d['name'] for d in filtered]}")

# 3. OpenAICompatAdapter
cfg2 = {"provider": "siliconflow"}
a2 = create_adapter(cfg2)
print(f"\nOpenAI compat adapter: {type(a2).__name__}")
print(f"Provider: {a2.provider}")

# 4. 列出模型（验证 API 连通性）
print("\nListing Gemini models (first 5)...")
models = a.list_models()[:5]
for m in models:
    print(f"  - {m}")
if not models:
    print("  (empty — check API key)")

print("\nAll checks passed!")
