"""快速验证 ToolBudgetManager 逻辑"""
from tools import TOOL_DECLARATIONS, TOOL_REGISTRY
from provider import ToolBudgetManager
from prompt import build_tool_budget_prompt

mgr = ToolBudgetManager(TOOL_DECLARATIONS)

print("=== 初始状态 ===")
budget = mgr.get_budget_dict()
print("配额字典:", budget)
print("get_device_info 可用:", mgr.is_available("get_device_info"))
print("任何工具可用:", mgr.any_available())

print("\n=== Dashboard 显示 ===")
print(build_tool_budget_prompt(budget))

print("\n=== 过滤后的声明 ===")
filtered = mgr.filter_declarations(TOOL_DECLARATIONS)
print(f"可用声明数: {len(filtered)}")
for d in filtered:
    print(f"  - {d['function']['name']} (有 max_calls_per_response: {'max_calls_per_response' in d})")

print("\n=== 消耗一次 get_device_info ===")
mgr.consume("get_device_info")
budget = mgr.get_budget_dict()
print("配额字典:", budget)
print("get_device_info 可用:", mgr.is_available("get_device_info"))
print("任何工具可用:", mgr.any_available())

print("\n=== 耗尽后 Dashboard ===")
print(build_tool_budget_prompt(budget))

print("\n=== 耗尽后的声明 ===")
filtered = mgr.filter_declarations(TOOL_DECLARATIONS)
print(f"可用声明数: {len(filtered)}")

print("\n=== 测试通过 ===")
