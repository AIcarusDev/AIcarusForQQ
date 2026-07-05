# 代码清理审计 - 2026-07-05

范围：对 `E:\Aic_forQ\core` 做一次静态、只读的代码清理审计，重点找三类问题：

1. 重复实现、可以提取成公共逻辑的代码。
2. 已经没有实际使用的死代码。
3. 现在理论上不会再走到、但还留在代码里的兼容逻辑。

本次使用的证据：

- `git status --short`：写报告前工作区是干净的。
- `git ls-files "*.py"`：仓库内有 256 个被 Git 跟踪的 Python 文件。
- 运行时探针：用 `PYTHONPATH=src; build_tools(...)` 分别检查 core 和 QQ 场景下真实工具集合。
- AST 重复代码扫描：扫描 `src/**/*.py` 的函数体重复。
- `ruff check src tests --select F401,F841,F821,F811`：检查未使用 import、未使用变量、未定义名称等基础问题。
- 定向 `rg` 搜索：搜索 `legacy`、`compat`、`fallback`、`not_used`、内部 forced tool、namespace manifest、schema repair hook 等关键词。

优先级说明：

- P1：明确死代码，或已经变成 no-op 的兼容路径。做一次聚焦测试后，大概率可以删除。
- P2：重复实现，或仍然接在线上路径里的兼容层。值得清理，但需要更谨慎。
- P3：小型卫生问题，例如未使用 import/局部变量、本地缓存、保留策略文档。

## P1

### 1. 旧的“最近窗口记忆归档”路径已经是空操作，但主循环还在调用

证据：

- `src/consciousness/main_loop.py:182` 定义了 `_schedule_archive(...)`，函数体直接 `return`，注释写着它是旧的 recent-window archive trigger。
- `src/consciousness/main_loop.py:608` 每轮结束后仍然调用 `_schedule_archive(session, result.tool_calls_log)`。
- `src/memory/archiver.py:886` 的 `schedule_archive(...)` 也是 no-op，直接返回 `None`。
- `src/memory/archiver.py:266` 附近的 `schedule_compression_archive(...)` / `archive_compression_summary(...)` 也只是 legacy shim，只打日志，不做实际归档。
- `tests/test_memory_v2.py:488-489` 只是断言这些旧调度器返回 `None`。

为什么重要：主循环每轮都保留一个已经无效的分支；`memory.archiver` 里也暴露了看起来像公开 API、但实际什么都不做的函数。

建议清理：

1. 删除 `consciousness/main_loop.py` 里的 `_schedule_archive` 和调用点。
2. 确认脚本没有依赖后，删除或取消导出 `schedule_archive`、`schedule_compression_archive`、`archive_compression_summary`。
3. 把现有“旧函数返回 None”的测试，替换成证明 V2 归档只来自 cognition-flow compression 的测试。

### 2. 旧 forced-tool 记忆归档栈看起来已经不用了

证据：

- `src/memory/archive_memories.py:8` 声明了一个名为 `archive_memories` 的 forced function tool。
- `src/memory/archive_memories.py:147` 把它包装成 `InternalToolSpec`。
- `src/memory/archive_prompt.py` 保存了旧 forced-tool prompt。
- 当前归档流程在 `src/memory/archiver.py:32-33` 导入的是 `parse_archive_output` 和 `prompt_v2.ARCHIVE_SYSTEM_PROMPT`，之后在 `src/memory/archiver.py:679` 解析文本输出。
- 仓库搜索没有发现 `memory.archive_memories` 的导入，没有发现 `archive_memories.TOOL` 的使用，也没有发现对 `RoundRunner._call_forced_tool` 的调用。
- `src/llm/core/round_runner.py:873` 仍然定义了 `_call_forced_tool(...)`。
- `src/llm/core/internal_tool.py:11` 似乎只为旧 forced-tool 路径服务。

为什么重要：同一个“记忆归档”领域里保留了两套模型输出合同。后续改 prompt/schema 时容易误判，而且旧枚举也可能误导人。

建议清理：

1. 删除 `src/memory/archive_memories.py` 和 `src/memory/archive_prompt.py`，或把其中仍有参考价值的历史说明搬到文档。
2. 如果确认没有外部调用者，删除 `InternalToolSpec` 和 `RoundRunner._call_forced_tool`。
3. 删除后跑 `tests/test_memory_v2.py`，再做一次 prompt snapshot / archiver smoke 检查。

### 3. `src/tools/not_used/*` 是被跟踪的代码，但不会进入运行时工具集合

证据：

- 这些文件被 Git 跟踪：`check_physical_state.py`、`delete_memory.py`、`get_self_image.py`、`suggest_person_merge.py`、`update_person_profile.py`。
- 在 QQ 场景下打开所有可见 namespace 后，`ToolCollection.all_specs` 里没有任何 `not_used` 工具。
- `src/tools/CONVENTIONS.md:8` 明确说 `not_used/` 不参与 namespace 扫描。
- `docs/namespace.md:259` 和 `docs/namespace.md:544` 明确说 `get_self_image` 已下线 / 待清理。

为什么重要：这些是旧式 `DECLARATION` 工具实现。继续留在 `src/tools` 会增加审计噪音，也让工具合同搜索结果看起来比真实运行时更乱。

建议清理：

1. 如果没有手动恢复流程依赖它们，直接删除整个文件夹。
2. 如果某些内容需要留作参考，搬到 `docs/legacy/`，并保持为不可 import 的文档文本。

### 4. `src/memory_backends` 本地只剩被忽略的 bytecode

证据：

- `git ls-files src/memory_backends` 没有返回任何被跟踪的源码。
- `git status --short --ignored=matching` 显示 `!! src/memory_backends/__pycache__/`。

为什么重要：这是已删除包留下的本地残留，会干扰大范围扫描。

建议清理：做本地卫生清理时删除 `src/memory_backends/__pycache__/`。

## P2

### 5. token 统计和工具统计重复实现了同一套时间桶逻辑

证据：

- `src/token_usage_stats.py:34` / `src/tool_usage_stats.py:40`：都有 `_bucket_start_ms`。
- `src/token_usage_stats.py:46` / `src/tool_usage_stats.py:52`：都有 `_next_bucket_start_ms`。
- `src/token_usage_stats.py:60` / `src/tool_usage_stats.py:66`：都有 `_apply_range_preset`。
- AST 函数体哈希也确认 `_bucket_start_ms` 和 `_next_bucket_start_ms` 是重复实现。

建议清理：提取一个公共 helper，例如 `stats_time.py` 或 `runtime/stats_time.py`。各 service 只保留自己的 SQL 和结果组装逻辑。

### 6. JSON Schema `$ref` 解析重复实现

证据：

- `src/tools/prompt_signatures.py:66` 定义了 `_resolve_ref`。
- `src/llm/core/tool_calling/schema.py:120` 定义了同样的 resolver。
- AST 函数体哈希确认两者实现相同。

为什么重要：这两个文件都在工具合同链路上。`$ref` 行为如果改一边不改另一边，模型可见签名和后端校验/修复可能漂移。

建议清理：把 JSON Pointer `$ref` 解析提取到一个小的 schema 工具模块，然后两个地方都 import 它。

### 7. Chrome 可执行文件探测重复实现

证据：

- `src/browser/session.py:164` 定义 `_system_chrome_path`。
- `src/web/routes_settings.py:263` 定义 `_browser_login_chrome_path`。
- AST 函数体哈希确认两者实现相同。

建议清理：提取一个 browser runtime helper，让 browser session 和 settings 登录路由共用。

### 8. prompt/log 根目录解析重复实现

证据：

- `src/llm/prompt_snapshot.py:134` 定义 `_resolve_root_dir`。
- `src/llm/discarded_response_log.py:138` 定义 `_resolve_root_dir`。
- AST 函数体哈希确认两者实现相同。

建议清理：提取一个共享的 log path helper，供 prompt snapshot 和 discarded-response log 共用。

### 9. `runtime.emergency_reset` 是兼容 wrapper，但当前代码仍在 import 它

证据：

- `src/runtime/emergency_reset.py` 文件说明自己是旧 emergency reset API 的 compatibility wrapper。
- 当前仍有 import：`src/consciousness/main_loop.py:44`、`src/llm/compression/worker.py:13`、`src/web/routes_runtime.py:9`。

为什么重要：这个 wrapper 还不是死代码，但新代码仍然 import 旧模块，说明兼容边界已经反过来了。

建议清理：把这些 import 迁移到 `runtime.maintenance` / `maintenance_service`，然后单独删除 wrapper。

### 10. 多个 schema repair 兼容路径只是在防御旧模型/旧工具形状

证据：

- `src/tools/core/runtime_manage.py:193` 仍把旧字段 `timeout -> seconds`、`duration -> minutes`；但当前 prompt signature 已经只展示 `seconds` / `minutes`。
- `tests/test_wait_contract.py:24` 专门测试这套 legacy mapping。
- `src/platforms/qq/tools/qq_runtime/enter_qq_session.py:192` 把旧的 `type='temp'` 映射成 `private`。
- `src/platforms/qq/tools/qq_runtime/enter_qq_session.py:350` 和 `:379` 兼容旧测试 mock：目标解析返回 `None` 时 fallback。
- `src/platforms/qq/tools/qq_social/send_message/send_message.py:226` 保留 array-shaped `repair_schema_args` wrapper；而实际运行时用的是 `make_schema_repairer(config)`，会按配置决定 message shape。

建议清理：先定一个兼容截止线。如果旧 prompt log 不再需要通过 live validation 重放，就可以删除这些 repair，并把测试改成只断言当前合同。

### 11. QQ 旧配置迁移仍然有效，但需要保留期限

证据：

- `src/platforms/qq/adapter/config.py:56` 合并旧 `qq_adapter` 配置。
- `src/platforms/qq/adapter/config.py:97` 合并旧 `alerting.qq_adapter_restart` 配置。
- `src/platforms/qq/adapter/config.py:115` 暴露 `remove_legacy`。
- `tests/test_config_normalization.py` 覆盖了旧配置删除。
- `src/web/routes_updates.py` 也有面向用户的旧 QQ 配置整理流程。

为什么重要：这段代码大概率仍对用户配置升级有用，所以不是 P1 死代码。但如果没有保留期限，它会永远留着。

建议清理：现在先保留，但补一段文档：从哪个配置版本/日期之后，不再迁移这些旧 key。

### 12. 数据库一次性迁移和旧表兼容逻辑很多，但缺少统一说明

证据：

- `src/database.py:24` 说明旧 `profiles / group_cards` 表保留用于迁移。
- `src/database.py:1096` 执行 `_migrate_legacy`。
- `src/database.py:1173` 使用迁移 key `rename_persons_accounts_v1`。
- `src/database.py:2017` 保留 `upsert_group_card` 兼容 wrapper，而当前运行时调用者主要使用 `upsert_group`。

为什么重要：数据库迁移不能随便删，但它们让 `database.py` 很难审计。

建议清理：补一个 migration registry / retention 说明，标出哪些迁移永久保留、哪些是一次性但暂时保留、哪些在备份/导出边界后可以删除。

### 13. 工具合同迁移后，`src/tools` 下只剩 `not_used` 还在用旧 declaration 风格

证据：

- 当前活跃工具基本使用 `ToolContract` / 生成式 prompt signature。
- `src/tools/not_used/*` 是 `src/tools` 下剩下的旧式 `DECLARATION` + prompt-signature 风格业务工具。
- `tests/test_tool_prompt_signatures.py:460` 仍扫描 `src/tools` 里的 legacy declaration 文件，主要是在约束这些旧文件。

建议清理：删除或移动 `not_used` 后，可以把这个测试变成更严格的规则：活跃工具要么 Python-first，要么明确声明自己是动态工具。

## P3

### 14. Ruff 发现了一些小的未使用 import / 局部变量

命令：

```powershell
python -m ruff check src tests --select F401,F841,F821,F811 --output-format concise
```

发现：

- `src/consciousness/main_loop.py:40`：`session_key_for_focus` import 后未使用。
- `src/consciousness/sources.py:7`：`typing.Any` import 后未使用。
- `src/platforms/qq/adapter/segments.py:288`：`_build_miniapp_card` 里的局部变量 `payload` 赋值后未使用。
- `src/platforms/qq/tools/qq_runtime/enter_qq_session.py:282` 和 `:312`：局部 `import app_state` 未使用。
- `src/platforms/qq/tools/qq_social/send_voice.py:66`：局部变量 `preferred` 赋值后未使用。
- `src/tools/core/goal_manage.py:8`：`VALID_RESOLUTIONS` import 后未使用。
- `src/web/routes_memory.py:441`：`exc` 赋值后未使用。

建议清理：可以做一次机械 lint cleanup。注意 `send_voice.preferred`：它可能原本是想在动态 schema 里标记默认 TTS worker，删除前最好确认意图。

### 15. 工作区里有大量生成的 bytecode 缓存

证据：

- 扫描发现 `src` / `tests` 下有 41 个 `__pycache__` 目录。
- 扫描发现 505 个 `.pyc` 文件。
- 它们被 ignore，不影响 Git，但会出现在 `git status --ignored=matching` 和文件扫描里。

建议清理：后续做大范围审计前，可以先删除这些本地 bytecode cache。这不影响被 Git 跟踪的代码。

## 建议处理顺序

1. 先清理 P1 memory archive：删除 no-op 调度调用、旧 forced-tool 归档文件、旧 forced-tool 支持。
2. 再清理 P1 工具：删除或归档 `src/tools/not_used`。
3. 做 P2 提取：stats 时间 helper、schema `$ref`、Chrome 路径、log 根目录路径。
4. 做 P2 兼容层整理：迁移 `runtime.emergency_reset` import，并决定 schema repair shim 的保留截止线。
5. 最后做 P3 lint/cache 卫生清理。

## 后续实际清理时的验证清单

```powershell
python -m ruff check src tests --select F401,F841,F821,F811
python -m pytest tests/test_memory_v2.py
python -m pytest tests/test_tool_prompt_signatures.py tests/test_tool_namespaces.py tests/test_wait_contract.py
```

还建议补一个运行时探针：

```powershell
$env:PYTHONPATH='src'
# 分别用 build_tools(...) 检查 core-only 和 QQ-session 场景下的真实工具集合
```
