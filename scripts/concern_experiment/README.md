# Concern 离线提取实验

本目录是独立实验，不被正式 agent 导入。`prompt.md` 是完整的 system prompt；`run.py` 直接调用官方 DeepSeek Chat Completions API；`controls.json` 是人工编写的边界样本，不能当作真实认知记录。

当前契约：主观视角、每组 5 条认知、只生成候选，输出 `source_ids` / `context` / `question`。`context` 是来源中的连续原文，`question` 保留原有的不确定性、主体和行动进度。不读取已有 MemoryEvents，不做记忆对照、历史 Concern 去重、修改或删除。

## 输入与调用

从 `CognitionSources` 以只读事务导出已经进入记忆提取的完整历史批次。用 `origin_id` 与 `created_at` 共同标识一次批次，避免进程重启后区间编号重用；只选择恰有 5 条记录的批次，不拼接残缺批次。

```powershell
python -B scripts/concern_experiment/run.py export --out tmp/concern_trial/samples.json --limit 12
python -B scripts/concern_experiment/run.py run --samples tmp/concern_trial/samples.json --out tmp/concern_trial/concern --batches real_01,real_02,real_03 --concern-thinking disabled --reasoning-effort low --verbatim-context
python -B scripts/concern_experiment/run.py run --samples scripts/concern_experiment/controls.json --out tmp/concern_trial/controls --concern-thinking disabled --reasoning-effort low --verbatim-context
```

每次使用新的输出目录，已有结果不会被覆盖。依赖为 `httpx`、`python-dotenv`、`PyYAML`，沿用项目现有环境。

`--with-memory` 在同一个离线调度中，为同一份 `<task>` 分别调用 Concern prompt 和现有记忆提取 prompt。后者通过 AST 读取字符串常量，避免导入应用；仅保存原始事件输出，不执行业务工作流和数据库写入。并发上限限制实际请求开始时间，不保证两个 HTTP 请求在同一毫秒开始。

这是历史触发批次的回放，不是在线监听，也没有给压缩 worker 添加回调。当前线上触发位置参考 `src/llm/compression/worker.py`，输入序列化参考 `src/memory/event_extraction/workflow.py`。

## 参数与凭据

- 固定模型：`deepseek-v4-flash`，不自动替换模型。
- Concern 默认开启 thinking；`--concern-thinking disabled` 可关闭。`--reasoning-effort low|high|max` 默认 `high`，报告注明每次实际设置。
- 记忆侧默认关闭原生 thinking，保留原 prompt 自带的分析与提取输出；可用 `--memory-thinking enabled` 比较。它与生产环境的实际生成配置不是同一项保证。
- 最大输出 16000 tokens；遇到 `length` 记为失败。
- 默认请求 `json_object`；`--response-format text` 仅用于接口对照，prompt 仍要求 JSON。结构验证同样执行。
- 从本地 `config/config_user.yaml` 找到 HTTPS 官方 `api.deepseek.com` provider，再读取其对应 `.env` 或进程环境变量中的 key；不记录 key、不导出认证头、不跟随重定向。

API 参数依据：[Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)、[JSON Output](https://api-docs.deepseek.com/guides/json_mode/)。官方说明 JSON 模式可能返回空正文；实验将它计为失败，不转换成空 Concern。

## 检查与证据

每次运行保留 system prompt 快照及 SHA-256、输入、无认证信息的请求体、完整 API 响应、解析结果、耗时和 token 用量。真实认知和原始响应只存放在 git 忽略的 `tmp/` 中。

自动检查 JSON 形状、字段类型、来源 ID、重复来源 ID 和结束原因。`--verbatim-context` 另外检查 context 是否为所引用认知的连续子串，不归一化标点、不自动修复。失败仍完整保存原始候选并使程序以非零状态退出；这不是业务过滤器，也不能检测问句中的语义偏移。记忆侧只检查事件 JSON 可解析，不声称完成了正式记忆结构化验证。语义质量由逐项对照原始认知判断，结构通过不代表问题合理。

不自动修复模型 JSON，不自动补来源 ID，不按关键词删问题，不把重试覆盖到失败样本上。`controls.json` 是本次评估材料，不是锁定 prompt 措辞的单元测试。

本次结果见 `docs/concern_extraction_experiment_20260905.md`；当前 prompt 是试验版本，尚未证明所有候选都符合边界。

保持 v5 prompt 不变的关闭思考对照见 `docs/concern_no_thinking_experiment_20260905.md`。该轮只调用 Concern，使用 `--concern-thinking disabled --reasoning-effort low`，与保存的开启 low 请求逐字段核对，唯一变化为 `thinking.type`。

后续语义保真迭代见 `docs/concern_semantic_iteration_20260905.md`。当前 prompt 是该轮 v9，要求先引用再提简短问题；仍有实测反例，尚未通过语义保真验收。v6–v9 的全部调用、失败和重复试验均保留，未接入正式 agent。
