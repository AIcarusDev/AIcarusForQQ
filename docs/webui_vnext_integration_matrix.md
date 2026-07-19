# AIcarus WebUI vNext 集成覆盖矩阵

日期：2026-07-19

状态：M0–M5 基线已完成真实集成并通过最终回归；旧 UI 与 vNext 并行保留。

## 1. 当前结论

- 旧 UI 继续位于 `/`，vNext 通过同一 Quart 服务位于 `/new/`。两者不覆盖路由、静态资源或浏览器偏好，可用于渐进迁移。
- vNext 的 11 个主目的地均已连接真实契约。15 个设置领域中 14 个已实现真实读写或受控资源操作；“记忆系统”配置页因配置结构仍在演进而明确保留为待版本化领域。
- 首页、会话、资源、可观测性、Memory Schema/MemoryQL、设置与维护不再依赖原型模拟数据。
- 页面组件不直接调用 `fetch`。`webui-vnext/src/api/http.js` 是统一 HTTP 边界；各领域 Adapter 负责响应转换、错误、认证与取消。
- LocalStorage 只保存侧栏与主题偏好，不保存凭据、Secret、真实记忆、查询结果或表单业务数据。
- 本轮不是复制 grok-reg 后台。实现保留其“有边界、低噪声、纸张式、专注主任务”的设计理念，同时服从 AIcarus 的信息密度和真实业务结构。

## 2. 运行边界

```text
legacy UI /
vNext SPA /new/
        │
        ├─ domain adapters
        │    └─ shared HTTP/realtime transport
        │
        ├─ existing compatible APIs
        └─ versioned /api/ui/v1/* view models and commands
                     │
                     ├─ settings domains
                     ├─ observability
                     ├─ semantic memory query
                     └─ maintenance
```

核心边界：

- 复用同源 Session 认证，不在前端建立第二套凭据体系。
- HTTP 与 WebSocket 共用同一 Session 认证边界。Agent/日志流携带进程级 `stream_id`；服务重启后旧游标会被明确重置并补发快照，不把新进程的数据误判为旧流增量。
- 旧接口行为保持不变；vNext 需要不同语义时新增 `/api/ui/v1/*`，不让组件绑定存储表或旧模板字段。
- 重型 Memory 和可观测性页面按路由懒加载。
- 破坏性操作的目标、影响、备份策略、可用性与确认字符串全部由服务端给出。

## 3. 里程碑状态

| 阶段 | 交付范围 | 最终状态 |
|---|---|---|
| M0 | 同源壳层、认证、能力发现、首页状态、主题与响应式导航 | 完成 |
| M1 | 工具统计、Token 趋势、延迟分位、feature/model 分组、真实图表 | 完成 |
| M2 | Core 聊天、Focus、Agent、实时日志、更新、表情包、自身形象、Workspace 读取 | 完成 |
| M3 | 领域设置、revision 冲突保护、Secret 命令、安全设置 | 完成；记忆配置单独延后 |
| M4 | Memory Schema、MemoryQL 1.0、隔离结果、预算、provenance、Explain、DOC | 完成 |
| M5 | 维护概览、缓存、Workspace 命令/job、服务端确认与轮询 | 完成 |

## 4. 主功能覆盖

| 领域 | vNext 状态 | 契约与保护 |
|---|---|---|
| 全局壳层与认证 | 真实集成 | 同源 Session；401 回到现有认证流；HTTP/WS 继续受全局认证保护 |
| 首页 | 真实集成 | 读取 `/api/status`、`/api/core/status` 与 Memory Schema 聚合；不伪造健康状态 |
| Core 聊天 | 真实读写 | 发送携带幂等 `client_id`；区分 pending、accepted、failed |
| Focus | 真实读取 | 对焦点会话、上下文与媒体引用建立前端 View Model |
| Agent | 真实实时读取 | snapshot + cursor + `stream_id` + reconnect 去重；跨进程旧游标重置；展示规划、工具与执行时间线 |
| 运行日志 | 真实实时读取 | 沿用 `since` 增量语义并增加进程代际；跨重启补发快照；提供断线、认证失效与空状态 |
| 表情包 | 真实资源管理 | 列表、上传、编辑、删除、reconcile；图片最大 8 MiB 并由 Pillow 验证 |
| 工具统计 | 真实读取 | summary/timeline、成功失败、avg/p50/p95/max、24h/7d/30d、筛选与 tooltip |
| Token 用量 | 真实读取 | input/output/cached/reasoning；`group_by=feature\|model`；unknown usage 独立展示 |
| 记忆 | 真实只读 | Schema 优先；MemoryQL Graph/Table/Raw/Explain；结果隔离且有硬预算 |
| 设置 | 14/15 领域真实 | 领域 GET/PATCH、opaque revision、409 冲突、Secret keep/replace/clear；记忆配置待版本化 |
| 维护 | 真实受控命令 | 精确确认词、服务端 effects/preserves/backup、Workspace job 轮询；浏览器验收未执行任何破坏性动作 |

## 5. 设置领域覆盖

| 设置领域 | 状态 | 说明 |
|---|---|---|
| 模型供应商 | 真实 | Provider 元数据、模型选项、Secret 状态与显式替换/清除 |
| 主模型 | 真实 | 模型、生成参数、速率限制、revision 冲突保护 |
| 专用模型 | 真实 | Vision、工具守门、压缩、慢思考等独立绑定 |
| 记忆系统 | 延后 | 当前仍为明确的非持久化占位；待配置 schema/version 稳定后接入领域契约 |
| 角色与身份 | 真实 | Persona 与身份数据按领域保存 |
| 自身形象 | 真实 | 受限资源上传/删除；阻止路径穿越、保留名与意外覆盖 |
| QQ / Adapter | 真实 | 连接、范围与恢复策略；保存结果区分 saved/applied/restart-required |
| TTS | 真实 | 端点、并发和运行相关设置；Secret 不回显 |
| 外部服务 | 真实 | 搜索、浏览器、天气等依赖配置 |
| 告警与邮件 | 真实 | 告警、SMTP/IMAP、邮件控制；保存与测试语义分离 |
| Linux 工作区 | 真实 | 状态、配置、目录选择与受控 job；写操作移交维护页 |
| 缓存 | 真实只读 + 维护入口 | 展示真实占用；清理统一进入维护确认流 |
| 网络与高级 | 真实 | 代理采用显式保留、替换与清除；清除同步移除持久化键和当前进程值 |
| 外观 | 真实前端偏好 | 默认跟随系统；可选纸张、雾蓝、石墨、午夜、OLED 等配色 |
| 面板安全 | 真实 | 认证状态、密码与登出复用现有安全契约 |

## 6. Memory：面向演进结构的设计

Memory 页不再尝试显示全量实例图。默认视图只展示语义 Schema，用户通过 MemoryQL 精确投影需要的节点和关系。

当前契约：

- `GET /api/ui/v1/memory/schema`
- `POST /api/ui/v1/memory/query`
- `schema_version=memory-semantic-v1`
- `language_version=1.0`
- 硬限制：80 节点、120 边、100 行、深度 2、查询 8,000 字符、执行 500ms。
- 默认“最近事件关系”只请求 8 节点 / 4 边，避免自动适配后标签再次缩小。
- 使用 SQLite 只读连接、参数化执行计划、每次查询独立投影，并返回 budget、truncated、provenance 与 explain。
- Schema 检查同时验证表和必需列。记忆库缺失或结构不兼容时显式降级，不猜测字段。
- Schema 与查询执行进入工作线程，避免阻塞 Quart 事件循环。
- `MemoryEventRelations` 的窄关系查询先受最近记录限制，再加载关联事件正文；索引 `idx_MemoryEventRelations_recent(status, updated_at_ms, relation_id)` 保证真实库冷启动查询稳定。

这使记忆系统无需“开发完成后才能优化”：可先稳定 UI 与查询语言边界，后续结构变化通过新的 schema/language 版本演进。尚未稳定的“记忆配置写入”则继续延后，避免把未定内部结构固化为公共表单契约。

## 7. 设置与 Secret 安全

- 领域读写入口：`GET/PATCH /api/ui/v1/settings/<domain>`。
- 响应只暴露 `{configured, masked_hint}`，真实 Secret 不进入响应、DOM 初值或 LocalStorage。
- 写入只接受 `keep`、`replace`、`clear` 三种显式命令。
- 代理消费者当前统一使用 `strip() or None`，因此缺失值与空字符串在运行时都表示直连；vNext 不虚构不存在的第三种代理状态。`clear` 删除 `.env` 键并同步清除当前进程值，外部环境若另有配置则会在下次启动时重新继承。
- revision 由服务端生成并保持 opaque；缺失 revision 返回 428，过期 revision 返回 409 和最新可见快照。
- 页面保持 unsaved、conflict、restart-required、loading、empty 与 error 状态，不静默覆盖外部修改。
- 前端偏好只包括主题/配色与侧栏状态；业务设置不以浏览器存储代替后端。

## 8. 维护安全

- `GET /api/ui/v1/maintenance` 返回领域、动作、可用性、targets、effects、preserves、backup 与 expected confirmation。
- `POST /api/ui/v1/maintenance/actions/<domain>/<action>` 只接受服务端当前给出的确认字符串。
- Workspace job 通过 `/api/ui/v1/maintenance/workspace/jobs/<job_id>` 轮询，不假设命令已经完成。
- 缓存清理复用统一 scanner/clearer；旧缓存接口与 vNext 不再各自推导目标。
- 缓存动作在检查与执行之间持有同一服务锁；空缓存或已不可用的动作即使绕过前端直接调用也返回 409，避免并发状态漂移。
- Workspace 的 `describe_actions` 与 `start_job` 共用同一组可用性 guard，页面描述与实际执行不会产生两套判断。
- 最终浏览器 QA 只打开并取消确认，没有保存设置、上传、删除、清缓存、认知重置或启动 Workspace job。

## 9. 验收结果

- Python：`537 passed, 7 skipped`。跳过项均为需要显式 `AICQ_WORKSPACE_INTEGRATION=1` 的外部 Workspace 集成测试。
- vNext 聚焦契约：`91 passed`，覆盖路由注册、前端 Adapter、设置 revision/Secret、维护 guard、业务幂等与实时流代际。
- Memory 契约：`tests/test_vnext_memory_contracts.py` 10 项通过。
- 前端：`npm run lint` 通过；`npm run build` 通过。
- 构建输出：`src/static/new/`；主包约 250 kB，Memory 与可观测性独立懒加载。
- HTTP：重启后 `http://127.0.0.1:5000/new/` 返回 200，监听 PID 从 40224 更新为 26860；旧 `/` 保持可用。
- 认证与实时性：测试证明 HTTP/WS 共用 Session；实机重启后登录状态保留，Agent 显示“已连接”，日志流持续接收新数据。
- 浏览器：1440 × 900、390 × 844 与恢复默认视口均已验收；无横向溢出；最终控制台 warning/error 为 0。
- 最终 MemoryQL 默认查询：8 节点、4 关系、15ms、隔离结果可视化正常；DOC 入口与四种结果视图保留。
- 视觉证据与比较历史：根目录 `design-qa.md`；最终结果为 `passed`。

## 10. 后续迭代边界

下一阶段不应再做一次大规模重写。建议将 `/new/` 作为 beta 入口进入真实使用观察：

1. 收集每个目的地的真实任务完成率、错误与用户反馈，不用“页面点击量”替代可用性判断。
2. 等记忆配置 schema 稳定后，为 `memory-system` 增加版本化领域契约；不要复用跨域 `POST /settings/full`。
3. 按真实使用证据修正信息密度、移动端长表单和图表钻取，不提前扩展无验证的控制项。
4. 旧 UI 至少保留一个过渡版本；只有高风险路径有等价回退、文档和迁移提示后再讨论默认入口切换。

## 11. 证据索引

- vNext 静态入口：`src/web/routes_vnext.py`
- v1 能力、Memory 与可观测性：`src/web/routes_ui_v1.py`
- v1 设置：`src/web/routes_ui_v1_settings.py`
- v1 维护：`src/web/routes_ui_v1_maintenance.py`
- 设置领域边界：`src/web/settings_domains.py`
- MemoryQL：`src/memory/semantic_query.py`
- 缓存维护：`src/runtime/cache_maintenance.py`
- 前端 API 边界：`webui-vnext/src/api/`
- 前端主壳层：`webui-vnext/src/App.jsx`
- Memory UI 与内嵌文档：`webui-vnext/src/memory/`
- 可观测性 UI：`webui-vnext/src/observability/`
- 维护 UI：`webui-vnext/src/maintenance/MaintenancePage.jsx`
- 契约测试：`tests/test_vnext_*_contracts.py`、`tests/test_vnext_routes.py`
- 视觉 QA：`design-qa.md`、`artifacts/webui-vnext-final-qa/`、`artifacts/webui-vnext-final-qa-2026-07-19/`
