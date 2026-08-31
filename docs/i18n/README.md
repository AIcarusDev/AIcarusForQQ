# 工具名称本地化维护约定

工具统计页的本地化采用“公开运行词表 + 非静态溯源表”两部分：

- `src/static/i18n/tool-name-glossary.zh-CN.json` 只保存页面运行所需的 `zh_name` 与 `zh_definition`。
- `docs/i18n/tool-name-glossary.zh-CN.provenance.json` 保存协作者审查所需的 `source` 与 `status`，不会由 `/static/` 匿名提供。

维护规则：

1. key 必须是历史日志或当前工具注册表中的完整 raw name，只允许精确匹配，不拆前缀、不猜测、不机翻。
2. 未收录、词条损坏或词表加载失败时，页面必须原样显示 raw name，且不得影响统计请求。
3. 中文名称和定义应来自当前 `ToolContract`、源码注释；已移除工具可引用 Git 历史对象。
4. 新增、改名或删除词条时必须同步更新运行词表与溯源表；两份文件的 key 集合保持一致。
5. `status` 只允许 `active` 或 `historical`。`source` 使用 `path#symbol` 或 `git:<revision>:path#symbol`。
6. 公开运行词表不得包含源码路径、Git revision、凭据或业务数据。

默认界面显示原始英文工具名；中文切换仅改变展示文本，API 参数、筛选状态和 `data-*` 属性继续使用 raw name。
