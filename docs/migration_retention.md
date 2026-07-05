# 迁移兼容保留策略

更新时间：2026-07-05

这份文档记录仍然保留在运行时里的历史迁移逻辑。目标不是把迁移代码立刻删掉，而是给它们明确边界，避免后续清理时把“仍需升级用户数据”的代码误判成普通死代码。

## QQ 配置迁移

涉及代码：

- `src/platforms/qq/adapter/config.py`
- `src/web/routes_updates.py`
- `src/web/routes_settings.py`
- `src/config_loader.py`

当前保留内容：

- 顶层旧配置 `qq_adapter` 会被归一化到 `platforms.qq.*`。
- 旧告警配置 `alerting.qq_adapter_restart` 会被归一化到新的 QQ 平台配置结构。
- `normalize_qq_platform_config(..., remove_legacy=True)` 仍会删除旧 key，保证保存后的配置不再继续携带旧字段。
- Web 更新页仍保留 napcat / qq_adapter 迁移入口，用于用户主动整理旧配置。

保留原因：用户可能从旧配置文件直接升级到当前版本；这类迁移属于配置加载边界，不能只按当前仓库默认配置判断是否可删。

删除条件：

- 发布说明明确要求用户已完成 QQ 配置迁移。
- 当前支持的最早配置版本已经晚于 `platforms.qq` 配置结构上线版本。
- `tests/test_config_normalization.py` 中旧 key 删除用例已被替换成“旧 key 不再受支持”的错误提示或文档检查。

删除前验证：

```powershell
python -m pytest tests/test_config_normalization.py
python -m pytest tests/test_tool_namespaces.py
```

## 数据库旧表与一次性迁移

涉及代码：

- `src/database.py`

当前保留内容：

- 旧表 `profiles` / `group_cards` 只用于迁移，不再写入。
- `_migrate_legacy(...)` 负责把旧表数据迁移到当前实体/账号结构。
- `_migrate_persons_accounts_to_entities(...)` 使用 `_migrations` 表里的 `rename_persons_accounts_v1` 作为幂等哨兵。
- `upsert_group_card(...)` 作为旧调用名 wrapper，转发到当前 group/entity 写入路径。

保留原因：数据库迁移直接影响用户本地长期数据，删除前必须能证明不再支持旧数据库跨版本启动。

保留策略：

- `_migrations` 表记录的一次性迁移默认长期保留，除非项目明确停止支持对应旧数据库版本。
- 旧表读取迁移可以在“最早支持数据库 schema 版本”晚于旧表版本后删除。
- 兼容 wrapper 只有在仓库内、插件/脚本边界、文档都不再提到旧函数名时才删除。

删除前验证：

```powershell
python -m pytest tests/test_memory_v2.py tests/test_config_normalization.py
python -m pytest tests/test_tool_namespaces.py
```

还应使用一份旧数据库副本做启动迁移 smoke test，确认当前版本要么能迁移，要么给出清晰的不支持提示。
