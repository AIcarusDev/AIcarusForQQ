# 表情包收藏与统一图片引用

图片的身份、原图、别名、通用描述和精查结果统一由 [统一图片机制](unified_images.md) 管理。收藏只记录主 `image_ref`、用途描述和收藏时间，不另存一份图片。

- `save_sticker(image_ref, description)`：将已登记图片加入收藏；重复收藏保留已有用途描述。
- `list_stickers()`：返回主 ref、用途描述和可选的标注网格。
- `update_sticker(image_ref, description)`：修改收藏用途描述，不覆盖图片的通用视觉描述。
- `delete_sticker(image_ref)`：取消收藏关系；原图及全部历史 ref 仍能通过统一查询读取。

主 ref 和别名均可用于操作同一收藏。`sticker_id` 与三位编号继续拒绝使用，不依据现在的收藏顺序猜测历史图片。

普通图片发送、表情发送、`view_image`、`examine_image`、`save_image` 共用图片读取入口。浏览器资源固化后使用相同的图片身份，既有发送确认规则不会因别名或去重绕过。GIF 原始字节保留；模型用到的抽帧和网格属于派生展示。

`data/stickers/index.json` 使用 v4：`stickers` 以主 ref 为键，每项只有 `description` 与 `created_at`。别名及内容绑定只存于统一图片索引。旧索引和原图通过显式迁移工具导入，不在启动或普通查询时自动改名、移动或删除文件。

取消收藏不删除统一原图，也不删除通过 `save_image` 导出的工作区副本。导出依然校验格式、禁止覆盖已有文件。

Web API 保持 `/api/sticker/<image_ref>` 读取收藏图片、`/api/stickers/<image_ref>` 修改或取消收藏；列表与上传响应继续使用 `image_ref`。没有收藏关系时，收藏专用 API 返回未找到，统一图片查询仍可读取原图。

业务测试使用隔离数据库和图片目录；迁移、备份、验证与清理步骤见统一图片机制文档。
