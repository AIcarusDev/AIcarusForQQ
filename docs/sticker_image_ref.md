# 表情包 image_ref

表情包收藏以原始图片的 image_ref 为身份。sticker_id 和三位收藏编号不再是可用接口。

## 工具与发送

- qq_stickers.save_sticker(image_ref, description)：收藏可见聊天、历史窗口、转发窗口中的图片。
- qq_stickers.list_stickers()：返回主 image_ref 和印象；视觉模型还会收到标注完整 ref 的网格。
- qq_stickers.update_sticker(image_ref, description)：修改印象。
- qq_stickers.delete_sticker(image_ref)：删除收藏及其全部别名。
- send_message 的表情片段为 {"command":"sticker","image_ref":"a1b2c3d4e5f6"}，单条和批量消息形状均支持。

直接发送可见聊天图片不要求先收藏。浏览器原图继续使用原来的 image/resource_ref 固化和确认流程。

首次收藏保留原 ref。重复图片按原始字节的 SHA-256 合并，保留已有主 ref、印象和创建时间，并登记新 ref 为别名。主 ref 和别名都可用于读取、编辑、删除；收藏工具结果返回主 ref。

收藏图片接入共用 ImageResolver，可通过 view_image、examine_image、save_image 和其他使用解析器的工具访问。即使原聊天离开上下文或普通图片缓存被清理，收藏原始字节仍然可用。

保存工作空间副本继续调用 save_image：

    {"image_ref":"a1b2c3d4e5f6","path":"/home/agent/images/reaction.gif"}

图片不转码，GIF 保留原始动图字节。路径扩展名应与实际格式一致；目标已有文件时返回 already_exists。工作空间副本独立于收藏，删除收藏不删除副本。

## 持久化与迁移

data/stickers/index.json 的 v2 格式包含：

- version: 2。
- stickers：以主 ref 为键，记录 aliases、description、created_at、filename、mime、sha256。
- ref_hashes：ref 与原始内容哈希的历史绑定；删除收藏后保留绑定，防止同一个 ref 被用于另一张图片。历史绑定不代表图片仍在收藏中。

新代码首次访问收藏时，在整理前自动迁移旧索引：

1. 校验旧索引和原始文件；错误会中止，不能按空库覆盖。
2. 原样保存 index.v1.backup.json。
3. 为每个旧收藏生成一次 12 位随机 ref，保留描述、创建时间、图片字节和内部文件名。
4. 原子提交 v2 索引。再次启动直接读取已有 ref。

迁移失败可修复原文件后重试，未提交的索引不会替换原索引。备份用于核查和人工恢复；新版收藏发生修改后，不应直接以迁移前备份覆盖当前索引。

收藏读写在同一进程内加锁，索引和新图片使用临时文件后原子替换。删除先隔离文件再提交索引；提交失败恢复原文件，进程中断后仍被索引引用的隔离文件会在读取时恢复。

读取收藏时检查 SHA-256。手动改名可通过整理找回；内容替换会撤销旧收藏引用，为新内容建立新 ref。整理不重编号，超出 30 个收藏上限的孤儿图片留在磁盘等待空位。

旧聊天记录中的编号无法可靠恢复原图片，只显示历史表情占位；不会按今天的收藏顺序猜测。

## Web API

列表与上传响应使用 image_ref 字段；图片读取为 /api/sticker/<image_ref>，修改和删除为 /api/stickers/<image_ref>。这些接口接受主 ref 或别名，拒绝旧编号。索引读取或迁移错误返回明确失败。

## 验证

业务回归位于 tests/test_sticker_collection.py 与 tests/test_sticker_ref_pipeline.py。全部 pytest 测试使用临时收藏目录，避免读取或迁移生产收藏；QQ 发送使用模拟客户端，工作空间导入使用隔离文件。

迁移副本、桌面与窄屏页面及聊天预览的一次性验收记录位于 output/playwright/sticker-ref-20260906/。它们是验证产物，不属于文档内容测试。
