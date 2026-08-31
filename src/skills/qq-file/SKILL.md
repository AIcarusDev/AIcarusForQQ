---
name: qq-file
description: 使用 qq_file 下载、读取、查找、枚举或永久删除当前 QQ 账号的文件。适用于根平台为 QQ 且需要处理 QQ file 消息或已保存的 QQ 文件时。
---

# QQ 文件操作

## 核心概念

`qq_file` 使用统一的 Linux 形态路径管理 QQ 文件。QQ 文件消息负责提供下载来源；下载、读取、枚举、搜索和删除都以 `qq_file` 返回的路径为准。

文件默认保存到：

```text
/home/agent/qq/{qq_id}/file/{conversation_type}_{conversation_id}/
```

- `qq_id` 是当前 QQ 账号的真实 QQ 号；
- `conversation_type` 是 `private` 或 `group`；
- `conversation_id` 在私聊中是对方真实 QQ 号，在群聊中是真实群号。

`message_id` 只在当前 QQ 会话中定位文件消息；本地路径操作可以跨会话目录。

本机 Linux 电脑存在时，文件实际保存在 Linux 中；Linux 电脑不存在时，`qq_file` 会把文件暂存在项目内的宿主机后备目录。后备目录不承诺持久化，宿主机物理路径不会返回。

后备文件仍可通过 `qq_file` 下载、读取受支持格式、枚举、搜索和删除，但不能交给 `computer` 访问或运行。Linux 电脑后来存在时，新的文件操作使用 Linux，既有后备文件不会自动迁移或合并，必要时重新下载。

## 下载

使用 `download({action:"start", message_id})` 下载当前会话中的文件消息。调用会观察任务最多 15 秒：

- 下载完成或精确记录的本地文件仍然存在时，结果包含可用的本地路径；
- 任务未完成时，结果包含 `download_id`、状态和进度，下载继续运行；
- 使用 `download({action:"poll", download_id})` 查询一个已知任务；
- 使用 `download({action:"list"})` 找回当前 QQ 账号跨会话的活跃任务和最近终态任务；
- 使用 `download({action:"stop", download_id})` 停止仍在运行的任务。

15 秒是观察窗口，不是下载时限。任务未完成时沿用原 `download_id` 查询，不需要重新启动下载。

## 读取

`read({source:{message_id}})` 已包含下载步骤：本地文件不存在时会复用或启动下载并观察最多 15 秒；下载及时完成则直接返回内容，否则返回 `download_pending` 和下载进度。

下载任务完成后，可以使用结果中的本地路径调用 `read({source:{path}})`；如果仍在来源会话中，也可以再次使用原 `message_id` 调用 `read`。

已有本地路径使用 `read({source:{path}, selection?})`。返回 `next_cursor` 时，使用 `read({cursor})` 继续读取，不再重复传入 `source` 或 `selection`。

`read` 直接解析 UTF-8 文本、PDF、DOCX、XLSX 和 PPTX 的原生文本与结构，不执行 OCR、宏、脚本或外部链接。不支持直接解析的文件先下载；只有本机 Linux 电脑存在时，才能再通过 `computer` 处理该路径。后备文件不能由 `computer` 处理或运行。

## 查找本地文件与历史消息

- `list_files` 枚举当前活动文件存储中实际存在的本地普通文件；省略 `scope` 时使用当前会话，也可指定会话或全部会话。
- `search({source:"local", ...})` 按文件名搜索实际存在的本地文件并返回路径。
- `search({source:"history", ...})` 按文件名搜索 AICQ 已同步的 QQ 文件消息并返回 `message_id`，不会向 QQ 拉取更多历史。
- `search` 只匹配文件名，可用 `file_types` 限定无点扩展名；省略 `scope` 时使用当前会话，也可指定会话或全部会话。
- 历史结果不在当前会话时，先进入结果所指会话，再用其 `message_id` 调用 `read` 或 `download({action:"start", ...})`。

分页结果使用对应工具返回的 `next_cursor` 继续，并只传 `cursor`。

## 删除

使用 `delete({path})` 永久删除当前 QQ 账号文件根目录中的一个本地普通文件。`path` 必须是 `qq_file` 返回的精确绝对 Linux 形态路径；不接受宿主机路径、目录、glob、符号链接或根目录外路径。
