# QQ 文件 namespace 设计

> 状态：IMPLEMENTED
> 日期：2026-08-31
> 当前完成度：五个公开工具、共享路径与双后端路由、持久下载任务、文档解析、数据库索引、namespace 和 Skill 均已实现，并有自动化合同与行为测试。

## 文档结构

本文件保存 namespace 的共享设计、跨工具不变量、当前未决事项和代码衔接点。配套模型操作说明见 [`qq-file/SKILL.md`](./qq-file/SKILL.md)。每个工具的完整参数 Schema、正式 TS 注入形态、返回合同与示例拆分到独立文档：

| 工具合同 | 文档 |
|---|---|
| `download` | [download.md](./download.md) |
| `read` | [read.md](./read.md) |
| `list_files` | [list_files.md](./list_files.md) |
| `search` | [search.md](./search.md) |
| `delete` | [delete.md](./delete.md) |

## 1. 文档目的

本设计把 QQ 文件操作建模为一个使用 Linux 路径形态的本地文件空间。Linux 电脑存在时，文件空间由 Linux 提供；Linux 电脑不存在时，由项目内不承诺持久化的宿主机后备目录提供。

QQ 负责提供文件消息和下载来源；`qq_file` 始终使用 `/home/agent/qq/...` 逻辑路径完成下载、读取、枚举、搜索和删除。宿主机物理路径不进入公开工具合同，常规 QQ 文件操作也不需要展开通用 `computer` namespace。

### 1.1 全局模型可见与业务返回文案约束

以下规则适用于所有 namespace description、工具 description、参数字段 description、生成后的 TypeScript-like 注释、运行时错误、warning 和其他业务返回：

1. 模型可见文案必须保持工具使用者的主观视角，不得用 `Agent`、`bot`、机器人等第三人称词语描述使用工具的主体。应使用“当前 QQ 账号”“当前会话”“本机 Linux 电脑”“此文件”等就地语义。
2. 业务返回不得泄露开发阶段、内部路线图或交付批次。禁止出现“第一阶段不实现”“后续再做”“本期暂缓”等内部过程表述。
3. 能力文案只陈述当前事实：支持的能力直接说明支持；不支持的能力统一说明“暂不支持”，并在必要时给出可执行替代路径。
4. 开发文档正文可以记录版本规划，但任何会进入模型上下文、日志业务结果或人类用户界面的字符串与示例都必须满足以上规则。

模型可见参数合同由自动化测试与本文档逐字段比对；业务返回的措辞约束同时纳入回归检查。

本文档记录已实现的产品语义、目录、生命周期、工具合同、安全边界、迁移和验证方式。

## 2. 已确认的核心原则

1. 只纳入 QQ `file` 消息，不纳入原生图片、视频和语音消息。
2. 每条真实 QQ 文件消息只包含一个文件，因此下载入口直接使用 `message_id`，不额外创造 file ref 或 file index。
3. 公开文件路径固定在 `/home/agent` 下；Linux 电脑不存在时，同一逻辑路径映射到项目内的宿主机后备目录。
4. 不同 QQ 会话使用不同目录。
5. 会话目录是归档边界，不是权限隔离；Agent 可以显式跨会话读取、搜索和删除文件。
6. `qq_file` 只在当前根平台为 QQ 时可见。
7. `qq_file` 及其 skill 只能由 Agent 按需打开；看到文件消息不会自动展开 namespace 或 skill。
8. 任意 QQ `file` 类型都允许下载；只有受支持格式可以由 `qq_file.read` 直接解析。
9. 不支持直接读取的格式只有在 Linux 电脑存在时才能按需通过 `computer` 继续处理；宿主机后备文件不能交给 `computer` 运行或处理。
10. 删除是永久删除，不提供回收站。
11. 搜索只匹配文件名，不搜索文件正文。
12. 下载任务的 15 秒是观察窗口，不是执行超时；观察结束绝不取消下载。
13. 超过观察窗口的任务静默完成或失败，不主动唤醒 Agent。如果 Agent 后续不再查询，该任务就被 Agent 遗忘。
14. `read` 可以通过当前会话 `message_id` 自动复用或启动下载；15 秒内完成时直接读取，未完成时返回任务进度。
15. 本地文件搜索与已同步历史搜索统一为 `search`，由必填 `source=local/history` 明确区分数据源。

## 3. 逻辑目录与存储路由合同

### 3.1 公开逻辑根目录

```text
/home/agent/qq/{agent_qq}/file/{conversation_type}_{conversation_id}/
```

字段含义：

- `agent_qq`：当前 Agent 登录的真实 QQ 号，由 QQ runtime 提供，不接受模型传参。
- `conversation_type`：只允许 `private` 或 `group`。
- `conversation_id`：私聊使用对方真实 QQ 号；群聊使用真实群号。

示例：

```text
/home/agent/qq/213628848/file/group_1090411227/
/home/agent/qq/213628848/file/private_123456789/
```

该路径是 `qq_file` 的稳定公开身份。所有工具入参、返回路径、数据库 `local_path` 和 cursor 都使用这一 Linux 形态的逻辑路径，不公开也不接受 Windows 宿主机绝对路径。

### 3.2 后端选择

每次需要开始新的文件系统操作时，先进行不启动 Linux 发行版的浅层安装态判断：

```text
确认 AICQ-Workspace 已存在
  -> 使用 linux 后端
  -> 发行版或容器暂未运行也不进入后备目录

确认 AICQ-Workspace 不存在
  -> 使用 host_fallback 后端

无法可靠判断安装态
  -> 返回 runtime_unavailable
  -> 不猜测为不存在
```

选择规则只看受管 Linux 电脑是否存在，不看当前是否运行，也不把 `enabled` 配置、健康状态或一次启动结果当成“不存在”。以下情况都不得自动降级到宿主机后备目录：

- Linux 电脑已经安装但当前停止；
- Linux 启动失败或 bridge 暂时不可用；
- 协议或受管版本需要更新；
- Linux 文件系统空间不足、权限异常或写入失败；
- 完整安装态探测自身失败，无法明确证明 Linux 不存在。

实现应从 Windows 当前用户的 WSL 注册信息中浅层判断是否存在名为 `AICQ-Workspace` 的发行版。不得为了选择后端调用会进入发行版执行命令的完整 `WorkspaceControlPlane.probe()`。

### 3.3 两种物理映射

Linux 后端保持原始路径：

```text
/home/agent/qq/{agent_qq}/file/{conversation_type}_{conversation_id}/
```

宿主机后备目录固定为：

```text
{project_root}\cache\qq_file_fallback\home\agent\qq\{agent_qq}\file\{conversation_type}_{conversation_id}\
```

例如逻辑路径：

```text
/home/agent/qq/213628848/file/group_1090411227/report.pdf
```

在后备模式下映射到：

```text
{project_root}\cache\qq_file_fallback\home\agent\qq\213628848\file\group_1090411227\report.pdf
```

宿主机后备目录的合同：

- 它是可清理缓存，不承诺跨清理、重装、项目迁移或外部删除后的持久性；
- 缓存字节消失后，明面存在性检查按文件不存在处理，允许重新下载；
- 宿主机绝对路径只在实现内部使用，不进入工具结果、业务错误、日志提示或模型上下文；
- 后备文件只能通过 `qq_file` 下载、读取受支持格式、枚举、搜索和删除，不能由 `computer` 访问或运行；
- 任意类型仍可下载，但后备模式不增加执行宿主机文件的能力；
- 文件大小上限、15 秒观察窗口、任务静默完成、精确删除和跨会话能力与 Linux 后端相同。

Windows 不能原样表示的 Linux 文件名由后备层使用可逆的物理组件编码，并把映射保存在内部 `storage_relpath` 中。公开 `local_filename` 和 `local_path` 仍保持既定 Linux 命名语义。

### 3.4 后端切换

Linux 电脑后来被确认存在时：

- 后续新文件系统操作立即选择 `linux`；
- 不自动迁移、合并或复制 `host_fallback` 中的既有文件；
- 后备文件退出当前活动文件树，后续明面存在性检查不会把它当成 Linux 文件；
- 需要同一来源文件时，允许按现有规则重新下载到 Linux；
- 旧后备文件可以随缓存清理消失，不因此产生主动唤醒或消息。

下载任务在创建时冻结 `storage_backend` 和 `storage_relpath`。安装态变化、15 秒观察窗口结束、`poll` 或 `list` 都不会切换正在运行任务的写入目标，也不会取消任务。任务终态记录保留其创建时的逻辑路径和后端身份，但不得因此重新激活已经退出活动文件树的后备目录。

### 3.5 文件名冲突

存储后端不会自动为同名文件增加业务序号，此规则由 `qq_file` 实现。

不按内容或哈希去重。目标目录中已存在同名文件时，在扩展名前递增 `(n)`：

```text
报告.pdf
报告(1).pdf
报告(2).pdf
```

无扩展名文件使用：

```text
README
README(1)
```

并发下载必须通过当前存储后端的原子占位或等价机制选定最终文件名，不能使用“先检查、后普通写入”的竞态实现。

### 3.6 路径安全

所有公开路径操作必须限制在：

```text
/home/agent/qq/{current_agent_qq}/file/
```

至少需要满足：

- 清理 QQ 原始文件名中的目录分隔符、NUL 和控制字符。
- 拒绝 `..` 路径逃逸。
- 拒绝通过符号链接逃出受管根目录。
- 宿主机后备层同时拒绝 Windows reparse point、junction 和其他可改变解析目标的目录项。
- 删除只接受普通文件，不接受目录、glob 或符号链接。
- 下载中的临时文件不出现在 `list_files` 和搜索结果中。

## 4. 消息身份、文件记录与去重

### 4.1 下载消息定位

`download(action="start")` 仅在当前 QQ 会话内查询：

```text
session_key + message_id
```

公开参数仍然只有 `message_id`。`session_key` 从当前 QQ 会话推导，不能由模型指定。

若当前不在任何具体 QQ 会话中，`start` 返回 `no_current_qq_session`。不会从其他会话递归查找同一个 `message_id`。

查询到的消息必须：

- 是真实 QQ 消息；
- 包含 `file` segment；
- 该消息只对应一个文件；
- 保留足够的 `file_id` / `busid` 等 adapter 定位信息，或能通过 QQ adapter 使用 `message_id` 重新取得下载入口。

### 4.2 已下载文件的明面去重

同一来源消息重复调用下载时，系统先查 `qq_file_records` 当前记录的精确 `local_path` 和 `storage_backend`。

判定“文件仍然存在”只做明面检测：

- 记录的 `storage_backend` 与当前活动后端一致；
- 检查记录中的精确逻辑路径及其当前物理映射；
- 路径仍位于当前 QQ 账号的受管根目录；
- 路径是普通文件；
- 路径不是符号链接。

明确不做：

- 不扫描其他目录寻找同名文件；
- 不计算哈希寻找移动后的副本；
- 不追踪 inode；
- 不因某个相似文件存在而推断它是原文件。

结果：

- 精确路径仍存在：返回 `already_exists` 和该路径，不重复下载。
- 文件被重命名、移动或删除，导致精确路径不存在：允许重新下载。
- 重新下载按照当前目录的同名规则选择文件名，并更新该来源消息当前记录的 `local_path`。
- 被人工移动或重命名的旧文件若仍在 QQ 文件根目录内，可以由物理目录枚举发现，但不再自动归属于原消息记录。

### 4.3 活跃任务去重（已确认）

同一 `agent_qq + session_key + message_id` 已经存在活跃下载任务时，不创建第二个任务，返回 `already_downloading`、现有 `download_id` 和当前进度。

此行为属于已确认的 `download` 合同，用于避免同一来源消息被并发创建为多个下载任务。

## 5. namespace 与 skill

### 5.1 namespace

建议 manifest 身份：

```yaml
qq_file:
  description: "QQ 文件管理：下载 QQ 文件消息，并读取、枚举、搜索或永久删除本地 QQ 文件。"
  activation:
    platform: qq
  skill: qq-file
```

行为边界：

- 只在根平台为 QQ 时可见。
- 默认折叠。
- Agent 使用 `namespace_manage.open` 主动打开。
- 文件消息到达不会自动打开 namespace。
- 文件消息到达不会自动加载 skill。
- 打开 namespace 不等于自动下载任何文件。

### 5.2 配套 Skill

配套 [`qq-file/SKILL.md`](./qq-file/SKILL.md) 是一个自包含的单文件 Skill，不依赖 references、assets 或外部脚本。它只说明如何使用 `qq_file` 完成下载、读取、任务找回、文件查找和精确删除，不加入人格、主动性或通用决策风格。

模型可见文案采用分工：

- Skill 保存跨工具操作流程，例如 15 秒观察窗口结束后的 `poll / list`、`read(message_id)` 的自动下载读取，以及本地与历史搜索的区别；
- 工具顶层 description 只说明函数职责；
- 参数 description 保留调用形态、字段来源、默认值与约束，避免单看函数签名时产生歧义。

### 5.3 当前工具规划

| 工具 | 状态 | 职责 |
|---|---|---|
| [`download`](./download.md) | 已实现 | 启动、轮询、列出和显式停止 QQ 文件下载任务 |
| [`read`](./read.md) | 已实现 | 从本地路径或当前会话文件消息读取文本、PDF、DOCX、XLSX、PPTX |
| [`list_files`](./list_files.md) | 已实现 | 枚举当前活动存储中的 QQ 文件并标明是否有索引记录 |
| [`search`](./search.md) | 已实现 | 按文件名搜索本地文件或 AICQ 已同步的 QQ 文件消息 |
| [`delete`](./delete.md) | 已实现 | 按精确逻辑路径永久删除一个 QQ 文件 |

这些工具底层统一经过 `QQFileStorageRouter`，由 `LinuxQQFileStorage` 或 `HostFallbackQQFileStorage` 实现文件操作。公开语义保持在 `qq_file`，不要求展开 `computer`，五个工具的参数和返回合同不增加后端选择字段。

## 6. 下载任务生命周期

### 6.1 状态机

```text
queued
  -> resolving
  -> downloading
  -> verifying
  -> completed

任意非终态 -> failed
任意可停止非终态 -> stopped（仅显式 stop）
```

终态：

```text
completed | failed | stopped
```

### 6.2 15 秒观察窗口

`download(action="start")` 启动任务后观察最多 15 秒：

```text
15 秒内完成
  -> 本次工具调用返回 completed 和最终逻辑路径

15 秒内未完成
  -> 本次工具调用返回当前状态、download_id 和进度
  -> 后台下载继续运行
```

15 秒观察结束不是下载失败，不使用 `timed_out` 作为任务状态。可单独返回 `observation_timeout: true` 表示这次观察窗口已经结束。

### 6.3 超时后的静默行为

一旦 `start` 已经因为观察窗口结束而返回：

- 后续完成或失败不发布 Agent 完成事件；
- 不写入 `<attention_events>`；
- 不自动唤醒 Agent；
- 不自动打开 `qq_file`；
- 不自动加载 `qq-file` skill；
- 不自动向任何 QQ 会话发送消息；
- Core 重启不得因为该任务的状态主动发起 Agent 回合。

Agent 后续只能通过：

```text
download(action="poll", download_id="...")
download(action="list")
```

主动查询。如果 Agent 不再查询，该任务即使已经完成或失败，也保持静默。

Core 重启会中断仍在传输的下载进程。首次再次使用 `qq_file` 时，持久记录中仍为活跃态的任务会校准为 `failed`，失败码为 `download_interrupted`；`poll/list` 可以查到该终态，随后可对原消息重新发起下载。该校准不产生主动模型唤醒，也不向 QQ 会话发送消息。

### 6.4 poll 与 list

`poll` 只查询一个明确的 `download_id`，不隐式包含其他任务。

`list` 是任务发现和找回入口：

- 不要求当前 QQ 会话；
- 限定为当前 Agent 的真实 QQ 号；
- 跨当前 QQ 账号的全部会话；
- 默认返回全部活跃任务；
- 同时返回最近 20 条终态任务；
- 支持按状态过滤；
- 支持分页；
- 返回已完成任务的最终逻辑路径；
- 不区分“未观察/已观察”，也不承担提醒语义。

### 6.5 下载大小与磁盘

默认单文件上限：

```text
4 GiB = 4 * 1024 * 1024 * 1024 bytes
```

规则：

- 作为可配置常量实现，但默认值固定为 4 GiB。
- QQ 元数据已经声明超过上限时，不启动任务，返回 `file_too_large`。
- 大小未知时允许开始流式下载；实际字节超过上限后任务失败。
- 不设置 QQ 文件总容量配额。
- QQ 文件与当前活动存储后端中的其他数据共享真实磁盘容量；后备模式检查项目所在宿主机卷的可用空间。
- 已知目标大小时应检查实际可用磁盘；不足时返回 `insufficient_disk_space`。
- 文件数据必须流式传输，不能整体载入内存或放入普通 JSON RPC 正文。

### 6.6 临时文件

- 下载写入隐藏临时文件或专用任务临时位置。
- 任务验证成功后，在同一物理存储后端内原子发布为最终路径，并拒绝覆盖已经存在的目标。
- `list_files`、`search({source:"local"})` 和按路径执行的 `read` 不暴露未完成文件。
- 任务终态为 `failed` 或 `stopped` 时删除残缺临时文件。
- 15 秒观察窗口结束不删除临时文件，因为任务仍在运行。

## 7. 下载记录与文件索引

### 7.1 `qq_file_records`

保存每一次已经完成并原子提交的下载记录：

```text
qq_file_records
  record_id
  agent_qq
  session_key
  conversation_type
  conversation_id
  message_id
  original_filename
  local_path
  storage_backend
  storage_relpath
  size_bytes
  downloaded_at
  deleted_at
```

同一来源可以有多条记录：精确记录路径被移动、重命名或删除后，重新下载会创建新记录。去重时按 `(agent_qq, session_key, message_id, storage_backend)` 查找最新未删除记录，再明面检查其精确 `local_path` 是否仍为普通文件。

`local_path` 保存公开逻辑路径；`storage_backend` 为内部枚举 `linux | host_fallback`；`storage_relpath` 保存相对于对应受信根目录的物理定位，不保存宿主机绝对路径。文件正文只存在对应物理后端，该表不是第二份文件存储。

### 7.2 `qq_file_downloads`

保存下载尝试和进度：

```text
qq_file_downloads
  download_id
  agent_qq
  session_key
  message_id
  conversation_type
  conversation_id
  original_filename
  source_file_id
  status
  bytes_downloaded
  total_bytes
  target_path
  local_path
  storage_backend
  storage_relpath
  created_at
  updated_at
  finished_at
  failure_code
  failure_message
  failure_retryable
```

任务创建时冻结 `storage_backend` 和 `storage_relpath`。此表不以意识流或 Agent 上下文为索引；`list` 必须能够在模型已经忘记 `download_id` 后重新发现任务。

暂不支持：

- 完成事件；
- 未读/已读任务状态；
- `observed_at` 驱动的提醒；
- Core 启动时的主动模型唤醒。

任务历史当前不自动清理，与 AICQ 数据库共同保留；`list` 通过分页限制单次返回量。

### 7.3 `qq_file_messages`

保存已经进入 AICQ 数据库的单文件 QQ 消息索引：

```text
qq_file_messages
  agent_qq
  session_key
  message_id
  conversation_type
  conversation_id
  filename
  extension
  size_bytes
  sender_id
  sender_name
  sent_at
  indexed_at
```

该表只由 QQ 消息同步链路写入；`search({source:"history"})` 只查询该表，不主动向 QQ 拉取更多历史。消息撤回时删除对应索引，消息 ID 回填时同步更新索引键。

## 8. 搜索合同边界

### 8.1 通用规则

- 只匹配文件名，不搜索正文。
- 支持按文件扩展名限定 `file_types`。
- `file_types` 使用不带点、大小写不敏感的扩展名，例如 `pdf`、`docx`、`zip`。
- `query` 与 `file_types` 至少提供一个，同时提供时采用 AND。
- 文件名匹配采用 Unicode NFKC 规范化与默认大小写折叠后的字面子串，不支持 glob 或正则。
- 搜索入参不增加发送者或时间过滤；历史搜索结果直接返回发送者与消息时间。

### 8.2 数据源与会话 scope

`search` 使用两个正交字段：

```text
source=local    当前活动存储后端中实际存在的普通文件
source=history  AICQ 已同步的 QQ 文件消息
```

会话范围统一使用嵌套 scope：

```text
current       当前会话
conversation  显式 private/group + 真实 ID
all           当前 QQ 账号的所有会话
```

规则：

- `scope` 省略时默认为 `{type:"current"}`；当前不在具体 QQ 会话中返回 `no_current_qq_session`。
- `{type:"conversation"}` 和 `{type:"all"}` 是显式范围，可以在 QQ 首页使用。
- 跨会话结果必须标明 `conversation_type` 和 `conversation_id`。

### 8.3 `source=local`

搜索当前活动 QQ 文件存储中的文件名，不解析文件正文；结果返回可直接用于 `read / delete` 的绝对逻辑路径。

### 8.4 `source=history`

只搜索已经同步到 AICQ 数据库的 QQ `file` 消息：

- 不为了搜索而主动向 QQ/NapCat 递归拉取更早历史；
- 结果必须说明搜索范围是 AICQ 当前已同步历史；
- 结果包含 `message_id`、文件名和来源会话；
- 只有记录属于当前活动存储后端且精确物理文件仍存在时才返回 `local_file`；存在时可以按路径读取，否则必须进入对应来源会话，再用 `message_id` 调用 `read` 或 `download(start)`。
- 不把历史 `ref`、`file_id`、`busid` 或数据库内部行号作为公开下载索引。
- 两种数据源的完整参数、返回联合、排序和 cursor 合同见 [search.md](./search.md)。

## 9. 读取合同边界

`qq_file.read` 使用统一入口并自动识别受支持格式。开始读取的来源为：

```text
source.path        当前 QQ 账号 file 根目录中的现有文件
source.message_id  当前 QQ 会话中的文件消息
```

`source.message_id` 的精确记录路径不存在时，复用或启动下载任务并观察最多 15 秒：完成后直接解析；未完成返回 `download_pending` 和任务进度，不创建读取 cursor，下载继续运行。

当前支持范围：

```text
UTF-8 文本 / Markdown / 代码文本
PDF
DOCX
XLSX
PPTX
```

建议统一返回：

```text
file_type
content
structure
next_cursor
```

分页维度：

- 文本：行；
- PDF：页；
- DOCX：标题、段落和表格；
- XLSX：工作表和区域；
- PPTX：幻灯片。

直接读取文档原生结构和文本：

- 扫描版 PDF 不做 OCR，返回 `ocr_required`；
- 不支持格式返回 `unsupported_file_type` 和精确逻辑路径；
- `qq_file` 不自动打开 `computer`；
- Linux 电脑存在时可以按需展开 `computer` 继续处理；后备文件不能由 `computer` 访问或运行。

具体参数和返回体见 [read.md](./read.md)。

## 10. 文件枚举与删除

### 10.1 `list_files`

保留独立 `list_files` 工具。它直接枚举当前活动存储后端的物理目录，并区分：

```text
managed=true   精确匹配 qq_file_records
managed=false  文件存在，但没有精确索引记录
```

被人工移动、重命名或通过其他本地文件能力创建的文件，只要仍在当前活动存储的 QQ 文件根目录内，就可以作为 `managed=false` 文件出现。

具体参数和返回体见 [list_files.md](./list_files.md)。

### 10.2 `delete`

公开删除入口只接受：

```text
delete(path)
```

不接受 `message_id` 删除，不接受目录或批量 glob。

规则：

- 永久删除，不进入回收站；
- 允许删除 `managed=true` 和 `managed=false` 文件；
- 必须是当前 QQ 账号文件根目录内的精确普通文件路径；
- 拒绝目录、glob、路径逃逸和符号链接；
- 若精确匹配 `qq_file_records`，同步设置 `deleted_at`；
- 若文件不存在，返回明确的 `file_not_found`，不猜测移动后的路径。

具体 JSON Schema 和 TS 注入形态见 [delete.md](./delete.md)。

## 11. 合同状态与实现未决事项

### 11.1 `download` 实现细节尚需确认

- 已确认保留 `list.statuses` 多选数组，以及只作用于终态历史的 `offset / limit`。
- 已确认采用“`ok` 表示工具操作、`job.status` 表示下载任务”的双层语义。
- 已确认保留 `DownloadJob.created_at / updated_at / finished_at`。
- 任务历史保留和清理周期。
- Core 重启时活跃下载任务的具体延续/惰性校准机制。
- 下载来源 URL 失效时由 Core 刷新还是由任务终止并要求重新 start。
- 自动重试次数、退避和是否支持 HTTP Range 断点续传。
- 下载完成验证只依赖字节数，还是还需要 adapter 元数据校验。

### 11.2 `read` 已确认

- 使用 `{source:{path}|{message_id}, selection?} | {cursor}`，不增加 `action="start" / "continue"`；
- `message_id` 只在当前会话定位；精确本地路径不存在时复用或启动下载并观察最多 15 秒；
- 下载未完成返回 `outcome=download_pending` 和进度，任务继续运行；只有开始返回内容后才创建读取 cursor；
- 单次正文固定 8000 字符，不把 `max_chars` 暴露为工具参数。
- 直接解析默认上限为 256 MiB；超限时保留文件，只有 Linux 后端文件可以按需使用 `computer`。
- 游标采用跨会话、跨 Core 重启的无状态认证形态，不设计 cursor 列表。
- DOCX 只把主文档标题、段落和表格计入 block。
- 宏启用 Office 文件暂不支持直接读取。

### 11.3 `list_files` 已确认

- `scope` 省略时默认当前会话，并保留显式 `conversation / all` 两种跨会话范围；
- 始终递归枚举普通文件，不增加目录条目或 `recursive` 参数；
- 固定按 `relative_path` 升序分页，并采用文件系统实时弱一致语义；
- `managed` 只表示精确路径记录，不代表文件内容经过 hash 验证；
- 单页默认 50、最大 200，使用无状态 cursor 继续。

### 11.4 `search` 已确认

- `source=local/history` 是首次搜索必填字段，一次调用只搜索一个数据源；
- query、file_types、limit 和嵌套 scope 只声明一次；`query / file_types` 至少提供一个，两者同时存在时采用 AND；
- query 对完整 basename 做 NFKC + Unicode 大小写折叠后的字面子串匹配，不支持 glob 或正则；
- 两种数据源都先按完整文件名相等、前缀、子串排序；local 再按 `relative_path` 升序，history 再按消息有效时间倒序；
- 扩展名采用最后一个点后的后缀，单一前导点 dotfile 视为无扩展名；
- 单页默认 50、最大 200，使用绑定 source、scope、条件与排序位置的无状态 cursor；
- 只搜索 `content_segments` 中真实 file segment，不从消息正文反推，也不向 QQ 补拉历史；
- 每条结果返回发送者、QQ 消息时间、`in_current_session` 和可选精确 `local_file`；
- 跨会话历史结果必须显式进入来源会话后，再用 `message_id` 调用 `read` 或 `download(start)`；
- 使用带 `agent_qq` 的 AICQ 派生索引，拒绝把归属不明确的旧记录混入当前账号搜索。

### 11.5 `delete` 已确认

- 唯一入参为精确绝对 `path`，不增加确认、force、递归或 scope 参数；
- 当前会话不是前置条件，允许删除当前 QQ 账号 file 根目录中的 managed / unmanaged 普通文件；
- 路径不存在返回 `file_not_found`，不采用幂等成功语义；
- 活跃下载占用最终路径时返回 `file_busy + blocking_download_id`，不会隐式 stop；
- 文件系统删除成功但数据库记录未同步时仍返回成功，并附带 `record_state_unsynchronized` warning；
- 成功只删除精确目录项，不清理空目录、硬链接、复制副本或其他进程打开的句柄。

### 11.6 存储路由已确认

- 公开路径始终使用 `/home/agent/qq/...` Linux 形态，不返回或接受 Windows 绝对路径；
- 只有明确确认 `AICQ-Workspace` 不存在时才选择 `host_fallback`；停止、启动失败、版本不匹配和探测不确定都不触发降级；
- 后备目录固定在 `{project_root}\cache\qq_file_fallback\home\agent\qq\...`，不承诺持久化；
- Linux 后来存在时立即成为新操作的活动后端，不迁移或合并后备缓存；
- 下载任务创建时冻结物理后端，15 秒观察结束不切换、不取消；
- 工具 JSON Schema、正式 TS 形态和返回 Schema 不增加 `storage_backend`，后端身份只保存在内部记录；
- 后备文件可以由 `qq_file.read` 解析受支持格式，但不能由 `computer` 访问或运行。

### 11.7 工具合同覆盖状态

`download / read / list_files / search / delete` 五个公开工具均已有已确认的参数、正式 TS 注入形态和返回合同。实现位于 `src/platforms/qq/files` 与 `src/platforms/qq/tools/qq_file`，数据库表由 `init_db` 创建，自动化验证位于 `tests/test_qq_file.py`。

当前模型可见合同尺寸：

- 五个工具完整 namespace 为 3207 个字符、UTF-8 5155 字节；
- 其中 `search` 为 707 个字符、UTF-8 1135 字节；
- `qq-file/SKILL.md` 原始文件为 2133 个字符、UTF-8 4059 字节；去掉 YAML frontmatter 后的正文为 2018 个字符、UTF-8 3846 字节；
- namespace 与 Skill 正文的纯文本合计为 5225 个字符、UTF-8 9001 字节，不包含实际 Skill 注入器可能增加的包装文本。

## 12. 实现衔接点

- QQ namespace manifest：`src/platforms/qq/tools_manifest.yaml`
- QQ 文件消息 metadata 构造：`src/platforms/qq/adapter/segments.py`
- QQ 历史消息持久化：`src/database.py` 的 `chat_messages`
- 当前通用 Linux namespace：`src/tools/modules.yaml` 中的 `computer`
- Workspace 服务：`src/workspace/service.py`
- Workspace 15 秒命令观察合同：`src/workspace/tools/command.py`
- Python-first 工具合同：`src/tools/contract.py`
- 模型可见 TS-like 签名生成：`src/tools/prompt_signatures.py`
- namespace 模型注入：`src/llm/core/tool_calling/aic_action.py`
- `requirements.txt` 已包含 `pypdf`、`python-docx`、`openpyxl` 和 `python-pptx`；解析运行在禁用网络访问的受限工作进程中。
- `qq_file.read` 对未经信任的文件正文启用 `RESULT_CDATA = True`，避免正文破坏意识流 XML 结构。
- QQ 文件实现：`src/platforms/qq/files/`
- 五个公开工具：`src/platforms/qq/tools/qq_file/`
- 运行时 Skill：`src/skills/qq-file/SKILL.md`
- 合同、行为和 Linux bridge 测试：`tests/test_qq_file.py`

本设计不要求公开 `computer` 工具来完成正常 QQ 文件操作。Linux 后端可以复用其安全路径、bridge、后台任务和分页设计；宿主机后备实现必须使用独立受信根目录和等价的路径安全检查。
