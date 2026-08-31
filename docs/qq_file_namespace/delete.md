# `qq_file.delete`

[← 主文档](./README.md) · [Skill](./qq-file/SKILL.md) · [download](./download.md) · [read](./read.md) · [list_files](./list_files.md) · [search](./search.md) · [delete](./delete.md)

> 本合同已实现。参数 JSON Schema 和 TypeScript-like 签名遵循当前仓库的 `ToolContract` 生成规则，并由合同测试逐字段校验。

## 1. 职责与路径边界

`delete` 永久删除当前 QQ 账号文件根目录中的一个精确普通文件：

```text
/home/agent/qq/{current_agent_qq}/file/
```

它只有一个公开参数 `path`。不接受 `message_id`、`download_id`、scope、目录、文件数组、glob、递归开关、确认开关或回收站选项。

调用不依赖当前所在会话：

- 可以在 QQ 首页使用；
- 可以删除当前会话、其他会话目录或 file 根目录直属的普通文件；
- 当前真实 QQ 账号由 runtime 提供，不能通过参数指定；
- 无法取得当前 QQ 账号身份或当前活动文件存储不可用时，删除不会开始。

允许删除：

- `managed=true` 的已下载文件；
- `managed=false` 的人工改名、移动或通过其他本地文件能力创建的文件；
- 普通隐藏文件，但不包括 `qq_file` 保留的内部任务文件。

拒绝删除：

- 当前 QQ 账号 file 根目录外的任何路径；
- file 根目录本身或任意目录；
- 最终路径是符号链接的条目；
- 任一父路径经过符号链接解析的条目；
- socket、FIFO、设备等非普通文件；
- 下载临时文件、路径锁或其他 `qq_file` 保留内部文件。

## 2. 唯一调用形态

```json
{
  "path": "/home/agent/qq/213628848/file/group_1090411227/report.pdf"
}
```

没有 `action="delete"` 包装，也不允许同时传其他字段。

## 3. 参数 JSON Schema declaration

```json
{
  "name": "delete",
  "description": "永久删除当前 QQ 账号文件根目录中的一个本地普通文件。",
  "parameters": {
    "additionalProperties": false,
    "properties": {
      "path": {
        "description": "要永久删除的绝对 Linux 普通文件路径，必须位于当前 QQ 账号的 file 根目录内。",
        "minLength": 1,
        "type": "string"
      }
    },
    "required": ["path"],
    "type": "object"
  }
}
```

绝对路径、Linux 路径形态、UTF-8 编码后的 Linux 路径字节限制、受管根目录和文件类型由业务校验完成。JSON Schema 只承担非空字符串和额外字段约束。

## 4. 模型正式可见的 TypeScript-like 形态

```ts
// 永久删除当前 QQ 账号文件根目录中的一个本地普通文件。
delete(args: {
  path: string; // 要永久删除的绝对 Linux 普通文件路径，必须位于当前 QQ 账号的 file 根目录内。
})
```

当前实际测得：

- `delete` 单独函数签名：115 个字符，UTF-8 221 字节；
- 五个工具完整 namespace：3207 个字符，UTF-8 5155 字节；
- 返回 Schema 不进入常驻提示词。

## 5. 精确路径与删除语义

`path` 是 `/home/agent/qq/...` Linux 形态的逻辑路径。实际删除目标由[主文档的存储路由合同](./README.md#32-后端选择)决定：Linux 电脑存在时删除 Linux 文件；只有明确确认 Linux 电脑不存在时才删除宿主机后备文件。工具不接受或返回宿主机绝对路径，也不跨两个后端猜测同名文件。

输入路径必须满足：

- 是以 `/` 开始的绝对 Linux 路径；
- 不接受 `~`、环境变量、`file://` URI 或 Windows 路径；
- 不包含 NUL，也不包含 `.` 或 `..` 路径组件；
- 按路径组件判断是否位于当前 QQ 账号 file 根目录内，不能使用容易产生前缀混淆的字符串判断；
- 使用 Linux 大小写敏感语义，不进行文件名大小写修复；
- `*`、`?`、`[]` 等字符永远不作为 glob 展开；若它们出现在文件名中，只按字面路径查找。

删除目标以调用执行时当前活动存储后端中，该精确逻辑路径的物理状态为准：

- 路径不存在时返回 `file_not_found`，不扫描其他目录、不查找同名文件，也不把不存在视为成功；
- 路径已经人工移动或改名时，旧路径返回 `file_not_found`；必须重新通过 `list_files` 或 `search({source:"local"})` 取得新路径后显式删除；
- 成功删除只移除该精确目录项，不删除空的会话目录；
- 若文件存在其他硬链接或复制副本，它们继续存在；`delete` 不做 hash 搜索或内容级清理；
- 成功返回表示该精确路径已经被永久 unlink，不存在回收站或撤销 token。

删除前读取并返回目标的 `name` 和 `size_bytes`，便于确认实际删除对象。调用方不能通过这些返回字段恢复文件内容。

## 6. 活跃下载、并发与打开句柄

下载完成前使用隐藏临时文件并在最终提交时原子发布。`download` 与 `delete` 共享路径级锁：

- 若精确 path 是非终态下载任务已经保留的最终目标，`delete` 返回 `file_busy` 和 `blocking_download_id`，不停止下载，也不删除任何文件；
- 需要删除该目标时，先显式 `download(stop)`，等待任务进入终态，再重新调用 `delete(path)`；
- `delete` 不接受 `force`，也不会隐式停止任务；
- 内部下载临时路径即使被猜中也返回 `protected_internal_path`。

已知的 QQ 下载写入使用上述锁。Linux 后端遵循原生 unlink 语义：路径会立即消失，但已打开句柄可能继续访问旧 inode，磁盘空间也可能到最后一个句柄关闭后才释放。宿主机后备层遵循 Windows 文件共享和删除语义；被其他进程以不允许删除的方式占用时返回明确文件系统错误，不扫描或终止其他进程。

读取是同步操作；若实现为短期共享文件租约，删除可以等待安全检查完成或返回 `file_busy`，但不能让删除与同一路径的受管写入提交交错。

## 7. `qq_file_records` 与部分状态

文件系统是删除结果的事实来源，数据库记录是可修复索引：

1. 删除前按当前 `agent_qq + storage_backend + 精确 local_path + deleted_at IS NULL` 查询下载记录；
2. 安全检查通过后永久 unlink 文件；
3. 若存在精确记录，将其 `deleted_at` 设置为实际删除时间；
4. 不删除 QQ `chat_messages`、`qq_file_downloads` 或历史搜索索引。

下载目标选择、提交和删除使用同一进程内路径锁；数据库索引允许保留同一路径的历史记录，删除时把该精确路径的全部未删除记录统一标记为已删除。

成功结果中的字段表示删除前状态：

```text
was_managed=true   删除前精确匹配一条有效下载记录，source 非 null
was_managed=false  已确认没有精确记录，source 为 null
was_managed=null   本次无法读取记录状态，source 为 null
```

SQLite 与当前活动文件存储不能形成一个原子事务。若物理删除已成功但记录读取或更新失败：

- 仍返回 `ok=true, deleted=true`，因为精确路径已经消失；
- 返回 `record_state_unsynchronized` warning；
- 不返回可重试的删除失败，避免重试时把 `file_not_found` 误解为文件仍在；
- 后续记录校准可以根据精确路径不存在补齐 `deleted_at`，但业务结果不依赖该校准完成。

删除后，同一来源消息再次调用 `download(start)` 时，精确路径存在性检查失败，因此允许重新下载并建立新的当前映射。`search({source:"history"})` 仍可找到原 QQ 文件消息；其 `local_file` 变为 `null`。人工移动或复制的等价文件不受记录更新影响。

## 8. 返回 JSON Schema（第一版）

```json
{
  "$defs": {
    "Conversation": {
      "additionalProperties": false,
      "properties": {
        "type": {
          "enum": ["private", "group"],
          "type": "string"
        },
        "id": {
          "minLength": 1,
          "type": "string"
        }
      },
      "required": ["type", "id"],
      "type": "object"
    },
    "DeleteSource": {
      "additionalProperties": false,
      "properties": {
        "message_id": {
          "minLength": 1,
          "type": "string"
        },
        "conversation": {
          "$ref": "#/$defs/Conversation"
        },
        "original_filename": {
          "minLength": 1,
          "type": "string"
        },
        "recorded_size_bytes": {
          "minimum": 0,
          "type": "integer"
        },
        "downloaded_at": {
          "format": "date-time",
          "type": "string"
        }
      },
      "required": [
        "message_id",
        "conversation",
        "original_filename",
        "recorded_size_bytes",
        "downloaded_at"
      ],
      "type": "object"
    },
    "DeleteWarning": {
      "additionalProperties": false,
      "properties": {
        "code": {
          "const": "record_state_unsynchronized",
          "type": "string"
        },
        "message": {
          "minLength": 1,
          "type": "string"
        }
      },
      "required": ["code", "message"],
      "type": "object"
    },
    "DeleteSuccess": {
      "additionalProperties": false,
      "properties": {
        "ok": {
          "const": true,
          "type": "boolean"
        },
        "deleted": {
          "const": true,
          "type": "boolean"
        },
        "path": {
          "minLength": 1,
          "type": "string"
        },
        "name": {
          "minLength": 1,
          "type": "string"
        },
        "size_bytes": {
          "minimum": 0,
          "type": "integer"
        },
        "was_managed": {
          "anyOf": [
            {
              "type": "boolean"
            },
            {
              "type": "null"
            }
          ]
        },
        "source": {
          "anyOf": [
            {
              "$ref": "#/$defs/DeleteSource"
            },
            {
              "type": "null"
            }
          ]
        },
        "deleted_at": {
          "format": "date-time",
          "type": "string"
        },
        "warnings": {
          "items": {
            "$ref": "#/$defs/DeleteWarning"
          },
          "type": "array"
        }
      },
      "required": [
        "ok",
        "deleted",
        "path",
        "name",
        "size_bytes",
        "was_managed",
        "source",
        "deleted_at",
        "warnings"
      ],
      "type": "object"
    },
    "DeleteError": {
      "additionalProperties": false,
      "properties": {
        "code": {
          "enum": [
            "invalid_path",
            "path_outside_qq_file_root",
            "file_not_found",
            "directory_not_allowed",
            "symlink_not_allowed",
            "not_a_regular_file",
            "protected_internal_path",
            "file_busy",
            "permission_denied",
            "filesystem_unavailable",
            "runtime_unavailable",
            "internal_error"
          ],
          "type": "string"
        },
        "message": {
          "minLength": 1,
          "type": "string"
        },
        "retryable": {
          "type": "boolean"
        },
        "blocking_download_id": {
          "anyOf": [
            {
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ]
        }
      },
      "required": [
        "code",
        "message",
        "retryable",
        "blocking_download_id"
      ],
      "type": "object"
    },
    "DeleteErrorResult": {
      "additionalProperties": false,
      "properties": {
        "ok": {
          "const": false,
          "type": "boolean"
        },
        "error": {
          "$ref": "#/$defs/DeleteError"
        }
      },
      "required": ["ok", "error"],
      "type": "object"
    }
  },
  "anyOf": [
    {
      "$ref": "#/$defs/DeleteSuccess"
    },
    {
      "$ref": "#/$defs/DeleteErrorResult"
    }
  ]
}
```

返回不变量：

```text
ok=true   <=> deleted=true
was_managed=true   => source 非 null
was_managed=false  => source=null
was_managed=null   => source=null 且 warnings 包含 record_state_unsynchronized
error.code=file_busy => blocking_download_id 非 null
其他错误              => blocking_download_id=null
```

## 9. 典型运行时返回

永久删除一个精确匹配下载记录的文件：

```json
{
  "ok": true,
  "deleted": true,
  "path": "/home/agent/qq/213628848/file/group_1090411227/report.pdf",
  "name": "report.pdf",
  "size_bytes": 1048576,
  "was_managed": true,
  "source": {
    "message_id": "1803394108",
    "conversation": {
      "type": "group",
      "id": "1090411227"
    },
    "original_filename": "report.pdf",
    "recorded_size_bytes": 1048576,
    "downloaded_at": "2026-08-30T14:21:00+08:00"
  },
  "deleted_at": "2026-08-31T10:00:00+08:00",
  "warnings": []
}
```

文件已删除，但本次无法同步下载记录状态：

```json
{
  "ok": true,
  "deleted": true,
  "path": "/home/agent/qq/213628848/file/private_123456789/notes.txt",
  "name": "notes.txt",
  "size_bytes": 4096,
  "was_managed": null,
  "source": null,
  "deleted_at": "2026-08-31T10:05:00+08:00",
  "warnings": [
    {
      "code": "record_state_unsynchronized",
      "message": "文件已永久删除，但本次无法读取或更新下载记录状态。"
    }
  ]
}
```

目标正在被下载任务使用：

```json
{
  "ok": false,
  "error": {
    "code": "file_busy",
    "message": "该路径当前由下载任务占用。请先停止对应下载任务后重试。",
    "retryable": true,
    "blocking_download_id": "qfd_01JY..."
  }
}
```

目标路径不存在：

```json
{
  "ok": false,
  "error": {
    "code": "file_not_found",
    "message": "指定的文件路径不存在。请重新枚举或搜索后提供当前精确路径。",
    "retryable": false,
    "blocking_download_id": null
  }
}
```

## 10. 双后端删除与安全实现建议

Linux 后端已在 `WorkspaceService / WorkspaceBackend` 中提供面向单文件的固定结构化操作，不拼接 shell 命令：

1. Windows/Core 侧取得当前真实 QQ 账号并构造允许的 file 根目录；
2. 将已经拆分并校验的相对路径组件通过结构化 stdin / RPC 发送给 Linux helper；
3. Linux 侧从受信根目录 fd 开始，逐级使用 `O_NOFOLLOW`、`openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` 或等价 dirfd 方案遍历父目录；
4. 对最终条目执行 `lstat / fstatat(..., AT_SYMLINK_NOFOLLOW)`，确认是普通文件且不是保留内部路径；
5. 在共享路径锁内使用 `unlinkat` 删除精确目录项，不经过 shell、glob 或字符串命令插值；
6. 返回删除前的文件名、大小和明确 errno 映射；Core 再更新可选数据库记录。

宿主机后备层使用同一逻辑路径校验和路径锁，但把相对组件锚定到 `{project_root}\cache\qq_file_fallback\home\agent`。逐级拒绝符号链接、junction、reparse point 和根目录逃逸，最终只删除普通文件；不能把逻辑路径直接拼接后交给 shell、PowerShell 或通配符 API。

安全不变量：

- 检查和 unlink 都锚定同一个已验证父目录 fd，不能先解析完整字符串路径再交给另一个 shell；
- 即使外部进程并发替换最终目录项，操作也不能跟随符号链接逃出 file 根目录；
- 下载最终原子发布与删除必须使用同一锁域；
- 内部临时文件名称规则只在实现中维护，不出现在模型可见 description 或成功结果中；
- 任何失败都不得退化为递归删除、目录删除或宽泛清理。

## 11. 返回合同与上下文消耗

- 返回 JSON Schema、路径安全规则、记录同步和错误枚举不随 namespace 注入；
- `delete` 常驻增量是 115 个字符、UTF-8 221 字节；
- 五个工具完整 namespace 是 3207 个字符、UTF-8 5155 字节；
- 成功结果只返回删除前轻量 metadata 和可选来源记录，不返回文件内容、哈希、inode、内部临时路径或数据库主键。
