# `qq_file.download`

[← 主文档](./README.md) · [Skill](./qq-file/SKILL.md) · [download](./download.md) · [read](./read.md) · [list_files](./list_files.md) · [search](./search.md) · [delete](./delete.md)

> 本合同已实现。JSON Schema 和 TS 签名由当前仓库的 `ToolContract` 生成规则生成，并由合同测试逐字段校验。

## 1. action 语义

| action | 是否需要当前会话 | 行为 |
|---|---:|---|
| `start` | 是 | 按当前会话的 `message_id` 启动下载并观察最多 15 秒 |
| `poll` | 否 | 立即查询一个 `download_id`，不等待，也不包含其他任务 |
| `list` | 否 | 跨当前 QQ 账号的所有会话发现活跃和最近终态任务 |
| `stop` | 否 | 显式停止一个仍在运行的下载任务 |

`start` 在创建任务前按[主文档的存储路由合同](./README.md#32-后端选择)选择物理后端：Linux 电脑存在时使用 Linux；只有明确确认 Linux 电脑不存在时才使用项目内的宿主机后备目录。任务创建后冻结后端，15 秒观察窗口结束、`poll` 和 `list` 都不会改变写入目标。

`target_path` 和 `local_path` 始终是 `/home/agent/qq/...` Linux 形态的逻辑路径，不返回宿主机绝对路径。后端身份和物理相对路径只保存在内部下载记录中，不增加公开参数或返回字段。

## 2. JSON Schema declaration

```json
{
  "name": "download",
  "description": "启动、查询、列出或停止 QQ 文件下载任务。",
  "parameters": {
    "$defs": {
      "DownloadListArgs": {
        "additionalProperties": false,
        "properties": {
          "action": {
            "const": "list",
            "description": "列出当前 QQ 账号的跨会话下载任务。",
            "type": "string"
          },
          "statuses": {
            "anyOf": [
              {
                "items": {
                  "enum": [
                    "queued",
                    "resolving",
                    "downloading",
                    "verifying",
                    "completed",
                    "failed",
                    "stopped"
                  ],
                  "type": "string"
                },
                "type": "array"
              },
              {
                "type": "null"
              }
            ],
            "default": null,
            "description": "可选状态过滤，同时作用于活跃任务和终态任务；省略时返回全部活跃任务和最近的终态任务。",
            "uniqueItems": true
          },
          "offset": {
            "default": 0,
            "description": "终态任务历史的分页偏移，默认 0；活跃任务始终全部返回。",
            "minimum": 0,
            "type": "integer"
          },
          "limit": {
            "default": 20,
            "description": "终态任务历史最多返回数量，默认 20，最大 100；活跃任务不受此限制。",
            "maximum": 100,
            "minimum": 1,
            "type": "integer"
          }
        },
        "required": [
          "action"
        ],
        "type": "object"
      },
      "DownloadPollArgs": {
        "additionalProperties": false,
        "properties": {
          "action": {
            "const": "poll",
            "description": "立即查询一个下载任务的当前状态和进度。",
            "type": "string"
          },
          "download_id": {
            "description": "start 或 list 返回的下载任务 ID。",
            "minLength": 1,
            "type": "string"
          }
        },
        "required": [
          "action",
          "download_id"
        ],
        "type": "object"
      },
      "DownloadStartArgs": {
        "additionalProperties": false,
        "properties": {
          "action": {
            "const": "start",
            "description": "启动当前 QQ 会话中指定文件消息的下载。",
            "type": "string"
          },
          "message_id": {
            "description": "当前 QQ 会话中的文件消息 ID。",
            "minLength": 1,
            "type": "string",
            "x-coerce-integer": true
          }
        },
        "required": [
          "action",
          "message_id"
        ],
        "type": "object"
      },
      "DownloadStopArgs": {
        "additionalProperties": false,
        "properties": {
          "action": {
            "const": "stop",
            "description": "显式停止一个仍在运行的下载任务。",
            "type": "string"
          },
          "download_id": {
            "description": "start 或 list 返回的下载任务 ID。",
            "minLength": 1,
            "type": "string"
          }
        },
        "required": [
          "action",
          "download_id"
        ],
        "type": "object"
      }
    },
    "discriminator": {
      "mapping": {
        "list": "#/$defs/DownloadListArgs",
        "poll": "#/$defs/DownloadPollArgs",
        "start": "#/$defs/DownloadStartArgs",
        "stop": "#/$defs/DownloadStopArgs"
      },
      "propertyName": "action"
    },
    "oneOf": [
      {
        "$ref": "#/$defs/DownloadStartArgs"
      },
      {
        "$ref": "#/$defs/DownloadPollArgs"
      },
      {
        "$ref": "#/$defs/DownloadListArgs"
      },
      {
        "$ref": "#/$defs/DownloadStopArgs"
      }
    ]
  }
}
```

## 3. 模型正式可见的 TypeScript-like 形态

当 `qq_file` 已打开且当前只加入此工具时，namespace 增量形态为：

```ts
<namespace name="qq_file" active="true">// 启动、查询、列出或停止 QQ 文件下载任务。
download(args: {
  action: "start"; // 启动当前 QQ 会话中指定文件消息的下载。
  message_id: string; // 当前 QQ 会话中的文件消息 ID。
} | {
  action: "poll"; // 立即查询一个下载任务的当前状态和进度。
  download_id: string; // start 或 list 返回的下载任务 ID。
} | {
  action: "list"; // 列出当前 QQ 账号的跨会话下载任务。
  statuses?: ("queued" | "resolving" | "downloading" | "verifying" | "completed" | "failed" | "stopped")[]; // 可选状态过滤，同时作用于活跃任务和终态任务；省略时返回全部活跃任务和最近的终态任务。 数组项不可重复
  offset?: number; // 终态任务历史的分页偏移，默认 0；活跃任务始终全部返回。
  limit?: number; // 终态任务历史最多返回数量，默认 20，最大 100；活跃任务不受此限制。 范围 1~100
} | {
  action: "stop"; // 显式停止一个仍在运行的下载任务。
  download_id: string; // start 或 list 返回的下载任务 ID。
})</namespace>
```

按当前仓库的实际生成与 namespace 包装规则测得：

- 单独函数签名（含工具 description）：653 个字符，UTF-8 1093 字节。
- 加当前 namespace 外壳：705 个字符，UTF-8 1145 字节。
- 不包含配套 Skill 的上下文；Skill 在按需加载时另计。

## 4. 实际调用信封

```json
{
  "namespace": "qq_file",
  "name": "download",
  "arguments": {
    "action": "start",
    "message_id": "123456789"
  }
}
```

## 5. 返回合同原则

返回体统一用：

```text
ok: true   -> 本次工具操作成功执行；任务本身仍可能是 failed
ok: false  -> 工具无法完成这次 start / poll / list / stop 操作
```

因此，成功查询到一个 `status="failed"` 的任务仍返回 `ok: true`，失败原因位于 `job.failure`。这可以明确区分“下载任务失败”和“工具调用失败”。

`start` 的 `outcome` 只有三种：

| outcome | 含义 |
|---|---|
| `started` | 本次新建了任务；查看 `job.status` 判断 15 秒内已经完成、失败，还是仍在运行 |
| `already_downloading` | 同一来源已有活跃任务，不创建第二个任务，立即返回原任务 |
| `already_exists` | 来源记录中的精确 `local_path` 仍是普通文件，不创建任务，返回现有文件 |

`observation_timeout` 只出现在带 `job` 的 `start` 成功结果中：

- 新任务观察 15 秒后仍为活跃状态：`true`；
- 新任务在 15 秒内进入终态：`false`；
- 命中已有活跃任务并立即返回：`false`。

`DownloadJob` 同时保留两个路径字段：

- `target_path`：任务准备写入的最终目标路径；活跃、失败或停止时不能据此声称文件已经可用；
- `local_path`：只有 `status="completed"` 时才是非 `null`，表示已经原子提交且可读取的本地文件。

其余字段不变量：

- `total_bytes` 未知时为 `null`；
- `progress_percent` 仅在 `total_bytes` 已知时计算，否则为 `null`；
- `failure` 只有 `status="failed"` 时非 `null`；
- `finished_at` 只有终态 `completed / failed / stopped` 时非 `null`；
- `stop` 返回 `outcome="stopped"` 前，必须已经结束任务并删除临时文件；若调用时任务已经是终态，返回 `outcome="already_terminal"` 和该任务的真实终态；
- `list.active` 返回所有符合状态过滤的活跃任务，不分页；`offset / limit` 只分页 `list.terminal`。

## 6. 返回 JSON Schema（第一版）

> 这是 `DownloadResult` 的运行时合同。当前通过实现分支与合同测试保证字段和不变量；`ToolContract.result_model` 本身不负责运行时返回校验。

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
    "DownloadedFile": {
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
        "local_path": {
          "minLength": 1,
          "type": "string"
        },
        "size_bytes": {
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
        "local_path",
        "size_bytes",
        "downloaded_at"
      ],
      "type": "object"
    },
    "DownloadFailure": {
      "additionalProperties": false,
      "properties": {
        "code": {
          "enum": [
            "source_unavailable",
            "file_too_large",
            "insufficient_disk_space",
            "transport_error",
            "write_failed",
            "size_mismatch",
            "verification_failed",
            "download_interrupted",
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
        }
      },
      "required": ["code", "message", "retryable"],
      "type": "object"
    },
    "DownloadJob": {
      "additionalProperties": false,
      "properties": {
        "download_id": {
          "minLength": 1,
          "type": "string"
        },
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
        "status": {
          "enum": [
            "queued",
            "resolving",
            "downloading",
            "verifying",
            "completed",
            "failed",
            "stopped"
          ],
          "type": "string"
        },
        "bytes_downloaded": {
          "minimum": 0,
          "type": "integer"
        },
        "total_bytes": {
          "anyOf": [
            {
              "minimum": 0,
              "type": "integer"
            },
            {
              "type": "null"
            }
          ]
        },
        "progress_percent": {
          "anyOf": [
            {
              "maximum": 100,
              "minimum": 0,
              "type": "number"
            },
            {
              "type": "null"
            }
          ]
        },
        "target_path": {
          "minLength": 1,
          "type": "string"
        },
        "local_path": {
          "anyOf": [
            {
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ]
        },
        "created_at": {
          "format": "date-time",
          "type": "string"
        },
        "updated_at": {
          "format": "date-time",
          "type": "string"
        },
        "finished_at": {
          "anyOf": [
            {
              "format": "date-time",
              "type": "string"
            },
            {
              "type": "null"
            }
          ]
        },
        "failure": {
          "anyOf": [
            {
              "$ref": "#/$defs/DownloadFailure"
            },
            {
              "type": "null"
            }
          ]
        }
      },
      "required": [
        "download_id",
        "message_id",
        "conversation",
        "original_filename",
        "status",
        "bytes_downloaded",
        "total_bytes",
        "progress_percent",
        "target_path",
        "local_path",
        "created_at",
        "updated_at",
        "finished_at",
        "failure"
      ],
      "type": "object"
    },
    "ToolErrorDetails": {
      "additionalProperties": false,
      "properties": {
        "size_bytes": {
          "minimum": 0,
          "type": "integer"
        },
        "limit_bytes": {
          "minimum": 0,
          "type": "integer"
        },
        "required_bytes": {
          "minimum": 0,
          "type": "integer"
        },
        "available_bytes": {
          "minimum": 0,
          "type": "integer"
        },
        "current_status": {
          "enum": [
            "queued",
            "resolving",
            "downloading",
            "verifying",
            "completed",
            "failed",
            "stopped"
          ],
          "type": "string"
        }
      },
      "type": "object"
    },
    "ToolError": {
      "additionalProperties": false,
      "properties": {
        "code": {
          "enum": [
            "no_current_qq_session",
            "message_not_found",
            "not_file_message",
            "qq_adapter_unavailable",
            "download_not_found",
            "file_too_large",
            "insufficient_disk_space",
            "invalid_download_state",
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
        "details": {
          "$ref": "#/$defs/ToolErrorDetails"
        }
      },
      "required": ["code", "message", "retryable"],
      "type": "object"
    },
    "StartAlreadyExistsResult": {
      "additionalProperties": false,
      "properties": {
        "ok": {
          "const": true,
          "type": "boolean"
        },
        "action": {
          "const": "start",
          "type": "string"
        },
        "outcome": {
          "const": "already_exists",
          "type": "string"
        },
        "file": {
          "$ref": "#/$defs/DownloadedFile"
        }
      },
      "required": ["ok", "action", "outcome", "file"],
      "type": "object"
    },
    "StartJobResult": {
      "additionalProperties": false,
      "properties": {
        "ok": {
          "const": true,
          "type": "boolean"
        },
        "action": {
          "const": "start",
          "type": "string"
        },
        "outcome": {
          "enum": ["started", "already_downloading"],
          "type": "string"
        },
        "observation_timeout": {
          "type": "boolean"
        },
        "job": {
          "$ref": "#/$defs/DownloadJob"
        }
      },
      "required": [
        "ok",
        "action",
        "outcome",
        "observation_timeout",
        "job"
      ],
      "type": "object"
    },
    "PollResult": {
      "additionalProperties": false,
      "properties": {
        "ok": {
          "const": true,
          "type": "boolean"
        },
        "action": {
          "const": "poll",
          "type": "string"
        },
        "job": {
          "$ref": "#/$defs/DownloadJob"
        }
      },
      "required": ["ok", "action", "job"],
      "type": "object"
    },
    "ListResult": {
      "additionalProperties": false,
      "properties": {
        "ok": {
          "const": true,
          "type": "boolean"
        },
        "action": {
          "const": "list",
          "type": "string"
        },
        "active": {
          "items": {
            "$ref": "#/$defs/DownloadJob"
          },
          "type": "array"
        },
        "terminal": {
          "items": {
            "$ref": "#/$defs/DownloadJob"
          },
          "type": "array"
        },
        "offset": {
          "minimum": 0,
          "type": "integer"
        },
        "limit": {
          "maximum": 100,
          "minimum": 1,
          "type": "integer"
        },
        "terminal_has_more": {
          "type": "boolean"
        },
        "next_offset": {
          "anyOf": [
            {
              "minimum": 0,
              "type": "integer"
            },
            {
              "type": "null"
            }
          ]
        }
      },
      "required": [
        "ok",
        "action",
        "active",
        "terminal",
        "offset",
        "limit",
        "terminal_has_more",
        "next_offset"
      ],
      "type": "object"
    },
    "StopResult": {
      "additionalProperties": false,
      "properties": {
        "ok": {
          "const": true,
          "type": "boolean"
        },
        "action": {
          "const": "stop",
          "type": "string"
        },
        "outcome": {
          "enum": ["stopped", "already_terminal"],
          "type": "string"
        },
        "job": {
          "$ref": "#/$defs/DownloadJob"
        }
      },
      "required": ["ok", "action", "outcome", "job"],
      "type": "object"
    },
    "ErrorResult": {
      "additionalProperties": false,
      "properties": {
        "ok": {
          "const": false,
          "type": "boolean"
        },
        "action": {
          "enum": ["start", "poll", "list", "stop"],
          "type": "string"
        },
        "error": {
          "$ref": "#/$defs/ToolError"
        }
      },
      "required": ["ok", "action", "error"],
      "type": "object"
    }
  },
  "oneOf": [
    {
      "$ref": "#/$defs/StartAlreadyExistsResult"
    },
    {
      "$ref": "#/$defs/StartJobResult"
    },
    {
      "$ref": "#/$defs/PollResult"
    },
    {
      "$ref": "#/$defs/ListResult"
    },
    {
      "$ref": "#/$defs/StopResult"
    },
    {
      "$ref": "#/$defs/ErrorResult"
    }
  ]
}
```

## 7. 典型运行时返回

新任务在 15 秒内完成：

```json
{
  "ok": true,
  "action": "start",
  "outcome": "started",
  "observation_timeout": false,
  "job": {
    "download_id": "qfd_01K4...",
    "message_id": "123456789",
    "conversation": {
      "type": "group",
      "id": "1090411227"
    },
    "original_filename": "example.pdf",
    "status": "completed",
    "bytes_downloaded": 7340032,
    "total_bytes": 7340032,
    "progress_percent": 100,
    "target_path": "/home/agent/qq/213628848/file/group_1090411227/example.pdf",
    "local_path": "/home/agent/qq/213628848/file/group_1090411227/example.pdf",
    "created_at": "2026-08-30T10:00:00Z",
    "updated_at": "2026-08-30T10:00:04Z",
    "finished_at": "2026-08-30T10:00:04Z",
    "failure": null
  }
}
```

观察 15 秒后仍在下载：

```json
{
  "ok": true,
  "action": "start",
  "outcome": "started",
  "observation_timeout": true,
  "job": {
    "download_id": "qfd_01K4...",
    "message_id": "123456789",
    "conversation": {
      "type": "group",
      "id": "1090411227"
    },
    "original_filename": "example.zip",
    "status": "downloading",
    "bytes_downloaded": 193986560,
    "total_bytes": 734003200,
    "progress_percent": 26.43,
    "target_path": "/home/agent/qq/213628848/file/group_1090411227/example.zip",
    "local_path": null,
    "created_at": "2026-08-30T10:00:00Z",
    "updated_at": "2026-08-30T10:00:15Z",
    "finished_at": null,
    "failure": null
  }
}
```

已有记录的精确文件仍存在：

```json
{
  "ok": true,
  "action": "start",
  "outcome": "already_exists",
  "file": {
    "message_id": "123456789",
    "conversation": {
      "type": "group",
      "id": "1090411227"
    },
    "original_filename": "example.pdf",
    "local_path": "/home/agent/qq/213628848/file/group_1090411227/example.pdf",
    "size_bytes": 7340032,
    "downloaded_at": "2026-08-30T10:00:04Z"
  }
}
```

任务已经被模型遗忘后，`list` 仍可发现它：

```json
{
  "ok": true,
  "action": "list",
  "active": [],
  "terminal": [
    {
      "download_id": "qfd_01K4...",
      "message_id": "123456789",
      "conversation": {
        "type": "group",
        "id": "1090411227"
      },
      "original_filename": "example.zip",
      "status": "completed",
      "bytes_downloaded": 734003200,
      "total_bytes": 734003200,
      "progress_percent": 100,
      "target_path": "/home/agent/qq/213628848/file/group_1090411227/example.zip",
      "local_path": "/home/agent/qq/213628848/file/group_1090411227/example.zip",
      "created_at": "2026-08-30T10:00:00Z",
      "updated_at": "2026-08-30T10:02:31Z",
      "finished_at": "2026-08-30T10:02:31Z",
      "failure": null
    }
  ],
  "offset": 0,
  "limit": 20,
  "terminal_has_more": false,
  "next_offset": null
}
```

工具层错误：

```json
{
  "ok": false,
  "action": "start",
  "error": {
    "code": "no_current_qq_session",
    "message": "当前不在任何具体 QQ 会话中，无法按 message_id 下载文件。",
    "retryable": false
  }
}
```

## 8. 返回合同与模型上下文消耗

当前正式注入的 TypeScript-like 签名仍然只有 11.3 节的参数形态，不包含 `DownloadResult` 返回类型。也就是说：

- 11.6 节的返回 JSON Schema 不产生常驻提示词消耗；
- `DownloadJob`、错误枚举和示例不会随 namespace 一起注入；
- 工具真正调用后，实际返回的那一份 JSON 会作为工具结果进入上下文；
- `list` 不返回内部 `session_key`、当前 QQ 账号内部索引、临时文件路径、adapter 原始响应或调试堆栈。
