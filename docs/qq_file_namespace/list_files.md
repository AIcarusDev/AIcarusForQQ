# `qq_file.list_files`

[← 主文档](./README.md) · [Skill](./qq-file/SKILL.md) · [download](./download.md) · [read](./read.md) · [list_files](./list_files.md) · [search](./search.md) · [delete](./delete.md)

> 本合同已实现。参数 JSON Schema 和 TypeScript-like 签名由当前仓库的 `ToolContract` 生成规则生成，并由合同测试逐字段校验。

## 1. 职责与枚举范围

`list_files` 直接递归枚举当前活动存储后端中，当前 QQ 账号文件根目录下的本地普通文件：

```text
/home/agent/qq/{current_agent_qq}/file/
```

该路径是公开逻辑根目录。Linux 电脑存在时枚举 Linux；只有明确确认 Linux 电脑不存在时才枚举项目内的宿主机后备目录。它不合并两个后端，也不返回宿主机绝对路径。完整选择规则见[主文档的存储路由合同](./README.md#32-后端选择)。

它不查询 QQ 文件消息历史、不访问 QQ 网络、不自动下载，也不按正文或文件名进行搜索。文件名和类型过滤由 `search({source:"local"})` 负责。

支持三个起始范围：

| scope | 范围 | 是否要求当前会话 |
|---|---|---:|
| `current` | 当前 QQ 会话对应的 `private_*` 或 `group_*` 目录 | 是 |
| `conversation` | 明确指定的好友或群目录 | 否 |
| `all` | 当前 QQ 账号整个 `file` 根目录 | 否 |

规则：

- `scope` 省略时等同于 `current`；QQ 首页调用会返回 `no_current_qq_session`；
- `conversation` 和 `all` 可以在 QQ 首页或任意会话中使用；
- `current` / `conversation` 递归枚举选定会话目录；`all` 递归枚举整个文件根目录，也包含未归入标准会话目录的普通文件；
- 目标会话目录尚不存在时返回成功空列表，不把“尚未下载文件”当成错误；
- 只返回普通文件，不返回目录；
- 不遍历或返回符号链接；无法安全读取的条目通过 `warnings` 汇总；
- 下载中的内部临时文件永远不作为业务文件返回；除此之外，普通隐藏文件照常枚举；
- 每个分页内按 `relative_path` 升序返回，大小写保持逻辑路径原样；
- 返回的是调用时的实时文件系统状态。文件随后仍可能被移动、修改或删除，`read / delete` 会再次检查精确路径。

## 2. 四种互斥调用形态

当前会话，允许直接传空参数：

```json
{}
```

指定会话：

```json
{
  "scope": "conversation",
  "conversation_type": "group",
  "conversation_id": "1090411227",
  "limit": 50
}
```

全部目录：

```json
{
  "scope": "all",
  "limit": 50
}
```

继续下一页：

```json
{
  "cursor": "qfl_..."
}
```

`cursor` 不能与 `scope / conversation_type / conversation_id / limit` 同时出现。

## 3. 参数 JSON Schema declaration

```json
{
  "name": "list_files",
  "description": "枚举当前 QQ 账号文件根目录中的本地普通文件。",
  "parameters": {
    "$defs": {
      "ListAllArgs": {
        "additionalProperties": false,
        "properties": {
          "scope": {
            "const": "all",
            "description": "递归枚举当前 QQ 账号的全部文件目录。",
            "type": "string"
          },
          "limit": {
            "default": 50,
            "description": "本页最多返回的文件数，默认 50，最大 200。",
            "maximum": 200,
            "minimum": 1,
            "type": "integer"
          }
        },
        "required": ["scope"],
        "type": "object"
      },
      "ListContinueArgs": {
        "additionalProperties": false,
        "properties": {
          "cursor": {
            "description": "上次 list_files 返回的 next_cursor；使用时不能传其他字段。",
            "maxLength": 2048,
            "minLength": 1,
            "type": "string"
          }
        },
        "required": ["cursor"],
        "type": "object"
      },
      "ListConversationArgs": {
        "additionalProperties": false,
        "properties": {
          "scope": {
            "const": "conversation",
            "description": "递归枚举指定 QQ 会话的文件目录。",
            "type": "string"
          },
          "conversation_type": {
            "description": "会话类型。",
            "enum": ["private", "group"],
            "type": "string"
          },
          "conversation_id": {
            "description": "好友 QQ 号或群号。",
            "minLength": 1,
            "type": "string",
            "x-coerce-integer": true
          },
          "limit": {
            "default": 50,
            "description": "本页最多返回的文件数，默认 50，最大 200。",
            "maximum": 200,
            "minimum": 1,
            "type": "integer"
          }
        },
        "required": [
          "scope",
          "conversation_type",
          "conversation_id"
        ],
        "type": "object"
      },
      "ListCurrentArgs": {
        "additionalProperties": false,
        "properties": {
          "scope": {
            "const": "current",
            "default": "current",
            "description": "递归枚举当前会话的文件目录。",
            "type": "string"
          },
          "limit": {
            "default": 50,
            "description": "本页最多返回的文件数，默认 50，最大 200。",
            "maximum": 200,
            "minimum": 1,
            "type": "integer"
          }
        },
        "type": "object"
      }
    },
    "anyOf": [
      {
        "$ref": "#/$defs/ListCurrentArgs"
      },
      {
        "$ref": "#/$defs/ListConversationArgs"
      },
      {
        "$ref": "#/$defs/ListAllArgs"
      },
      {
        "$ref": "#/$defs/ListContinueArgs"
      }
    ]
  }
}
```

## 4. 模型正式可见的 TypeScript-like 形态

```ts
// 枚举当前 QQ 账号文件根目录中的本地普通文件。
list_files(args: {
  scope?: "current"; // 递归枚举当前会话的文件目录。
  limit?: number; // 本页最多返回的文件数，默认 50，最大 200。 范围 1~200
} | {
  scope: "conversation"; // 递归枚举指定 QQ 会话的文件目录。
  conversation_type: "private" | "group"; // 会话类型。
  conversation_id: string; // 好友 QQ 号或群号。
  limit?: number; // 本页最多返回的文件数，默认 50，最大 200。 范围 1~200
} | {
  scope: "all"; // 递归枚举当前 QQ 账号的全部文件目录。
  limit?: number; // 本页最多返回的文件数，默认 50，最大 200。 范围 1~200
} | {
  cursor: string; // 上次 list_files 返回的 next_cursor；使用时不能传其他字段。 最多 2048 个字符
})
```

当前实际测得：

- `list_files` 单独函数签名：526 个字符，UTF-8 836 字节；
- `download + read + list_files` 完整 namespace：2665 个字符，UTF-8 4445 字节；
- 返回 Schema 不进入常驻提示词。

## 5. `managed` 与来源记录

每个返回文件都包含：

```text
managed=true   精确路径仍匹配一条未删除的 qq_file_records 记录
managed=false  文件实际存在，但当前没有精确路径记录
```

不变量：

- `managed=true` 时 `source` 必须非 `null`；
- `managed=false` 时 `source` 必须为 `null`；
- 人工重命名或移动文件后，新路径返回 `managed=false`，不扫描旧记录猜测来源；
- 这里只校验精确路径映射，不计算 hash、不跟踪 inode，也不声明文件内容从下载后从未变化；
- `source.recorded_size_bytes` 是下载记录中的原始大小，`size_bytes` 是当前活动存储中文件的实际大小，可用于发现明显差异；
- `conversation` 从当前相对路径的标准首级目录推导；`scope=all` 下未归入 `private_* / group_*` 的文件返回 `conversation=null`。

## 6. cursor 与实时一致性

`next_cursor` 是经过认证的无状态分页游标，绑定：

```text
current_agent_qq
原始 scope
current 已解析出的实际会话
limit
上一条 relative_path
```

规则：

- `scope=current` 的第一页一旦解析出实际会话，后续 cursor 始终继续该会话；切换界面不会改变游标范围；
- cursor 不要求当前处于具体 QQ 会话，可以跨会话和 Core 重启继续；
- cursor 由其他 QQ 账号产生时返回 `cursor_scope_mismatch`；被篡改时返回 `invalid_cursor`；
- 继续页从上一条 `relative_path` 之后按升序枚举；
- 这是实时弱一致列表，不创建文件快照。分页期间新增、删除或重命名文件时，结果可能移动；需要最新完整视图时从第一页重新调用；
- cursor 丢失后重新发起对应 scope，不提供 cursor 找回列表。

分页不变量：

```text
count = files.length
has_more=true   <=> next_cursor 为非 null
has_more=false  <=> next_cursor 为 null
```

## 7. 成功与错误返回原则

成功结果包含：

```text
ok
scope
files
count
has_more
next_cursor
warnings
```

`scope` 会返回解析后的实际范围：

- `current`：包含调用时解析出的实际会话；
- `conversation`：回显明确指定的会话；
- `all`：不附带单一会话。

文件条目包含绝对路径、相对路径、文件名、无点小写扩展名、当前大小、修改时间、归属会话、`managed` 和可选下载来源。符号链接或不可读条目不会混入 `files`，而是以汇总 warning 返回。

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
    "ManagedSource": {
      "additionalProperties": false,
      "properties": {
        "message_id": {
          "minLength": 1,
          "type": "string"
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
        "original_filename",
        "recorded_size_bytes",
        "downloaded_at"
      ],
      "type": "object"
    },
    "FileEntry": {
      "additionalProperties": false,
      "properties": {
        "name": {
          "minLength": 1,
          "type": "string"
        },
        "path": {
          "minLength": 1,
          "type": "string"
        },
        "relative_path": {
          "minLength": 1,
          "type": "string"
        },
        "extension": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ]
        },
        "size_bytes": {
          "minimum": 0,
          "type": "integer"
        },
        "modified_at": {
          "format": "date-time",
          "type": "string"
        },
        "conversation": {
          "anyOf": [
            {
              "$ref": "#/$defs/Conversation"
            },
            {
              "type": "null"
            }
          ]
        },
        "managed": {
          "type": "boolean"
        },
        "source": {
          "anyOf": [
            {
              "$ref": "#/$defs/ManagedSource"
            },
            {
              "type": "null"
            }
          ]
        }
      },
      "required": [
        "name",
        "path",
        "relative_path",
        "extension",
        "size_bytes",
        "modified_at",
        "conversation",
        "managed",
        "source"
      ],
      "type": "object"
    },
    "CurrentScope": {
      "additionalProperties": false,
      "properties": {
        "type": {
          "const": "current",
          "type": "string"
        },
        "conversation": {
          "$ref": "#/$defs/Conversation"
        }
      },
      "required": ["type", "conversation"],
      "type": "object"
    },
    "ConversationScope": {
      "additionalProperties": false,
      "properties": {
        "type": {
          "const": "conversation",
          "type": "string"
        },
        "conversation": {
          "$ref": "#/$defs/Conversation"
        }
      },
      "required": ["type", "conversation"],
      "type": "object"
    },
    "AllScope": {
      "additionalProperties": false,
      "properties": {
        "type": {
          "const": "all",
          "type": "string"
        }
      },
      "required": ["type"],
      "type": "object"
    },
    "ListScope": {
      "discriminator": {
        "mapping": {
          "all": "#/$defs/AllScope",
          "conversation": "#/$defs/ConversationScope",
          "current": "#/$defs/CurrentScope"
        },
        "propertyName": "type"
      },
      "oneOf": [
        {
          "$ref": "#/$defs/CurrentScope"
        },
        {
          "$ref": "#/$defs/ConversationScope"
        },
        {
          "$ref": "#/$defs/AllScope"
        }
      ]
    },
    "ListWarning": {
      "additionalProperties": false,
      "properties": {
        "code": {
          "enum": ["unsafe_entry_skipped", "unreadable_entry_skipped"],
          "type": "string"
        },
        "count": {
          "minimum": 1,
          "type": "integer"
        },
        "message": {
          "minLength": 1,
          "type": "string"
        }
      },
      "required": ["code", "count", "message"],
      "type": "object"
    },
    "ListSuccess": {
      "additionalProperties": false,
      "properties": {
        "ok": {
          "const": true,
          "type": "boolean"
        },
        "scope": {
          "$ref": "#/$defs/ListScope"
        },
        "files": {
          "items": {
            "$ref": "#/$defs/FileEntry"
          },
          "type": "array"
        },
        "count": {
          "minimum": 0,
          "type": "integer"
        },
        "has_more": {
          "type": "boolean"
        },
        "next_cursor": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ]
        },
        "warnings": {
          "items": {
            "$ref": "#/$defs/ListWarning"
          },
          "type": "array"
        }
      },
      "required": [
        "ok",
        "scope",
        "files",
        "count",
        "has_more",
        "next_cursor",
        "warnings"
      ],
      "type": "object"
    },
    "ListError": {
      "additionalProperties": false,
      "properties": {
        "code": {
          "enum": [
            "no_current_qq_session",
            "invalid_conversation",
            "invalid_cursor",
            "cursor_scope_mismatch",
            "filesystem_unavailable",
            "database_unavailable",
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
        }
      },
      "required": ["code", "message", "retryable"],
      "type": "object"
    },
    "ListErrorResult": {
      "additionalProperties": false,
      "properties": {
        "ok": {
          "const": false,
          "type": "boolean"
        },
        "error": {
          "$ref": "#/$defs/ListError"
        }
      },
      "required": ["ok", "error"],
      "type": "object"
    }
  },
  "anyOf": [
    {
      "$ref": "#/$defs/ListSuccess"
    },
    {
      "$ref": "#/$defs/ListErrorResult"
    }
  ]
}
```

## 9. 典型运行时返回

当前群会话存在一个仍受记录管理的文件，以及一个已经人工改名的文件：

```json
{
  "ok": true,
  "scope": {
    "type": "current",
    "conversation": {
      "type": "group",
      "id": "1090411227"
    }
  },
  "files": [
    {
      "name": "example.pdf",
      "path": "/home/agent/qq/213628848/file/group_1090411227/example.pdf",
      "relative_path": "group_1090411227/example.pdf",
      "extension": "pdf",
      "size_bytes": 7340032,
      "modified_at": "2026-08-30T10:00:04Z",
      "conversation": {
        "type": "group",
        "id": "1090411227"
      },
      "managed": true,
      "source": {
        "message_id": "123456789",
        "original_filename": "example.pdf",
        "recorded_size_bytes": 7340032,
        "downloaded_at": "2026-08-30T10:00:04Z"
      }
    },
    {
      "name": "renamed.docx",
      "path": "/home/agent/qq/213628848/file/group_1090411227/renamed.docx",
      "relative_path": "group_1090411227/renamed.docx",
      "extension": "docx",
      "size_bytes": 28672,
      "modified_at": "2026-08-30T11:20:00Z",
      "conversation": {
        "type": "group",
        "id": "1090411227"
      },
      "managed": false,
      "source": null
    }
  ],
  "count": 2,
  "has_more": false,
  "next_cursor": null,
  "warnings": []
}
```

QQ 首页省略 scope：

```json
{
  "ok": false,
  "error": {
    "code": "no_current_qq_session",
    "message": "当前不在具体 QQ 会话中。请指定 scope=conversation 或 scope=all。",
    "retryable": false
  }
}
```

## 10. 返回合同与上下文消耗

- 返回 JSON Schema、`managed` 判定规则和错误枚举不随 namespace 注入；
- `list_files` 常驻增量是 526 个字符、UTF-8 836 字节；
- `download + read + list_files` 完整 namespace 是 2665 个字符、UTF-8 4445 字节；
- 单页默认最多 50 个文件，调用方可提高到 200；
- 不返回目录、内部临时文件、数据库内部主键、原始记录行或文件内容。
