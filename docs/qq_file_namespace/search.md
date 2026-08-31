# `qq_file.search`

[← 主文档](./README.md) · [Skill](./qq-file/SKILL.md) · [download](./download.md) · [read](./read.md) · [list_files](./list_files.md) · [search](./search.md) · [delete](./delete.md)

> 本合同已实现。参数 JSON Schema 和 TypeScript-like 签名遵循当前仓库的 `ToolContract` 生成规则，并由合同测试逐字段校验。

## 1. 职责、数据源与边界

`search` 按文件名搜索两种明确分离的数据源：

| source | 搜索对象 | 主要定位结果 |
|---|---|---|
| `local` | 当前 QQ 账号活动文件存储中实际存在的普通文件 | 绝对逻辑 `path` |
| `history` | AICQ 数据库中已经同步的真实 QQ `file` 消息 | `conversation + message_id` |

本地搜索根目录：

```text
/home/agent/qq/{current_agent_qq}/file/
```

共同边界：

- `source` 是首次搜索的必填字段，一次调用只搜索一个数据源；
- 只匹配文件名，不搜索文件正文、消息正文、发送者或会话名称；
- 支持按无点扩展名 `file_types` 过滤；
- 搜索本身不下载文件、不读取文件内容，也不主动访问 QQ / NapCat；
- `source="local"` 以当前活动存储后端的实时文件状态为准，不合并 Linux 与宿主机后备目录；
- `source="history"` 只覆盖 AICQ 已同步历史，不补拉更早 QQ 消息；
- 两种数据源各自返回独立结果类型，不合并排序，也不对同名或对应记录进行跨源去重。

`local` 使用[主文档的存储路由合同](./README.md#32-后端选择)：Linux 电脑存在时搜索 Linux；只有明确确认 Linux 电脑不存在时才搜索项目内的宿主机后备目录。结果始终返回 `/home/agent/qq/...` 逻辑路径，不返回宿主机绝对路径。`history` 只依赖 AICQ 数据库，不因文件存储后端不可用而停止搜索。

## 2. 两种顶层调用形态

搜索当前会话中的本地 PDF：

```json
{
  "source": "local",
  "query": "report",
  "file_types": ["pdf"]
}
```

搜索全部已同步 QQ 会话中的 DOCX 文件消息：

```json
{
  "source": "history",
  "file_types": ["docx"],
  "scope": {
    "type": "all"
  }
}
```

搜索指定群的本地文件：

```json
{
  "source": "local",
  "query": "预算",
  "scope": {
    "type": "conversation",
    "conversation_type": "group",
    "conversation_id": "1090411227"
  },
  "limit": 20
}
```

继续上一页：

```json
{
  "cursor": "qfs_..."
}
```

顶层只有 `{source, query?, file_types?, limit?, scope?}` 与 `{cursor}` 两种互斥形态。首次搜索时 `query` 与 `file_types` 至少提供一个；cursor 已绑定原 `source`、范围和条件，继续时不能重复提交其他字段。

## 3. 参数 JSON Schema declaration

```json
{
  "name": "search",
  "description": "按文件名搜索本机 QQ 文件或 AICQ 已同步的 QQ 文件消息。",
  "parameters": {
    "$defs": {
      "AllSearchScope": {
        "additionalProperties": false,
        "properties": {
          "type": {
            "const": "all",
            "description": "搜索当前 QQ 账号的全部会话。",
            "type": "string"
          }
        },
        "required": ["type"],
        "type": "object"
      },
      "ConversationSearchScope": {
        "additionalProperties": false,
        "properties": {
          "type": {
            "const": "conversation",
            "description": "搜索指定 QQ 会话。",
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
          }
        },
        "required": ["type", "conversation_type", "conversation_id"],
        "type": "object"
      },
      "CurrentSearchScope": {
        "additionalProperties": false,
        "properties": {
          "type": {
            "const": "current",
            "description": "搜索当前 QQ 会话。",
            "type": "string"
          }
        },
        "required": ["type"],
        "type": "object"
      },
      "SearchContinueArgs": {
        "additionalProperties": false,
        "properties": {
          "cursor": {
            "description": "上次 search 返回的 next_cursor；使用时不能传其他字段。",
            "maxLength": 2048,
            "minLength": 1,
            "type": "string"
          }
        },
        "required": ["cursor"],
        "type": "object"
      },
      "SearchStartArgs": {
        "additionalProperties": false,
        "properties": {
          "source": {
            "description": "local 搜索实际存在的本机文件；history 搜索 AICQ 已同步的 QQ 文件消息。",
            "enum": ["local", "history"],
            "type": "string"
          },
          "query": {
            "description": "文件名文字，按 Unicode 大小写不敏感的字面子串匹配；与 file_types 至少提供一个。",
            "maxLength": 255,
            "minLength": 1,
            "type": "string"
          },
          "file_types": {
            "description": "可选无点扩展名列表，例如 pdf、docx；大小写不敏感，与 query 至少提供一个。",
            "items": {
              "maxLength": 32,
              "minLength": 1,
              "type": "string"
            },
            "maxItems": 20,
            "minItems": 1,
            "type": "array",
            "uniqueItems": true
          },
          "limit": {
            "default": 50,
            "description": "本页最多返回的结果数，默认 50，最大 200。",
            "maximum": 200,
            "minimum": 1,
            "type": "integer"
          },
          "scope": {
            "default": {
              "type": "current"
            },
            "description": "搜索的会话范围；省略时使用当前会话。",
            "discriminator": {
              "mapping": {
                "all": "#/$defs/AllSearchScope",
                "conversation": "#/$defs/ConversationSearchScope",
                "current": "#/$defs/CurrentSearchScope"
              },
              "propertyName": "type"
            },
            "oneOf": [
              {
                "$ref": "#/$defs/CurrentSearchScope"
              },
              {
                "$ref": "#/$defs/ConversationSearchScope"
              },
              {
                "$ref": "#/$defs/AllSearchScope"
              }
            ]
          }
        },
        "required": ["source"],
        "anyOf": [
          {
            "required": ["query"]
          },
          {
            "required": ["file_types"]
          }
        ],
        "type": "object"
      }
    },
    "anyOf": [
      {
        "$ref": "#/$defs/SearchStartArgs"
      },
      {
        "$ref": "#/$defs/SearchContinueArgs"
      }
    ]
  }
}
```

`file_types` 的“无点、无目录分隔符、无控制字符”约束由 schema repair 或业务校验补充；违反时返回 `invalid_file_type`。`SearchStartArgs.anyOf` 保证首次调用至少存在一个搜索条件。

## 4. 模型正式可见的 TypeScript-like 形态

```ts
// 按文件名搜索本机 QQ 文件或 AICQ 已同步的 QQ 文件消息。
search(args: {
  source: "local" | "history"; // local 搜索实际存在的本机文件；history 搜索 AICQ 已同步的 QQ 文件消息。
  query?: string; // 文件名文字，按 Unicode 大小写不敏感的字面子串匹配；与 file_types 至少提供一个。 最多 255 个字符
  file_types?: string[]; // 可选无点扩展名列表，例如 pdf、docx；大小写不敏感，与 query 至少提供一个。 最多 20 项；数组项不可重复
  limit?: number; // 本页最多返回的结果数，默认 50，最大 200。 范围 1~200
  scope?: {
    type: "current"; // 搜索当前 QQ 会话。
  } | {
    type: "conversation"; // 搜索指定 QQ 会话。
    conversation_type: "private" | "group"; // 会话类型。
    conversation_id: string; // 好友 QQ 号或群号。
  } | {
    type: "all"; // 搜索当前 QQ 账号的全部会话。
  }; // 搜索的会话范围；省略时使用当前会话。
} | {
  cursor: string; // 上次 search 返回的 next_cursor；使用时不能传其他字段。 最多 2048 个字符
})
```

上下文尺寸见第 13 节最终校验。返回 Schema、排序规则和索引设计不进入 namespace 常驻提示词。

## 5. 文件名、扩展名与共同匹配规则

- `query` 对完整 basename（含扩展名）执行 Unicode NFKC 规范化和默认大小写折叠后的字面子串匹配；
- `query` 不解释为 glob 或正则表达式；
- `file_types` 是不带前导点的扩展名数组，同时存在 `query` 时采用 AND；
- `archive.tar.gz` 的扩展名是 `gz`；无扩展名文件和单一前导点 dotfile 的扩展名为 `null`；`.config.json` 的扩展名是 `json`；
- `file_types` 不能匹配 `extension=null`；
- 查询条件回显为 `filters.query` 和规范化、去重、排序后的 `filters.file_types`。

存在 `query` 时，两个数据源都先按匹配类型排序：

1. 完整文件名与 query 完全相等：`match_type=exact`；
2. 完整文件名以 query 开头：`match_type=prefix`；
3. 完整文件名包含 query：`match_type=substring`。

只提供 `file_types` 时使用 `match_type=type_only`。同一匹配类型内，`local` 按 `relative_path` 升序；`history` 按消息有效时间倒序，再用数据库内部稳定键倒序。

## 6. scope、cursor 与实时一致性

三个 scope 对两种数据源含义一致：

```text
current       调用当下的当前 QQ 会话
conversation  显式 private/group + 真实 ID
all           当前 QQ 账号全部会话
```

- `scope` 省略时默认 `{type:"current"}`；当前不在具体 QQ 会话中返回 `no_current_qq_session`；
- `conversation` 和 `all` 可以在 QQ 首页使用；
- 首次搜索解析出的真实 scope 会在成功结果中返回；
- cursor 是经过认证的无状态游标，绑定当前 QQ 账号、`source`、已解析 scope、规范化条件、limit 和对应数据源的排序位置；
- cursor 可以跨会话和 Core 重启继续；`scope=current` 在首次调用时固定，之后切换会话不改变其范围；
- 其他 QQ 账号产生的 cursor 返回 `cursor_scope_mismatch`；被篡改或无法解码返回 `invalid_cursor`；
- cursor 调用只传 token；丢失后重新提交原搜索条件，不提供 cursor 列表；
- 两种搜索都是实时弱一致视图，不创建文件系统或数据库快照。分页期间数据变化可能影响后续页，需要最新完整结果时从第一页重新搜索。

分页不变量：

```text
count = files.length     当 source=local
count = messages.length  当 source=history
has_more=true   <=> next_cursor 为非 null
has_more=false  <=> next_cursor 为 null
```

## 7. `source="local"` 合同

`local` 递归搜索当前 QQ 账号文件根目录中的普通文件：

- 只返回调用时实际存在的普通文件；
- 不跟随任何路径组件中的符号链接，不返回目录、临时下载文件、socket、FIFO 或设备节点；
- `scope=current / conversation` 只搜索对应标准会话目录；`scope=all` 搜索整个 file 根目录；
- `scope=all` 下不在标准 `private_* / group_*` 首级目录中的普通文件可以返回，`conversation=null`；
- 路径在分页期间消失时跳过，并通过 warning 汇总；
- `managed=true` 只表示当前精确路径匹配未删除的 `qq_file_records` 记录，不计算 hash、不追踪 inode，也不猜测重命名或移动后的来源；
- `managed=false` 时 `source=null`；
- 文件条目返回 `name`、绝对 `path`、`relative_path`、`extension`、`match_type`、当前大小、修改时间、会话、`managed` 和可选下载来源。

本地 warning：

- `unsafe_entry_skipped`：发现符号链接或不安全目录项；
- `unreadable_entry_skipped`：搜索期间无法读取或已经消失的目录项。

## 8. `source="history"` 合同

`history` 只搜索 AICQ 数据库中已经同步的真实 QQ `file` 消息：

- 只接受 `content_segments` 中结构有效且唯一的 file segment，不从正文、旧 `ref` 或其他 metadata 猜测文件消息；
- 不主动向 QQ / NapCat 补拉历史，成功结果固定返回 `history_coverage="aicq_synced_only"`；
- 每条真实消息独立返回；相同文件名不合并，同一 `session_key + message_id` 的重复数据库行只返回一个规范结果；
- 有效时间优先采用可解析的 QQ 消息 `timestamp`，否则使用数据库 `created_at` 排序；`sent_at` 只回显真实可解析的 QQ 消息时间，不能用数据库写入时间冒充；
- 条目返回 `message_id`、`filename`、`extension`、`match_type`、声明大小、来源会话、发送者、QQ 消息时间、`in_current_session` 和可选 `local_file`；
- `local_file` 仅在记录属于当前活动存储后端，且精确记录路径仍存在、是普通文件、不是符号链接并位于当前 QQ 账号文件根目录内时返回，否则为 `null`；
- `local_file=null` 不证明从未下载，也不扫描改名、移动或复制后的等价文件；如需查找实际文件，使用 `search({source:"local"})`。

读取历史结果：

1. `local_file` 非 `null` 时，可以用 `read({source:{path: local_file.path}})`；
2. `local_file=null` 且 `in_current_session=true` 时，可以用 `read({source:{message_id}})`，由 `read` 自动下载并观察最多 15 秒；
3. `in_current_session=false` 时，必须先进入结果中的来源会话，再用同一 `message_id` 调用 `read` 或 `download(start)`；
4. `search` 不替代会话切换，也不赋予 `message_id` 跨会话定位能力。

历史 warning：

- `invalid_file_message_skipped`：结构化 file segment 缺失、数量异常或 filename 无效；
- `unresolved_conversation_skipped`：无法安全解析为真实 QQ private/group 会话；
- `local_file_state_unavailable`：历史搜索成功，但本次无法检查可选本地文件状态，相关结果按 `local_file=null` 返回。

## 9. 成功与错误返回原则

成功结果以 `source` 判别：

```text
source=local
  files

source=history
  history_coverage=aicq_synced_only
  messages
```

两者都返回实际 `scope`、规范化 `filters`、`count`、`has_more`、`next_cursor` 和与数据源对应的 `warnings`。不把本地文件与历史消息压成包含大量空字段的统一条目。

错误结果统一为 `ok=false + error`。`filesystem_unavailable` 只适用于 `local`；`database_unavailable` 只适用于 `history`；cursor 无法安全恢复 source 时，错误结果不猜测数据源。

## 10. 返回 JSON Schema

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
        },
        "name": {
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
      "required": ["message_id", "original_filename", "recorded_size_bytes", "downloaded_at"],
      "type": "object"
    },
    "LocalFileEntry": {
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
        "match_type": {
          "enum": ["exact", "prefix", "substring", "type_only"],
          "type": "string"
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
        "match_type",
        "size_bytes",
        "modified_at",
        "conversation",
        "managed",
        "source"
      ],
      "type": "object"
    },
    "Sender": {
      "additionalProperties": false,
      "properties": {
        "id": {
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
        "display_name": {
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
      "required": ["id", "display_name"],
      "type": "object"
    },
    "HistoryLocalFile": {
      "additionalProperties": false,
      "properties": {
        "path": {
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
      "required": ["path", "size_bytes", "downloaded_at"],
      "type": "object"
    },
    "HistoryMessageEntry": {
      "additionalProperties": false,
      "properties": {
        "message_id": {
          "minLength": 1,
          "type": "string"
        },
        "filename": {
          "minLength": 1,
          "type": "string"
        },
        "extension": {
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
        "match_type": {
          "enum": ["exact", "prefix", "substring", "type_only"],
          "type": "string"
        },
        "declared_size_bytes": {
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
        "conversation": {
          "$ref": "#/$defs/Conversation"
        },
        "sender": {
          "$ref": "#/$defs/Sender"
        },
        "sent_at": {
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
        "in_current_session": {
          "type": "boolean"
        },
        "local_file": {
          "anyOf": [
            {
              "$ref": "#/$defs/HistoryLocalFile"
            },
            {
              "type": "null"
            }
          ]
        }
      },
      "required": [
        "message_id",
        "filename",
        "extension",
        "match_type",
        "declared_size_bytes",
        "conversation",
        "sender",
        "sent_at",
        "in_current_session",
        "local_file"
      ],
      "type": "object"
    },
    "CurrentResultScope": {
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
    "ConversationResultScope": {
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
    "AllResultScope": {
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
    "SearchResultScope": {
      "discriminator": {
        "mapping": {
          "all": "#/$defs/AllResultScope",
          "conversation": "#/$defs/ConversationResultScope",
          "current": "#/$defs/CurrentResultScope"
        },
        "propertyName": "type"
      },
      "oneOf": [
        {
          "$ref": "#/$defs/CurrentResultScope"
        },
        {
          "$ref": "#/$defs/ConversationResultScope"
        },
        {
          "$ref": "#/$defs/AllResultScope"
        }
      ]
    },
    "SearchFilters": {
      "additionalProperties": false,
      "properties": {
        "query": {
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
        "file_types": {
          "items": {
            "minLength": 1,
            "type": "string"
          },
          "type": "array",
          "uniqueItems": true
        }
      },
      "required": ["query", "file_types"],
      "type": "object"
    },
    "LocalSearchWarning": {
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
    "HistorySearchWarning": {
      "additionalProperties": false,
      "properties": {
        "code": {
          "enum": [
            "invalid_file_message_skipped",
            "unresolved_conversation_skipped",
            "local_file_state_unavailable"
          ],
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
    "LocalSearchSuccess": {
      "additionalProperties": false,
      "properties": {
        "ok": {
          "const": true,
          "type": "boolean"
        },
        "source": {
          "const": "local",
          "type": "string"
        },
        "scope": {
          "$ref": "#/$defs/SearchResultScope"
        },
        "filters": {
          "$ref": "#/$defs/SearchFilters"
        },
        "files": {
          "items": {
            "$ref": "#/$defs/LocalFileEntry"
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
            "$ref": "#/$defs/LocalSearchWarning"
          },
          "type": "array"
        }
      },
      "required": ["ok", "source", "scope", "filters", "files", "count", "has_more", "next_cursor", "warnings"],
      "type": "object"
    },
    "HistorySearchSuccess": {
      "additionalProperties": false,
      "properties": {
        "ok": {
          "const": true,
          "type": "boolean"
        },
        "source": {
          "const": "history",
          "type": "string"
        },
        "history_coverage": {
          "const": "aicq_synced_only",
          "type": "string"
        },
        "scope": {
          "$ref": "#/$defs/SearchResultScope"
        },
        "filters": {
          "$ref": "#/$defs/SearchFilters"
        },
        "messages": {
          "items": {
            "$ref": "#/$defs/HistoryMessageEntry"
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
            "$ref": "#/$defs/HistorySearchWarning"
          },
          "type": "array"
        }
      },
      "required": [
        "ok",
        "source",
        "history_coverage",
        "scope",
        "filters",
        "messages",
        "count",
        "has_more",
        "next_cursor",
        "warnings"
      ],
      "type": "object"
    },
    "SearchError": {
      "additionalProperties": false,
      "properties": {
        "code": {
          "enum": [
            "no_current_qq_session",
            "search_filter_required",
            "invalid_file_type",
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
    "SearchErrorResult": {
      "additionalProperties": false,
      "properties": {
        "ok": {
          "const": false,
          "type": "boolean"
        },
        "error": {
          "$ref": "#/$defs/SearchError"
        }
      },
      "required": ["ok", "error"],
      "type": "object"
    }
  },
  "anyOf": [
    {
      "$ref": "#/$defs/LocalSearchSuccess"
    },
    {
      "$ref": "#/$defs/HistorySearchSuccess"
    },
    {
      "$ref": "#/$defs/SearchErrorResult"
    }
  ]
}
```

## 11. 典型运行时返回

本地搜索：

```json
{
  "ok": true,
  "source": "local",
  "scope": {
    "type": "current",
    "conversation": {
      "type": "group",
      "id": "1090411227"
    }
  },
  "filters": {
    "query": "report",
    "file_types": ["pdf"]
  },
  "files": [
    {
      "name": "report.pdf",
      "path": "/home/agent/qq/213628848/file/group_1090411227/report.pdf",
      "relative_path": "group_1090411227/report.pdf",
      "extension": "pdf",
      "match_type": "exact",
      "size_bytes": 1048576,
      "modified_at": "2026-08-30T14:21:00+08:00",
      "conversation": {
        "type": "group",
        "id": "1090411227"
      },
      "managed": true,
      "source": {
        "message_id": "1803394108",
        "original_filename": "report.pdf",
        "recorded_size_bytes": 1048576,
        "downloaded_at": "2026-08-30T14:21:00+08:00"
      }
    }
  ],
  "count": 1,
  "has_more": false,
  "next_cursor": null,
  "warnings": []
}
```

历史搜索：

```json
{
  "ok": true,
  "source": "history",
  "history_coverage": "aicq_synced_only",
  "scope": {
    "type": "all"
  },
  "filters": {
    "query": "report",
    "file_types": ["pdf"]
  },
  "messages": [
    {
      "message_id": "99887766",
      "filename": "monthly-report.pdf",
      "extension": "pdf",
      "match_type": "substring",
      "declared_size_bytes": null,
      "conversation": {
        "type": "private",
        "id": "123456789",
        "name": "示例好友"
      },
      "sender": {
        "id": "123456789",
        "display_name": "示例好友"
      },
      "sent_at": "2026-08-29T09:00:00+08:00",
      "in_current_session": false,
      "local_file": null
    }
  ],
  "count": 1,
  "has_more": false,
  "next_cursor": null,
  "warnings": []
}
```

当前不在具体 QQ 会话中且省略 scope：

```json
{
  "ok": false,
  "error": {
    "code": "no_current_qq_session",
    "message": "当前不在具体 QQ 会话中。请指定 scope.type=conversation 或 scope.type=all。",
    "retryable": false
  }
}
```

## 12. AICQ 历史索引

文件元数据原文保存在 `chat_messages.content_segments` JSON 文本中，会话信息保存在 `chat_sessions`；账号隔离的派生索引 `qq_file_messages` 保存历史搜索所需字段：

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

主键为：

```text
(agent_qq, session_key, message_id)
```

- `chat_messages + chat_sessions` 仍是已同步历史来源，派生索引不产生数据库中不存在的 QQ 文件消息；
- 新消息写库时同步写入索引；消息撤回时删除索引；消息 ID 回填时同步更新索引键；
- 无法无歧义绑定真实 QQ 账号的旧记录不进入任何账号分区；
- `conversation_name` 在查询时从 `chat_sessions` 联接，不复制到派生表；
- NFKC、Unicode 大小写折叠和匹配等级在查询层计算，不依赖 SQLite 默认 `NOCASE`；
- cursor 使用签名状态保存过滤条件和分页偏移，不返回数据库内部行号。

## 13. 返回合同与上下文消耗

- 返回 JSON Schema、排序规则、索引建议、错误枚举和文件系统安全检查不随 namespace 注入；
- `search` 单独函数签名是 707 个字符、UTF-8 1135 字节；
- 原两个搜索签名合计 2234 个字符、UTF-8 3938 字节；合并后减少 1409 个字符、UTF-8 2549 字节；
- 五个工具的完整 namespace 是 3207 个字符、UTF-8 5155 字节；
- 合并后只常驻一份 query、file_types、limit 和 scope 声明；
- 单页默认最多 50 条，可提高到 200；
- `local` 不返回文件正文、hash、inode 或下载临时文件；
- `history` 不返回消息正文、文件正文、内部 adapter locator、数据库内部行号、原始 JSON 行或内部匹配分数。
