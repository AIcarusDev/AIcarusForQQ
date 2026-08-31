# `qq_file.read`

[← 主文档](./README.md) · [Skill](./qq-file/SKILL.md) · [download](./download.md) · [read](./read.md) · [list_files](./list_files.md) · [search](./search.md) · [delete](./delete.md)

> 本合同已实现。参数 JSON Schema 和 TypeScript-like 签名由当前仓库的 `ToolContract` 生成规则生成，并由合同测试逐字段校验。

## 1. 职责、来源与路径边界

`read` 从两种来源开始读取：

- `source.path`：读取当前 QQ 账号文件根目录中已经存在的本地文件；
- `source.message_id`：在当前 QQ 会话中定位文件消息；精确记录路径不存在时，复用或启动对应下载任务并观察最多 15 秒。

公开逻辑文件根目录：

```text
/home/agent/qq/{current_agent_qq}/file/
```

路径来源规则：

- `path` 必须是该根目录内的绝对 Linux 形态逻辑路径；
- 允许跨 `private_*` / `group_*` 会话目录读取；
- 允许读取 `managed=true` 和 `managed=false` 的文件；
- 不要求当前位于任何具体 QQ 会话，QQ 首页也可以按 `path` 读取；
- 必须是精确普通文件；拒绝目录、glob、路径逃逸和任意路径组件中的符号链接；
- 读取不修改文件，也不改变 `qq_file_records`；
- `path` 已被移动或重命名时，旧路径按不存在处理，不追踪新位置。

消息来源规则：

- `message_id` 只在调用当下的当前 QQ 会话解析；当前不在具体会话中返回 `no_current_qq_session`；
- 如果同一来源记录属于当前活动存储后端，且精确本地路径存在并通过普通文件安全检查，直接读取；
- 如果精确路径不存在，调用与 `download(action="start")` 相同的来源解析、活跃任务去重、目标命名和下载限制；
- 15 秒内完成下载时，立即从完成后的 `local_path` 开始解析；
- 15 秒内未完成时返回 `outcome="download_pending"` 和任务进度，任务继续运行；
- 返回下载进度时不创建读取 cursor，也不主动唤醒；后续使用 `download.poll / download.list` 查找状态，或再次调用 `read`；
- 已完成记录的路径被移动、重命名或删除时视为不存在，允许重新下载；
- `message_id` 不提供跨会话隐式定位。跨会话历史结果必须先进入来源会话，再调用 `read({source:{message_id}})`。

物理文件由[主文档的存储路由](./README.md#32-后端选择)提供。Linux 电脑不存在时，`read` 可以直接解析宿主机后备目录中的受支持格式，但仍只接受和返回上述逻辑路径。后备文件不能交给 `computer` 访问或运行；Linux 后来存在时，后续新读取使用 Linux 活动文件树，不自动迁移或合并旧后备文件。

文件类型不由调用参数声明。后端按内容识别：

- PDF 使用文件签名；
- DOCX / XLSX / PPTX 使用 OOXML 容器和内部 content type；
- 其余文件只有在能够严格解码为 UTF-8 文本且不呈现二进制特征时才按文本读取；
- 扩展名只作为提示，不覆盖内容检测结果。

## 2. 两种互斥调用形态

使用本地路径开始读取：

```json
{
  "source": {
    "path": "/home/agent/qq/213628848/file/group_1090411227/report.pdf"
  },
  "selection": {
    "type": "pdf_pages",
    "start_page": 5,
    "end_page": 8
  }
}
```

使用当前会话文件消息开始读取：

```json
{
  "source": {
    "message_id": "1803394108"
  }
}
```

继续上次已经开始的内容读取：

```json
{
  "cursor": "qfr_..."
}
```

顶层只有 `{source, selection?}` 与 `{cursor}` 两种互斥形态。`source` 内只有 `{path}` 与 `{message_id}` 两种互斥形态。选择器只负责指定第一次内容读取的起点或边界；如果单次输出装不下，后续始终使用返回的 `next_cursor`。

可选 `selection`：

| type | 定位维度 | 省略 selection 时 |
|---|---|---|
| `text_lines` | UTF-8 文本行号 | 从第 1 行开始 |
| `pdf_pages` | PDF 页码 | 从第 1 页开始 |
| `docx_blocks` | DOCX 正文块；标题、段落、表格各占一个块 | 从第 1 块开始 |
| `xlsx_range` | 精确工作表名称和可选 A1 区域 | 从第 1 个工作表的已用区域开始 |
| `pptx_slides` | 幻灯片编号 | 从第 1 张开始 |

范围结束值均为包含式。结束值小于起点返回 `invalid_selection`；结束值超过文档末尾时读取到真实末尾为止。选择器类型与实际文件类型不一致时返回 `selection_type_mismatch`，不按扩展名强行解释。

## 3. 参数 JSON Schema declaration

```json
{
  "name": "read",
  "description": "读取当前 QQ 账号文件根目录中的 UTF-8 文本、PDF、DOCX、XLSX 或 PPTX。",
  "parameters": {
    "$defs": {
      "DocxBlocksSelection": {
        "additionalProperties": false,
        "properties": {
          "type": {
            "const": "docx_blocks",
            "description": "按 DOCX 正文块定位；标题、段落和表格各占一个块。",
            "type": "string"
          },
          "start_block": {
            "default": 1,
            "description": "起始块编号，从 1 开始，默认 1。",
            "minimum": 1,
            "type": "integer"
          },
          "end_block": {
            "anyOf": [
              {
                "minimum": 1,
                "type": "integer"
              },
              {
                "type": "null"
              }
            ],
            "default": null,
            "description": "可选结束块编号（包含），必须不小于 start_block。"
          }
        },
        "required": ["type"],
        "type": "object"
      },
      "MessageReadSource": {
        "additionalProperties": false,
        "properties": {
          "message_id": {
            "description": "当前 QQ 会话中的文件消息 ID。",
            "minLength": 1,
            "type": "string",
            "x-coerce-integer": true
          }
        },
        "required": ["message_id"],
        "type": "object"
      },
      "PathReadSource": {
        "additionalProperties": false,
        "properties": {
          "path": {
            "description": "要读取的绝对 Linux 文件路径，必须位于当前 QQ 账号的 file 根目录内。",
            "minLength": 1,
            "type": "string"
          }
        },
        "required": ["path"],
        "type": "object"
      },
      "PdfPagesSelection": {
        "additionalProperties": false,
        "properties": {
          "type": {
            "const": "pdf_pages",
            "description": "按 PDF 页码定位。",
            "type": "string"
          },
          "start_page": {
            "default": 1,
            "description": "起始页码，从 1 开始，默认 1。",
            "minimum": 1,
            "type": "integer"
          },
          "end_page": {
            "anyOf": [
              {
                "minimum": 1,
                "type": "integer"
              },
              {
                "type": "null"
              }
            ],
            "default": null,
            "description": "可选结束页码（包含），必须不小于 start_page。"
          }
        },
        "required": ["type"],
        "type": "object"
      },
      "PptxSlidesSelection": {
        "additionalProperties": false,
        "properties": {
          "type": {
            "const": "pptx_slides",
            "description": "按 PPTX 幻灯片编号定位。",
            "type": "string"
          },
          "start_slide": {
            "default": 1,
            "description": "起始幻灯片编号，从 1 开始，默认 1。",
            "minimum": 1,
            "type": "integer"
          },
          "end_slide": {
            "anyOf": [
              {
                "minimum": 1,
                "type": "integer"
              },
              {
                "type": "null"
              }
            ],
            "default": null,
            "description": "可选结束幻灯片编号（包含），必须不小于 start_slide。"
          }
        },
        "required": ["type"],
        "type": "object"
      },
      "ReadContinueArgs": {
        "additionalProperties": false,
        "properties": {
          "cursor": {
            "description": "上次 read 返回的 next_cursor；使用时不能传 source 或 selection。",
            "maxLength": 2048,
            "minLength": 1,
            "type": "string"
          }
        },
        "required": ["cursor"],
        "type": "object"
      },
      "ReadStartArgs": {
        "additionalProperties": false,
        "properties": {
          "source": {
            "description": "从现有本地路径或当前会话文件消息开始读取。",
            "oneOf": [
              {
                "$ref": "#/$defs/PathReadSource"
              },
              {
                "$ref": "#/$defs/MessageReadSource"
              }
            ]
          },
          "selection": {
            "anyOf": [
              {
                "discriminator": {
                  "mapping": {
                    "docx_blocks": "#/$defs/DocxBlocksSelection",
                    "pdf_pages": "#/$defs/PdfPagesSelection",
                    "pptx_slides": "#/$defs/PptxSlidesSelection",
                    "text_lines": "#/$defs/TextLinesSelection",
                    "xlsx_range": "#/$defs/XlsxRangeSelection"
                  },
                  "propertyName": "type"
                },
                "oneOf": [
                  {
                    "$ref": "#/$defs/TextLinesSelection"
                  },
                  {
                    "$ref": "#/$defs/PdfPagesSelection"
                  },
                  {
                    "$ref": "#/$defs/DocxBlocksSelection"
                  },
                  {
                    "$ref": "#/$defs/XlsxRangeSelection"
                  },
                  {
                    "$ref": "#/$defs/PptxSlidesSelection"
                  }
                ]
              },
              {
                "type": "null"
              }
            ],
            "default": null,
            "description": "可选起点或范围；省略时从文档开头顺序读取。"
          }
        },
        "required": ["source"],
        "type": "object"
      },
      "TextLinesSelection": {
        "additionalProperties": false,
        "properties": {
          "type": {
            "const": "text_lines",
            "description": "按 UTF-8 文本行定位。",
            "type": "string"
          },
          "start_line": {
            "default": 1,
            "description": "起始行号，从 1 开始，默认 1。",
            "minimum": 1,
            "type": "integer"
          },
          "end_line": {
            "anyOf": [
              {
                "minimum": 1,
                "type": "integer"
              },
              {
                "type": "null"
              }
            ],
            "default": null,
            "description": "可选结束行号（包含），必须不小于 start_line。"
          }
        },
        "required": ["type"],
        "type": "object"
      },
      "XlsxRangeSelection": {
        "additionalProperties": false,
        "properties": {
          "type": {
            "const": "xlsx_range",
            "description": "按 XLSX 工作表和 A1 单元格区域定位。",
            "type": "string"
          },
          "sheet": {
            "description": "工作表名称，精确匹配。",
            "minLength": 1,
            "type": "string"
          },
          "cell_range": {
            "anyOf": [
              {
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "default": null,
            "description": "可选 A1 区域，例如 A1 或 A1:F50；省略时从该表已用区域开头读取。"
          }
        },
        "required": ["type", "sheet"],
        "type": "object"
      }
    },
    "anyOf": [
      {
        "$ref": "#/$defs/ReadStartArgs"
      },
      {
        "$ref": "#/$defs/ReadContinueArgs"
      }
    ]
  }
}
```

## 4. 模型正式可见的 TypeScript-like 形态

`qq_file` 打开后，`read` 追加的实际函数签名为：

```ts
// 读取当前 QQ 账号文件根目录中的 UTF-8 文本、PDF、DOCX、XLSX 或 PPTX。
read(args: {
  source: {
    path: string; // 要读取的绝对 Linux 文件路径，必须位于当前 QQ 账号的 file 根目录内。
  } | {
    message_id: string; // 当前 QQ 会话中的文件消息 ID。
  }; // 从现有本地路径或当前会话文件消息开始读取。
  selection?: {
    type: "text_lines"; // 按 UTF-8 文本行定位。
    start_line?: number; // 起始行号，从 1 开始，默认 1。
    end_line?: number; // 可选结束行号（包含），必须不小于 start_line。
  } | {
    type: "pdf_pages"; // 按 PDF 页码定位。
    start_page?: number; // 起始页码，从 1 开始，默认 1。
    end_page?: number; // 可选结束页码（包含），必须不小于 start_page。
  } | {
    type: "docx_blocks"; // 按 DOCX 正文块定位；标题、段落和表格各占一个块。
    start_block?: number; // 起始块编号，从 1 开始，默认 1。
    end_block?: number; // 可选结束块编号（包含），必须不小于 start_block。
  } | {
    type: "xlsx_range"; // 按 XLSX 工作表和 A1 单元格区域定位。
    sheet: string; // 工作表名称，精确匹配。
    cell_range?: string; // 可选 A1 区域，例如 A1 或 A1:F50；省略时从该表已用区域开头读取。
  } | {
    type: "pptx_slides"; // 按 PPTX 幻灯片编号定位。
    start_slide?: number; // 起始幻灯片编号，从 1 开始，默认 1。
    end_slide?: number; // 可选结束幻灯片编号（包含），必须不小于 start_slide。
  }; // 可选起点或范围；省略时从文档开头顺序读取。
} | {
  cursor: string; // 上次 read 返回的 next_cursor；使用时不能传 source 或 selection。 最多 2048 个字符
})
```

当前实际测得：

- `read` 单独函数签名：1146 个字符，UTF-8 1810 字节；
- 与 `download` 同时展开后的完整 namespace：2088 个字符，UTF-8 3474 字节；
- 返回 Schema 不进入常驻提示词。

## 5. cursor 合同

`next_cursor` 是只供 `qq_file.read` 原样回传的不透明游标，不是下载任务，也不需要 `poll / list`：

- 游标绑定当前 QQ 账号、最终精确本地路径、检测到的文件类型、原始 selection 边界、下一读取位置和文件指纹；
- 使用 `message_id` 开始读取时，只有下载已经完成并开始返回内容后才创建 cursor；cursor 绑定完成后的精确本地路径，不继续依赖原消息或当前会话；
- 游标不绑定当前 QQ 会话，因此可以跨会话继续；
- 目标形态为经过认证的无状态游标，不依赖模型上下文或内存中的读取任务，Core 重启后仍可使用；
- 重复提交同一个游标，在文件未变化时返回同一段结果，保证幂等；
- 文件被修改时返回 `file_changed`；精确路径被移动或删除时返回 `not_found`；
- 游标被篡改返回 `invalid_cursor`；由其他 QQ 账号产生的游标返回 `cursor_scope_mismatch`；
- 每次继续读取都重新执行当前 QQ 账号根目录与普通文件检查，游标不能绕过路径权限。

分页不变量：

```text
has_more=true   <=> next_cursor 为非 null
has_more=false  <=> next_cursor 为 null
```

`has_more` 只表示原始 selection 范围内是否还有内容；如果第一次没有 selection，则表示整个文档是否还有内容。游标丢失后，可以重新用 `source + selection` 定位，不提供游标找回列表。

## 6. 各格式的结构化文本

内容读取成功时统一返回 `content` 字符串和 `locations` 结构定位。`content` 是确定性结构化文本，不追求还原原应用的视觉排版。

| file_type | 原生读取内容 | location |
|---|---|---|
| `text` | 严格 UTF-8 文本，保留换行并增加行号；接受 UTF-8 BOM | `text_lines` |
| `pdf` | 按页提取原生文本，保留页边界和标题/作者元数据 | `pdf_pages` |
| `docx` | 主文档中的标题、段落、超链接文本/目标和表格；按正文顺序编号为 block | `docx_blocks` |
| `xlsx` | 工作表顺序、可见状态、单元格坐标、值、公式和可用的缓存值 | `xlsx_range` |
| `pptx` | 幻灯片标题、文本框、表格和演讲者备注；按幻灯片编号 | `pptx_slides` |

具体边界：

- PDF 不保证视觉列布局；如果所选范围全部没有可提取文字但含图像内容，返回 `ocr_required`；混合型 PDF 返回已有文字并附 `partial_ocr_required` warning；
- DOCX 的 block 覆盖主文档标题、段落和表格；页眉、页脚、批注、修订轨迹和嵌入对象不作为正文块；
- XLSX 按工作簿顺序遍历全部工作表，包括隐藏表，并明确返回 `sheet_state`；不执行公式，只返回公式文本以及文件中已经存在的缓存值；不刷新外部数据连接；
- PPTX 不解释图片、视频、图表视觉含义或动画时间线；
- 嵌入媒体、自动外链数据、嵌入对象或复杂布局被省略时，通过 `warnings` 明示；普通超链接只作为文本和 URL 返回，绝不打开或抓取；
- 旧版二进制 `DOC / XLS / PPT` 和宏启用的 `DOCM / XLSM / PPTM` 暂不支持直接读取。

单个逻辑单元也可能超过单次正文上限，例如超长 PDF 页、DOCX 段落或 XLSX 单元格。每个 `locations` 项使用：

```text
starts_mid_unit=true  当前 content 从一个逻辑单元的中间继续
ends_mid_unit=true    当前 content 在一个逻辑单元的中间结束
```

用于明确同一页或同一 block 的内容是否跨返回分页。

## 7. 资源与执行边界

默认限制：

| 项目 | 默认限制 |
|---|---:|
| 可直接解析的本地文件大小 | 256 MiB |
| OOXML 解压后累计内容 | 1 GiB |
| OOXML ZIP entry 数 | 10000 |
| 单次返回 `content` | 8000 个 Unicode 字符 |

这些限制独立于 4 GiB 下载上限：文件可以成功下载但因过大而不能由 `qq_file.read` 直接解析。所有值作为可配置常量实现，但不公开成模型参数。

执行原则：

- 内容解析是同步只读操作，不创建后台读取任务，不产生读取完成事件，也没有独立的读取 `poll / list`；
- `source.message_id` 可以复用或创建 `qq_file.download` 任务；15 秒只约束下载观察阶段，不缩短后续内容解析的既有限制；
- 下载观察结束时仍未完成，返回 `download_pending`；任务继续运行并由 `download.poll / download.list` 管理；
- 下载在观察窗口内完成后，从完成的 `local_path` 开始解析；若格式不受支持或解析失败，错误中保留该路径；
- 解析应在受限工作进程中完成，禁用网络，不执行宏、脚本、PDF JavaScript、外部链接或嵌入对象；
- 触发输入大小限制返回 `file_too_large_to_read`；
- 触发 OOXML 解压安全限制返回 `archive_safety_limit_exceeded`；
- 触发解析时间、内存或结构复杂度限制返回 `read_limit_exceeded`；
- 所有上述错误都保留本地文件并返回精确路径；`qq_file` 不自动展开或调用 `computer`。

## 8. 成功与错误返回原则

内容读取成功结果固定包含：

```text
ok
outcome=content
path
file_type
size_bytes
document
content
locations
has_more
next_cursor
warnings
```

下载尚未完成的成功受理结果固定包含：

```text
ok
outcome=download_pending
download.download_id
download.message_id
download.status
download.bytes_downloaded
download.total_bytes
download.progress_percent
download.target_path
download.updated_at
```

`download_pending` 表示文件仍在下载，不能把 `target_path` 当作可读取的完成文件；此结果没有 `content`、`locations` 或 `next_cursor`。

`starts_mid_unit / ends_mid_unit` 位于每个 `locations` 项内，而不是成功结果顶层。

`document` 提供轻量格式元数据：

- 文本：UTF-8 编码和总行数；
- PDF：页数、标题、作者；
- DOCX：正文 block 总数；
- XLSX：工作表总数；
- PPTX：幻灯片总数。

错误结果统一为 `ok=false + error`。路径解析错误、下载来源错误和内容解析错误使用同一错误信封；已经存在本地完成文件时，`unsupported_file_type`、`unsupported_text_encoding`、`password_required`、`ocr_required`、大小/安全限制和解析失败必须在 `error.path` 返回已验证的精确 Linux 形态逻辑路径。只有 Linux 电脑存在时，该路径才能继续交给 `computer` 处理。

## 9. 返回 JSON Schema（第一版）

> 当前 `ToolContract` 只注入参数合同；返回 Schema 作为本文档中的正式合同保存，并由自动化合同测试校验代表性运行时结果。

```json
{
  "$defs": {
    "TextDocumentInfo": {
      "additionalProperties": false,
      "properties": {
        "type": {
          "const": "text",
          "type": "string"
        },
        "encoding": {
          "const": "utf-8",
          "type": "string"
        },
        "total_lines": {
          "minimum": 0,
          "type": "integer"
        }
      },
      "required": ["type", "encoding", "total_lines"],
      "type": "object"
    },
    "PdfDocumentInfo": {
      "additionalProperties": false,
      "properties": {
        "type": {
          "const": "pdf",
          "type": "string"
        },
        "page_count": {
          "minimum": 0,
          "type": "integer"
        },
        "title": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ]
        },
        "author": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ]
        }
      },
      "required": ["type", "page_count", "title", "author"],
      "type": "object"
    },
    "DocxDocumentInfo": {
      "additionalProperties": false,
      "properties": {
        "type": {
          "const": "docx",
          "type": "string"
        },
        "block_count": {
          "minimum": 0,
          "type": "integer"
        }
      },
      "required": ["type", "block_count"],
      "type": "object"
    },
    "XlsxDocumentInfo": {
      "additionalProperties": false,
      "properties": {
        "type": {
          "const": "xlsx",
          "type": "string"
        },
        "sheet_count": {
          "minimum": 0,
          "type": "integer"
        }
      },
      "required": ["type", "sheet_count"],
      "type": "object"
    },
    "PptxDocumentInfo": {
      "additionalProperties": false,
      "properties": {
        "type": {
          "const": "pptx",
          "type": "string"
        },
        "slide_count": {
          "minimum": 0,
          "type": "integer"
        }
      },
      "required": ["type", "slide_count"],
      "type": "object"
    },
    "DocumentInfo": {
      "discriminator": {
        "mapping": {
          "docx": "#/$defs/DocxDocumentInfo",
          "pdf": "#/$defs/PdfDocumentInfo",
          "pptx": "#/$defs/PptxDocumentInfo",
          "text": "#/$defs/TextDocumentInfo",
          "xlsx": "#/$defs/XlsxDocumentInfo"
        },
        "propertyName": "type"
      },
      "oneOf": [
        {
          "$ref": "#/$defs/TextDocumentInfo"
        },
        {
          "$ref": "#/$defs/PdfDocumentInfo"
        },
        {
          "$ref": "#/$defs/DocxDocumentInfo"
        },
        {
          "$ref": "#/$defs/XlsxDocumentInfo"
        },
        {
          "$ref": "#/$defs/PptxDocumentInfo"
        }
      ]
    },
    "TextLinesLocation": {
      "additionalProperties": false,
      "properties": {
        "type": {
          "const": "text_lines",
          "type": "string"
        },
        "start_line": {
          "minimum": 1,
          "type": "integer"
        },
        "end_line": {
          "minimum": 1,
          "type": "integer"
        },
        "starts_mid_unit": {
          "type": "boolean"
        },
        "ends_mid_unit": {
          "type": "boolean"
        }
      },
      "required": [
        "type",
        "start_line",
        "end_line",
        "starts_mid_unit",
        "ends_mid_unit"
      ],
      "type": "object"
    },
    "PdfPagesLocation": {
      "additionalProperties": false,
      "properties": {
        "type": {
          "const": "pdf_pages",
          "type": "string"
        },
        "start_page": {
          "minimum": 1,
          "type": "integer"
        },
        "end_page": {
          "minimum": 1,
          "type": "integer"
        },
        "starts_mid_unit": {
          "type": "boolean"
        },
        "ends_mid_unit": {
          "type": "boolean"
        }
      },
      "required": [
        "type",
        "start_page",
        "end_page",
        "starts_mid_unit",
        "ends_mid_unit"
      ],
      "type": "object"
    },
    "DocxBlocksLocation": {
      "additionalProperties": false,
      "properties": {
        "type": {
          "const": "docx_blocks",
          "type": "string"
        },
        "start_block": {
          "minimum": 1,
          "type": "integer"
        },
        "end_block": {
          "minimum": 1,
          "type": "integer"
        },
        "starts_mid_unit": {
          "type": "boolean"
        },
        "ends_mid_unit": {
          "type": "boolean"
        }
      },
      "required": [
        "type",
        "start_block",
        "end_block",
        "starts_mid_unit",
        "ends_mid_unit"
      ],
      "type": "object"
    },
    "XlsxRangeLocation": {
      "additionalProperties": false,
      "properties": {
        "type": {
          "const": "xlsx_range",
          "type": "string"
        },
        "sheet": {
          "minLength": 1,
          "type": "string"
        },
        "sheet_index": {
          "minimum": 1,
          "type": "integer"
        },
        "sheet_state": {
          "enum": ["visible", "hidden", "very_hidden"],
          "type": "string"
        },
        "cell_range": {
          "minLength": 1,
          "type": "string"
        },
        "starts_mid_unit": {
          "type": "boolean"
        },
        "ends_mid_unit": {
          "type": "boolean"
        }
      },
      "required": [
        "type",
        "sheet",
        "sheet_index",
        "sheet_state",
        "cell_range",
        "starts_mid_unit",
        "ends_mid_unit"
      ],
      "type": "object"
    },
    "PptxSlidesLocation": {
      "additionalProperties": false,
      "properties": {
        "type": {
          "const": "pptx_slides",
          "type": "string"
        },
        "start_slide": {
          "minimum": 1,
          "type": "integer"
        },
        "end_slide": {
          "minimum": 1,
          "type": "integer"
        },
        "starts_mid_unit": {
          "type": "boolean"
        },
        "ends_mid_unit": {
          "type": "boolean"
        }
      },
      "required": [
        "type",
        "start_slide",
        "end_slide",
        "starts_mid_unit",
        "ends_mid_unit"
      ],
      "type": "object"
    },
    "ReadLocation": {
      "discriminator": {
        "mapping": {
          "docx_blocks": "#/$defs/DocxBlocksLocation",
          "pdf_pages": "#/$defs/PdfPagesLocation",
          "pptx_slides": "#/$defs/PptxSlidesLocation",
          "text_lines": "#/$defs/TextLinesLocation",
          "xlsx_range": "#/$defs/XlsxRangeLocation"
        },
        "propertyName": "type"
      },
      "oneOf": [
        {
          "$ref": "#/$defs/TextLinesLocation"
        },
        {
          "$ref": "#/$defs/PdfPagesLocation"
        },
        {
          "$ref": "#/$defs/DocxBlocksLocation"
        },
        {
          "$ref": "#/$defs/XlsxRangeLocation"
        },
        {
          "$ref": "#/$defs/PptxSlidesLocation"
        }
      ]
    },
    "ReadWarning": {
      "additionalProperties": false,
      "properties": {
        "code": {
          "enum": [
            "partial_ocr_required",
            "embedded_media_omitted",
            "formula_value_unavailable",
            "external_link_omitted",
            "layout_simplified"
          ],
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
    "ReadDownloadProgress": {
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
        "status": {
          "enum": ["queued", "resolving", "downloading", "verifying"],
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
        "updated_at": {
          "format": "date-time",
          "type": "string"
        }
      },
      "required": [
        "download_id",
        "message_id",
        "status",
        "bytes_downloaded",
        "total_bytes",
        "progress_percent",
        "target_path",
        "updated_at"
      ],
      "type": "object"
    },
    "ReadDownloadPending": {
      "additionalProperties": false,
      "properties": {
        "ok": {
          "const": true,
          "type": "boolean"
        },
        "outcome": {
          "const": "download_pending",
          "type": "string"
        },
        "download": {
          "$ref": "#/$defs/ReadDownloadProgress"
        }
      },
      "required": ["ok", "outcome", "download"],
      "type": "object"
    },
    "ReadSuccess": {
      "additionalProperties": false,
      "properties": {
        "ok": {
          "const": true,
          "type": "boolean"
        },
        "outcome": {
          "const": "content",
          "type": "string"
        },
        "path": {
          "minLength": 1,
          "type": "string"
        },
        "file_type": {
          "enum": ["text", "pdf", "docx", "xlsx", "pptx"],
          "type": "string"
        },
        "size_bytes": {
          "minimum": 0,
          "type": "integer"
        },
        "document": {
          "$ref": "#/$defs/DocumentInfo"
        },
        "content": {
          "maxLength": 8000,
          "type": "string"
        },
        "locations": {
          "items": {
            "$ref": "#/$defs/ReadLocation"
          },
          "type": "array"
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
            "$ref": "#/$defs/ReadWarning"
          },
          "type": "array"
        }
      },
      "required": [
        "ok",
        "outcome",
        "path",
        "file_type",
        "size_bytes",
        "document",
        "content",
        "locations",
        "has_more",
        "next_cursor",
        "warnings"
      ],
      "type": "object"
    },
    "ReadErrorDetails": {
      "additionalProperties": false,
      "properties": {
        "size_bytes": {
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
        "limit_bytes": {
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
        "detected_file_type": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ]
        },
        "selection_type": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ]
        },
        "actual_file_type": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ]
        }
      },
      "type": "object"
    },
    "ReadError": {
      "additionalProperties": false,
      "properties": {
        "code": {
          "enum": [
            "no_current_qq_session",
            "message_not_found",
            "not_file_message",
            "qq_adapter_unavailable",
            "file_too_large",
            "insufficient_disk_space",
            "source_unavailable",
            "transport_error",
            "write_failed",
            "size_mismatch",
            "verification_failed",
            "download_interrupted",
            "download_stopped",
            "not_found",
            "path_outside_qq_file_root",
            "not_regular_file",
            "symlink_not_allowed",
            "unsupported_file_type",
            "unsupported_text_encoding",
            "file_too_large_to_read",
            "archive_safety_limit_exceeded",
            "password_required",
            "ocr_required",
            "invalid_selection",
            "selection_type_mismatch",
            "invalid_cursor",
            "cursor_scope_mismatch",
            "file_changed",
            "read_limit_exceeded",
            "parse_failed",
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
        "path": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ]
        },
        "details": {
          "anyOf": [
            {
              "$ref": "#/$defs/ReadErrorDetails"
            },
            {
              "type": "null"
            }
          ]
        }
      },
      "required": ["code", "message", "retryable"],
      "type": "object"
    },
    "ReadErrorResult": {
      "additionalProperties": false,
      "properties": {
        "ok": {
          "const": false,
          "type": "boolean"
        },
        "error": {
          "$ref": "#/$defs/ReadError"
        }
      },
      "required": ["ok", "error"],
      "type": "object"
    }
  },
  "anyOf": [
    {
      "$ref": "#/$defs/ReadSuccess"
    },
    {
      "$ref": "#/$defs/ReadDownloadPending"
    },
    {
      "$ref": "#/$defs/ReadErrorResult"
    }
  ]
}
```

## 10. 典型运行时返回

读取 UTF-8 文本的第一段：

```json
{
  "ok": true,
  "outcome": "content",
  "path": "/home/agent/qq/213628848/file/private_123456789/notes.md",
  "file_type": "text",
  "size_bytes": 18234,
  "document": {
    "type": "text",
    "encoding": "utf-8",
    "total_lines": 420
  },
  "content": "1\t# 项目记录\n2\t\n3\t第一部分……",
  "locations": [
    {
      "type": "text_lines",
      "start_line": 1,
      "end_line": 120,
      "starts_mid_unit": false,
      "ends_mid_unit": false
    }
  ],
  "has_more": true,
  "next_cursor": "qfr_...",
  "warnings": []
}
```

指定 XLSX 区域：

```json
{
  "ok": true,
  "outcome": "content",
  "path": "/home/agent/qq/213628848/file/group_1090411227/data.xlsx",
  "file_type": "xlsx",
  "size_bytes": 93412,
  "document": {
    "type": "xlsx",
    "sheet_count": 4
  },
  "content": "[Sheet: 汇总 | A1:C3]\n\tA\tB\tC\n1\t项目\t金额\t公式\n2\tA\t12\t=SUM(B2:B3) [cached: 30]\n3\tB\t18\t",
  "locations": [
    {
      "type": "xlsx_range",
      "sheet": "汇总",
      "sheet_index": 1,
      "sheet_state": "visible",
      "cell_range": "A1:C3",
      "starts_mid_unit": false,
      "ends_mid_unit": false
    }
  ],
  "has_more": false,
  "next_cursor": null,
  "warnings": []
}
```

使用 `message_id` 开始读取，但下载在 15 秒内尚未完成：

```json
{
  "ok": true,
  "outcome": "download_pending",
  "download": {
    "download_id": "qfd_01K4...",
    "message_id": "1803394108",
    "status": "downloading",
    "bytes_downloaded": 2771384,
    "total_bytes": 10485760,
    "progress_percent": 26.43,
    "target_path": "/home/agent/qq/213628848/file/group_1090411227/report.pdf",
    "updated_at": "2026-08-30T14:21:15+08:00"
  }
}
```

扫描版 PDF 无可提取文本：

```json
{
  "ok": false,
  "error": {
    "code": "ocr_required",
    "message": "所选 PDF 页面没有可提取的原生文字，需要 OCR；qq_file.read 暂不支持 OCR。",
    "retryable": false,
    "path": "/home/agent/qq/213628848/file/group_1090411227/scan.pdf"
  }
}
```

不支持直接读取的文件：

```json
{
  "ok": false,
  "error": {
    "code": "unsupported_file_type",
    "message": "该文件类型不能由 qq_file.read 直接读取。",
    "retryable": false,
    "path": "/home/agent/qq/213628848/file/group_1090411227/archive.zip",
    "details": {
      "size_bytes": 10485760,
      "limit_bytes": null,
      "detected_file_type": "zip",
      "selection_type": null,
      "actual_file_type": null
    }
  }
}
```

## 11. 返回合同与上下文消耗

- 返回 JSON Schema、格式规则和错误枚举不会随 namespace 注入模型；
- `read` 常驻增量是 1146 个字符、UTF-8 1810 字节；
- `download + read` 的完整 namespace 是 2088 个字符、UTF-8 3474 字节；
- 工具实际执行后，单次 `content` 最多 8000 个 Unicode 字符，另加轻量结构元数据；
- 不返回 OOXML 原始 XML、PDF 对象树、样式全集、二进制内容、调试堆栈或 parser 内部状态；
- `cursor` 最长接受 2048 字符，但目标实现应保持为短且不透明的 token。
