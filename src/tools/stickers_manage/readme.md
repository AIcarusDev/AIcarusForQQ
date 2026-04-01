由于函数工具数量增多，且无法阉割功能，未来可能会把同类工具合并为高内聚的工具
例如将表情包的增删查改合并为：

```json
{
  "name": "manage_stickers",
  "description": "表情管理：用于增删查改自己的表情包收藏。",
  "parameters": {
    "type": "object",
    "properties": {
      "motivation": {
        "type": "string"
      },
      "action": {
        "type": "string",
        "enum": ["list", "save", "delete", "update"],
        "description": "要执行的具体动作"
      }
    },
    "required": ["motivation", "action"],
    "oneOf": [
      {
        "description": "查看收藏的表情包",
        "properties": {
          "action": { "const": "list" }
        }
      },
      {
        "description": "保存表情包",
        "properties": {
          "action": { "const": "save" },
          "image_ref": {
            "type": "string",
            "description": "目标图片/表情的 ref，12位十六进制字符串"
          },
          "description": {
            "type": "string",
            "description": "描述这个表情包的适用场景"
          }
        },
        "required": ["image_ref", "description"]
      },
      {
        "description": "删除表情包",
        "properties": {
          "action": { "const": "delete" },
          "sticker_id": {
            "type": "string",
            "description": "要删除的表情包 ID（如 '003'）"
          }
        },
        "required": ["sticker_id"]
      },
      {
        "description": "修改描述",
        "properties": {
          "action": { "const": "update" },
          "sticker_id": {
            "type": "string",
            "description": "要修改的表情包 ID（如 '003'）"
          },
          "description": {
            "type": "string",
            "description": "新的描述"
          }
        },
        "required": ["sticker_id", "description"]
      }
    ]
  }
}
```
