"""get_user_avatar.py — 通过 QQ 号获取用户头像

调用 QQ 公开头像接口（q1.qlogo.cn），无需 NapCat 连接。
条件启用：仅在 config["vision"] 为 True（默认）时加载。
最多每次 AI 响应调用 3 次（max_calls_per_response=3）。
"""

import logging
import urllib.request

logger = logging.getLogger("AICQ.tools")

_AVATAR_URL = "https://q1.qlogo.cn/g?b=qq&nk={qq}&s=640"

ALWAYS_AVAILABLE: bool = False

DECLARATION: dict = {
    "name": "get_user_avatar",
    "description": (
        "通过 QQ 号获取对应用户的头像图片。"
        "可用于查看任意用户（包括你自己）的当前 QQ 头像。"
        "返回内容仅自己可见。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "qq_number": {
                "type": "string",
                "description": "目标用户的 QQ 号码（纯数字字符串）。",
            },
            "motivation": {
                "type": "string",
            },
        },
        "required": ["qq_number", "motivation"],
    },
}


def condition(config: dict) -> bool:
    return config.get("vision", True)


def execute(**kwargs) -> dict:
    qq = str(kwargs.get("qq_number", "")).strip()
    if not qq or not qq.isdigit():
        logger.warning("[tools] get_user_avatar: 无效QQ号 qq=%s", qq)
        return {"error": f"无效的 QQ 号：{qq!r}，请传入纯数字字符串。"}

    url = _AVATAR_URL.format(qq=qq)
    try:
        logger.info("[tools] get_user_avatar: 开始获取头像 qq=%s", qq)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data: bytes = resp.read()
            content_type: str = resp.headers.get("Content-Type", "image/jpeg")
            mime = content_type.split(";")[0].strip() or "image/jpeg"
    except Exception as e:
        logger.warning("[tools] get_user_avatar: 获取失败 qq=%s — %s", qq, e)
        return {"error": f"获取头像失败：{e}"}

    if not data:
        logger.warning("[tools] get_user_avatar: 服务器返回为空 qq=%s", qq)
        return {"error": "服务器返回数据为空，可能该 QQ 号不存在。"}

    logger.info("[tools] get_user_avatar: 获取成功 qq=%s", qq)
    return {
        "qq_number": qq,
        "note": "头像已返回，仅你自己可见。",
        "_multimodal_parts": [
            {
                "mime_type": mime,
                "display_name": f"avatar_{qq}.jpg",
                "data": data,
            }
        ],
    }
