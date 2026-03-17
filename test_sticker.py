"""test_sticker.py — 测试以"收藏表情"形式发送图片

使用方法：
  1. 先停止主 bot（主 bot 占用 ws://127.0.0.1:8078）
  2. 运行本脚本：python test_sticker.py
  3. NapCat 会自动重连到本脚本的临时 WS 服务
  4. 发送成功后脚本自动退出
  5. 重新启动主 bot
"""

import asyncio
import base64
import json
import uuid
from pathlib import Path

import websockets
from websockets.asyncio.server import ServerConnection

# ── 配置 ──────────────────────────────────────────────
IMAGE_PATH = Path(r"E:\Aic_forQ\core\data\self_image\self_image.png")
TARGET_USER_ID = 2514624910
WS_HOST = "127.0.0.1"
WS_PORT = 8078
# ──────────────────────────────────────────────────────

_done = asyncio.Event()


async def _handler(ws: ServerConnection) -> None:
    print(f"[+] NapCat 已连接: {ws.remote_address}")

    # 读取图片并 base64 编码
    img_b64 = base64.b64encode(IMAGE_PATH.read_bytes()).decode()

    # 构造发送请求
    # 关键：data 中加 "sub_type": 1 → QQ 识别为收藏表情，而非普通图片
    echo_id = str(uuid.uuid4())
    payload = {
        "action": "send_msg",
        "params": {
            "user_id": TARGET_USER_ID,
            "message_type": "private",
            "message": [
                {
                    "type": "image",
                    "data": {
                        "file": f"base64://{img_b64}",
                        "sub_type": 1,
                    },
                }
            ],
        },
        "echo": echo_id,
    }

    print(f"[>] 发送私信给 {TARGET_USER_ID}（sub_type=1，收藏表情形式）...")
    await ws.send(json.dumps(payload))

    # 等待 API 响应
    async for raw in ws:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if data.get("echo") == echo_id:
            status = data.get("status", "?")
            retcode = data.get("retcode", "?")
            print(f"[<] API 响应: status={status}, retcode={retcode}")
            if status == "ok":
                print("[✓] 发送成功！请在 QQ 上确认消息是否显示为[动画表情]而非[图片]")
            else:
                print(f"[✗] 发送失败: {data}")
            break

    _done.set()


async def main() -> None:
    if not IMAGE_PATH.exists():
        print(f"[✗] 图片不存在: {IMAGE_PATH}")
        return

    print(f"[*] 测试图片: {IMAGE_PATH.name} ({IMAGE_PATH.stat().st_size} bytes)")
    print(f"[*] 目标账号: {TARGET_USER_ID}")
    print(f"[*] 启动临时 WS 服务: ws://{WS_HOST}:{WS_PORT}")
    print("[*] 等待 NapCat 连接（需先停主 bot）...\n")

    async with websockets.serve(_handler, WS_HOST, WS_PORT):
        await _done.wait()

    print("\n[*] 测试完毕，脚本退出。")


if __name__ == "__main__":
    asyncio.run(main())
