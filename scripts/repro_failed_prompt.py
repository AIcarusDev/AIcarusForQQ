"""读取 logs/failed_prompts/*.json，二分定位是哪张 image 触发 SiliconFlow 20015。

用法（项目根）:
    python scripts/repro_failed_prompt.py [dump 文件路径]
不指定则取 logs/failed_prompts 下最新一份。

策略:
  1. 完整重放一次 → 复现报错
  2. 抽掉所有图片 → 应当通过（证明问题确为图片）
  3. 二分搜索 image_url：逐步缩到单张元凶
  4. 元凶定位后，把它的 data URL 解码到 logs/failed_prompts/culprit.bin 供进一步分析
"""
from __future__ import annotations

import base64
import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

_load_env()

try:
    from openai import OpenAI
except ImportError:
    print("缺少 openai 包"); sys.exit(1)


def pick_latest_dump() -> Path | None:
    d = ROOT / "logs" / "failed_prompts"
    if not d.exists():
        return None
    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def iter_image_parts(messages):
    """生成 (msg_idx, part_idx, part) 给所有 image_url 类型 part"""
    for i, m in enumerate(messages):
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for j, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "image_url":
                yield i, j, part


def with_images_masked(messages, keep_indices: set[int]):
    """返回新 messages，仅保留 keep_indices 对应的 image part；其余 image part 移除（不留占位）"""
    out = copy.deepcopy(messages)
    counter = 0
    for m in out:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        new_content = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                if counter in keep_indices:
                    new_content.append(part)
                counter += 1
            else:
                new_content.append(part)
        m["content"] = new_content
    return out


def call(client, model, messages) -> tuple[bool, str]:
    """返回 (success, info)。info 是 错误描述 或 OK"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=8,
            temperature=0.1,
        )
        _ = resp.choices[0].message.content
        return True, "OK"
    except Exception as e:
        return False, str(e)


def main(argv: list[str]) -> int:
    dump_path = Path(argv[0]) if argv else pick_latest_dump()
    if not dump_path or not dump_path.exists():
        print("找不到 dump 文件，先让程序触发一次 20015 让 provider 自动 dump 到 logs/failed_prompts/")
        return 1
    print(f"加载 dump: {dump_path}")
    payload = json.loads(dump_path.read_text(encoding="utf-8"))
    messages = payload["messages"]
    model = payload.get("model") or "Pro/moonshotai/Kimi-K2.6"

    images = list(iter_image_parts(messages))
    print(f"原始报错: {payload.get('error','?')[:200]}")
    print(f"消息数: {len(messages)}  image_url 数: {len(images)}")
    if not images:
        print("dump 里没有 image_url，无法复现图片问题")
        return 1

    api_key = os.environ.get("MODEL_PROVIDER_CUSTOM_API_KEY") or os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        print("未找到 API key"); return 1
    client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

    # 1. 完整重放
    print("\n[1/4] 完整重放……")
    ok, info = call(client, model, messages)
    print(f"  -> {'OK' if ok else 'FAIL'}: {info[:160]}")
    if ok:
        print("  ✓ 重放未复现，可能是缓存或临时问题")
        return 0

    # 2. 无图
    print("\n[2/4] 抽掉所有图片重放……")
    ok2, info2 = call(client, model, with_images_masked(messages, set()))
    print(f"  -> {'OK' if ok2 else 'FAIL'}: {info2[:160]}")
    if not ok2:
        print("  ✗ 无图也失败，问题非图片，本脚本不适用")
        return 2

    # 3. 二分
    print(f"\n[3/4] 二分定位 {len(images)} 张图……")
    suspects = list(range(len(images)))
    while len(suspects) > 1:
        mid = len(suspects) // 2
        left, right = suspects[:mid], suspects[mid:]
        # 先测 left
        ok_l, _ = call(client, model, with_images_masked(messages, set(left)))
        print(f"  保留 {len(left)} 张 ({left[0]}..{left[-1]}) -> {'OK' if ok_l else 'FAIL'}")
        if not ok_l:
            suspects = left
        else:
            suspects = right
            ok_r, _ = call(client, model, with_images_masked(messages, set(right)))
            print(f"  保留 {len(right)} 张 ({right[0]}..{right[-1]}) -> {'OK' if ok_r else 'FAIL'}")
            if ok_r:
                print("  奇怪：两半都 OK，可能是组合效应，放弃二分")
                return 3
        time.sleep(0.3)

    culprit_idx = suspects[0]
    print(f"\n  ★ 元凶: image #{culprit_idx}")

    # 4. dump 元凶
    print("\n[4/4] 保存元凶……")
    msg_idx, part_idx, part = images[culprit_idx]
    url = part["image_url"]["url"]
    print(f"  位置: message[{msg_idx}].content[{part_idx}]")
    print(f"  URL 头: {url[:80]}...")
    if url.startswith("data:"):
        try:
            header, b64 = url.split(",", 1)
            mime = header.split(";")[0].replace("data:", "")
            ext = mime.split("/")[-1]
            raw = base64.b64decode(b64)
            out = dump_path.parent / f"culprit.{ext}"
            out.write_bytes(raw)
            print(f"  ✓ 已保存 {len(raw)} B -> {out}")
            # 用 PIL 打开看看
            try:
                from PIL import Image
                with Image.open(out) as im:
                    print(f"  PIL: format={im.format} mode={im.mode} size={im.size} frames={getattr(im,'n_frames',1)}")
            except Exception as e:
                print(f"  PIL 也打不开: {e}")
        except Exception as e:
            print(f"  解码失败: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
