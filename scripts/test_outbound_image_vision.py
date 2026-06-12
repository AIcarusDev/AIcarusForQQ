"""验证 outbound_image 规范化后能稳定通过 SiliconFlow 视觉接口。

用法（在项目根目录）：
    python scripts/test_outbound_image_vision.py
可选：
    python scripts/test_outbound_image_vision.py 路径1 路径2 ...

无参数时自动取 cache/image 下最近修改的若干张图（含动图/大图/静态图各类样本），
分别走 normalize_for_openai_compatible，再用 .env 里的 SiliconFlow Key 调主模型一次，
打印每张图的"规范化后大小 + 服务端响应"。
"""
from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 加载 .env
def _load_env():
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

from llm.media.outbound_image import normalize_for_openai_compatible  # noqa: E402

try:
    from openai import OpenAI
except ImportError:
    print("缺少 openai 包，pip install openai")
    sys.exit(1)


def pick_default_samples() -> list[Path]:
    cache = ROOT / "cache" / "image"
    if not cache.exists():
        return []
    images = [
        p for p in cache.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    ]
    images.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    picks: dict[str, Path] = {}
    for p in images:
        suf = p.suffix.lower()
        if suf not in picks:
            picks[suf] = p
        if len(picks) >= 5:
            break
    return list(picks.values())


def main(argv: list[str]) -> int:
    if argv:
        paths = [Path(p) for p in argv]
    else:
        paths = pick_default_samples()
    if not paths:
        print("未指定路径，且 cache/image 下未找到样本图")
        return 1

    api_key = os.environ.get("MODEL_PROVIDER_CUSTOM_API_KEY") or os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        print("未在 .env 中找到 MODEL_PROVIDER_CUSTOM_API_KEY 或 SILICONFLOW_API_KEY")
        return 1

    base_url = "https://api.siliconflow.cn/v1"
    model = "Pro/moonshotai/Kimi-K2.6"
    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"模型: {model} @ {base_url}")
    print(f"共 {len(paths)} 张图")
    print("-" * 80)

    fail = 0
    for p in paths:
        if not p.exists():
            print(f"[skip] 不存在: {p}")
            continue
        raw = p.read_bytes()
        b64 = base64.b64encode(raw).decode()
        norm = normalize_for_openai_compatible(b64, "image/jpeg")
        if not norm:
            print(f"[norm-FAIL] {p.name} ({len(raw)} B)")
            fail += 1
            continue
        out_b64, out_mime = norm
        out_size = len(base64.b64decode(out_b64))
        passthrough = out_b64 == b64
        print(f"\n► {p.name}")
        print(f"  原始: {len(raw)} B   -> 规范化: {out_size} B  mime={out_mime}  {'(透传)' if passthrough else '(重编码)'}")

        data_url = f"data:{out_mime};base64,{out_b64}"
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": "用不超过20字描述图中内容。"},
                    ],
                }],
                max_tokens=64,
                temperature=0.3,
            )
            text = (resp.choices[0].message.content or "").strip().replace("\n", " ")
            dt = time.time() - t0
            print(f"  [OK] {dt:.2f}s  → {text[:80]}")
        except Exception as e:
            dt = time.time() - t0
            print(f"  [FAIL] {dt:.2f}s  → {e}")
            fail += 1

    print("\n" + "=" * 80)
    print(f"完成。失败 {fail} / 总 {len(paths)}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
