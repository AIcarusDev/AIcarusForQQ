"""test_weather.py — 测试和风天气 API 是否正常工作

直接运行：
    python test_weather.py
    python test_weather.py 上海
    python test_weather.py Tokyo
"""

import json
import os
import sys

# 把 src 加入路径，方便直接 import tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# 手动加载 .env（正式运行时 main.py 会调 load_dotenv）
def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        print("[warn] .env 文件不存在，请确保环境变量已设置")
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

_load_env()

# 检查必要配置
_missing = []
if not os.environ.get("QWEATHER_API_KEY"):
    _missing.append("QWEATHER_API_KEY")
if not os.environ.get("QWEATHER_API_HOST"):
    _missing.append("QWEATHER_API_HOST  # 在控制台-设置中查看，格式如 abc123.xyz.qweatherapi.com")
if _missing:
    print("[ERROR] .env 缺少以下配置：")
    for m in _missing:
        print(f"  {m}")
    sys.exit(1)

from tools.get_weather import execute  # noqa: E402


def run_test(city: str):
    print(f"\n{'='*50}")
    print(f"查询城市：{city}")
    print("="*50)

    result = execute(city=city)

    if "error" in result:
        print(f"[ERROR] {result['error']}")
        return

    print(f"城市：{result['city']}  ({result['region']})")
    print()

    cur = result["current"]
    print(f"【实时天气】（观测时间：{cur['obs_time']}）")
    print(f"  天气状况：{cur['text']}")
    print(f"  温度：{cur['temp']}°C  体感：{cur['feels_like']}°C")
    print(f"  风向/风力：{cur['wind_dir']} {cur['wind_scale']}级")
    print(f"  湿度：{cur['humidity']}%")
    print()

    if result.get("forecast_3d"):
        print("【未来3天预报】")
        for day in result["forecast_3d"]:
            print(
                f"  {day['date']}  {day['temp_min']}~{day['temp_max']}°C  "
                f"白天:{day['text_day']} 夜间:{day['text_night']}  "
                f"UV:{day['uv_index']}  湿度:{day['humidity']}%"
            )

    print()
    print("原始返回数据：")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cities = sys.argv[1:] if len(sys.argv) > 1 else ["北京", "上海", "London"]
    for c in cities:
        run_test(c)
