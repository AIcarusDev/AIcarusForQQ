"""get_weather.py — 和风天气查询

调用和风天气 API，返回指定城市的实时天气 + 未来 3 天预报。
API Key 通过环境变量 QWEATHER_API_KEY 注入。
"""

import logging
import os

import httpx

logger = logging.getLogger("AICQ.tools")

# 请求超时（秒）
_TIMEOUT = 15


def _urls(api_host: str) -> tuple[str, str, str]:
    """根据 API Host 构造 GeoAPI 和天气接口的完整 URL。"""
    base = f"https://{api_host}"
    return (
        f"{base}/geo/v2/city/lookup",   # 城市搜索
        f"{base}/v7/weather/now",        # 实时天气
        f"{base}/v7/weather/3d",         # 3天预报
    )

DECLARATION: dict = {
    "max_calls_per_response": 2,
    "name": "get_weather",
    "description": (
        "查询指定城市的天气情况，包括实时天气（温度、体感温度、天气状况、风向风力、湿度）"
        "以及未来 3 天的天气预报（最高/最低温、天气状况）。"
        "当用户提到天气、温度、下雨、出行等相关话题时可以主动调用。"
        "返回内容仅自己可见。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "要查询的城市名称，例如「北京」「上海」「Tokyo」等，中英文均可。",
            },
            "motivation": {
                "type": "string",
                "description": "调用此工具的动机或原因。",
            },
        },
        "required": ["city"],
    },
}


def execute(city: str, **kwargs) -> dict:
    api_key = os.environ.get("QWEATHER_API_KEY", "").strip()
    api_host = os.environ.get("QWEATHER_API_HOST", "").strip()
    if not api_key:
        logger.warning("[tools] get_weather: QWEATHER_API_KEY 未配置")
        return {"error": "QWEATHER_API_KEY 未配置，无法查询天气"}
    if not api_host:
        logger.warning("[tools] get_weather: QWEATHER_API_HOST 未配置")
        return {"error": "QWEATHER_API_HOST 未配置，请在控制台-设置中查看你的 API Host 并填入 .env"}

    geo_url, now_url, daily_url = _urls(api_host)
    headers = {"X-QW-Api-Key": api_key}

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            # Step 1: 城市搜索，获取 location_id
            geo_resp = client.get(
                geo_url,
                params={"location": city, "number": 1},
                headers=headers,
            )
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()

            if geo_data.get("code") != "200" or not geo_data.get("location"):
                logger.warning("[tools] get_weather: 城市未找到 city=%r code=%s", city, geo_data.get("code"))
                return {"error": f"未找到城市：{city}（GeoAPI code={geo_data.get('code')}）"}

            loc = geo_data["location"][0]
            location_id = loc["id"]
            city_name = loc["name"]
            country = loc.get("country", "")
            adm1 = loc.get("adm1", "")
            logger.info("[tools] get_weather: 城市解析 city=%r -> %s (%s) id=%s", city, city_name, adm1, location_id)

            # Step 2: 实时天气
            now_resp = client.get(now_url, params={"location": location_id}, headers=headers)
            now_resp.raise_for_status()
            now_data = now_resp.json()

            # Step 3: 3天预报
            daily_resp = client.get(daily_url, params={"location": location_id}, headers=headers)
            daily_resp.raise_for_status()
            daily_data = daily_resp.json()

        if now_data.get("code") != "200":
            return {"error": f"实时天气查询失败（code={now_data.get('code')}）"}

        now = now_data["now"]
        current = {
            "temp": now.get("temp"),           # 温度 °C
            "feels_like": now.get("feelsLike"), # 体感温度 °C
            "text": now.get("text"),            # 天气描述
            "wind_dir": now.get("windDir"),     # 风向
            "wind_scale": now.get("windScale"), # 风力等级
            "humidity": now.get("humidity"),    # 相对湿度 %
            "obs_time": now.get("obsTime"),     # 观测时间
        }

        forecast = []
        if daily_data.get("code") == "200":
            for day in daily_data.get("daily", []):
                forecast.append({
                    "date": day.get("fxDate"),
                    "temp_max": day.get("tempMax"),
                    "temp_min": day.get("tempMin"),
                    "text_day": day.get("textDay"),
                    "text_night": day.get("textNight"),
                    "uv_index": day.get("uvIndex"),
                    "humidity": day.get("humidity"),
                })

        logger.info("[tools] get_weather: 查询完成 city=%r temp=%s°C text=%s", city_name, current["temp"], current["text"])
        return {
            "city": city_name,
            "region": f"{country} {adm1}".strip(),
            "current": current,
            "forecast_3d": forecast,
        }

    except httpx.HTTPStatusError as e:
        logger.warning("[tools] get_weather: HTTP 错误 city=%r — %s", city, e)
        return {"error": f"天气查询失败 (HTTP {e.response.status_code}): {e}"}
    except Exception as e:
        logger.warning("[tools] get_weather: 异常 city=%r — %s", city, e)
        return {"error": f"天气查询失败: {e}"}
