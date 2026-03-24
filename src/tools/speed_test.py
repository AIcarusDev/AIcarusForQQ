"""speed_test.py — 测试当前网络上下行速度与延迟"""

import logging
import time

import httpx

logger = logging.getLogger("AICQ.tools")

DECLARATION: dict = {
    "max_calls_per_response": 1,
    "name": "speed_test",
    "description": (
        "在需要时，测试当前设备的网络速度，包括下载速度、上传速度和延迟（ping）。"
        "返回内容仅自己可见。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "motivation": {
                "type": "string",
                "description": "调用此工具的动机或原因。",
            },
        },
    },
}

# 用于测速的公共资源（Cloudflare speed test 端点）
_DOWNLOAD_URL = "https://speed.cloudflare.com/__down?bytes=5000000"  # 5 MB
_UPLOAD_URL = "https://speed.cloudflare.com/__up"
_PING_URL = "https://speed.cloudflare.com/__down?bytes=1"
_TIMEOUT = 30


def _measure_ping(client: httpx.Client) -> float:
    """测量 HTTP 延迟（毫秒），取三次平均值。"""
    times: list[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        client.get(_PING_URL, timeout=10)
        times.append((time.perf_counter() - t0) * 1000)
    return round(sum(times) / len(times), 1)


def _measure_download(client: httpx.Client) -> float:
    """测量下载速度（Mbps）。"""
    t0 = time.perf_counter()
    resp = client.get(_DOWNLOAD_URL, timeout=_TIMEOUT)
    elapsed = time.perf_counter() - t0
    size_bits = len(resp.content) * 8
    return round(size_bits / elapsed / 1_000_000, 2)


def _measure_upload(client: httpx.Client) -> float:
    """测量上传速度（Mbps），上传 2 MB 随机数据。"""
    payload = b"\x00" * 2_000_000
    t0 = time.perf_counter()
    client.post(_UPLOAD_URL, content=payload, timeout=_TIMEOUT)
    elapsed = time.perf_counter() - t0
    size_bits = len(payload) * 8
    return round(size_bits / elapsed / 1_000_000, 2)


def execute(**kwargs) -> dict:
    logger.info("[tools] speed_test: 开始测速")
    result: dict = {}
    parts: list[str] = []

    try:
        with httpx.Client(http2=True) as client:
            try:
                ping_ms = _measure_ping(client)
                result["ping_ms"] = ping_ms
                parts.append(f"延迟 {ping_ms} ms")
                logger.info("[tools] speed_test: ping = %s ms", ping_ms)
            except Exception as e:
                result["ping_error"] = str(e)
                logger.warning("[tools] speed_test: ping 失败 - %s", e)

            try:
                dl = _measure_download(client)
                result["download_mbps"] = dl
                parts.append(f"下载 {dl} Mbps")
                logger.info("[tools] speed_test: 下载 = %s Mbps", dl)
            except Exception as e:
                result["download_error"] = str(e)
                logger.warning("[tools] speed_test: 下载测速失败 - %s", e)

            try:
                ul = _measure_upload(client)
                result["upload_mbps"] = ul
                parts.append(f"上传 {ul} Mbps")
                logger.info("[tools] speed_test: 上传 = %s Mbps", ul)
            except Exception as e:
                result["upload_error"] = str(e)
                logger.warning("[tools] speed_test: 上传测速失败 - %s", e)

    except Exception as e:
        result["error"] = str(e)
        logger.error("[tools] speed_test: 初始化失败 - %s", e)

    result["summary"] = "；".join(parts) if parts else "测速失败，请检查网络连接。"
    logger.info("[tools] speed_test: 完成，summary = %s", result["summary"])
    return result
