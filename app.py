import html
import json
import os
import platform
import subprocess
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import psutil
import yaml
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

from prompt import SYSTEM_PROMPT, get_formatted_time_for_llm

load_dotenv()

app = Flask(__name__)
app.json.sort_keys = False  # type: ignore[attr-defined]

# ── 加载配置 ──────────────────────────────────────────────
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

with open("persona.md", "r", encoding="utf-8") as f:
    persona = f.read()

# ── 配置项 ────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = config.get("model", "gemini-3.1-flash-lite")
MODEL_NAME = config.get("model_name", MODEL)
GEN = config.get("generation", {})
SAFETY_THRESHOLD = config.get("safety", {}).get("threshold", "OFF")
TIMEZONE = ZoneInfo(config.get("timezone", "Asia/Shanghai"))
MAX_CYCLES = config.get("max_cycles", 3)
THINKING_LEVEL = config.get("thinking", {}).get("level", None)
MAX_CONTEXT = 20
BOT_NAME = config.get("bot_name", "小懒猫")

# ── 结构化输出 Schema ────────────────────────────────────
RESPONSE_SCHEMA = {
    "type": "object",
    "description": "你的内心状态与要发送的消息。",
    "properties": {
        "mood": {
            "type": "string",
            "description": "你当前的情绪，是下意识的第一反应。"
        },
        "think": {
            "type": "string",
            "description": "你当前的内心想法，是自然真实且私密的，可以简短，也可以是非常丰富深度思考。"
        },
        "intent": {
            "type": "string",
            "description": "你当前最直接的、短期的意图或打算。"
        },
        "messages": {
            "type": "array",
            "description": "要发送的消息列表。数组中的每一个元素代表一条独立发送的消息。长消息可以分为多个 segment 发送。通常情况下建议：一条消息控制在 10 字以下，消息中的标点符号均可省略，并且自然口语化。如果不需要发送消息，则不需要输出此项。",
            "items": {
                "type": "object",
                "description": "单条消息的结构",
                "properties": {
                    "reply_message_id": {
                        "type": ["string", "null"],
                        "description": "要引用回复的目标消息ID。仅在需要明确上下文或特别提醒时使用。不需要时留空字符串。"
                    },
                    "segments": {
                        "type": "array",
                        "description": "该条消息的具体内容片段（文本或@某人）",
                        "items": {
                            "oneOf": [
                                {
                                    "title": "@某人",
                                    "type": "object",
                                    "properties": {
                                        "command": {
                                            "type": "string",
                                            "enum": ["at"]
                                        },
                                        "params": {
                                            "type": "object",
                                            "properties": {
                                                "user_id": {
                                                    "type": "string",
                                                    "description": "被 @ 用户的 ID"
                                                }
                                            },
                                            "required": ["user_id"]
                                        }
                                    },
                                    "required": ["command", "params"]
                                },
                                {
                                    "title": "文本",
                                    "type": "object",
                                    "properties": {
                                        "command": {
                                            "type": "string",
                                            "enum": ["text"]
                                        },
                                        "params": {
                                            "type": "object",
                                            "properties": {
                                                "content": {
                                                    "type": "string",
                                                    "description": "文本内容，建议控制长度（例如 10 字以下），省略标点符号，省略主语。"
                                                }
                                            },
                                            "required": ["content"]
                                        }
                                    },
                                    "required": ["command", "params"]
                                }
                            ]
                        }
                    }
                },
                "required": ["segments"]
            }
        },
        "motivation": {
            "type": "string",
            "description": "你发送消息或选择不发送消息的原因。"
        },
        "cycle_action": {
            "type": "string",
            "enum": ["continue", "stop"],
            "description": (
                "循环管理。"
                "'continue'：在本轮所有消息确认发出并同步后，立即激活下一轮循环（消耗一次剩余循环次数）。"
                "'stop'：结束当前循环，等待被动激活。"
                "当剩余连续循环次数为 0 时，你必须选择 'stop'。"
            ),
        },
    },
    "required": ["mood", "think", "intent", "motivation", "cycle_action"],
}

# ── 安全设置 ──────────────────────────────────────────────
SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory[cat],
        threshold=types.HarmBlockThreshold[SAFETY_THRESHOLD],
    )
    for cat in [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
]

# ── 上下文存储（demo 用内存） ─────────────────────────────
context_messages: list[dict] = []
previous_cycle_json: dict | None = None
remaining_cycles: int = 0


def add_to_context(entry: dict) -> None:
    context_messages.append(entry)
    while len(context_messages) > MAX_CONTEXT:
        context_messages.pop(0)


def build_chat_log_xml() -> str:
    if not context_messages:
        return "<chat_log>\n</chat_log>"
    lines = ["<chat_log>"]
    for msg in context_messages:
        safe_content = html.escape(msg["content"], quote=False)
        safe_name = html.escape(msg["sender_name"])
        lines.append(
            f'  <message id="{msg["message_id"]}" '
            f'sender_id="{msg["sender_id"]}" '
            f'sender_name="{safe_name}" '
            f'timestamp="{msg["timestamp"]}">'
        )
        lines.append(f"    {safe_content}")
        lines.append("  </message>")
    lines.append("</chat_log>")
    return "\n".join(lines)


def build_system_prompt() -> str:
    now = datetime.now(TIMEZONE)
    prev = (
        json.dumps(previous_cycle_json, ensure_ascii=False, indent=2)
        if previous_cycle_json
        else "null"
    )
    return SYSTEM_PROMPT.format(
        persona=persona,
        time=get_formatted_time_for_llm(now),
        model_name=MODEL_NAME,
        number=remaining_cycles,
        device_info=DEVICE_INFO_STR,
        previous_cycle_json=prev,
    )


def extract_grounding(response):
    try:
        meta = (
            response.candidates[0].grounding_metadata
            if response.candidates
            else None
        )
        if meta:
            g = {}
            if meta.web_search_queries:
                g["search_queries"] = list(meta.web_search_queries)
            if meta.grounding_chunks:
                g["sources"] = [
                    {
                        "title": getattr(c.web, "title", ""),
                        "uri": getattr(c.web, "uri", ""),
                    }
                    for c in meta.grounding_chunks
                    if hasattr(c, "web")
                ]
            return g
    except Exception:
        pass
    return None


def extract_bot_messages(result: dict) -> list[str]:
    """从模型输出中提取每条消息的文本内容。"""
    messages = []
    for msg in result.get("messages", []):
        parts = []
        for seg in msg.get("segments", []):
            cmd = seg.get("command")
            params = seg.get("params", {})
            if cmd == "text":
                parts.append(params.get("content", ""))
            elif cmd == "at":
                parts.append(f"@{params.get('user_id', '')}")
        text = "".join(parts)
        if text:
            messages.append(text)
    return messages


# ── 设备信息（启动时采集一次） ────────────────────────────
def _collect_device_info() -> str:
    parts = [f"{platform.system()} {platform.version()} ({platform.machine()})"]
    try:
        vm = psutil.virtual_memory()
        total = round(vm.total / (1024 ** 3), 1)
        available = round(vm.available / (1024 ** 3), 1)
        parts.append(f"RAM {total}GB 总计 / {available}GB 可用 ({vm.percent}% 已用)")
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) == 3:
                    parts.append(f"GPU {p[0]} 显存 {p[1]}MB 总计 / {p[2]}MB 空闲")
    except Exception:
        pass
    return "；".join(parts)


DEVICE_INFO_STR: str = _collect_device_info()


def call_model_and_process() -> tuple[dict | None, dict | None, str, str]:
    """调用模型、更新上下文、返回 (result, grounding, system_prompt, user_prompt)。"""
    global previous_cycle_json

    system_prompt = build_system_prompt()
    chat_log = build_chat_log_xml()

    google_search_tool = types.Tool(google_search=types.GoogleSearch())

    response = client.models.generate_content(
        model=MODEL,
        contents=chat_log,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=GEN.get("temperature", 1.0),
            top_p=GEN.get("top_p", 0.95),
            top_k=GEN.get("top_k", 40),
            max_output_tokens=GEN.get("max_output_tokens", 8192),
            presence_penalty=GEN.get("presence_penalty", 0.0),
            frequency_penalty=GEN.get("frequency_penalty", 0.0),
            response_mime_type="application/json",
            response_json_schema=RESPONSE_SCHEMA,
            tools=[google_search_tool],
            safety_settings=SAFETY_SETTINGS,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel[THINKING_LEVEL]
            ) if THINKING_LEVEL else None,
        ),
    )

    if not response.text:
        return None, None, system_prompt, chat_log

    result = json.loads(response.text)

    # 剩余为 0 → 强制 stop
    if remaining_cycles <= 0:
        result["cycle_action"] = "stop"

    # 将机器人消息逐条写入上下文
    now_ts = datetime.now(TIMEZONE).isoformat()
    for text in extract_bot_messages(result):
        add_to_context({
            "role": "bot",
            "message_id": f"msg_{uuid.uuid4().hex[:8]}",
            "sender_id": "bot",
            "sender_name": BOT_NAME,
            "timestamp": now_ts,
            "content": text,
        })

    previous_cycle_json = result
    grounding = extract_grounding(response)
    return result, grounding, system_prompt, chat_log


# ── 路由 ──────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    global remaining_cycles

    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"success": False, "error": "消息不能为空"}), 400

    user_id = data.get("user_id", "user_001")
    user_name = data.get("user_name", "测试用户")
    message_id = f"msg_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now(TIMEZONE).isoformat()

    add_to_context({
        "role": "user",
        "message_id": message_id,
        "sender_id": user_id,
        "sender_name": user_name,
        "timestamp": timestamp,
        "content": user_message,
    })

    # 被动激活：重置循环预算
    remaining_cycles = MAX_CYCLES

    try:
        result, grounding, system_prompt, user_prompt = call_model_and_process()
        if result is None:
            return jsonify({"success": False, "error": "模型返回为空（可能被安全过滤拦截）"}), 502

        return jsonify({
            "success": True,
            "data": result,
            "message_id": message_id,
            "grounding": grounding,
            "remaining_cycles": remaining_cycles,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/cycle", methods=["POST"])
def cycle():
    """主动循环：前端确认上一轮消息已全部渲染后调用。"""
    global remaining_cycles

    if remaining_cycles <= 0:
        return jsonify({"success": False, "error": "没有剩余循环次数"}), 400

    remaining_cycles -= 1

    try:
        result, grounding, system_prompt, user_prompt = call_model_and_process()
        if result is None:
            return jsonify({"success": False, "error": "模型返回为空（可能被安全过滤拦截）"}), 502

        return jsonify({
            "success": True,
            "data": result,
            "grounding": grounding,
            "remaining_cycles": remaining_cycles,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear_context():
    global previous_cycle_json, remaining_cycles
    context_messages.clear()
    previous_cycle_json = None
    remaining_cycles = 0
    return jsonify({"success": True})


if __name__ == "__main__":
    srv = config.get("server", {})
    app.run(debug=srv.get("debug", True), port=srv.get("port", 5000))
