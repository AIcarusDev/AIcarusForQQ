import html
import json
import logging
import platform
import subprocess
import traceback
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from zoneinfo import ZoneInfo

import psutil
import yaml
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from prompt import SYSTEM_PROMPT, get_formatted_time_for_llm
from provider import create_adapter

load_dotenv()

# ── 日志配置 ──────────────────────────────────────────────
_log_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d\n%(message)s\n",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log_handler = RotatingFileHandler(
    "mita.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_log_handler.setFormatter(_log_formatter)
logging.root.setLevel(logging.DEBUG)
logging.root.addHandler(_log_handler)
# 同时输出到控制台
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_formatter)
logging.root.addHandler(_console_handler)

logger = logging.getLogger("mita.app")

app = Flask(__name__)
app.json.sort_keys = False  # type: ignore[attr-defined]

# ── 加载配置 ──────────────────────────────────────────────
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

with open("persona.md", "r", encoding="utf-8") as f:
    persona = f.read()

# ── 运行时覆盖（保存上次 switch_provider 的选择，重启后不丢失）──────
_RUNTIME_OVERRIDE_FILE = ".model_override.json"
try:
    with open(_RUNTIME_OVERRIDE_FILE, "r", encoding="utf-8") as _f:
        _ov = json.load(_f)
    config["provider"] = _ov["provider"]
    config["model"] = _ov["model"]
    config["model_name"] = _ov.get("model_name", _ov["model"])
    if _ov.get("base_url"):
        config["base_url"] = _ov["base_url"]
    elif "base_url" in config:
        del config["base_url"]
    logger.info("已应用运行时覆盖: provider=%s model=%s", config["provider"], config["model"])
except FileNotFoundError:
    pass
except Exception as _e:
    logger.warning("运行时覆盖文件无效，已忽略: %s", _e)

# ── 配置项 ────────────────────────────────────────────────
MODEL = config.get("model", "gemini-2.0-flash")
MODEL_NAME = config.get("model_name", MODEL)
GEN = config.get("generation", {})
TIMEZONE = ZoneInfo(config.get("timezone", "Asia/Shanghai"))
MAX_CYCLES = config.get("max_cycles", 3)
MAX_CONTEXT = 20
BOT_NAME = config.get("bot_name", "小懒猫")
adapter = create_adapter(config)

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
        previous_cycle_json=prev,
    )


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


# ── 自定义工具 ───────────────────────────────────────────
def get_device_info() -> dict:
    """获取设备基本信息：操作系统、内存（RAM）使用情况、GPU 显存情况。"""
    info: dict = {
        "os": f"{platform.system()} {platform.version()}",
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
    }
    parts = [f"{platform.system()} {platform.version()} ({platform.machine()})"]
    try:
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / (1024 ** 3), 1)
        info["ram_available_gb"] = round(vm.available / (1024 ** 3), 1)
        info["ram_used_percent"] = vm.percent
        parts.append(f"RAM {info['ram_total_gb']}GB 总计 / {info['ram_available_gb']}GB 可用 ({vm.percent}% 已用)")
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0:
            gpus = []
            for line in proc.stdout.strip().splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) == 3:
                    gpus.append({"name": p[0], "vram_total_mb": int(p[1]), "vram_free_mb": int(p[2])})
                    parts.append(f"GPU {p[0]} 显存 {p[1]}MB 总计 / {p[2]}MB 空闲")
            if gpus:
                info["gpus"] = gpus
    except Exception:
        pass
    info["summary"] = "；".join(parts)
    return info


# ── 工具注册表 ────────────────────────────────────────────
TOOL_DECLARATIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_device_info",
            "description": "获取当前运行设备的基本信息，包括操作系统版本、内存（RAM）使用情况和 GPU 显存情况。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_REGISTRY: dict = {
    "get_device_info": get_device_info,
}


def call_model_and_process() -> tuple[dict | None, dict | None, str, str, bool]:
    """调用模型、更新上下文、返回 (result, grounding, system_prompt, user_prompt, repaired)。"""
    global previous_cycle_json

    system_prompt = build_system_prompt()
    chat_log = build_chat_log_xml()

    result, grounding, repaired = adapter.call(
        system_prompt, chat_log, GEN, RESPONSE_SCHEMA,
        tool_declarations=TOOL_DECLARATIONS,
        tool_registry=TOOL_REGISTRY,
    )

    if result is None:
        return None, None, system_prompt, chat_log, False

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
    return result, grounding, system_prompt, chat_log, repaired


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
        result, grounding, system_prompt, user_prompt, repaired = call_model_and_process()
        if result is None:
            logger.warning("[/chat] 模型返回为空（可能被安全过滤拦截）")
            return jsonify({"success": False, "error": "模型返回为空（可能被安全过滤拦截）"}), 502

        return jsonify({
            "success": True,
            "data": result,
            "message_id": message_id,
            "grounding": grounding,
            "remaining_cycles": remaining_cycles,
            "json_repaired": repaired,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })
    except Exception as e:
        logger.error(
            "[/chat] 异常\n"
            "user_message: %s\n"
            "user_id: %s\n"
            "%s",
            user_message, user_id, traceback.format_exc(),
        )
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/cycle", methods=["POST"])
def cycle():
    """主动循环：前端确认上一轮消息已全部渲染后调用。"""
    global remaining_cycles

    if remaining_cycles <= 0:
        return jsonify({"success": False, "error": "没有剩余循环次数"}), 400

    remaining_cycles -= 1

    try:
        result, grounding, system_prompt, user_prompt, repaired = call_model_and_process()
        if result is None:
            remaining_cycles += 1  # 模型失败，回滚次数
            logger.warning("[/cycle] 模型返回为空（可能被安全过滤拦截），remaining_cycles 已回滚至 %d", remaining_cycles)
            return jsonify({"success": False, "error": "模型返回为空（可能被安全过滤拦截）"}), 502

        return jsonify({
            "success": True,
            "data": result,
            "grounding": grounding,
            "remaining_cycles": remaining_cycles,
            "json_repaired": repaired,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })
    except Exception as e:
        remaining_cycles += 1  # 异常，回滚次数
        logger.error(
            "[/cycle] 异常，remaining_cycles 已回滚至 %d\n%s",
            remaining_cycles, traceback.format_exc(),
        )
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear_context():
    global previous_cycle_json, remaining_cycles
    context_messages.clear()
    previous_cycle_json = None
    remaining_cycles = 0
    return jsonify({"success": True})


@app.route("/config", methods=["GET"])
def get_config():
    """返回当前 provider 配置（供前端初始化用）。"""
    return jsonify({
        "provider": config.get("provider", "gemini"),
        "model": MODEL,
        "base_url": config.get("base_url", ""),
    })


@app.route("/models", methods=["POST"])
def list_models_route():
    """返回指定 provider / base_url 下的可用模型列表（切换前可先预览）。"""
    data = request.get_json() or {}
    provider = (data.get("provider") or config.get("provider", "gemini")).strip()
    base_url = (data.get("base_url") or "").strip()

    tmp_cfg = dict(config)
    tmp_cfg["provider"] = provider
    if base_url:
        tmp_cfg["base_url"] = base_url
    elif "base_url" in tmp_cfg:
        del tmp_cfg["base_url"]

    try:
        tmp_adapter = create_adapter(tmp_cfg)
    except ValueError as e:
        # 未知 provider 等客户端参数错误
        return jsonify({"success": False, "error": str(e), "models": []}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "models": []}), 500

    try:
        models = tmp_adapter.list_models()
        return jsonify({"success": True, "models": models})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "models": []}), 500


@app.route("/switch_provider", methods=["POST"])
def switch_provider():
    """运行时切换 provider / model，无需重启。"""
    global adapter, MODEL, MODEL_NAME, config

    data = request.get_json() or {}
    provider = (data.get("provider") or "").strip()
    model = (data.get("model") or "").strip()
    base_url = (data.get("base_url") or "").strip()

    if not provider or not model:
        return jsonify({"success": False, "error": "provider 和 model 不能为空"}), 400

    new_cfg = dict(config)
    new_cfg["provider"] = provider
    new_cfg["model"] = model
    new_cfg["model_name"] = model
    if base_url:
        new_cfg["base_url"] = base_url
    elif "base_url" in new_cfg:
        del new_cfg["base_url"]

    try:
        new_adapter = create_adapter(new_cfg)
    except ValueError as e:
        # 未知 provider 等客户端参数错误 → 400
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        # 依赖包缺失，属于服务端环境问题 → 500
        return jsonify({"success": False, "error": f"服务端缺少依赖: {e}"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    config = new_cfg
    adapter = new_adapter
    MODEL = model
    MODEL_NAME = model

    # 持久化到运行时覆盖文件，防止服务重启后丢失选择
    try:
        with open(_RUNTIME_OVERRIDE_FILE, "w", encoding="utf-8") as _f:
            json.dump({
                "provider": provider,
                "model": model,
                "model_name": model,
                "base_url": base_url or None,
            }, _f, ensure_ascii=False, indent=2)
    except Exception as _e:
        logger.warning("写入运行时覆盖文件失败: %s", _e)

    return jsonify({"success": True, "provider": provider, "model": model})


if __name__ == "__main__":
    srv = config.get("server", {})
    app.run(debug=srv.get("debug", True), port=srv.get("port", 5000))
