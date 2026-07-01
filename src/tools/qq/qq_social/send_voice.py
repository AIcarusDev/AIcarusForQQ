"""send_voice - send one synthesized voice message."""

from __future__ import annotations

import asyncio
import logging
import uuid
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tools._async_bridge import run_coroutine_sync
from qq_adapter.conversation import format_adapter_error

logger = logging.getLogger("AICQ.tools")

# 工具的静态基础 schema，不含 TTS Worker 动态参数
DECLARATION: dict = {
    "name": "send_voice",
    "description": (
        "向当前会话发送一条语音消息。"
        "当你想用语音而不是文字表达时使用。"
        "仅在 TTS 连接且有效时可用。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "语音内容，注意不要带任何括号，简短。"
            },
        },
        "required": ["text"],
    },
}

PROMPT_SIGNATURE = """
// 向当前会话发送一条语音消息。
// 当你想用语音而不是文字表达时使用。
// 仅在 TTS 连接且有效时可用。
send_voice(args: {
  text: string; // 语音内容，注意不要带任何括号，简短。
})
"""

EXTERNALLY_PERCEPTIBLE: bool = True
TOOL_EFFECT: dict[str, str] = {"surface": "qq", "kind": "session_write"}


def get_declaration(**_kwargs: Any) -> dict:
    """运行时动态构建工具 schema，聚合所有在线 Worker 的 llm_schema。"""
    import app_state
    import copy

    decl = copy.deepcopy(DECLARATION)
    tts_server = app_state.tts_server
    if tts_server is None:
        return decl

    plugins = tts_server.list_plugins()
    if not plugins:
        return decl

    if len(plugins) == 1:
        # 单插件：原有行为，直接合并 llm_schema 属性
        extra_props = (plugins[0].get("llm_schema") or {}).get("properties") or {}
        decl["parameters"]["properties"].update(extra_props)
        return decl

    # 多插件：聚合所有 schema，添加 plugin_id 选择参数
    tts_cfg = app_state.tts_cfg or {}
    preferred = str(tts_cfg.get("default_plugin_id") or "").strip() or None

    plugin_ids = [p["plugin_id"] for p in plugins]
    plugin_desc_parts: list[str] = []
    merged_props: dict[str, Any] = {}
    for p in plugins:
        pid = p["plugin_id"]
        schema = p.get("llm_schema") or {}
        desc = schema.get("description") or pid
        plugin_desc_parts.append(f"{pid}（{desc}）")
        merged_props.update(schema.get("properties") or {})

    decl["parameters"]["properties"]["plugin_id"] = {
        "type": "string",
        "enum": plugin_ids,
        "description": "选择使用哪个 Worker：" + "；".join(plugin_desc_parts),
    }
    decl["parameters"]["properties"].update(merged_props)
    # 多插件时 plugin_id 必填，text 变为可选（歌声合成不需要 text）
    decl["parameters"]["required"] = ["plugin_id"]
    decl["description"] = (
        "向当前会话发送一条语音消息。"
        "通过 plugin_id 参数选择使用哪个 Worker，各 Worker 的参数见对应说明。"
    )
    return decl


def get_prompt_signature(**_kwargs: Any) -> str:
    """Return a handwritten prompt signature that mirrors dynamic TTS workers."""
    try:
        import app_state

        tts_server = app_state.tts_server
    except Exception:
        tts_server = None
    if tts_server is None:
        return PROMPT_SIGNATURE

    plugins = tts_server.list_plugins()
    if not plugins:
        return PROMPT_SIGNATURE

    if len(plugins) == 1:
        extra_lines = _worker_param_lines(plugins[0].get("llm_schema") or {}, required=set())
        return "\n".join([
            "// 向当前会话发送一条语音消息。",
            "// 当你想用语音而不是文字表达时使用。",
            "// 仅在 TTS 连接且有效时可用。",
            "send_voice(args: {",
            "  text: string; // 语音内容，注意不要带任何括号，简短。",
            *extra_lines,
            "})",
        ])

    plugin_ids = [str(p.get("plugin_id") or "").strip() for p in plugins]
    plugin_ids = [pid for pid in plugin_ids if pid]
    plugin_union = " | ".join(_ts_string_literal(pid) for pid in plugin_ids) or "string"
    plugin_desc_parts: list[str] = []
    merged_props: dict[str, Any] = {}
    for plugin in plugins:
        pid = str(plugin.get("plugin_id") or "").strip()
        schema = plugin.get("llm_schema") or {}
        desc = str(schema.get("description") or pid).strip()
        if pid:
            plugin_desc_parts.append(f"{pid}（{desc}）")
        props = schema.get("properties") if isinstance(schema, dict) else None
        if isinstance(props, dict):
            merged_props.update(props)
    worker_desc = "；".join(plugin_desc_parts)
    return "\n".join([
        "// 向当前会话发送一条语音消息。",
        "// 通过 plugin_id 参数选择使用哪个 Worker，各 Worker 的参数见对应说明。",
        "send_voice(args: {",
        f"  plugin_id: {plugin_union}; // 选择使用哪个 Worker：{worker_desc}",
        "  text?: string; // 语音内容，注意不要带任何括号，简短；多插件时 text 可选（歌声合成不需要 text）。",
        *_worker_param_lines({"properties": merged_props}, required=set()),
        "})",
    ])


def _worker_param_lines(schema: dict[str, Any], *, required: set[str]) -> list[str]:
    return [line for line, _key in _worker_param_line_items(schema, required=required)]


def _worker_param_line_items(schema: dict[str, Any], *, required: set[str]) -> list[tuple[str, str]]:
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict):
        return []
    lines: list[tuple[str, str]] = []
    for key, prop_schema in props.items():
        key_text = str(key or "").strip()
        if not key_text or key_text in {"text", "plugin_id"}:
            continue
        marker = "" if key_text in required else "?"
        field_type = _worker_schema_to_ts(prop_schema if isinstance(prop_schema, dict) else {}, 1)
        comment = _worker_schema_comment(prop_schema if isinstance(prop_schema, dict) else {})
        comment_text = f" // {comment}" if comment else ""
        lines.append((f"  {_ts_property_name(key_text)}{marker}: {field_type};{comment_text}", key_text))
    return lines


def _worker_schema_to_ts(schema: dict[str, Any], indent: int) -> str:
    if "const" in schema:
        return _ts_literal(schema["const"])
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return " | ".join(_ts_literal(item) for item in enum_values)

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return " | ".join(_worker_schema_to_ts({"type": item}, indent) for item in schema_type)
    if schema_type == "string":
        return "string"
    if schema_type in {"integer", "number"}:
        return "number"
    if schema_type == "boolean":
        return "boolean"
    if schema_type == "array":
        item_schema = schema.get("items")
        item_type = _worker_schema_to_ts(item_schema if isinstance(item_schema, dict) else {}, indent)
        if "\n" in item_type or "|" in item_type:
            return f"({item_type})[]"
        return f"{item_type}[]"
    if schema_type == "object" or isinstance(schema.get("properties"), dict):
        return _worker_object_to_ts(schema, indent)

    for key in ("anyOf", "oneOf"):
        variants = schema.get(key)
        if isinstance(variants, list) and variants:
            rendered = [
                _worker_schema_to_ts(item, indent)
                for item in variants
                if isinstance(item, dict)
            ]
            if rendered:
                return " | ".join(rendered)
    return "unknown"


def _worker_object_to_ts(schema: dict[str, Any], indent: int) -> str:
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        return "Record<string, unknown>"

    required = set(schema.get("required") or [])
    child_pad = "  " * (indent + 1)
    close_pad = "  " * indent
    lines = ["{"]
    for key, child_schema in props.items():
        key_text = str(key or "").strip()
        if not key_text:
            continue
        child = child_schema if isinstance(child_schema, dict) else {}
        marker = "" if key_text in required else "?"
        field_type = _worker_schema_to_ts(child, indent + 1)
        comment = _worker_schema_comment(child)
        comment_text = f" // {comment}" if comment else ""
        lines.append(f"{child_pad}{_ts_property_name(key_text)}{marker}: {field_type};{comment_text}")
    lines.append(f"{close_pad}}}")
    return "\n".join(lines)


def _worker_schema_comment(schema: dict[str, Any]) -> str:
    parts: list[str] = []
    description = " ".join(str(schema.get("description") or "").split())
    if description:
        parts.append(description)

    constraints: list[str] = []
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if minimum is not None and maximum is not None:
        constraints.append(f"范围 {minimum}~{maximum}")
    elif minimum is not None:
        constraints.append(f"最小 {minimum}")
    elif maximum is not None:
        constraints.append(f"最大 {maximum}")

    min_items = schema.get("minItems")
    max_items = schema.get("maxItems")
    if min_items is not None and max_items is not None:
        constraints.append(f"数组长度 {min_items}~{max_items}")
    elif min_items is not None:
        constraints.append(f"至少 {min_items} 项")
    elif max_items is not None:
        constraints.append(f"最多 {max_items} 项")
    if schema.get("uniqueItems") is True:
        constraints.append("数组项不可重复")

    for constraint in constraints:
        if constraint not in "；".join(parts):
            parts.append(constraint)
    return "；".join(parts)


def _ts_property_name(value: str) -> str:
    if value.replace("_", "").isalnum() and not value[:1].isdigit():
        return value
    return _ts_string_literal(value)


def _ts_string_literal(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _ts_literal(value: object) -> str:
    if isinstance(value, str):
        return _ts_string_literal(value)
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return str(value)

REQUIRES_CONTEXT: list[str] = ["session", "qq_adapter_client"]


def condition(config: dict) -> bool:
    return True


def sanitize_semantic_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str], str | None]:
    changes: list[str] = []
    # 根据当前动态 schema 判断 text 是否必填，避免硬编码插件 ID
    decl = get_declaration()
    required = (decl.get("parameters") or {}).get("required") or []
    if "text" in required:
        text = str(args.get("text") or "").strip()
        if not text:
            return args, changes, "text is empty"
        if text != args.get("text"):
            args = dict(args)
            args["text"] = text
            changes.append("trimmed text")
    return args, changes, None


def _tts_cache_dir() -> Path:
    core_dir = Path(__file__).resolve().parents[2]
    path = core_dir / "cache" / "tts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_pcm_wav(pcm: bytes, audio_format: dict[str, Any]) -> Path:
    sample_rate = int(audio_format.get("sample_rate", 16000) or 16000)
    channels = int(audio_format.get("channels", 1) or 1)
    bit_depth = int(audio_format.get("bit_depth", 16) or 16)
    sample_width = max(1, bit_depth // 8)
    wav_path = _tts_cache_dir() / f"tts_{uuid.uuid4().hex}.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return wav_path


def _wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        if frame_rate <= 0:
            return 0.0
        return wav_file.getnframes() / frame_rate


async def _synthesize_to_wav(text: str, *, plugin_id: str | None = None, **kwargs: Any) -> tuple[Path, str, dict[str, Any]]:
    import app_state

    tts_server = app_state.tts_server
    if tts_server is None:
        raise RuntimeError("TTS 服务端未启用")

    tts_cfg = app_state.tts_cfg or {}
    preferred_plugin_id = plugin_id or str(tts_cfg.get("default_plugin_id") or "").strip() or None
    resolved_id = tts_server.select_plugin_id(preferred_plugin_id)
    if not resolved_id:
        raise RuntimeError("没有在线 TTS Worker")

    plugin_info = tts_server.get_plugin_info(resolved_id)
    if plugin_info is None:
        raise RuntimeError(f"TTS Worker {resolved_id!r} 不在线")

    task_id = await tts_server.dispatch_task(resolved_id, text, kwargs or {})
    try:
        await tts_server.wait_task(task_id, timeout=float(tts_cfg.get("task_timeout", 60)))
        pcm = bytes(app_state.tts_audio_buffers.pop(task_id, b""))
    except Exception:
        app_state.tts_audio_buffers.pop(task_id, None)
        raise

    if not pcm:
        raise RuntimeError("TTS Worker 未返回音频数据")

    wav_path = _write_pcm_wav(pcm, plugin_info.get("audio_format") or {})
    return wav_path, resolved_id, plugin_info


def make_handler(session: Any, qq_adapter_client: Any) -> Callable:
    def execute(text: str = "", **kwargs) -> dict:
        import app_state
        from database import save_chat_message
        from llm.core.round_context import get_current_inner_state
        from web.debug_server import broadcast_chat_event

        plugin_id: str | None = kwargs.pop("plugin_id", None) or None

        loop: asyncio.AbstractEventLoop | None = getattr(app_state, "main_loop", None)
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}
        if not qq_adapter_client or not qq_adapter_client.connected:
            return {"error": "QQ adapter 未连接"}

        conv_type = session.conv_type
        conv_id = session.conv_id
        try:
            group_id = int(conv_id) if conv_type == "group" else None
            user_id = int(conv_id) if conv_type in {"private", "temp"} else None
            temp_source_group_id = None
            if conv_type == "temp":
                source_group_id = str(getattr(session, "temp_source_group_id", "") or "").strip()
                if not source_group_id:
                    return {"error": "临时会话缺少来源群，无法发送语音。请先从可用群聊打开该临时会话。"}
                temp_source_group_id = int(source_group_id)
        except (ValueError, TypeError):
            return {"error": f"会话 ID 无效: {conv_id}"}
        if conv_type not in {"group", "private", "temp"}:
            return {"error": f"当前会话类型不支持发送 QQ 语音: {conv_type or 'unknown'}"}

        try:
            wav_path, plugin_id, plugin_info = run_coroutine_sync(
                _synthesize_to_wav(text, plugin_id=plugin_id, **kwargs),
                loop,
                timeout=float((app_state.tts_cfg or {}).get("task_timeout", 60)) + 5,
            )
            duration_seconds = _wav_duration_seconds(wav_path)
        except Exception as exc:
            logger.warning("[send_voice] TTS 合成失败: %s", exc)
            return {"error": f"TTS 合成失败: {exc}"}

        message = [{"type": "record", "data": {"file": str(wav_path)}}]
        try:
            send_result = run_coroutine_sync(
                qq_adapter_client.send_message(
                    group_id=group_id,
                    user_id=user_id,
                    temp_source_group_id=temp_source_group_id,
                    message=message,
                    llm_elapsed=0.0,
                ),
                loop,
                timeout=30,
            )
        except Exception as exc:
            logger.warning("[send_voice] 语音发送失败: %s", exc)
            send_result = None

        conversation_id = f"{conv_type}_{conv_id}"
        now_ts = datetime.now(app_state.TIMEZONE).isoformat()
        if send_result and send_result.get("message_id") is not None:
            real_id = str(send_result["message_id"])
            content_type = "voice"
        else:
            real_id = f"failed_{uuid.uuid4().hex[:8]}"
            content_type = "send_failed"

        entry = {
            "role": "bot",
            "message_id": real_id,
            "sender_id": session._qq_id or "bot",
            "sender_name": session._qq_name or "",
            "sender_role": "",
            "timestamp": now_ts,
            "content": text,
            "content_type": content_type,
            "content_segments": [
                {"type": "voice", "duration": duration_seconds, "transcript": text},
            ],
        }
        session.add_to_context(entry)
        asyncio.run_coroutine_threadsafe(save_chat_message(conversation_id, entry), loop)
        asyncio.run_coroutine_threadsafe(
            broadcast_chat_event({
                "type": "bot_turn",
                "conv_id": conversation_id,
                "conv_name": session.conv_name or conversation_id,
                "conv_type": conv_type or "unknown",
                "entries": [entry],
                "inner_state": get_current_inner_state(),
            }),
            loop,
        )

        result = {
            "success": content_type == "voice",
            "message_id": real_id,
            "plugin_id": plugin_id,
            "audio_format": plugin_info.get("audio_format") or {},
            "duration": duration_seconds,
        }
        if content_type != "voice":
            result["error"] = format_adapter_error(
                getattr(qq_adapter_client, "last_api_error", None),
                "语音消息发送失败",
            )
        return result

    return execute
