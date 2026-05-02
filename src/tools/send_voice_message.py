"""send_voice_message.py - send one synthesized voice message."""

from __future__ import annotations

import asyncio
import logging
import uuid
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tools._async_bridge import run_coroutine_sync

logger = logging.getLogger("AICQ.tools")

DECLARATION: dict = {
    "name": "send_voice_message",
    "description": (
        "向当前会话发送一条语音消息。"
        "当你想用语音而不是文字表达时使用。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "motivation": {
                "type": "string"
            },
            "text": {
                "type": "string",
                "description": "语音内容，注意不要带带任何括号，简短。"
            },
        },
        "required": ["motivation", "text"],
    },
}

REQUIRES_CONTEXT: list[str] = ["session", "napcat_client"]


def condition(config: dict) -> bool:
    return bool((config.get("tts") or {}).get("enabled", False))


def sanitize_semantic_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str], str | None]:
    changes: list[str] = []
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
    path = core_dir / "data" / "tts_cache"
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


async def _synthesize_to_wav(text: str) -> tuple[Path, str, dict[str, Any]]:
    import app_state

    tts_server = app_state.tts_server
    if tts_server is None:
        raise RuntimeError("TTS 服务端未启用")

    tts_cfg = app_state.tts_cfg or {}
    preferred_plugin_id = str(tts_cfg.get("default_plugin_id") or "").strip() or None
    plugin_id = tts_server.select_plugin_id(preferred_plugin_id)
    if not plugin_id:
        raise RuntimeError("没有在线 TTS Worker")

    plugin_info = tts_server.get_plugin_info(plugin_id)
    if plugin_info is None:
        raise RuntimeError(f"TTS Worker {plugin_id!r} 不在线")

    task_id = await tts_server.dispatch_task(plugin_id, text, {})
    try:
        await tts_server.wait_task(task_id, timeout=float(tts_cfg.get("task_timeout", 60)))
        pcm = bytes(app_state.tts_audio_buffers.pop(task_id, b""))
    except Exception:
        app_state.tts_audio_buffers.pop(task_id, None)
        raise

    if not pcm:
        raise RuntimeError("TTS Worker 未返回音频数据")

    wav_path = _write_pcm_wav(pcm, plugin_info.get("audio_format") or {})
    return wav_path, plugin_id, plugin_info


def make_handler(session: Any, napcat_client: Any) -> Callable:
    def execute(motivation: str, text: str, **kwargs) -> dict:
        del motivation, kwargs
        import app_state
        from database import save_chat_message
        from web.debug_server import broadcast_chat_event

        loop: asyncio.AbstractEventLoop | None = getattr(app_state, "main_loop", None)
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}
        if not napcat_client or not napcat_client.connected:
            return {"error": "NapCat 未连接"}

        conv_type = session.conv_type
        conv_id = session.conv_id
        try:
            group_id = int(conv_id) if conv_type == "group" else None
            user_id = int(conv_id) if conv_type == "private" else None
        except (ValueError, TypeError):
            return {"error": f"会话 ID 无效: {conv_id}"}

        try:
            wav_path, plugin_id, plugin_info = run_coroutine_sync(
                _synthesize_to_wav(text),
                loop,
                timeout=float((app_state.tts_cfg or {}).get("task_timeout", 60)) + 5,
            )
            duration_seconds = _wav_duration_seconds(wav_path)
        except Exception as exc:
            logger.warning("[send_voice_message] TTS 合成失败: %s", exc)
            return {"error": f"TTS 合成失败: {exc}"}

        message = [{"type": "record", "data": {"file": str(wav_path)}}]
        try:
            send_result = run_coroutine_sync(
                napcat_client.send_message(
                    group_id=group_id,
                    user_id=user_id,
                    message=message,
                    llm_elapsed=0.0,
                ),
                loop,
                timeout=30,
            )
        except Exception as exc:
            logger.warning("[send_voice_message] 语音发送失败: %s", exc)
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
                "inner_state": {},
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
            result["error"] = "语音消息发送失败"
        return result

    return execute