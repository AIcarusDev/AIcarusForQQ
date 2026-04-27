"""log_config.py — 彩色日志 & LLM 交互记录

提供：
  - ANSI 彩色控制台输出（按日志级别着色）
  - base64 / 长二进制数据自动压缩
  - LLM prompt / response / tool 专用格式化日志

Logger 命名规范（chip 显示靠它）：

  AICQ
  ├── app                   顶层业务编排（napcat_handler / lifecycle）
  ├── config / db / consciousness
  ├── napcat
  │   ├── client / events / debug / segments
  │   └── heartbeat        心跳（默认 INFO，不刷屏）
  ├── llm
  │   ├── session / retry / rate_limit / core / provider
  │   ├── tool_calling / is / history / quote_prefetch
  │   ├── media.{image_cache, sticker, vision}
  │   └── io.{prompt, response, tool}   ← LLM 输入输出
  ├── memory.archiver
  ├── web.{chat, memory, settings}
  └── tools
"""

import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

# ── ANSI 转义码 ─────────────────────────────────────────────────────

_RESET = "\033[0m"
_DIM   = "\033[2m"

_LEVEL_STYLES: dict[int, tuple[str, str]] = {
    logging.DEBUG:    ("\033[36m",   "DEBUG"),     # 青色
    logging.INFO:     ("\033[32m",   " INFO"),     # 绿色
    logging.WARNING:  ("\033[33m",   " WARN"),     # 黄色
    logging.ERROR:    ("\033[31m",   "ERROR"),     # 红色
    logging.CRITICAL: ("\033[1;31m", "FATAL"),     # 粗体红色
}

_PROMPT_STYLE   = "\033[95m"  # 亮洋红
_RESPONSE_STYLE = "\033[96m"  # 亮青色
_TOOL_STYLE     = "\033[93m"  # 亮黄
_BOX_STYLE      = "\033[90m"  # 暗灰


# ── base64 / 长数据压缩 ─────────────────────────────────────────────

# data URI（优先匹配）
_DATA_URI_RE = re.compile(
    r"data:[a-zA-Z0-9_.+/-]+;base64,[A-Za-z0-9+/\r\n]{64,}={0,2}"
)
# 裸 base64 块（≥128 字符，避免误判）
_RAW_B64_RE = re.compile(
    r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{128,}={0,2}(?![A-Za-z0-9+/=])"
)


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1048576:
        return f"{n / 1024:.1f}KB"
    return f"{n / 1048576:.1f}MB"


def compress_base64(text: str) -> str:
    """将文本中的 base64 / data-URI 替换为可读的大小摘要。"""

    def _sub_data_uri(m: re.Match) -> str:
        full = m.group(0)
        header, payload = full.split(",", 1) if "," in full else (full, "")
        size = len(payload) * 3 // 4
        return f"{header},[≈{_fmt_size(size)}]"

    def _sub_raw(m: re.Match) -> str:
        size = len(m.group(0)) * 3 // 4
        return f"[base64 ≈{_fmt_size(size)}]"

    text = _DATA_URI_RE.sub(_sub_data_uri, text)
    text = _RAW_B64_RE.sub(_sub_raw, text)
    return text


# ── ANSI 剥离（文件日志用）──────────────────────────────────────────

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


# ── Formatters ───────────────────────────────────────────────────────

class ColorFormatter(logging.Formatter):
    """彩色控制台 Formatter：按级别着色，自动压缩 base64。"""

    def format(self, record: logging.LogRecord) -> str:
        color, label = _LEVEL_STYLES.get(
            record.levelno, ("", record.levelname)
        )
        ts = self.formatTime(record, "%H:%M:%S")
        message = record.getMessage()

        # 异常堆栈
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            if exc_text:
                message += "\n" + exc_text
        if record.stack_info:
            message += "\n" + str(record.stack_info)

        header = (
            f"{_DIM}{ts}{_RESET} "
            f"{color}{label}{_RESET} "
            f"{_DIM}{record.name} "
            f"{record.filename}:{record.lineno}{_RESET}"
        )
        return compress_base64(f"{header}\n{message}")


class FileFormatter(logging.Formatter):
    """纯文本文件 Formatter：无色，自动压缩 base64，剥离 ANSI。"""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d\n%(message)s\n",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        output = super().format(record)
        output = _ANSI_RE.sub("", output)
        return compress_base64(output)


class BrowserLogHandler(logging.Handler):
    """将日志记录推送到前端日志页面的 WebSocket 客户端。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from web.debug_server import add_log_record
            message = record.getMessage()
            if record.exc_info:
                message += "\n" + self.formatException(record.exc_info)
            if record.stack_info:
                message += "\n" + str(record.stack_info)
            message = compress_base64(_ANSI_RE.sub("", message))
            add_log_record({
                "level": record.levelname,
                "name": record.name,
                "message": message,
                "time": logging.Formatter().formatTime(record, "%H:%M:%S"),
                "file": record.filename,
                "lineno": record.lineno,
            })
        except Exception:
            pass


# ── 初始化 ───────────────────────────────────────────────────────────

def setup_logging(log_file: Optional[str] = None, level: int = logging.DEBUG):
    """初始化全局日志：彩色控制台 + 轮转文件 + 浏览器 WebSocket。"""
    if sys.platform == "win32":
        os.system("")  # 启用 Windows VT100 ANSI 转义

    if log_file is None:
        _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(_BASE_DIR, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "AICQ.log")

    root = logging.root
    root.setLevel(level)
    root.handlers.clear()

    # 控制台
    console = logging.StreamHandler()
    console.setFormatter(ColorFormatter())
    root.addHandler(console)

    # 文件
    fh = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    fh.setFormatter(FileFormatter())
    root.addHandler(fh)

    # 浏览器 WebSocket（无格式化，直接推 JSON）
    bh = BrowserLogHandler()
    bh.setLevel(logging.DEBUG)
    root.addHandler(bh)

    # 降低第三方库噪音
    logging.getLogger("websockets").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    # NapCat 心跳：独立 logger，默认 INFO（DEBUG 心跳过于频繁，不入控制台/UI/文件）
    logging.getLogger("AICQ.napcat.heartbeat").setLevel(logging.INFO)

    # 屏蔽 hypercorn access log 中的状态轮询心跳请求（每 10s 一次，无意义噪音）
    class _StatusPollFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "/debug/api/status" not in record.getMessage()

    logging.getLogger("hypercorn.access").addFilter(_StatusPollFilter())


# ── LLM Prompt / Response / Tool 专用日志 ──────────────────────────

# 拆成 3 个子 logger，前端 chip 可分别显示/过滤：
#   AICQ.llm.io.prompt    模型输入
#   AICQ.llm.io.response  模型输出
#   AICQ.llm.io.tool      模型发起的工具调用
_prompt_logger   = logging.getLogger("AICQ.llm.io.prompt")
_response_logger = logging.getLogger("AICQ.llm.io.response")
_tool_logger     = logging.getLogger("AICQ.llm.io.tool")

_BOX_W = 70
_BOX_H = "─" * _BOX_W


def _format_user_content(user_content) -> str:
    """将 user_content（str 或多模态 list）转为可读文本。

    多模态 parts 之间不再额外插入换行——image_url 是被夹在两段 text
    之间的内联占位，原 text 段自身已含必要的换行结构，强行 join('\n')
    会把 [图片 ref="..."[内嵌图片]] 撑成 3 行，难看且容易误读。
    """
    if isinstance(user_content, str):
        return user_content
    if isinstance(user_content, list):
        parts = []
        for item in user_content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item["text"])
                elif item.get("type") == "image_url":
                    parts.append("[内嵌图片]")
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(user_content)


def log_prompt(provider: str, system_prompt: str, user_content):
    """DEBUG 级别记录发送给 LLM 的完整 prompt（保留空格和换行）。

    使用 stacklevel=2 让 record.filename:lineno 反映真实调用方。
    """
    user_text = _format_user_content(user_content)
    _prompt_logger.debug(
        "%s",
        f"\n{_BOX_STYLE}┌{_BOX_H}┐{_RESET}\n"
        f"{_BOX_STYLE}│{_RESET} {_PROMPT_STYLE}📤 PROMPT → {provider}{_RESET}\n"
        f"{_BOX_STYLE}├{_BOX_H}┤{_RESET}\n"
        f"{_BOX_STYLE}│{_RESET} {_DIM}SYSTEM:{_RESET}\n"
        f"{system_prompt}\n"
        f"{_BOX_STYLE}├{_BOX_H}┤{_RESET}\n"
        f"{_BOX_STYLE}│{_RESET} {_DIM}USER:{_RESET}\n"
        f"{user_text}\n"
        f"{_BOX_STYLE}└{_BOX_H}┘{_RESET}",
        stacklevel=2,
    )


def log_response(provider: str, raw_text: str | None):
    """DEBUG 级别记录 LLM 的原始输出。"""
    if raw_text is None:
        _response_logger.debug("[%s] 📥 模型返回空响应", provider, stacklevel=2)
        return
    _response_logger.debug(
        "%s",
        f"\n{_BOX_STYLE}┌{_BOX_H}┐{_RESET}\n"
        f"{_BOX_STYLE}│{_RESET} {_RESPONSE_STYLE}📥 RESPONSE ← {provider}{_RESET}\n"
        f"{_BOX_STYLE}├{_BOX_H}┤{_RESET}\n"
        f"{raw_text}\n"
        f"{_BOX_STYLE}└{_BOX_H}┘{_RESET}",
        stacklevel=2,
    )


def log_tool_call(provider: str, fn_name: str, args: dict):
    """DEBUG 级别记录 LLM 发起的工具调用原文，以格式化方框展示。"""
    import json as _json
    args_text = _json.dumps(args, ensure_ascii=False, indent=2)
    _tool_logger.debug(
        "%s",
        f"\n{_BOX_STYLE}┌{_BOX_H}┐{_RESET}\n"
        f"{_BOX_STYLE}│{_RESET} {_TOOL_STYLE}🔧 TOOL CALL  {fn_name}  [{provider}]{_RESET}\n"
        f"{_BOX_STYLE}├{_BOX_H}┤{_RESET}\n"
        f"{args_text}\n"
        f"{_BOX_STYLE}└{_BOX_H}┘{_RESET}",
        stacklevel=2,
    )
