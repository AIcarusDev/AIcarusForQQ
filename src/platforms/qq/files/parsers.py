"""QQ compatibility policy for the shared local document parser."""

from pathlib import Path
from typing import Any

from file_reading.parsers import (
    MAX_ARCHIVE_BYTES,
    MAX_ARCHIVE_ENTRIES,
    MAX_EXTRACTED_CHARS,
    MAX_READ_BYTES,
    ParseContractError,
    ParserMessages,
    detect_type as _detect_type,
    parse_document as _parse_document,
    parse_document_safe as _parse_document_safe,
)


QQ_PARSER_MESSAGES = ParserMessages(
    unsupported_file_type="该文件类型不能由 qq_file.read 直接读取。",
    ocr_required="所选 PDF 页面没有可提取的原生文字，需要 OCR；qq_file.read 暂不支持 OCR。",
    network_disabled="network access is disabled while reading QQ files",
)


def detect_type(path: Path) -> str:
    return _detect_type(path, messages=QQ_PARSER_MESSAGES)


def parse_document(path_value: str, selection: dict[str, Any] | None) -> dict[str, Any]:
    return _parse_document(path_value, selection, messages=QQ_PARSER_MESSAGES)


def parse_document_safe(path_value: str, selection: dict[str, Any] | None) -> dict[str, Any]:
    return _parse_document_safe(path_value, selection, QQ_PARSER_MESSAGES)

__all__ = [
    "MAX_ARCHIVE_BYTES",
    "MAX_ARCHIVE_ENTRIES",
    "MAX_EXTRACTED_CHARS",
    "MAX_READ_BYTES",
    "ParseContractError",
    "ParserMessages",
    "QQ_PARSER_MESSAGES",
    "detect_type",
    "parse_document",
    "parse_document_safe",
]
