from __future__ import annotations

from file_reading.parsers import parse_document as parse_shared_document
from platforms.qq.files.parsers import parse_document as parse_qq_document


def test_shared_and_qq_document_readers_keep_the_same_content_contract(tmp_path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    selection = {"type": "text_lines", "start_line": 2, "end_line": 2}

    shared = parse_shared_document(str(path), selection)
    qq_compat = parse_qq_document(str(path), selection)

    assert shared == qq_compat == {
        "file_type": "text",
        "document": {"type": "text", "encoding": "utf-8", "total_lines": 2},
        "text": "2\tbeta",
        "location": {"type": "text_lines", "start_line": 2, "end_line": 2},
        "warnings": [],
    }
