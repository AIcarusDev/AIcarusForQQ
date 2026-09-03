from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FILE_OPS_PATH = ROOT / "scripts/workspace/appliance/opt/aicq-workspace/image/file-ops.py"


def load_file_ops():
    spec = importlib.util.spec_from_file_location("aicq_workspace_file_ops", FILE_OPS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_file_numbers_lines_without_truncation_and_rejects_binary(
    tmp_path, monkeypatch
) -> None:
    file_ops = load_file_ops()
    text_path = tmp_path / "notes.txt"
    text_path.write_text("alpha\n" + "x" * 2100 + "\nomega", encoding="utf-8")
    monkeypatch.setattr(file_ops, "resolve_path", lambda _value: text_path)

    result = file_ops.read_file({"path": "notes.txt", "start_line": 2, "line_count": 1})
    assert result["content"] == "2\t" + "x" * 2100
    assert result["truncated_lines"] == []
    assert result["has_more"] is True
    assert result["next_line"] == 3

    text_path.write_bytes(b"valid\x00utf8")
    with pytest.raises(file_ops.OperationError) as exc_info:
        file_ops.read_file({"path": "notes.txt"})
    assert exc_info.value.code == "binary_file"


def test_read_file_rejects_more_than_5000_source_characters_without_returning_content(
    tmp_path, monkeypatch
) -> None:
    file_ops = load_file_ops()
    text_path = tmp_path / "large.txt"
    lines = ["界" * 3002, *(["界"] * 999)]
    text_path.write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setattr(file_ops, "resolve_path", lambda _value: text_path)

    allowed = file_ops.read_file({"path": "large.txt", "line_count": 1000})

    assert len("\n".join(lines)) == 5000
    assert len(allowed["content"]) > file_ops.MAX_READ_CONTENT_CHARS
    assert allowed["content"].startswith("1\t" + "界" * 3002)
    assert allowed["content"].endswith("1000\t界")

    lines[0] += "界"
    text_path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(file_ops.OperationError) as exc_info:
        file_ops.read_file({"path": "large.txt", "line_count": 1000})

    assert exc_info.value.code == "content_too_large"
    assert exc_info.value.message

    output = io.StringIO()
    monkeypatch.setattr(
        file_ops.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "operation": "read_file",
                    "params": {"path": "large.txt", "line_count": 1000},
                }
            )
        ),
    )
    monkeypatch.setattr(file_ops.sys, "stdout", output)
    assert file_ops.main() == 0
    payload = json.loads(output.getvalue())
    assert payload["ok"] is False
    assert payload["error"]["code"] == "content_too_large"
    assert payload["error"]["message"]


def test_edit_file_is_batch_atomic_and_preserves_bom_and_crlf(tmp_path, monkeypatch) -> None:
    file_ops = load_file_ops()
    path = tmp_path / "notes.txt"
    original = b"\xef\xbb\xbffirst\r\nsecond\r\n"
    path.write_bytes(original)
    monkeypatch.setattr(file_ops, "resolve_path", lambda _value: path)
    captured: dict = {}

    def fake_atomic(target, raw, **kwargs):
        captured.update({"target": target, "raw": raw, **kwargs})

    monkeypatch.setattr(file_ops, "atomic_write", fake_atomic)
    expected = file_ops.revision(original)
    result = file_ops.edit_file(
        {
            "path": "notes.txt",
            "expected_revision": expected,
            "edits": [
                {"old_text": "first", "new_text": "one"},
                {"old_text": "second", "new_text": "two"},
            ],
        }
    )
    assert result["replacements"] == 2
    assert captured["raw"] == b"\xef\xbb\xbfone\r\ntwo\r\n"
    assert captured["expected_revision"] == expected

    captured.clear()
    with pytest.raises(file_ops.OperationError) as exc_info:
        file_ops.edit_file(
            {
                "path": "notes.txt",
                "expected_revision": expected,
                "edits": [
                    {"old_text": "first", "new_text": "one"},
                    {"old_text": "missing", "new_text": "never"},
                ],
            }
        )
    assert exc_info.value.code == "ambiguous_edit"
    assert captured == {}


def test_search_builds_stable_ripgrep_contract_with_context(monkeypatch, tmp_path) -> None:
    file_ops = load_file_ops()
    monkeypatch.setattr(file_ops, "resolve_path", lambda _value: tmp_path)
    captured: list[str] = []

    def fake_rg(argv, *, skip_lines, max_lines):
        captured.extend(argv)
        assert skip_lines == 0
        assert max_lines == 11
        return ["/home/agent/a.py:2:Needle"]

    monkeypatch.setattr(file_ops, "run_rg", fake_rg)
    result = file_ops.search(
        {
            "pattern": "Needle",
            "path": "/home/agent",
            "literal": True,
            "case_sensitive": True,
            "context_before": 2,
            "context_after": 3,
            "glob": "*.py",
            "limit": 10,
        }
    )
    assert "--context-separator" not in captured
    assert captured[-3:] == ["--", "Needle", str(tmp_path)]
    assert captured[captured.index("--before-context") + 1] == "2"
    assert captured[captured.index("--after-context") + 1] == "3"
    assert result["content"] == "/home/agent/a.py:2:Needle"


def test_search_result_page_bounds_a_single_oversized_line(tmp_path) -> None:
    file_ops = load_file_ops()
    result = file_ops.paginate(["界" * 30000], 0, 10, tmp_path)

    assert len(result["content"].encode("utf-8")) <= file_ops.MAX_LIST_BYTES
    assert result["count"] == 1
    assert result["truncated"] is True
