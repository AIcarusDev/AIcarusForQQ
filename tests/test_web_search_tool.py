import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools.web_search import MAX_CONTENT_CHARS, _compact_content


def test_compact_content_collapses_whitespace_and_truncates() -> None:
    raw = ("alpha\n\n  beta\t" * 80).strip()

    content, truncated, original_chars = _compact_content(raw)

    assert truncated is True
    assert len(content) <= MAX_CONTENT_CHARS
    assert content.endswith("...")
    assert "\n" not in content
    assert "\t" not in content
    assert original_chars > len(content)


def test_compact_content_keeps_short_content_unchanged() -> None:
    content, truncated, original_chars = _compact_content(" short   summary ")

    assert content == "short summary"
    assert truncated is False
    assert original_chars == len("short summary")
