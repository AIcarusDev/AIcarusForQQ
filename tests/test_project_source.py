from __future__ import annotations

import os
from pathlib import Path

import pytest

from project_source.service import MAX_READ_CHARS, ProjectSourceService
from tools import build_tools
from tools.namespaces import NamespaceRuntimeState, load_namespace_registry


PROTECTED_ENV_NAME = "." + "env"


def test_list_exposes_protected_source_but_read_denies_before_open(monkeypatch, tmp_path: Path):
    protected = tmp_path / PROTECTED_ENV_NAME
    protected.write_text("CANARY=secret-value\n", encoding="utf-8")
    service = ProjectSourceService(tmp_path)

    listed = service.list_directory(".")
    row = next(item for item in listed["entries"] if item["name"] == PROTECTED_ENV_NAME)
    assert row["content_access"] == "denied"

    monkeypatch.setattr(
        service,
        "_read_file_bytes",
        lambda *_args, **_kwargs: pytest.fail("protected source was opened"),
    )
    denied = service.read_file(PROTECTED_ENV_NAME)
    assert denied["ok"] is False
    assert denied["code"] == "permission_denied"


def test_read_cursor_returns_every_character_and_preserves_source_text(tmp_path: Path):
    token_like_comment = "# sk-this-is-ordinary-source-text\n"
    content = "x" * (MAX_READ_CHARS + 17) + "\n" + token_like_comment + "tail\n"
    (tmp_path / "sample.py").write_bytes(content.encode("utf-8"))
    service = ProjectSourceService(tmp_path)

    chunks: list[str] = []
    result = service.read_file("sample.py", line_count=10)
    while True:
        assert result["ok"] is True
        chunks.append(result["content"])
        cursor = result.get("next_cursor")
        if not cursor:
            break
        result = service.read_file("sample.py", line_count=10, cursor=cursor)

    assert "".join(chunks) == content
    assert token_like_comment in "".join(chunks)


def test_known_config_secret_field_is_masked_without_scanning_other_text(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    token_like_comment = "sk-comment-is-not-a-protected-location"
    config_text = (
        "tts:\n"
        "  enabled: true\n"
        "  secret_token: |\n"
        "    real-secret-line-one\n"
        "    real-secret-line-two\n"
        f"note: {token_like_comment}\n"
    )
    (config_dir / "config_user.yaml").write_text(config_text, encoding="utf-8")
    service = ProjectSourceService(tmp_path)

    result = service.read_file("config/config_user.yaml")

    assert result["ok"] is True
    assert result["filtered_fields"] == ["tts.secret_token"]
    assert "real-secret-line-one" not in result["content"]
    assert "real-secret-line-two" not in result["content"]
    assert token_like_comment in result["content"]


def test_content_search_uses_the_same_protection_boundary(tmp_path: Path):
    (tmp_path / PROTECTED_ENV_NAME).write_text("hidden-canary\n", encoding="utf-8")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config_user.yaml").write_text(
        "tts:\n  secret_token: hidden-canary\n",
        encoding="utf-8",
    )
    (tmp_path / "source.py").write_text("# visible-canary\n", encoding="utf-8")
    service = ProjectSourceService(tmp_path)

    hidden = service.search("hidden-canary")
    visible = service.search("visible-canary")
    paths = service.search(PROTECTED_ENV_NAME, mode="path")

    assert hidden["matches"] == []
    assert hidden["skipped"]["permission_denied"] == 1
    assert visible["matches"] == [
        {"path": "source.py", "line": 1, "text": "# visible-canary"}
    ]
    assert any(item["path"] == PROTECTED_ENV_NAME for item in paths["matches"])


def test_recursive_glob_also_matches_files_at_the_search_root(tmp_path: Path):
    (tmp_path / "root.py").write_text("needle\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "child.py").write_text("needle\n", encoding="utf-8")
    service = ProjectSourceService(tmp_path)

    result = service.search("needle", glob="**/*.py")

    assert [item["path"] for item in result["matches"]] == ["root.py", "nested/child.py"]


def test_database_and_multimodal_cache_text_are_unsupported(tmp_path: Path):
    (tmp_path / "sample.sqlite3").write_text("plain text with a database suffix", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "image.meta.json").write_text('{"description":"cached"}', encoding="utf-8")
    service = ProjectSourceService(tmp_path)

    database = service.read_file("sample.sqlite3")
    cached = service.read_file("cache/image.meta.json")
    cache_row = next(item for item in service.list_directory(".")["entries"] if item["name"] == "cache")

    assert database["code"] == "unsupported_file_type"
    assert cached["code"] == "unsupported_source"
    assert cache_row["content_access"] == "unsupported"


def test_paths_cannot_escape_or_follow_links(tmp_path: Path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside", encoding="utf-8")
    service = ProjectSourceService(tmp_path)

    escaped = service.read_file("../outside.txt")
    assert escaped["code"] == "path_outside_project"

    link = tmp_path / "outside-link.txt"
    try:
        os.symlink(outside, link)
    except OSError:
        return
    linked = service.read_file("outside-link.txt")
    assert linked["code"] == "permission_denied"
    link_row = next(item for item in service.list_directory(".")["entries"] if item["name"] == link.name)
    assert link_row["kind"] == "link"
    assert link_row["content_access"] == "denied"


def test_namespace_registers_three_read_only_tools_and_skill():
    registry = load_namespace_registry()
    spec = registry.get("project_source")
    assert spec is not None
    assert spec.tools == ("list", "read", "search")
    assert spec.skill == "project-source"
    assert spec.activation.platform == ""

    state = NamespaceRuntimeState()
    state.open("project_source", registry, 1)
    collection = build_tools(
        {},
        namespace_state=state,
        current_round=1,
        default_ttl_rounds=5,
    )

    assert "project_source" in collection.active_namespace_names()
    assert {
        "project_source.list",
        "project_source.read",
        "project_source.search",
    }.issubset(collection.active_names())
    assert all(
        collection.active_specs[name].result_cdata is True
        for name in (
            "project_source.list",
            "project_source.read",
            "project_source.search",
        )
    )
