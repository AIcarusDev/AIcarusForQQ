from __future__ import annotations

import re
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent
LONG_NUMERIC_ID_RE = re.compile(r"\b\d{8,12}\b")
ENV_FILE_NAME = "".join([".", "env"])
ENV_FILE_RE = re.compile(rf"(?<!\w){re.escape(ENV_FILE_NAME)}(?!\w)", re.IGNORECASE)
FORBIDDEN_SNIPPETS = (
    "data/AICQ.db",
    "data\\AICQ.db",
    "real_chat",
    "real group",
)


def test_committed_tests_do_not_embed_private_runtime_data():
    offenders: list[str] = []
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        if path.name == "test_privacy_guard.py":
            continue
        text = path.read_text(encoding="utf-8")
        if LONG_NUMERIC_ID_RE.search(text):
            offenders.append(f"{path.name}: long numeric id")
        if ENV_FILE_RE.search(text):
            offenders.append(f"{path.name}: {ENV_FILE_NAME}")
        lowered = text.lower()
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet.lower() in lowered:
                offenders.append(f"{path.name}: {snippet}")

    assert offenders == []


def test_legacy_test_artifacts_are_not_left_outside_pytest_suite():
    legacy_paths = [
        *REPO_ROOT.glob("test_*.py"),
        *REPO_ROOT.glob("test_*.bat"),
        *(REPO_ROOT / "scripts").glob("test_*.py"),
        *(REPO_ROOT / "scripts" / "dialogues").glob("*test*.yaml"),
    ]

    assert legacy_paths == []
