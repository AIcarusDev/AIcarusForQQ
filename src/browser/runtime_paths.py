"""Runtime path helpers for browser integrations."""

from __future__ import annotations

import os
from pathlib import Path


def system_chrome_path() -> str:
    candidates = [
        os.environ.get("AICQ_BROWSER_CHROME_PATH", "").strip(),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return "chrome.exe"
