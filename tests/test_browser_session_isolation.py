from __future__ import annotations

from pathlib import Path

import pytest

from browser import session


class _FakeProcess:
    pid = 43210

    def __init__(self) -> None:
        self.terminated = False

    def poll(self):
        return None

    def terminate(self) -> None:
        self.terminated = True


class _FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeOpener:
    def __init__(self, opened: list[object]) -> None:
        self.opened = opened

    def open(self, request, timeout: float):
        self.opened.append((request, timeout))
        return _FakeResponse()


def test_login_url_is_opened_only_after_isolated_chrome_is_verified(monkeypatch, tmp_path: Path) -> None:
    launched: list[list[str]] = []
    opened: list[object] = []
    process = _FakeProcess()

    class Gateway:
        proxy_url = "http://127.0.0.1:45678"

    def fake_popen(argv, **_kwargs):
        launched.append(list(argv))
        return process

    monkeypatch.setattr(session, "get_browser_gateway", lambda: Gateway())
    monkeypatch.setattr(session, "system_chrome_path", lambda: "chrome.exe")
    monkeypatch.setattr(session, "_reserve_loopback_port", lambda: 41234)
    monkeypatch.setattr(session, "_wait_for_cdp", lambda endpoint: None)
    monkeypatch.setattr(session, "build_opener", lambda *_args: _FakeOpener(opened))
    monkeypatch.setattr(session.subprocess, "Popen", fake_popen)

    browser_path, pid = session.launch_isolated_login_browser(
        profile_dir=tmp_path / "profile",
        url="https://accounts.example/login?a=1&b=2",
    )

    assert browser_path == "chrome.exe"
    assert pid == 43210
    assert launched and "https://accounts.example" not in " ".join(launched[0])
    assert "--proxy-server=http://127.0.0.1:45678" in launched[0]
    assert "--proxy-bypass-list=<-loopback>" in launched[0]
    request, timeout = opened[0]
    assert request.full_url.startswith("http://127.0.0.1:41234/json/new?")
    assert "https%3A%2F%2Faccounts.example%2Flogin%3Fa%3D1%26b%3D2" in request.full_url
    assert request.get_method() == "PUT"
    assert timeout == 3.0
    assert process.terminated is False


def test_login_browser_is_terminated_when_cdp_policy_cannot_be_verified(monkeypatch, tmp_path: Path) -> None:
    process = _FakeProcess()

    class Gateway:
        proxy_url = "http://127.0.0.1:45678"

    monkeypatch.setattr(session, "get_browser_gateway", lambda: Gateway())
    monkeypatch.setattr(session, "system_chrome_path", lambda: "chrome.exe")
    monkeypatch.setattr(session, "_reserve_loopback_port", lambda: 41234)
    monkeypatch.setattr(session.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        session,
        "_wait_for_cdp",
        lambda _endpoint: (_ for _ in ()).throw(RuntimeError("not isolated")),
    )

    with pytest.raises(RuntimeError):
        session.launch_isolated_login_browser(
            profile_dir=tmp_path / "profile",
            url="https://accounts.example/login",
        )
    assert process.terminated is True
