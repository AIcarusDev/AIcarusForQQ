from __future__ import annotations

import asyncio
import socket

import pytest

from platforms.qq.adapter.errors import QQFileStreamError
from platforms.qq.files.transport import _validate_public_http_url


def _dns_result(address: str, port: int) -> list[tuple]:
    return [
        (
            socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            (address, port),
        )
    ]


def test_qq_ftn_download_accepts_proxy_fake_ip_dns(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _host, port, *, type: _dns_result("198.18.0.42", port),
    )

    asyncio.run(
        _validate_public_http_url(
            "https://edge.dc.ftn.qq.com/ftn_handler/signed-token/?fname=report.zip"
        )
    )


def test_qq_ftn_fake_ip_does_not_hide_a_mixed_private_dns_answer(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _host, port, *, type: (
            _dns_result("198.18.0.42", port) + _dns_result("127.0.0.1", port)
        ),
    )

    with pytest.raises(QQFileStreamError):
        asyncio.run(
            _validate_public_http_url(
                "https://edge.dc.ftn.qq.com/ftn_handler/signed-token/"
            )
        )


@pytest.mark.parametrize(
    "url,address",
    [
        ("https://attacker.example/ftn_handler/token/", "198.18.0.42"),
        ("https://edge.dc.ftn.qq.com/private-api", "198.18.0.42"),
        ("http://edge.dc.ftn.qq.com/ftn_handler/token/", "198.18.0.42"),
        ("https://edge.dc.ftn.qq.com:8443/ftn_handler/token/", "198.18.0.42"),
        ("https://198.18.0.42/ftn_handler/token/", "198.18.0.42"),
        ("https://edge.dc.ftn.qq.com/ftn_handler/token/", "127.0.0.1"),
        ("https://edge.dc.ftn.qq.com/ftn_handler/token/", "192.168.1.10"),
    ],
)
def test_proxy_fake_ip_exception_does_not_weaken_ssrf_checks(
    monkeypatch,
    url: str,
    address: str,
):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _host, port, *, type: _dns_result(address, port),
    )

    with pytest.raises(QQFileStreamError) as error:
        asyncio.run(_validate_public_http_url(url))

    assert error.value.failure_code == "source_unavailable"
    assert error.value.retryable is False
