from __future__ import annotations

import asyncio
import socket

import pytest

from platforms.qq.adapter.errors import QQFileStreamError
from platforms.qq.files.transport import _validate_public_http_url


def _dns_result(address: str, port: int) -> list[tuple]:
    return [
        (
            socket.AF_INET6 if ":" in address else socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            (address, port),
        )
    ]


@pytest.mark.parametrize("addresses", [
    ["198.18.0.42"],
    ["fdfe:dcba:9876::f"],
    ["198.18.0.42", "fdfe:dcba:9876::f"],
    ["1.1.1.1", "2606:4700:4700::1111"],
])
def test_qq_ftn_download_accepts_proxy_fake_ip_dns(monkeypatch, addresses):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _host, port, *, type: [entry for address in addresses for entry in _dns_result(address, port)],
    )

    asyncio.run(
        _validate_public_http_url(
            "https://edge.dc.ftn.qq.com/ftn_handler/signed-token/?fname=report.zip"
        )
    )


@pytest.mark.parametrize("private_address", ["127.0.0.1", "::1", "fd00::1", "192.168.1.10"])
def test_qq_ftn_fake_ip_does_not_hide_a_mixed_private_dns_answer(monkeypatch, private_address):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _host, port, *, type: (
            _dns_result("198.18.0.42", port)
            + _dns_result("fdfe:dcba:9876::f", port)
            + _dns_result(private_address, port)
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
        ("https://edge.dc.ftn.qq.com/ftn_handler/token/", "fdfe:dcba:9877::1"),
        ("https://attacker.example/ftn_handler/token/", "fdfe:dcba:9876::f"),
        ("https://edge.dc.ftn.qq.com.attacker.example/ftn_handler/token/", "fdfe:dcba:9876::f"),
        ("http://edge.dc.ftn.qq.com/ftn_handler/token/", "fdfe:dcba:9876::f"),
        ("https://edge.dc.ftn.qq.com:8443/ftn_handler/token/", "fdfe:dcba:9876::f"),
        ("https://edge.dc.ftn.qq.com/private-api", "fdfe:dcba:9876::f"),
        ("https://[fdfe:dcba:9876::f]/ftn_handler/token/", "fdfe:dcba:9876::f"),
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
