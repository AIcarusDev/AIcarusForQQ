from __future__ import annotations

import socket
import threading

import pytest

from browser.gateway import (
    BrowserGateway,
    BrowserNetworkError,
    WorkspaceProcessTunnel,
    _configured_upstream_proxy,
    chromium_gateway_args,
    classify_target,
    resolve_public_addresses,
    validate_browser_url,
)


def _proxy_request(proxy_url: str, request: bytes) -> bytes:
    port = int(proxy_url.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=2.0) as client:
        client.sendall(request)
        client.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)


def test_browser_url_contract_rejects_host_files_and_non_network_schemes() -> None:
    assert validate_browser_url("https://example.com/path") == "https://example.com/path"
    with pytest.raises(ValueError):
        validate_browser_url("file:///C:/Users/example/secret.txt")
    with pytest.raises(ValueError):
        validate_browser_url("data:text/plain,secret")


@pytest.mark.parametrize(
    "host",
    ["localhost", "service.localhost", "127.0.0.1", "127.9.8.7", "::1"],
)
def test_loopback_targets_are_classified_as_agent_workspace(host: str) -> None:
    target = classify_target(host, 7860)
    assert target.workspace_loopback is True


def test_public_resolution_rejects_private_and_mixed_dns_answers() -> None:
    def private_resolver(*_args, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.10", 443))]

    with pytest.raises(BrowserNetworkError):
        resolve_public_addresses("private.example", 443, resolver=private_resolver)

    def mixed_resolver(*_args, **_kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ]

    with pytest.raises(BrowserNetworkError):
        resolve_public_addresses("rebind.example", 443, resolver=mixed_resolver)


def test_synthetic_proxy_dns_range_is_only_accepted_with_an_upstream_proxy() -> None:
    def fake_ip_resolver(*_args, **_kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("198.18.0.42", 443)),
            (
                socket.AF_INET6,
                socket.SOCK_STREAM,
                6,
                "",
                ("fdfe:dcba:9876::42", 443, 0, 0),
            ),
        ]

    with pytest.raises(BrowserNetworkError):
        resolve_public_addresses("public.example", 443, resolver=fake_ip_resolver)
    addresses = resolve_public_addresses(
        "public.example",
        443,
        resolver=fake_ip_resolver,
        allow_synthetic_proxy_ips=True,
    )
    assert [address.sockaddr for address in addresses] == [
        ("198.18.0.42", 443),
        ("fdfe:dcba:9876::42", 443, 0, 0),
    ]


def test_browser_proxy_has_priority_over_generic_process_proxies(monkeypatch) -> None:
    monkeypatch.setenv("BROWSER_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("HTTP_PROXY", "http://generic-http.test:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://generic-https.test:8443")
    monkeypatch.setenv("TAVILY_PROXY", "http://search-only.test:9000")

    assert _configured_upstream_proxy() == "http://127.0.0.1:7890"


def test_search_proxy_is_not_reused_as_browser_proxy(monkeypatch) -> None:
    monkeypatch.delenv("BROWSER_PROXY", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.setenv("TAVILY_PROXY", "http://search-only.test:9000")
    monkeypatch.setattr("browser.gateway._windows_system_proxy", lambda: None)

    assert _configured_upstream_proxy() is None


def test_upstream_proxy_receives_original_public_hostname_after_fake_ip_validation() -> None:
    upstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    upstream.bind(("127.0.0.1", 0))
    upstream.listen()
    upstream_port = int(upstream.getsockname()[1])
    received: list[bytes] = []

    def serve_proxy() -> None:
        connection, _address = upstream.accept()
        with connection:
            header = bytearray()
            while b"\r\n\r\n" not in header:
                header.extend(connection.recv(4096))
            received.append(bytes(header))
            body = b"through-configured-proxy" * 10_000
            connection.sendall(
                b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: "
                + str(len(body)).encode("ascii")
                + b"\r\n\r\n"
                + body
            )

    worker = threading.Thread(target=serve_proxy, daemon=True)
    worker.start()

    def fake_ip_resolver(*_args, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("198.18.0.42", 80))]

    gateway = BrowserGateway(
        workspace_enabled=lambda: False,
        upstream_proxy=f"127.0.0.1:{upstream_port}",
        resolver=fake_ip_resolver,
    )
    try:
        response = _proxy_request(
            gateway.proxy_url,
            b"GET http://public.example/probe?q=1 HTTP/1.1\r\nHost: public.example\r\n\r\n",
        )
    finally:
        gateway.close()
        upstream.close()
        worker.join(timeout=1.0)

    assert response.endswith(b"through-configured-proxy" * 10_000)
    assert received and received[0].startswith(
        b"GET http://public.example/probe?q=1 HTTP/1.1\r\n"
    )


def test_gateway_denies_a_public_name_that_resolves_to_windows_loopback() -> None:
    def resolver(*_args, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    gateway = BrowserGateway(workspace_enabled=lambda: False, resolver=resolver)
    try:
        response = _proxy_request(
            gateway.proxy_url,
            b"CONNECT rebinding.example:443 HTTP/1.1\r\nHost: rebinding.example:443\r\n\r\n",
        )
    finally:
        gateway.close()
    assert response.startswith(b"HTTP/1.1 502")
    assert b"non-public IP" in response


def test_gateway_routes_localhost_to_workspace_without_touching_windows_port() -> None:
    trap = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    trap.bind(("127.0.0.1", 0))
    trap.listen()
    trap.settimeout(0.2)
    windows_port = int(trap.getsockname()[1])
    captured: list[bytes] = []

    class FakeTunnel:
        def pump(self, client: socket.socket, *, initial: bytes = b"") -> None:
            captured.append(initial)
            body = b"agent-local-response"
            client.sendall(
                b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: "
                + str(len(body)).encode("ascii")
                + b"\r\n\r\n"
                + body
            )

        def close(self) -> None:
            return None

    class TestGateway(BrowserGateway):
        def _workspace_tunnel(self, target):
            assert target.port == windows_port
            return FakeTunnel()

    gateway = TestGateway(workspace_enabled=lambda: True)
    try:
        response = _proxy_request(
            gateway.proxy_url,
            f"GET http://127.0.0.1:{windows_port}/probe HTTP/1.1\r\nHost: 127.0.0.1:{windows_port}\r\n\r\n".encode(
                "ascii"
            ),
        )
        with pytest.raises(TimeoutError):
            trap.accept()
    finally:
        gateway.close()
        trap.close()

    assert b"agent-local-response" in response
    assert captured and captured[0].startswith(b"GET /probe HTTP/1.1")


def test_unavailable_workspace_localhost_returns_renderable_error_page() -> None:
    class TestGateway(BrowserGateway):
        def _workspace_tunnel(self, _target):
            raise BrowserNetworkError("Agent localhost connection failed: <unavailable>")

    gateway = TestGateway(workspace_enabled=lambda: True)
    try:
        response = _proxy_request(
            gateway.proxy_url,
            b"GET http://localhost:8765/preview.html HTTP/1.1\r\nHost: localhost:8765\r\n\r\n",
        )
    finally:
        gateway.close()

    assert response.startswith(b"HTTP/1.1 502 Bad Gateway")
    assert b"Content-Type: text/html; charset=utf-8" in response
    assert b"<title>Agent localhost unavailable</title>" in response
    assert b"&lt;unavailable&gt;" in response


def test_disabled_workspace_fails_before_starting_wsl(monkeypatch) -> None:
    called = False

    def unexpected_popen(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("wsl.exe must not start")

    monkeypatch.setattr("browser.gateway.subprocess.Popen", unexpected_popen)
    with pytest.raises(BrowserNetworkError):
        WorkspaceProcessTunnel("127.0.0.1", 7860, enabled=lambda: False)
    assert called is False


def test_chromium_gateway_flags_remove_loopback_bypass_and_direct_udp() -> None:
    args = chromium_gateway_args("http://127.0.0.1:32123")
    assert "--proxy-server=http://127.0.0.1:32123" in args
    assert "--proxy-bypass-list=<-loopback>" in args
    assert "--disable-quic" in args
    assert "--force-webrtc-ip-handling-policy=disable_non_proxied_udp" in args
