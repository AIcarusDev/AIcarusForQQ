"""Loopback proxy that gives the native browser an Agent-scoped network view.

The browser remains a lightweight Windows process.  Every browser protocol
that Chromium can proxy is forced through this gateway:

* public destinations are resolved here and accepted only when every returned
  address is globally routable;
* Agent-loopback destinations are opened through the optional Linux workspace
  tunnel and never by connecting to Windows loopback;
* private, link-local, multicast, reserved and otherwise non-global targets
  fail closed.

This boundary protects browser navigations and web-platform requests.  The
Chromium process is still trusted; command-line launch policy disables the
known direct UDP paths separately in ``browser.session``.
"""

from __future__ import annotations

import base64
import ipaddress
import logging
import os
import socket
import socketserver
import subprocess
import threading
from dataclasses import dataclass
from typing import BinaryIO, Callable, Iterable, Sequence
from urllib.parse import SplitResult, urlsplit, urlunsplit


logger = logging.getLogger("AICQ.browser.gateway")

MAX_PROXY_HEADER_BYTES = 64 * 1024
CONNECT_TIMEOUT_SECONDS = 15.0
WORKSPACE_HANDSHAKE = b"AICQ-WORKSPACE-TUNNEL/1\n"
_LOOPBACK_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "loopback"}
_SYNTHETIC_PROXY_IPV4 = ipaddress.ip_network("198.18.0.0/15")


class BrowserNetworkError(RuntimeError):
    """A destination or transport failed the browser network policy."""


@dataclass(frozen=True, slots=True)
class ResolvedAddress:
    family: int
    socktype: int
    proto: int
    sockaddr: tuple


@dataclass(frozen=True, slots=True)
class ProxyTarget:
    host: str
    port: int
    workspace_loopback: bool


@dataclass(frozen=True, slots=True)
class UpstreamProxy:
    host: str
    port: int
    tls: bool = False
    authorization: str = ""

    @classmethod
    def parse(cls, raw: str | None) -> "UpstreamProxy | None":
        value = str(raw or "").strip()
        if not value:
            return None
        if "://" not in value:
            value = f"http://{value}"
        parsed = urlsplit(value)
        scheme = parsed.scheme.casefold()
        if scheme not in {"http", "https"} or not parsed.hostname:
            raise BrowserNetworkError("browser upstream proxy must be an http:// or https:// URL")
        try:
            port = parsed.port or (443 if scheme == "https" else 80)
        except ValueError as exc:
            raise BrowserNetworkError("browser upstream proxy has an invalid port") from exc
        authorization = ""
        if parsed.username is not None:
            username = parsed.username
            password = parsed.password or ""
            token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
            authorization = f"Basic {token}"
        return cls(parsed.hostname, port, scheme == "https", authorization)


def _normalize_host(host: str) -> str:
    value = str(host or "").strip().rstrip(".").casefold()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return value


def _literal_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(_normalize_host(host))
    except ValueError:
        return None


def is_workspace_loopback_host(host: str) -> bool:
    """Return whether a URL host belongs to the Agent computer's loopback."""

    normalized = _normalize_host(host)
    if normalized in _LOOPBACK_NAMES or normalized.endswith(".localhost"):
        return True
    literal = _literal_ip(normalized)
    return bool(literal is not None and literal.is_loopback)


def _validate_port(value: int) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise BrowserNetworkError("browser destination port is invalid") from exc
    if port < 1 or port > 65535:
        raise BrowserNetworkError("browser destination port is outside 1..65535")
    return port


def classify_target(host: str, port: int) -> ProxyTarget:
    normalized = _normalize_host(host)
    if not normalized or "\x00" in normalized:
        raise BrowserNetworkError("browser destination host is invalid")
    return ProxyTarget(
        host=normalized,
        port=_validate_port(port),
        workspace_loopback=is_workspace_loopback_host(normalized),
    )


def validate_browser_url(url: str) -> str:
    """Accept only browser-network URLs; local files are outside this world."""

    value = str(url or "").strip()
    parsed = urlsplit(value)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("url must be an absolute http:// or https:// URL")
    return value


def resolve_public_addresses(
    host: str,
    port: int,
    *,
    resolver: Callable[..., Sequence[tuple]] = socket.getaddrinfo,
    allow_synthetic_proxy_ips: bool = False,
) -> list[ResolvedAddress]:
    """Resolve one destination and reject the whole answer on any non-global IP.

    Rejecting mixed public/private answers prevents an attacker from relying on
    address ordering or a later DNS rebind to cross the boundary.
    """

    target = classify_target(host, port)
    if target.workspace_loopback:
        raise BrowserNetworkError("Agent loopback must use the workspace tunnel")
    try:
        raw = resolver(target.host, target.port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise BrowserNetworkError("browser destination DNS resolution failed") from exc
    addresses: list[ResolvedAddress] = []
    seen: set[tuple[int, tuple]] = set()
    for family, socktype, proto, _canonname, sockaddr in raw:
        if family not in {socket.AF_INET, socket.AF_INET6} or not sockaddr:
            continue
        try:
            literal = ipaddress.ip_address(str(sockaddr[0]).split("%", 1)[0])
        except ValueError as exc:
            raise BrowserNetworkError("browser destination resolved to an invalid IP") from exc
        synthetic_proxy_ip = bool(
            allow_synthetic_proxy_ips
            and isinstance(literal, ipaddress.IPv4Address)
            and literal in _SYNTHETIC_PROXY_IPV4
        )
        if not literal.is_global and not synthetic_proxy_ip:
            raise BrowserNetworkError("browser destination resolved to a local or non-public IP")
        key = (family, tuple(sockaddr))
        if key in seen:
            continue
        seen.add(key)
        addresses.append(ResolvedAddress(family, socktype, proto, tuple(sockaddr)))
    if not addresses:
        raise BrowserNetworkError("browser destination has no public TCP address")
    return addresses


def _connect_addresses(addresses: Iterable[ResolvedAddress], *, timeout: float) -> socket.socket:
    last_error: OSError | None = None
    for address in addresses:
        candidate = socket.socket(address.family, address.socktype, address.proto)
        candidate.settimeout(timeout)
        try:
            candidate.connect(address.sockaddr)
            candidate.settimeout(None)
            return candidate
        except OSError as exc:
            last_error = exc
            candidate.close()
    raise BrowserNetworkError("browser destination connection failed") from last_error


def _read_header(sock: socket.socket, *, limit: int = MAX_PROXY_HEADER_BYTES) -> tuple[bytes, bytes]:
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(min(8192, limit + 1 - len(data)))
        if not chunk:
            raise BrowserNetworkError("browser proxy client closed before sending headers")
        data.extend(chunk)
        if len(data) > limit:
            raise BrowserNetworkError("browser proxy headers exceed the 64 KiB limit")
    marker = data.index(b"\r\n\r\n") + 4
    return bytes(data[:marker]), bytes(data[marker:])


def _split_authority(authority: str, default_port: int) -> tuple[str, int]:
    value = str(authority or "").strip()
    if value.startswith("["):
        closing = value.find("]")
        if closing < 0:
            raise BrowserNetworkError("browser proxy received an invalid IPv6 authority")
        host = value[1:closing]
        suffix = value[closing + 1 :]
        port = default_port if not suffix else _validate_port(suffix.removeprefix(":"))
        return host, port
    if value.count(":") == 1:
        host, raw_port = value.rsplit(":", 1)
        return host, _validate_port(raw_port)
    if ":" in value:
        # An unbracketed IPv6 literal has no explicit port.
        return value, default_port
    return value, default_port


def _proxy_error(status: int, message: str) -> bytes:
    safe = message.encode("utf-8", errors="replace")[:1024]
    reason = {
        400: b"Bad Request",
        403: b"Forbidden",
        502: b"Bad Gateway",
        503: b"Service Unavailable",
    }.get(status, b"Proxy Error")
    return (
        f"HTTP/1.1 {status} ".encode("ascii")
        + reason
        + b"\r\nContent-Type: text/plain; charset=utf-8\r\nConnection: close\r\nContent-Length: "
        + str(len(safe)).encode("ascii")
        + b"\r\n\r\n"
        + safe
    )


def _read_exact_fd(fd: int, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = os.read(fd, size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def _write_all_fd(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise BrokenPipeError("workspace tunnel pipe closed")
        view = view[written:]


def _rewrite_http_request(header: bytes, parsed: SplitResult, *, absolute_url: str | None) -> bytes:
    lines = header.decode("iso-8859-1").split("\r\n")
    request_parts = lines[0].split(" ", 2)
    if len(request_parts) != 3:
        raise BrowserNetworkError("browser proxy received an invalid request line")
    path = absolute_url or urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    rewritten = [f"{request_parts[0]} {path} {request_parts[2]}"]
    upgrade = False
    for line in lines[1:]:
        if not line:
            continue
        name, separator, value = line.partition(":")
        if not separator:
            raise BrowserNetworkError("browser proxy received an invalid header")
        lowered = name.strip().casefold()
        if lowered in {"proxy-authorization", "proxy-connection", "connection"}:
            if lowered == "connection" and "upgrade" in value.casefold():
                upgrade = True
            continue
        rewritten.append(f"{name}:{value}")
    rewritten.append("Connection: Upgrade" if upgrade else "Connection: close")
    rewritten.extend(["", ""])
    return "\r\n".join(rewritten).encode("iso-8859-1")


def _pump_sockets(left: socket.socket, right: socket.socket, *, initial_right: bytes = b"") -> None:
    if initial_right:
        right.sendall(initial_right)

    def copy(source: socket.socket, destination: socket.socket) -> None:
        try:
            while True:
                chunk = source.recv(65536)
                if not chunk:
                    break
                destination.sendall(chunk)
        except OSError:
            pass
        finally:
            try:
                destination.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    worker = threading.Thread(target=copy, args=(left, right), daemon=True)
    worker.start()
    copy(right, left)
    worker.join(timeout=1.0)


class WorkspaceProcessTunnel:
    """One binary TCP stream opened inside the optional Agent workspace."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        enabled: Callable[[], bool],
        wsl_executable: str = "wsl.exe",
        distro_name: str = "AICQ-Workspace",
        appliance_user: str = "aicqws",
    ) -> None:
        if not enabled():
            raise BrowserNetworkError("Agent workspace is disabled; its localhost is unavailable")
        target = classify_target(host, port)
        if not target.workspace_loopback:
            raise BrowserNetworkError("workspace tunnel accepts loopback destinations only")
        argv = [
            wsl_executable,
            "--distribution",
            distro_name,
            "--user",
            appliance_user,
            "--exec",
            "/usr/local/bin/aicq-workspace-browser-connect",
            target.host,
            str(target.port),
        ]
        try:
            self.process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (FileNotFoundError, OSError) as exc:
            raise BrowserNetworkError("Agent workspace tunnel could not start") from exc
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self.stdin: BinaryIO = self.process.stdin
        self.stdout: BinaryIO = self.process.stdout
        self.stderr: BinaryIO = self.process.stderr
        timer = threading.Timer(CONNECT_TIMEOUT_SECONDS, self._kill)
        timer.daemon = True
        timer.start()
        try:
            handshake = _read_exact_fd(self.stdout.fileno(), len(WORKSPACE_HANDSHAKE))
        finally:
            timer.cancel()
        if handshake != WORKSPACE_HANDSHAKE:
            self._kill()
            diagnostic = self.stderr.read(4096).decode("utf-8", errors="replace").strip()
            raise BrowserNetworkError(diagnostic or "Agent workspace localhost connection failed")
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="aicq-browser-tunnel-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        try:
            while os.read(self.stderr.fileno(), 4096):
                pass
        except (OSError, ValueError):
            pass

    def _kill(self) -> None:
        if getattr(self, "process", None) is not None and self.process.poll() is None:
            try:
                self.process.kill()
            except OSError:
                pass

    def close(self) -> None:
        for stream in (getattr(self, "stdin", None), getattr(self, "stdout", None), getattr(self, "stderr", None)):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self._kill()
        try:
            self.process.wait(timeout=2.0)
        except (subprocess.TimeoutExpired, OSError):
            self._kill()
        stderr_thread = getattr(self, "_stderr_thread", None)
        if stderr_thread is not None and stderr_thread != threading.current_thread():
            stderr_thread.join(timeout=0.5)

    def pump(self, client: socket.socket, *, initial: bytes = b"") -> None:
        if initial:
            _write_all_fd(self.stdin.fileno(), initial)
        def client_to_workspace() -> None:
            try:
                while True:
                    chunk = client.recv(65536)
                    if not chunk:
                        break
                    _write_all_fd(self.stdin.fileno(), chunk)
            except (OSError, ValueError):
                pass
            finally:
                try:
                    self.stdin.close()
                except OSError:
                    pass

        worker = threading.Thread(target=client_to_workspace, daemon=True)
        worker.start()
        try:
            while True:
                chunk = os.read(self.stdout.fileno(), 65536)
                if not chunk:
                    break
                client.sendall(chunk)
        except (OSError, ValueError):
            pass
        finally:
            self._kill()
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            worker.join(timeout=1.0)


class _ThreadingProxyServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = False
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], gateway: "BrowserGateway") -> None:
        self.gateway = gateway
        super().__init__(server_address, _ProxyHandler)


class _ProxyHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        gateway = self.server.gateway  # type: ignore[attr-defined]
        if not gateway._connections.acquire(blocking=False):
            self.request.sendall(_proxy_error(503, "browser gateway connection limit reached"))
            return
        try:
            gateway._handle_client(self.request)
        finally:
            gateway._connections.release()


class BrowserGateway:
    """A lazy native proxy; starting it never inspects or starts WSL."""

    def __init__(
        self,
        *,
        workspace_enabled: Callable[[], bool],
        upstream_proxy: str | None = None,
        resolver: Callable[..., Sequence[tuple]] = socket.getaddrinfo,
        max_connections: int = 128,
    ) -> None:
        self._workspace_enabled = workspace_enabled
        self._upstream = UpstreamProxy.parse(upstream_proxy)
        self._resolver = resolver
        self._connections = threading.BoundedSemaphore(max(1, int(max_connections)))
        self._server: _ThreadingProxyServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def proxy_url(self) -> str:
        self.ensure_started()
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def ensure_started(self) -> None:
        with self._lock:
            if self._server is not None:
                return
            server = _ThreadingProxyServer(("127.0.0.1", 0), self)
            thread = threading.Thread(target=server.serve_forever, name="aicq-browser-gateway", daemon=True)
            thread.start()
            self._server = server
            self._thread = thread
            logger.info("[browser] isolated gateway listening on 127.0.0.1:%d", server.server_address[1])

    def close(self) -> None:
        with self._lock:
            server = self._server
            thread = self._thread
            self._server = None
            self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread != threading.current_thread():
            thread.join(timeout=2.0)

    def _connect_upstream(self) -> socket.socket:
        assert self._upstream is not None
        addresses = socket.getaddrinfo(self._upstream.host, self._upstream.port, type=socket.SOCK_STREAM)
        normalized = [ResolvedAddress(f, st, p, tuple(sa)) for f, st, p, _c, sa in addresses if f in {socket.AF_INET, socket.AF_INET6}]
        if not normalized:
            raise BrowserNetworkError("configured browser upstream proxy could not be resolved")
        connection = _connect_addresses(normalized, timeout=CONNECT_TIMEOUT_SECONDS)
        connection.settimeout(CONNECT_TIMEOUT_SECONDS)
        if self._upstream.tls:
            import ssl

            context = ssl.create_default_context()
            connection = context.wrap_socket(connection, server_hostname=self._upstream.host)
        return connection

    def _connect_public(self, target: ProxyTarget, *, tunnel: bool) -> tuple[socket.socket, ResolvedAddress]:
        addresses = resolve_public_addresses(
            target.host,
            target.port,
            resolver=self._resolver,
            allow_synthetic_proxy_ips=self._upstream is not None,
        )
        selected = addresses[0]
        if self._upstream is None:
            return _connect_addresses(addresses, timeout=CONNECT_TIMEOUT_SECONDS), selected
        connection = self._connect_upstream()
        if tunnel:
            authority = f"[{target.host}]:{target.port}" if ":" in target.host else f"{target.host}:{target.port}"
            headers = [
                f"CONNECT {authority} HTTP/1.1",
                f"Host: {authority}",
                "Proxy-Connection: Keep-Alive",
            ]
            if self._upstream.authorization:
                headers.append(f"Proxy-Authorization: {self._upstream.authorization}")
            connection.sendall(("\r\n".join([*headers, "", ""])).encode("iso-8859-1"))
            response, extra = _read_header(connection)
            status = response.split(b"\r\n", 1)[0].split(b" ", 2)
            if len(status) < 2 or status[1] != b"200" or extra:
                connection.close()
                raise BrowserNetworkError("configured browser upstream proxy rejected CONNECT")
        connection.settimeout(None)
        return connection, selected

    def _workspace_tunnel(self, target: ProxyTarget) -> WorkspaceProcessTunnel:
        return WorkspaceProcessTunnel(target.host, target.port, enabled=self._workspace_enabled)

    def _handle_client(self, client: socket.socket) -> None:
        client.settimeout(CONNECT_TIMEOUT_SECONDS)
        try:
            header, body_prefix = _read_header(client)
            first_line = header.split(b"\r\n", 1)[0].decode("iso-8859-1")
            method, raw_target, _version = first_line.split(" ", 2)
            if method.casefold() == "connect":
                self._handle_connect(client, raw_target, body_prefix)
            else:
                self._handle_http(client, header, body_prefix, raw_target)
        except BrowserNetworkError as exc:
            logger.info("[browser] gateway denied request: %s", exc)
            try:
                client.sendall(_proxy_error(502, str(exc)))
            except OSError:
                pass
        except (OSError, ValueError) as exc:
            logger.debug("[browser] gateway transport failed: %s", exc)
            try:
                client.sendall(_proxy_error(400, "invalid browser proxy request"))
            except OSError:
                pass

    def _handle_connect(self, client: socket.socket, raw_target: str, body_prefix: bytes) -> None:
        if body_prefix:
            raise BrowserNetworkError("CONNECT request contained unexpected data before approval")
        host, port = _split_authority(raw_target, 443)
        target = classify_target(host, port)
        client.settimeout(None)
        if target.workspace_loopback:
            tunnel = self._workspace_tunnel(target)
            try:
                client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                tunnel.pump(client)
            finally:
                tunnel.close()
            return
        upstream, _address = self._connect_public(target, tunnel=True)
        try:
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            _pump_sockets(client, upstream)
        finally:
            upstream.close()

    def _handle_http(self, client: socket.socket, header: bytes, body_prefix: bytes, raw_target: str) -> None:
        parsed = urlsplit(raw_target)
        if parsed.scheme.casefold() not in {"http", "ws"} or not parsed.hostname:
            raise BrowserNetworkError("browser proxy accepts absolute http:// or ws:// requests only")
        try:
            port = parsed.port or 80
        except ValueError as exc:
            raise BrowserNetworkError("browser proxy received an invalid URL port") from exc
        target = classify_target(parsed.hostname, port)
        client.settimeout(None)
        if target.workspace_loopback:
            tunnel = self._workspace_tunnel(target)
            try:
                request = _rewrite_http_request(header, parsed, absolute_url=None) + body_prefix
                tunnel.pump(client, initial=request)
            finally:
                tunnel.close()
            return
        upstream, _address = self._connect_public(target, tunnel=False)
        try:
            absolute_url: str | None = None
            if self._upstream is not None:
                absolute_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))
            request = _rewrite_http_request(header, parsed, absolute_url=absolute_url)
            if self._upstream is not None and self._upstream.authorization:
                request = request[:-2] + f"Proxy-Authorization: {self._upstream.authorization}\r\n\r\n".encode("iso-8859-1")
            _pump_sockets(client, upstream, initial_right=request + body_prefix)
        finally:
            upstream.close()


_GATEWAY: BrowserGateway | None = None
_GATEWAY_LOCK = threading.Lock()


def _configured_upstream_proxy() -> str | None:
    configured = (
        os.environ.get("HTTP_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("TAVILY_PROXY", "").strip()
        or None
    )
    if configured:
        return configured
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        ) as key:
            enabled = int(winreg.QueryValueEx(key, "ProxyEnable")[0])
            raw = str(winreg.QueryValueEx(key, "ProxyServer")[0] or "").strip()
    except (OSError, TypeError, ValueError):
        return None
    if not enabled or not raw:
        return None
    if ";" in raw or "=" in raw:
        entries: dict[str, str] = {}
        for item in raw.split(";"):
            name, separator, value = item.partition("=")
            if separator and value.strip():
                entries[name.strip().casefold()] = value.strip()
        raw = entries.get("https") or entries.get("http") or ""
    return raw or None


def _workspace_is_enabled() -> bool:
    try:
        import app_state
        from workspace.config import workspace_enabled

        return workspace_enabled(getattr(app_state, "config", {}) or {})
    except Exception:
        return False


def get_browser_gateway() -> BrowserGateway:
    global _GATEWAY
    with _GATEWAY_LOCK:
        if _GATEWAY is None:
            _GATEWAY = BrowserGateway(
                workspace_enabled=_workspace_is_enabled,
                upstream_proxy=_configured_upstream_proxy(),
            )
        return _GATEWAY


def close_browser_gateway() -> bool:
    global _GATEWAY
    with _GATEWAY_LOCK:
        gateway = _GATEWAY
        _GATEWAY = None
    if gateway is None:
        return False
    gateway.close()
    return True


def chromium_gateway_args(proxy_url: str) -> list[str]:
    """Return the fixed Chromium flags that prevent web traffic bypasses."""

    return [
        f"--proxy-server={proxy_url}",
        "--proxy-bypass-list=<-loopback>",
        "--disable-quic",
        "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
    ]


__all__ = [
    "BrowserGateway",
    "BrowserNetworkError",
    "ProxyTarget",
    "ResolvedAddress",
    "WorkspaceProcessTunnel",
    "classify_target",
    "chromium_gateway_args",
    "close_browser_gateway",
    "get_browser_gateway",
    "is_workspace_loopback_host",
    "resolve_public_addresses",
    "validate_browser_url",
]
