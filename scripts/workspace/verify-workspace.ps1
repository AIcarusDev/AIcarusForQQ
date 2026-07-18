[CmdletBinding()]
param(
    [switch]$Full
)

$ErrorActionPreference = 'Stop'
$DistroName = 'AICQ-Workspace'
$Bridge = '/usr/local/bin/aicq-workspace-bridge'
$ProtocolVersion = 5

function Invoke-WorkspaceRpc {
    param(
        [Parameter(Mandatory)][string]$Method,
        [hashtable]$Params = @{}
    )
    $request = @{
        version = $ProtocolVersion
        request_id = [Guid]::NewGuid().ToString('N')
        method = $Method
        params = $Params
    }
    $json = $request | ConvertTo-Json -Compress -Depth 8
    $transferId = [Guid]::NewGuid().ToString('N')
    $requestPath = Join-Path ([IO.Path]::GetTempPath()) "aicq-workspace-rpc-$transferId.in"
    $responsePath = Join-Path ([IO.Path]::GetTempPath()) "aicq-workspace-rpc-$transferId.out"
    $errorPath = Join-Path ([IO.Path]::GetTempPath()) "aicq-workspace-rpc-$transferId.err"
    try {
        $utf8 = New-Object Text.UTF8Encoding($false)
        [IO.File]::WriteAllText($requestPath, $json + "`n", $utf8)
        $bridgeProcess = Start-Process -FilePath wsl.exe -ArgumentList @(
            '--distribution', $DistroName,
            '--user', 'aicqws',
            '--exec', $Bridge
        ) -RedirectStandardInput $requestPath -RedirectStandardOutput $responsePath -RedirectStandardError $errorPath -NoNewWindow -Wait -PassThru
        if ($bridgeProcess.ExitCode -ne 0) {
            $bridgeError = [IO.File]::ReadAllText($errorPath)
            throw "Agent computer bridge failed for $Method (exit $($bridgeProcess.ExitCode)): $bridgeError"
        }
        $responseText = [IO.File]::ReadAllText($responsePath, $utf8)
    } finally {
        Remove-Item -LiteralPath $requestPath, $responsePath, $errorPath -Force -ErrorAction SilentlyContinue
    }
    $response = ($responseText | Out-String) | ConvertFrom-Json
    if ($response.version -ne $ProtocolVersion -or $response.request_id -ne $request.request_id) {
        throw "Agent computer protocol mismatch for $Method."
    }
    if (-not $response.ok) {
        throw "$($response.error.code): $($response.error.message)"
    }
    return $response.result
}

function Invoke-WorkspaceCommand {
    param(
        [Parameter(Mandatory)][string]$Command
    )
    $started = Invoke-WorkspaceRpc -Method start_command -Params @{
        workspace_id = 'default'
        command = $Command
        cwd = '/home/agent'
        stdin = ''
    }
    $null = Invoke-WorkspaceRpc -Method wait_command -Params @{
        workspace_id = 'default'
        command_id = $started.command_id
    }
    $result = Invoke-WorkspaceRpc -Method poll_command -Params @{
        workspace_id = 'default'
        command_id = $started.command_id
        cursor = 0
    }
    if ($result.exit_code -ne 0) {
        throw "Agent computer command failed ($($result.exit_code)): $($result.content)"
    }
    return $result
}

function Get-WorkspaceFirewallBlockPackets {
    $rules = & wsl.exe --distribution $DistroName --user root --exec /usr/sbin/nft list chain inet aicq_workspace output
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect Agent computer firewall counters.' }
    $total = 0L
    foreach ($line in $rules) {
        if ($line -match 'counter packets ([0-9]+).*comment "aicq-block-private-v[46]"') {
            $total += [long]$Matches[1]
        }
    }
    return $total
}

$health = Invoke-WorkspaceRpc -Method health
if (-not $health.firewall_active) { throw 'Agent computer firewall marker is not active.' }
if ($health.image_digest -ne 'sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90') {
    throw 'Installed Agent computer base-image digest does not match the repository manifest.'
}

$ensure = Invoke-WorkspaceRpc -Method ensure_default -Params @{ workspace_id = 'default' }
if ($ensure.container_name -ne 'aicq-workspace-default') { throw 'Unexpected Agent computer container name.' }
$probe = Invoke-WorkspaceCommand -Command 'test "$(id -un)" = agent && test "$(id -u)" = 1000 && id -nG | tr " " "\n" | grep -Fqx sudo && test "$(hostname)" = agent-computer && test "$HOME" = /home/agent && test "$PWD" = /home/agent && test -f ~/.profile && test -f ~/.bashrc && sudo -n true && test "$(sudo id -u)" = 0 && printf foundation-ok'
if ($probe.content.Trim() -ne 'foundation-ok') { throw 'Basic agent home/sudo/Bash probe failed.' }
$createCommandText = (& wsl.exe --distribution $DistroName --user aicqws --exec /usr/bin/env XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus /usr/bin/podman inspect --format '{{json .Config.CreateCommand}}' aicq-workspace-default | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect Agent computer network configuration.' }
$createCommand = @()
foreach ($argument in ($createCommandText | ConvertFrom-Json)) {
    $createCommand += [string]$argument
}
$isolatedNetworkReady = $false
for ($index = 0; $index -lt $createCommand.Count; $index++) {
    if ($createCommand[$index] -in @('--publish', '-p')) {
        throw 'Agent computer must not use explicit container port publishing.'
    }
    if ($createCommand[$index] -eq '--network' -and $index + 1 -lt $createCommand.Count) {
        $isolatedNetworkReady = $createCommand[$index + 1] -eq 'slirp4netns:allow_host_loopback=false'
    }
}
if (-not $isolatedNetworkReady) {
    throw 'Agent computer must use its isolated rootless network namespace.'
}
$published = @(& wsl.exe --distribution $DistroName --user aicqws --exec /usr/bin/env XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus /usr/bin/podman port aicq-workspace-default)
if ($LASTEXITCODE -ne 0 -or @($published | Where-Object { "$($_)".Trim() }).Count -ne 0) {
    throw 'Agent computer must not retain legacy explicit port mappings.'
}
$firewallRules = (& wsl.exe --distribution $DistroName --user root --exec /usr/sbin/nft list table inet aicq_workspace | Out-String)
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect computer egress firewall rules.' }
if ($firewallRules -match 'aicq-web-projection-return') {
    throw 'Agent computer firewall still contains the retired host-network Web projection exception.'
}
if ([regex]::Matches($firewallRules, 'ip daddr @blocked_ipv4 .*comment "aicq-block-private-v4"').Count -lt 2 -or
    [regex]::Matches($firewallRules, 'ip6 daddr @blocked_ipv6 .*comment "aicq-block-private-v6"').Count -lt 2) {
    throw 'Agent computer firewall does not cover both container root and Agent private egress.'
}
if ($firewallRules -notmatch 'iifname != "lo" meta l4proto tcp ct state new .*comment "aicq-block-nonloopback-inbound"') {
    throw 'Agent computer firewall does not block non-loopback inbound TCP traffic.'
}
& wsl.exe --distribution $DistroName --user aicqws --exec /usr/bin/test -x /usr/local/bin/aicq-workspace-browser-connect
if ($LASTEXITCODE -ne 0) { throw 'Agent computer browser tunnel helper is missing.' }

$tunnelPort = 0
$tunnelToken = [Guid]::NewGuid().ToString('N')
$tunnelStatusPath = "/tmp/aicq-browser-tunnel-$tunnelToken.status"
$tunnelServerPath = "/tmp/aicq-browser-tunnel-$tunnelToken-server.py"
$tunnelWaiterPath = "/tmp/aicq-browser-tunnel-$tunnelToken-waiter.py"
$tunnelCleanupPath = "/tmp/aicq-browser-tunnel-$tunnelToken-cleanup.py"
$tunnelExpectedBody = "aicq-browser-tunnel-ok:$tunnelToken"
$tunnelServer = @'
import http.server
import os
import pathlib
import sys

port = int(sys.argv[1])
token = sys.argv[2]
status_path = pathlib.Path(sys.argv[3])

def publish_status(value):
    temporary = status_path.with_name(status_path.name + ".new")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, status_path)

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = ("aicq-browser-tunnel-ok:" + token).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass

try:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    publish_status("ready:" + str(os.getpid()) + ":" + str(server.server_address[1]))
    server.serve_forever(poll_interval=0.1)
except BaseException as exc:
    publish_status("error:" + type(exc).__name__ + ": " + str(exc))
    raise
finally:
    if "server" in globals():
        server.server_close()
'@
$tunnelWaiter = @'
import pathlib
import sys
import time

status_path = pathlib.Path(sys.argv[1])
deadline = time.monotonic() + float(sys.argv[2])
while time.monotonic() < deadline:
    try:
        status = status_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        status = str()
    if status.startswith("ready:"):
        print(status)
        raise SystemExit(0)
    if status.startswith("error:"):
        print(status)
        raise SystemExit(2)
    time.sleep(0.1)
print("timeout: probe process did not publish startup status")
raise SystemExit(3)
'@
$tunnelCleanup = @'
import os
import pathlib
import signal
import sys
import time

token = sys.argv[1].encode("utf-8")
status_path = pathlib.Path(sys.argv[2])
self_pid = os.getpid()

def matching_pids():
    result = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == self_pid:
            continue
        try:
            command_line = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if token in command_line:
            result.append(pid)
    return result

for pid in matching_pids():
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

deadline = time.monotonic() + 3
remaining = matching_pids()
while remaining and time.monotonic() < deadline:
    time.sleep(0.1)
    remaining = matching_pids()

status_path.unlink(missing_ok=True)
status_path.with_name(status_path.name + ".new").unlink(missing_ok=True)
for raw_path in sys.argv[3:]:
    pathlib.Path(raw_path).unlink(missing_ok=True)
if remaining:
    print("probe processes still running: " + ",".join(str(pid) for pid in remaining))
    raise SystemExit(2)
'@
$podmanBaseArguments = @(
    '--distribution', $DistroName,
    '--user', 'aicqws',
    '--exec', '/usr/bin/env',
    'XDG_RUNTIME_DIR=/run/user/1000',
    'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus',
    '/usr/bin/podman'
)
function Copy-TunnelProbeScriptToContainer {
    param(
        [Parameter(Mandatory)][string]$Content,
        [Parameter(Mandatory)][string]$Destination
    )
    $transferId = [Guid]::NewGuid().ToString('N')
    $inputPath = Join-Path ([IO.Path]::GetTempPath()) "aicq-browser-tunnel-$transferId.in"
    $outputPath = Join-Path ([IO.Path]::GetTempPath()) "aicq-browser-tunnel-$transferId.out"
    $errorPath = Join-Path ([IO.Path]::GetTempPath()) "aicq-browser-tunnel-$transferId.err"
    try {
        $utf8 = New-Object Text.UTF8Encoding($false)
        [IO.File]::WriteAllText($inputPath, $Content + "`n", $utf8)
        $arguments = $podmanBaseArguments + @(
            'exec', '--interactive', 'aicq-workspace-default', '/usr/bin/tee', $Destination
        )
        $transfer = Start-Process -FilePath wsl.exe -ArgumentList $arguments `
            -RedirectStandardInput $inputPath -RedirectStandardOutput $outputPath `
            -RedirectStandardError $errorPath -NoNewWindow -Wait -PassThru
        if ($transfer.ExitCode -ne 0) {
            $transferError = [IO.File]::ReadAllText($errorPath)
            throw "Could not stage Agent computer browser-tunnel probe script at ${Destination}: $transferError"
        }
    } finally {
        Remove-Item -LiteralPath $inputPath, $outputPath, $errorPath -Force -ErrorAction SilentlyContinue
    }
}
$tunnelFailure = $null
$tunnelCleanupFailure = ''
try {
    Copy-TunnelProbeScriptToContainer -Content $tunnelCleanup -Destination $tunnelCleanupPath
    Copy-TunnelProbeScriptToContainer -Content $tunnelWaiter -Destination $tunnelWaiterPath
    Copy-TunnelProbeScriptToContainer -Content $tunnelServer -Destination $tunnelServerPath

    & wsl.exe @podmanBaseArguments exec --detach aicq-workspace-default /usr/bin/python3 $tunnelServerPath $tunnelPort $tunnelToken $tunnelStatusPath | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Could not launch Agent computer browser-tunnel probe process.' }

    $tunnelStartup = (& wsl.exe @podmanBaseArguments exec aicq-workspace-default /usr/bin/python3 $tunnelWaiterPath $tunnelStatusPath 10 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $tunnelStartup.StartsWith('ready:')) {
        throw "Agent computer browser-tunnel probe did not become ready: $tunnelStartup"
    }
    $tunnelStartupParts = @($tunnelStartup.Split(':'))
    if ($tunnelStartupParts.Count -ne 3 -or -not [int]::TryParse($tunnelStartupParts[2], [ref]$tunnelPort) -or $tunnelPort -lt 1) {
        throw "Agent computer browser-tunnel probe published an invalid port: $tunnelStartup"
    }

    $tunnelTransferId = [Guid]::NewGuid().ToString('N')
    $tunnelInputPath = Join-Path ([IO.Path]::GetTempPath()) "aicq-browser-tunnel-$tunnelTransferId.in"
    $tunnelOutputPath = Join-Path ([IO.Path]::GetTempPath()) "aicq-browser-tunnel-$tunnelTransferId.out"
    $tunnelErrorPath = Join-Path ([IO.Path]::GetTempPath()) "aicq-browser-tunnel-$tunnelTransferId.err"
    try {
        $requestBytes = [Text.Encoding]::ASCII.GetBytes("GET / HTTP/1.1`r`nHost: localhost:$tunnelPort`r`nConnection: close`r`n`r`n")
        [IO.File]::WriteAllBytes($tunnelInputPath, $requestBytes)
        $tunnelProcess = Start-Process -FilePath wsl.exe -ArgumentList @(
            '--distribution', $DistroName,
            '--user', 'aicqws',
            '--exec', '/usr/local/bin/aicq-workspace-browser-connect',
            '127.0.0.1', "$tunnelPort"
        ) -RedirectStandardInput $tunnelInputPath -RedirectStandardOutput $tunnelOutputPath `
          -RedirectStandardError $tunnelErrorPath -NoNewWindow -PassThru
        if (-not $tunnelProcess.WaitForExit(15000)) {
            $tunnelProcess.Kill()
            throw 'Agent computer browser tunnel probe timed out.'
        }
        $tunnelOutput = [IO.File]::ReadAllBytes($tunnelOutputPath)
        $handshake = [Text.Encoding]::ASCII.GetBytes("AICQ-WORKSPACE-TUNNEL/1`n")
        if ($tunnelProcess.ExitCode -ne 0 -or $tunnelOutput.Length -lt $handshake.Length) {
            $tunnelError = [IO.File]::ReadAllText($tunnelErrorPath)
            throw "Agent computer browser tunnel failed: $tunnelError"
        }
        $handshakeText = [Text.Encoding]::ASCII.GetString($tunnelOutput, 0, $handshake.Length)
        if ($handshakeText -ne "AICQ-WORKSPACE-TUNNEL/1`n") {
            throw 'Agent computer browser tunnel returned an invalid handshake.'
        }
        $httpText = [Text.Encoding]::UTF8.GetString($tunnelOutput, $handshake.Length, $tunnelOutput.Length - $handshake.Length)
        if ($httpText -notmatch [regex]::Escape($tunnelExpectedBody)) {
            throw 'Agent computer browser tunnel did not return the Agent-local HTTP response.'
        }
    } finally {
        Remove-Item -LiteralPath $tunnelInputPath, $tunnelOutputPath, $tunnelErrorPath -Force -ErrorAction SilentlyContinue
    }
} catch {
    $tunnelFailure = $_
} finally {
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $tunnelCleanupOutput = (& wsl.exe @podmanBaseArguments exec aicq-workspace-default /usr/bin/python3 $tunnelCleanupPath $tunnelToken $tunnelStatusPath $tunnelServerPath $tunnelWaiterPath $tunnelCleanupPath 2>&1 | Out-String).Trim()
        $tunnelCleanupExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($tunnelCleanupExitCode -ne 0) {
        $tunnelCleanupFailure = "Could not clean up Agent computer browser-tunnel probe: $tunnelCleanupOutput"
    }
}
if ($tunnelFailure) {
    if ($tunnelCleanupFailure) {
        throw "$($tunnelFailure.Exception.Message) $tunnelCleanupFailure"
    }
    throw $tunnelFailure
}
if ($tunnelCleanupFailure) { throw $tunnelCleanupFailure }

$wslIpv4 = ((& wsl.exe --distribution $DistroName --user root --exec /bin/hostname -I | Out-String).Trim() -split '\s+' | Where-Object { $_ -match '^\d+\.\d+\.\d+\.\d+$' } | Select-Object -First 1)
if (-not $wslIpv4) { throw 'Could not determine the Agent computer private WSL address.' }
$blockedBefore = Get-WorkspaceFirewallBlockPackets
Invoke-WorkspaceCommand -Command "! curl -fsS --connect-timeout 2 --max-time 4 http://${wslIpv4}:9/" | Out-Null
$blockedAfter = Get-WorkspaceFirewallBlockPackets
if ($blockedAfter -le $blockedBefore) {
    throw 'Agent private-egress probe did not hit the mapped-UID nftables rule.'
}

$wslConf = & wsl.exe --distribution $DistroName --user root --exec /bin/cat /etc/wsl.conf
foreach ($required in @('enabled=false', 'appendWindowsPath=false', 'systemd=true', 'default=aicqws')) {
    if (($wslConf | Out-String) -notmatch [regex]::Escape($required)) { throw "wsl.conf is missing $required" }
}
$rootBlocks = [long](& wsl.exe --distribution $DistroName --user root --exec /bin/df --output=size -B1 / | Select-Object -Last 1).Trim()
$resourceConfigText = (& wsl.exe --distribution $DistroName --user root --exec /bin/cat /etc/aicq-workspace-config.json | Out-String).Trim()
$resourceConfig = $resourceConfigText | ConvertFrom-Json
$diskCeiling = ([long]$resourceConfig.disk_gib + 6) * 1GB
if ($rootBlocks -gt $diskCeiling) { throw 'Agent computer root filesystem exceeds its configured VHD ceiling.' }

if ($Full) {
    Invoke-WorkspaceCommand -Command "printf 'alpha\nbeta\n' | grep beta | tr a-z A-Z | grep -qx BETA" | Out-Null
    Invoke-WorkspaceCommand -Command "set -e; probe_dir=`$(mktemp -d); trap 'rm -rf `"`$probe_dir`"' EXIT; python -m venv `"`$probe_dir/venv`"; timeout --signal=TERM 180 `"`$probe_dir/venv/bin/python`" -m pip install --no-cache-dir --disable-pip-version-check --quiet --retries 5 --timeout 30 packaging; `"`$probe_dir/venv/bin/python`" -c 'import packaging'" | Out-Null
    $compileCommand = @'
set -e
probe_dir=$(mktemp -d)
trap 'rm -rf "$probe_dir"' EXIT
cat > "$probe_dir/hello.c" <<'EOF'
#include <stdio.h>
int main(void){puts("compiled");}
EOF
gcc "$probe_dir/hello.c" -o "$probe_dir/hello"
test "$("$probe_dir/hello")" = compiled
'@
    Invoke-WorkspaceCommand -Command $compileCommand | Out-Null
    $gitCommand = @'
set -e
probe_dir=$(mktemp -d)
trap 'rm -rf "$probe_dir"' EXIT
for attempt in 1 2 3; do
    rm -rf "$probe_dir/public-repo"
    timeout --signal=TERM 60 git clone --depth 1 https://github.com/octocat/Hello-World.git "$probe_dir/public-repo" && break
    test "$attempt" -lt 3
    sleep "$((attempt * 2))"
done
test -d "$probe_dir/public-repo/.git"
'@
    Invoke-WorkspaceCommand -Command $gitCommand | Out-Null
    $aptCommand = @'
set -e
if ! command -v tree >/dev/null 2>&1; then
    for attempt in 1 2 3; do
        sudo rm -f /var/cache/apt/archives/*.deb /var/cache/apt/archives/partial/*
        if timeout --signal=TERM 300 sudo apt-get -o Acquire::Retries=5 -o Acquire::ForceIPv4=true -o Acquire::http::Timeout=30 -o Acquire::http::No-Cache=true update -qq \
            && timeout --signal=TERM 300 sudo apt-get -o Acquire::Retries=5 -o Acquire::ForceIPv4=true -o Acquire::http::Timeout=30 -o Acquire::http::No-Cache=true install -y -qq tree; then
            break
        fi
        test "$attempt" -lt 3
        sudo dpkg --configure -a || true
        sleep "$((attempt * 2))"
    done
fi
tree --version >/dev/null
'@
    Invoke-WorkspaceCommand -Command $aptCommand | Out-Null
    Invoke-WorkspaceCommand -Command "test ! -e /mnt/c && ! command -v cmd.exe && test ! -S /run/podman/podman.sock && test ! -S /run/user/1000/podman/podman.sock && test ! -e /dev/dxg" | Out-Null
    $blockedBefore = Get-WorkspaceFirewallBlockPackets
    Invoke-WorkspaceCommand -Command "! curl -fsS --connect-timeout 2 --max-time 4 http://169.254.169.254/ && ! curl -fsS --connect-timeout 2 --max-time 4 http://192.168.0.1/" | Out-Null
    $blockedAfter = Get-WorkspaceFirewallBlockPackets
    if ($blockedAfter -le $blockedBefore) { throw 'Private-egress probes did not hit the enforced nftables rules.' }
}

Write-Host "Agent computer verification passed (full=$Full)."
