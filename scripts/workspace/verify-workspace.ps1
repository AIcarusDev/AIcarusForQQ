[CmdletBinding()]
param(
    [switch]$Full
)

$ErrorActionPreference = 'Stop'
$DistroName = 'AICQ-Workspace'
$Bridge = '/usr/local/bin/aicq-workspace-bridge'
$ProtocolVersion = 2

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
            throw "Workspace bridge failed for $Method (exit $($bridgeProcess.ExitCode)): $bridgeError"
        }
        $responseText = [IO.File]::ReadAllText($responsePath, $utf8)
    } finally {
        Remove-Item -LiteralPath $requestPath, $responsePath, $errorPath -Force -ErrorAction SilentlyContinue
    }
    $response = ($responseText | Out-String) | ConvertFrom-Json
    if ($response.version -ne $ProtocolVersion -or $response.request_id -ne $request.request_id) {
        throw "Workspace protocol mismatch for $Method."
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
        cwd = '/workspace'
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
        throw "Workspace command failed ($($result.exit_code)): $($result.content)"
    }
    return $result
}

function Get-WorkspaceFirewallBlockPackets {
    $rules = & wsl.exe --distribution $DistroName --user root --exec /usr/sbin/nft list chain inet aicq_workspace output
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect workspace firewall counters.' }
    $total = 0L
    foreach ($line in $rules) {
        if ($line -match 'counter packets ([0-9]+).*comment "aicq-block-private-v[46]"') {
            $total += [long]$Matches[1]
        }
    }
    return $total
}

$health = Invoke-WorkspaceRpc -Method health
if (-not $health.firewall_active) { throw 'Workspace firewall marker is not active.' }
if ($health.image_digest -ne 'sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90') {
    throw 'Installed workspace base-image digest does not match the repository manifest.'
}

$ensure = Invoke-WorkspaceRpc -Method ensure_default -Params @{ workspace_id = 'default' }
if ($ensure.container_name -ne 'aicq-workspace-default') { throw 'Unexpected workspace container name.' }
$probe = Invoke-WorkspaceCommand -Command 'test "$(id -u)" = 0 && test "$PWD" = /workspace && printf foundation-ok'
if ($probe.content.Trim() -ne 'foundation-ok') { throw 'Basic root/Bash probe failed.' }

$wslConf = & wsl.exe --distribution $DistroName --user root --exec /bin/cat /etc/wsl.conf
foreach ($required in @('enabled=false', 'appendWindowsPath=false', 'systemd=true', 'default=aicqws')) {
    if (($wslConf | Out-String) -notmatch [regex]::Escape($required)) { throw "wsl.conf is missing $required" }
}
$rootBlocks = [long](& wsl.exe --distribution $DistroName --user root --exec /bin/df --output=size -B1 / | Select-Object -Last 1).Trim()
$resourceConfigText = (& wsl.exe --distribution $DistroName --user root --exec /bin/cat /etc/aicq-workspace-config.json | Out-String).Trim()
$resourceConfig = $resourceConfigText | ConvertFrom-Json
$diskCeiling = ([long]$resourceConfig.disk_gib + 6) * 1GB
if ($rootBlocks -gt $diskCeiling) { throw 'Workspace root filesystem exceeds its configured VHD ceiling.' }

if ($Full) {
    Invoke-WorkspaceCommand -Command "printf 'alpha\nbeta\n' | grep beta | tr a-z A-Z | grep -qx BETA" | Out-Null
    Invoke-WorkspaceCommand -Command "timeout --signal=TERM 180 python -m pip install --disable-pip-version-check --quiet --retries 5 --timeout 30 packaging && python -c 'import packaging'" | Out-Null
    $compileCommand = @'
cat > hello.c <<'EOF'
#include <stdio.h>
int main(void){puts("compiled");}
EOF
gcc hello.c -o hello && test "$(./hello)" = compiled
'@
    Invoke-WorkspaceCommand -Command $compileCommand | Out-Null
    $gitCommand = @'
set -e
for attempt in 1 2 3; do
    rm -rf public-repo
    timeout --signal=TERM 60 git clone --depth 1 https://github.com/octocat/Hello-World.git public-repo && break
    test "$attempt" -lt 3
    sleep "$((attempt * 2))"
done
test -d public-repo/.git
'@
    Invoke-WorkspaceCommand -Command $gitCommand | Out-Null
    $aptCommand = @'
set -e
if ! command -v tree >/dev/null 2>&1; then
    for attempt in 1 2 3; do
        rm -f /var/cache/apt/archives/*.deb /var/cache/apt/archives/partial/*
        if timeout --signal=TERM 300 apt-get -o Acquire::Retries=5 -o Acquire::ForceIPv4=true -o Acquire::http::Timeout=30 -o Acquire::http::No-Cache=true update -qq \
            && timeout --signal=TERM 300 apt-get -o Acquire::Retries=5 -o Acquire::ForceIPv4=true -o Acquire::http::Timeout=30 -o Acquire::http::No-Cache=true install -y -qq tree; then
            break
        fi
        test "$attempt" -lt 3
        dpkg --configure -a || true
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
    $published = & wsl.exe --distribution $DistroName --user aicqws --exec /usr/bin/env XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus /usr/bin/podman port aicq-workspace-default
    if ($LASTEXITCODE -ne 0 -or ($published | Out-String).Trim()) { throw 'Workspace container unexpectedly publishes an inbound port.' }
}

Write-Host "Workspace verification passed (full=$Full)."
