[CmdletBinding()]
param(
    [switch]$Full
)

$ErrorActionPreference = 'Stop'
$DistroName = 'AICQ-Workspace'
$Bridge = '/usr/local/bin/aicq-workspace-bridge'
$ProtocolVersion = 3

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
$published = @(& wsl.exe --distribution $DistroName --user aicqws --exec /usr/bin/env XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus /usr/bin/podman port aicq-workspace-default)
$publishedExitCode = $LASTEXITCODE
$published = @($published | ForEach-Object { "$($_)".Trim() } | Where-Object { $_ })
if ($publishedExitCode -ne 0 -or $published.Count -ne 1) {
    throw 'Agent computer container must publish exactly one inbound port.'
}
if ($published[0] -notmatch '^6080/tcp -> 127\.0\.0\.1:([0-9]{1,5})$') {
    throw 'Agent computer preview must publish container port 6080 on WSL loopback only.'
}
$previewHostPort = [int]$Matches[1]
if ($previewHostPort -lt 1 -or $previewHostPort -gt 65535) {
    throw 'Agent computer preview host port is outside the valid TCP port range.'
}
$firewallRules = (& wsl.exe --distribution $DistroName --user root --exec /usr/sbin/nft list chain inet aicq_workspace output | Out-String)
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect computer preview firewall rules.' }
$previewPortPattern = [regex]::Escape([string]$previewHostPort)
if ($firewallRules -notmatch "ip daddr 127\.0\.0\.1 tcp dport $previewPortPattern .*comment `"aicq-preview-loopback`"") {
    throw 'Agent computer firewall does not allow the fixed loopback preview listener.'
}
if ($firewallRules -notmatch "ip saddr 127\.0\.0\.1 tcp sport $previewPortPattern ct state established .*comment `"aicq-preview-loopback-return`"") {
    throw 'Agent computer firewall does not allow established preview return traffic.'
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
