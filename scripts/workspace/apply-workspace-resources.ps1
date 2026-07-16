[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$InstallRoot,
    [ValidateRange(1, 32)][int]$Cpus = 4,
    [ValidateRange(2, 64)][int]$MemoryGiB = 8,
    [ValidateRange(32, 512)][int]$DiskGiB = 64
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$DistroName = 'AICQ-Workspace'
$ApplianceUser = 'aicqws'
$ProtocolVersion = 3
$BrokerVersion = '0.4.0'
$InstallLocation = Join-Path $InstallRoot $DistroName
$ManagedMarker = Join-Path $InstallLocation '.aicq-workspace-managed.json'
$VerifyScript = Join-Path $PSScriptRoot 'verify-workspace.ps1'

function Write-WorkspaceStage {
    param([Parameter(Mandatory)][string]$Name)
    Write-Host "[computer][stage] $Name"
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @(),
        [ValidateRange(1, 120)][int]$MaxAttempts = 1,
        [ValidateRange(0, 30)][int]$RetryDelaySeconds = 2
    )
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        & $FilePath @Arguments
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) { return }
        if ($attempt -lt $MaxAttempts) { Start-Sleep -Seconds $RetryDelaySeconds }
    }
    throw "$FilePath exited with code $exitCode after $MaxAttempts attempt(s)"
}

function Get-DistroNames {
    @(& wsl.exe --list --quiet) | ForEach-Object { ($_ -replace "`0", '').Trim() } | Where-Object { $_ }
}

function Get-RunningDistroNames {
    $names = @(& wsl.exe --list --running --quiet 2>$null)
    if ($LASTEXITCODE -ne 0) { throw 'Could not query running WSL distributions.' }
    @($names) | ForEach-Object { ($_ -replace "`0", '').Trim() } | Where-Object { $_ }
}

function Stop-DistroAndWait {
    if (@(Get-RunningDistroNames) -notcontains $DistroName) { return }
    Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--terminate', $DistroName)
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    while (@(Get-RunningDistroNames) -contains $DistroName) {
        if ([DateTime]::UtcNow -ge $deadline) { throw "Timed out waiting for $DistroName to stop." }
        Start-Sleep -Milliseconds 250
    }
}

function Stop-WslVmForVhdManagement {
    Stop-DistroAndWait
    $otherRunning = @(Get-RunningDistroNames | Where-Object { $_ -ine $DistroName })
    if ($otherRunning.Count -gt 0) {
        throw "Cannot resize the Agent computer while other WSL distributions are running: $($otherRunning -join ', ')."
    }
    Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--shutdown') -MaxAttempts 30 -RetryDelaySeconds 2
}

function Invoke-WslWithUtf8Stdin {
    param(
        [Parameter(Mandatory)][string]$Content,
        [Parameter(Mandatory)][string[]]$Arguments
    )
    $inputPath = Join-Path ([IO.Path]::GetTempPath()) ('aicq-computer-config-{0}.py' -f [Guid]::NewGuid().ToString('N'))
    try {
        $utf8 = New-Object Text.UTF8Encoding($false)
        [IO.File]::WriteAllText($inputPath, $Content + "`n", $utf8)
        $process = Start-Process -FilePath wsl.exe -ArgumentList $Arguments `
            -RedirectStandardInput $inputPath -NoNewWindow -Wait -PassThru
        if ($process.ExitCode -ne 0) { throw "WSL configuration update exited with code $($process.ExitCode)." }
    } finally {
        Remove-Item -LiteralPath $inputPath -Force -ErrorAction SilentlyContinue
    }
}

Write-WorkspaceStage -Name 'applying_resources'
if ($env:OS -ne 'Windows_NT') { throw 'Agent computer resource updates require Windows.' }
if (-not ((Get-DistroNames) -contains $DistroName)) { throw 'Agent computer is not built.' }
if (-not (Test-Path -LiteralPath $ManagedMarker -PathType Leaf)) {
    throw 'Managed Agent computer marker is missing; update the system before applying resources.'
}

$marker = Get-Content -LiteralPath $ManagedMarker -Raw | ConvertFrom-Json
if ([int]$marker.protocol_version -ne $ProtocolVersion) {
    throw 'Agent computer system must be updated before applying resources in place.'
}
$installedDiskGiB = [int]$marker.resources.disk_gib
if ($DiskGiB -lt $installedDiskGiB) { throw 'Agent computer disk shrinking requires a full uninstall and rebuild.' }
if ($DiskGiB -gt $installedDiskGiB) {
    Write-WorkspaceStage -Name 'expanding_disk'
    Stop-WslVmForVhdManagement
    Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--manage', $DistroName, '--resize', "${DiskGiB}GB") -MaxAttempts 60 -RetryDelaySeconds 2
}

$serviceEnvironment = @(
    'XDG_RUNTIME_DIR=/run/user/1000',
    'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus'
)
$stopBrokerArgs = @('--distribution', $DistroName, '--user', $ApplianceUser, '--exec', '/usr/bin/env')
$stopBrokerArgs += $serviceEnvironment
$stopBrokerArgs += @('/usr/bin/systemctl', '--user', 'stop', 'aicq-workspace-broker.service')
Invoke-NativeChecked -FilePath wsl.exe -Arguments $stopBrokerArgs

$updater = @'
import json
import os
import pathlib
import sys

cpus = int(sys.argv[1])
memory_gib = int(sys.argv[2])
disk_gib = int(sys.argv[3])

manifest_path = pathlib.Path('/opt/aicq-workspace/protocol-manifest.json')
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
manifest['limits']['cpus'] = cpus
manifest['limits']['memory_bytes'] = memory_gib * 1024 * 1024 * 1024
temporary = manifest_path.with_suffix('.json.new')
temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
os.replace(temporary, manifest_path)

config_path = pathlib.Path('/etc/aicq-workspace-config.json')
temporary = config_path.with_suffix('.json.new')
temporary.write_text(json.dumps({
    'cpus': cpus,
    'memory_gib': memory_gib,
    'disk_gib': disk_gib,
}, separators=(',', ':')) + '\n', encoding='utf-8')
os.replace(temporary, config_path)
os.chmod(config_path, 0o644)
'@
Invoke-WslWithUtf8Stdin -Content $updater -Arguments @(
    '--distribution', $DistroName, '--user', 'root', '--exec', '/usr/bin/python3', '-', $Cpus, $MemoryGiB, $DiskGiB
)

$containerSettingsArgs = @('--distribution', $DistroName, '--user', $ApplianceUser, '--exec', '/usr/bin/env')
$containerSettingsArgs += $serviceEnvironment
$containerSettingsArgs += '/opt/aicq-workspace/apply-container-settings.sh'
Invoke-NativeChecked -FilePath wsl.exe -Arguments $containerSettingsArgs
Invoke-NativeChecked -FilePath wsl.exe -Arguments @(
    '--distribution', $DistroName, '--user', 'root', '--exec', '/bin/systemctl', 'restart', 'aicq-workspace-firewall.service'
)
$brokerArgs = @('--distribution', $DistroName, '--user', $ApplianceUser, '--exec', '/usr/bin/env')
$brokerArgs += $serviceEnvironment
$brokerArgs += @('/usr/bin/systemctl', '--user', 'restart', 'aicq-workspace-broker.service')
Invoke-NativeChecked -FilePath wsl.exe -Arguments $brokerArgs

$marker.protocol_version = $ProtocolVersion
$marker.broker_version = $BrokerVersion
$marker.resources = @{ cpus = $Cpus; memory_gib = $MemoryGiB; disk_gib = $DiskGiB }
$marker | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ManagedMarker -Encoding utf8

& $VerifyScript
if ($LASTEXITCODE -ne 0) { throw 'Agent computer verification failed after applying resources.' }
Write-WorkspaceStage -Name 'completed'
Write-Host '[computer] Resources applied in place; the Agent computer container was not replaced.'
