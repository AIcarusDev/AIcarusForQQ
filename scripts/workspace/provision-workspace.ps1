[CmdletBinding()]
param(
    [string]$InstallRoot = '',
    [string]$ConfigPath = '',
    [switch]$Recreate,
    [switch]$SkipVerification
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$DistroName = 'AICQ-Workspace'
$Assets = Join-Path $PSScriptRoot 'appliance'
$VerifyScript = Join-Path $PSScriptRoot 'verify-workspace.ps1'
$InstallRootResolver = Join-Path $PSScriptRoot 'resolve_install_root.py'

function Resolve-ConfiguredInstallRoot {
    if (-not [string]::IsNullOrWhiteSpace($InstallRoot)) {
        return $InstallRoot
    }
    if (-not (Test-Path $InstallRootResolver -PathType Leaf)) {
        throw "Workspace install-root resolver not found: $InstallRootResolver"
    }

    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    $pythonArgs = @()
    if (-not $python) {
        $python = Get-Command py.exe -ErrorAction SilentlyContinue
        $pythonArgs = @('-3')
    }
    if (-not $python) {
        throw 'Python is required to read workspace.install_root; pass -InstallRoot explicitly instead.'
    }

    $resolverArgs = @($InstallRootResolver)
    if (-not [string]::IsNullOrWhiteSpace($ConfigPath)) {
        $resolverArgs += @('--config', $ConfigPath)
    }
    $resolved = (& $python.Source @pythonArgs @resolverArgs | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($resolved)) {
        throw 'Could not resolve workspace.provisioning.install_root from configuration.'
    }
    return $resolved
}

$InstallRoot = Resolve-ConfiguredInstallRoot
$InstallLocation = Join-Path $InstallRoot $DistroName

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments)][string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath exited with code $LASTEXITCODE"
    }
}

function Get-DistroNames {
    @(& wsl.exe --list --quiet) | ForEach-Object { ($_ -replace "`0", '').Trim() } | Where-Object { $_ }
}

function Assert-SafeInstallLocation {
    $root = [IO.Path]::GetFullPath($InstallRoot).TrimEnd('\')
    $target = [IO.Path]::GetFullPath($InstallLocation).TrimEnd('\')
    if (-not $target.StartsWith("$root\", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe WSL install location: $target"
    }
    if ([IO.Path]::GetFileName($target) -ne $DistroName) {
        throw "Unexpected WSL install directory: $target"
    }
}

Write-Host '[workspace] Preflight checks'
if ($env:OS -ne 'Windows_NT') { throw 'Workspace provisioning requires Windows.' }
foreach ($command in @('wsl.exe', 'tar.exe')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required command is unavailable: $command"
    }
}

$version = (& wsl.exe --version | Out-String) -replace "`0", ''
if ($version -notmatch '2\.\d+\.\d+') { throw 'WSL 2.x command capabilities were not detected.' }
$help = (& wsl.exe --help | Out-String) -replace "`0", ''
foreach ($flag in @('--location', '--name', '--vhd-size', '--manage', '--set-sparse')) {
    if (-not $help.Contains($flag)) { throw "This WSL build does not support $flag." }
}

Assert-SafeInstallLocation
$pathRoot = [IO.Path]::GetPathRoot($InstallLocation)
if ([string]::IsNullOrWhiteSpace($pathRoot) -or $pathRoot.StartsWith('\\')) {
    throw "Workspace install location must use a local Windows drive: $InstallLocation"
}
$drive = Get-PSDrive -Name ($pathRoot.Substring(0, 1))
if ($drive.Free -lt 20GB) { throw 'At least 20 GiB free space is required before provisioning.' }
if (-not (Test-Path $Assets -PathType Container)) { throw "Appliance assets not found: $Assets" }

foreach ($uri in @(
    'https://archive.ubuntu.com/ubuntu/',
    'https://hub.docker.com/v2/repositories/library/ubuntu/tags/24.04'
)) {
    try {
        $null = Invoke-WebRequest -Uri $uri -Method Get -TimeoutSec 20 -MaximumRedirection 3 -UseBasicParsing
    } catch {
        throw "Network preflight failed for $uri : $($_.Exception.Message)"
    }
}

$existing = Get-DistroNames
$UpgradeExisting = $false
if ($existing -contains $DistroName) {
    if ($Recreate) {
        Write-Host "[workspace] Removing old workspace distro $DistroName"
        & wsl.exe --terminate $DistroName 2>$null
        Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--unregister', $DistroName)
    } else {
        $UpgradeExisting = $true
        Write-Host "[workspace] Upgrading $DistroName in place; /workspace will be preserved"
    }
}

if ((Test-Path $InstallLocation) -and -not $UpgradeExisting) {
    if (-not $Recreate) { throw "$InstallLocation exists; pass -Recreate to remove it." }
    Assert-SafeInstallLocation
    Remove-Item -LiteralPath $InstallLocation -Recurse -Force
}
New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null

if (-not $UpgradeExisting) {
    Write-Host '[workspace] Installing a fresh Ubuntu 24.04 WSL2 appliance'
    Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--install', 'Ubuntu-24.04', '--name', $DistroName, '--location', $InstallLocation, '--version', '2', '--vhd-size', '64GB', '--no-launch')
    & wsl.exe --terminate $DistroName 2>$null
    Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--manage', $DistroName, '--set-sparse', 'true', '--allow-unsafe')
} else {
    Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'aicqws', '--exec', '/usr/bin/env', 'XDG_RUNTIME_DIR=/run/user/1000', 'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus', '/usr/bin/systemctl', '--user', 'stop', 'aicq-workspace-broker.service')
    & wsl.exe --distribution $DistroName --user aicqws --exec /usr/bin/env 'XDG_RUNTIME_DIR=/run/user/1000' 'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus' /usr/bin/podman rm -f aicq-workspace-default 2>$null
    & wsl.exe --distribution $DistroName --user aicqws --exec /usr/bin/env 'XDG_RUNTIME_DIR=/run/user/1000' 'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus' /usr/bin/podman image rm -f localhost/aicq-workspace-dev:1 localhost/aicq-workspace-dev:2 2>$null
    Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'root', '--exec', '/bin/bash', '-c', 'mkdir -p /var/lib/aicq-workspace/commands && find /var/lib/aicq-workspace/commands -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +')
}

Write-Host '[workspace] Starting first boot as root'
Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'root', '--exec', '/bin/true')
Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'root', '--exec', '/bin/mkdir', '-p', '/tmp/aicq-workspace-stage')

Write-Host '[workspace] Streaming appliance assets through tar/stdin'
& tar.exe -C $Assets -cf - . | & wsl.exe --distribution $DistroName --user root --exec /bin/tar -xf - -C /tmp/aicq-workspace-stage
if ($LASTEXITCODE -ne 0) { throw 'Asset streaming into WSL failed.' }
Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'root', '--exec', '/bin/bash', '/tmp/aicq-workspace-stage/bootstrap.sh')

Write-Host '[workspace] Restarting WSL so wsl.conf and systemd take effect'
Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--terminate', $DistroName)
Start-Sleep -Seconds 2
& wsl.exe --distribution $DistroName --user root --exec /bin/systemctl is-system-running --wait | Out-Host
Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'root', '--exec', '/bin/systemctl', 'restart', 'aicq-workspace-firewall.service')
Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'aicqws', '--exec', '/usr/bin/env', 'XDG_RUNTIME_DIR=/run/user/1000', 'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus', '/usr/bin/systemctl', '--user', 'restart', 'aicq-workspace-broker.service')
Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--manage', $DistroName, '--set-default-user', 'aicqws')

if (-not $SkipVerification) {
    Write-Host '[workspace] Running internal verification and creating the default container'
    & $VerifyScript
    if ($LASTEXITCODE -ne 0) { throw 'Workspace verification failed.' }
}

Write-Host '[workspace] Provisioning complete.'
