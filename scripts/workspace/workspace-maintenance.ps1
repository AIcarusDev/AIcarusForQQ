[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('Restart', 'Clear', 'Uninstall')]
    [string]$Action,
    [Parameter(Mandatory)]
    [string]$InstallRoot
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$DistroName = 'AICQ-Workspace'
$InstallLocation = Join-Path $InstallRoot $DistroName
$ManagedMarker = Join-Path $InstallLocation '.aicq-workspace-managed.json'
$VerifyScript = Join-Path $PSScriptRoot 'verify-workspace.ps1'

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments)][string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$FilePath exited with code $LASTEXITCODE" }
}

function Get-DistroNames {
    @(& wsl.exe --list --quiet) | ForEach-Object { ($_ -replace "`0", '').Trim() } | Where-Object { $_ }
}

if (-not ((Get-DistroNames) -contains $DistroName)) {
    throw 'Workspace is not built.'
}

if ($Action -eq 'Restart') {
    Write-Host '[workspace] Terminating the workspace distro'
    & wsl.exe --terminate $DistroName 2>$null
    Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'root', '--exec', '/bin/true')
    & $VerifyScript
    if ($LASTEXITCODE -ne 0) { throw 'Workspace health verification failed after restart.' }
    Write-Host '[workspace] Workspace restarted and verified.'
    exit 0
}

if ($Action -eq 'Clear') {
    Write-Host '[workspace] Stopping the existing container before clearing /workspace'
    & wsl.exe --distribution $DistroName --user aicqws --exec /usr/bin/env 'XDG_RUNTIME_DIR=/run/user/1000' 'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus' /usr/bin/podman stop --ignore aicq-workspace-default 2>$null
    Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'root', '--exec', '/bin/bash', '-c', 'find /var/lib/aicq-workspace/workspace -mindepth 1 -delete')
    & $VerifyScript
    if ($LASTEXITCODE -ne 0) { throw 'Workspace health verification failed after clearing data.' }
    Write-Host '[workspace] /workspace data cleared.'
    exit 0
}

$root = [IO.Path]::GetFullPath($InstallRoot).TrimEnd('\')
$target = [IO.Path]::GetFullPath($InstallLocation).TrimEnd('\')
if (-not $target.StartsWith("$root\", [StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe workspace install location: $target"
}
if ([IO.Path]::GetFileName($target) -ne $DistroName) {
    throw "Unexpected workspace install directory: $target"
}
if (-not (Test-Path -LiteralPath $ManagedMarker -PathType Leaf)) {
    throw 'Managed workspace ownership marker is missing; refusing to delete the install directory.'
}
$marker = Get-Content -LiteralPath $ManagedMarker -Raw | ConvertFrom-Json
if ($marker.distro_name -ne $DistroName -or [IO.Path]::GetFullPath([string]$marker.install_location).TrimEnd('\') -ne $target) {
    throw 'Managed workspace ownership marker does not match the configured install directory.'
}

Write-Host '[workspace] Unregistering the dedicated WSL distro'
& wsl.exe --terminate $DistroName 2>$null
Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--unregister', $DistroName)
if (Test-Path -LiteralPath $target) {
    Remove-Item -LiteralPath $target -Recurse -Force
}
Write-Host '[workspace] Workspace fully uninstalled; parent directory and configuration were preserved.'
