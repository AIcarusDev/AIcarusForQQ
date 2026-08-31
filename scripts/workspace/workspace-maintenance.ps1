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
    throw 'Agent computer is not built.'
}

if ($Action -eq 'Restart') {
    Write-Host '[computer] Terminating the Agent computer distro'
    & wsl.exe --terminate $DistroName 2>$null
    Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'root', '--exec', '/bin/true')
    & $VerifyScript
    if ($LASTEXITCODE -ne 0) { throw 'Agent computer verification failed after restart.' }
    Write-Host '[computer] Agent computer restarted and verified.'
    exit 0
}

if ($Action -eq 'Clear') {
    Write-Host '[computer] Stopping the existing container before erasing /home/agent'
    & wsl.exe --distribution $DistroName --user aicqws --exec /usr/bin/env 'XDG_RUNTIME_DIR=/run/aicq-workspace/user' /usr/bin/podman stop --ignore aicq-workspace-default 2>$null
    Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'root', '--exec', '/bin/bash', '-c', 'find /var/lib/aicq-workspace/home -mindepth 1 -delete')
    & $VerifyScript
    if ($LASTEXITCODE -ne 0) { throw 'Agent computer verification failed after erasing the Agent home.' }
    Write-Host '[computer] /home/agent was erased and its standard user files were recreated.'
    exit 0
}

$root = [IO.Path]::GetFullPath($InstallRoot).TrimEnd('\')
$target = [IO.Path]::GetFullPath($InstallLocation).TrimEnd('\')
if (-not $target.StartsWith("$root\", [StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe Agent computer install location: $target"
}
if ([IO.Path]::GetFileName($target) -ne $DistroName) {
    throw "Unexpected Agent computer install directory: $target"
}
if (-not (Test-Path -LiteralPath $ManagedMarker -PathType Leaf)) {
    throw 'Managed Agent computer ownership marker is missing; refusing to delete the install directory.'
}
$marker = Get-Content -LiteralPath $ManagedMarker -Raw | ConvertFrom-Json
if ($marker.distro_name -ne $DistroName -or [IO.Path]::GetFullPath([string]$marker.install_location).TrimEnd('\') -ne $target) {
    throw 'Managed Agent computer ownership marker does not match the configured install directory.'
}

Write-Host '[computer] Unregistering the dedicated WSL distro'
& wsl.exe --terminate $DistroName 2>$null
Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--unregister', $DistroName)
if (Test-Path -LiteralPath $target) {
    Remove-Item -LiteralPath $target -Recurse -Force
}
Write-Host '[computer] Agent computer fully uninstalled; parent directory and configuration were preserved.'
