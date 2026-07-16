[CmdletBinding()]
param(
    [string]$InstallRoot = '',
    [string]$ConfigPath = '',
    [ValidateRange(1, 32)][int]$Cpus = 4,
    [ValidateRange(2, 64)][int]$MemoryGiB = 8,
    [ValidateRange(32, 512)][int]$DiskGiB = 64,
    [switch]$Recreate,
    [switch]$Resume,
    [switch]$RebuildSystem,
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
        throw "Agent computer install-root resolver not found: $InstallRootResolver"
    }

    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    $pythonArgs = @()
    if (-not $python) {
        $python = Get-Command py.exe -ErrorAction SilentlyContinue
        $pythonArgs = @('-3')
    }
    if (-not $python) {
        throw 'Python is required to read the Agent computer install_root; pass -InstallRoot explicitly instead.'
    }

    $resolverArgs = @($InstallRootResolver)
    if (-not [string]::IsNullOrWhiteSpace($ConfigPath)) {
        $resolverArgs += @('--config', $ConfigPath)
    }
    $resolved = (& $python.Source @pythonArgs @resolverArgs | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($resolved)) {
        throw 'Could not resolve the Agent computer install_root from configuration.'
    }
    return $resolved
}

$InstallRoot = Resolve-ConfiguredInstallRoot
$InstallLocation = Join-Path $InstallRoot $DistroName
$ManagedMarker = Join-Path $InstallLocation '.aicq-workspace-managed.json'
$ProvisioningMarker = Join-Path $InstallRoot '.aicq-workspace-provisioning.json'

function Write-WorkspaceStage {
    param([Parameter(Mandatory)][string]$Name)
    Write-Host "[computer][stage] $Name"
}

function Format-NativeCommand {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @()
    )
    $displayArguments = @($Arguments | ForEach-Object {
        if ($_ -match '[\s"]') { '"{0}"' -f ($_ -replace '"', '\"') } else { $_ }
    })
    return (@($FilePath) + $displayArguments) -join ' '
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @(),
        [ValidateRange(1, 120)][int]$MaxAttempts = 1,
        [ValidateRange(0, 30)][int]$RetryDelaySeconds = 2
    )

    $command = Format-NativeCommand -FilePath $FilePath -Arguments $Arguments
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Write-Host "[computer][command] $command"
        & $FilePath @Arguments
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) {
            return
        }
        if ($attempt -lt $MaxAttempts) {
            Write-Host "[computer][retry] Command exited with code $exitCode; waiting ${RetryDelaySeconds}s before attempt $($attempt + 1)/$MaxAttempts"
            Start-Sleep -Seconds $RetryDelaySeconds
        }
    }
    throw "$FilePath exited with code $exitCode after $MaxAttempts attempt(s)"
}

function Get-DistroNames {
    @(& wsl.exe --list --quiet) | ForEach-Object { ($_ -replace "`0", '').Trim() } | Where-Object { $_ }
}

function Get-RunningDistroNames {
    $names = @(& wsl.exe --list --running --quiet 2>$null)
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not query running WSL distributions.'
    }
    @($names) | ForEach-Object { ($_ -replace "`0", '').Trim() } | Where-Object { $_ }
}

function Stop-DistroAndWait {
    param(
        [Parameter(Mandatory)][string]$Name,
        [ValidateRange(1, 120)][int]$TimeoutSeconds = 30
    )

    if (@(Get-RunningDistroNames) -notcontains $Name) {
        return
    }

    Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--terminate', $Name)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while (@(Get-RunningDistroNames) -contains $Name) {
        if ([DateTime]::UtcNow -ge $deadline) {
            throw "Timed out waiting for WSL distribution $Name to stop."
        }
        Start-Sleep -Milliseconds 250
    }
}

function Stop-WslVmForVhdManagement {
    # WSL can report a distro as Stopped while the shared VM still owns its
    # ext4.vhdx.  --manage then fails with ERROR_SHARING_VIOLATION and tells
    # the caller to use --shutdown.  A global shutdown is safe here only when
    # no unrelated distro is running; never interrupt somebody else's WSL
    # session just to apply workspace storage settings.
    Stop-DistroAndWait -Name $DistroName
    $otherRunning = @(Get-RunningDistroNames | Where-Object { $_ -ine $DistroName })
    if ($otherRunning.Count -gt 0) {
        throw "Cannot configure the Agent computer VHD while other WSL distributions are running: $($otherRunning -join ', '). Stop them and retry."
    }

    Write-Host '[computer] Releasing the stopped Agent computer VHD from the WSL shared VM'
    Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--shutdown') -MaxAttempts 30 -RetryDelaySeconds 2
}

function Get-RegisteredDistroLocation {
    $lxss = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss'
    foreach ($key in @(Get-ChildItem -LiteralPath $lxss -ErrorAction SilentlyContinue)) {
        $item = Get-ItemProperty -LiteralPath $key.PSPath -ErrorAction SilentlyContinue
        if ([string]$item.DistributionName -ieq $DistroName) {
            return [Environment]::ExpandEnvironmentVariables([string]$item.BasePath)
        }
    }
    return ''
}

function Test-PathsEqual {
    param(
        [Parameter(Mandatory)][string]$Left,
        [Parameter(Mandatory)][string]$Right
    )
    $normalizedLeft = [IO.Path]::GetFullPath(($Left -replace '^\\\\\?\\', '')).TrimEnd('\')
    $normalizedRight = [IO.Path]::GetFullPath(($Right -replace '^\\\\\?\\', '')).TrimEnd('\')
    return $normalizedLeft.Equals($normalizedRight, [StringComparison]::OrdinalIgnoreCase)
}

function Test-ProvisioningMarker {
    if (-not (Test-Path -LiteralPath $ProvisioningMarker -PathType Leaf)) {
        return $false
    }
    try {
        $marker = Get-Content -LiteralPath $ProvisioningMarker -Raw | ConvertFrom-Json
        return (
            [string]$marker.distro_name -eq $DistroName -and
            (Test-PathsEqual -Left ([string]$marker.install_location) -Right $InstallLocation)
        )
    } catch {
        return $false
    }
}

function Test-PristinePartialDistro {
    $probe = '[ ! -e /opt/aicq-workspace ] && [ ! -e /var/lib/aicq-workspace ] && ! id aicqws >/dev/null 2>&1'
    $previousPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 promotes successful native stderr (including
        # WSL's localhost proxy warning) to NativeCommandError when the script
        # preference is Stop. This probe only consumes the native exit code.
        $ErrorActionPreference = 'Continue'
        & wsl.exe --distribution $DistroName --user root --exec /bin/sh -c $probe 2>$null
        $probeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    return $probeExitCode -eq 0
}

function Assert-SafeRepairableDistro {
    $registeredLocation = Get-RegisteredDistroLocation
    if ([string]::IsNullOrWhiteSpace($registeredLocation)) {
        throw "Could not determine the registered location for $DistroName; refusing automatic cleanup."
    }
    if (-not (Test-PathsEqual -Left $registeredLocation -Right $InstallLocation)) {
        throw "Registered distro location does not match the configured Agent computer path; refusing automatic cleanup."
    }
    if ((Test-ProvisioningMarker) -or (Test-PristinePartialDistro)) {
        return
    }
    throw 'The existing distro is not an owned or pristine partial Agent computer install; refusing automatic cleanup.'
}

function Remove-InstallDirectoryWithRetry {
    if (-not (Test-Path -LiteralPath $InstallLocation)) {
        return
    }
    Assert-SafeInstallLocation
    for ($attempt = 1; $attempt -le 60; $attempt++) {
        try {
            Remove-Item -LiteralPath $InstallLocation -Recurse -Force -ErrorAction Stop
            return
        } catch {
            if ($attempt -eq 60) {
                throw
            }
            Write-Host "[computer][retry] WSL has not released the old install directory; waiting 2s before attempt $($attempt + 1)/60"
            Start-Sleep -Seconds 2
        }
    }
}

function Install-FreshDistro {
    $arguments = @('--install', 'Ubuntu-24.04', '--name', $DistroName, '--location', $InstallLocation, '--version', '2', '--vhd-size', "${DiskGiB}GB", '--no-launch')
    $command = Format-NativeCommand -FilePath 'wsl.exe' -Arguments $arguments
    Write-Host "[computer][command] $command"
    & wsl.exe @arguments
    $installExitCode = $LASTEXITCODE
    if ($installExitCode -eq 0) {
        return
    }

    # Some WSL releases can finish registration but report a transient VHD
    # sharing violation while their background installer still owns the disk.
    # Never invoke --install twice. Trust the partial success only after the
    # registered path matches and the new base distro becomes launchable.
    Write-Host "[computer][retry] WSL install exited with code $installExitCode; checking for a safely registered partial success"
    for ($attempt = 1; $attempt -le 60; $attempt++) {
        $registeredLocation = Get-RegisteredDistroLocation
        if (
            -not [string]::IsNullOrWhiteSpace($registeredLocation) -and
            (Test-PathsEqual -Left $registeredLocation -Right $InstallLocation)
        ) {
            $previousPreference = $ErrorActionPreference
            try {
                $ErrorActionPreference = 'Continue'
                & wsl.exe --distribution $DistroName --user root --exec /bin/true 2>$null
                $launchExitCode = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $previousPreference
            }
            if ($launchExitCode -eq 0) {
                Write-Host '[computer] WSL registration is healthy after the transient installer error; continuing the same build'
                return
            }
        }
        if ($attempt -lt 60) {
            Start-Sleep -Seconds 2
        }
    }
    throw "wsl.exe install exited with code $installExitCode and did not become safely launchable"
}

function Set-ProvisioningMarker {
    param([Parameter(Mandatory)][string]$Phase)
    @{
        distro_name = $DistroName
        install_location = [IO.Path]::GetFullPath($InstallLocation)
        phase = $Phase
        updated_at = [DateTime]::UtcNow.ToString('o')
    } | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $ProvisioningMarker -Encoding utf8
}

function ConvertTo-ValidDiskGiB {
    param([object]$Value)

    [long]$parsed = 0
    if ([long]::TryParse([string]$Value, [ref]$parsed) -and $parsed -ge 32 -and $parsed -le 512) {
        return [int]$parsed
    }
    return $null
}

function Get-InstalledDiskGiB {
    if (Test-Path -LiteralPath $ManagedMarker -PathType Leaf) {
        try {
            $marker = Get-Content -LiteralPath $ManagedMarker -Raw | ConvertFrom-Json
            $markerDiskGiB = ConvertTo-ValidDiskGiB $marker.resources.disk_gib
            if ($null -ne $markerDiskGiB) {
                return $markerDiskGiB
            }
        } catch {
            Write-Host '[computer] Ignoring an unreadable managed ownership marker while detecting disk configuration'
        }
    }

    # Missing resource config is expected for protocol v1. Keep the Linux probe
    # successful so Windows PowerShell 5.1 does not promote cat's stderr to a
    # terminating NativeCommandError under ErrorActionPreference=Stop.
    $readResourceConfig = 'if [ -f /etc/aicq-workspace-config.json ]; then /bin/cat /etc/aicq-workspace-config.json; fi'
    $resourceConfigText = (& wsl.exe --distribution $DistroName --user root --exec /bin/sh -c $readResourceConfig | Out-String).Trim()
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($resourceConfigText)) {
        try {
            $resourceConfig = $resourceConfigText | ConvertFrom-Json
            $resourceDiskGiB = ConvertTo-ValidDiskGiB $resourceConfig.disk_gib
            if ($null -ne $resourceDiskGiB) {
                return $resourceDiskGiB
            }
        } catch {
            Write-Host '[computer] Ignoring an unreadable installed resource configuration while detecting disk configuration'
        }
    }

    # Protocol v1 did not persist a resource config or ownership marker and was
    # always provisioned with the historical 64 GiB default.
    Write-Host '[computer] Legacy appliance has no disk record; assuming the v1 default of 64GiB'
    return 64
}

function Copy-ApplianceAssetsToDistro {
    param(
        [Parameter(Mandatory)][string]$Source,
        [Parameter(Mandatory)][string]$Destination
    )

    # Windows PowerShell 5.1 pipelines decode and re-encode native stdout, which
    # corrupts tar's binary stream. Materialize the small appliance archive and
    # let Windows redirect the file handle directly into WSL stdin instead.
    $archiveName = 'aicq-workspace-assets-{0}.tar' -f [Guid]::NewGuid().ToString('N')
    $archivePath = Join-Path ([IO.Path]::GetTempPath()) $archiveName
    try {
        Invoke-NativeChecked -FilePath tar.exe -Arguments @('-C', $Source, '-cf', $archivePath, '.')
        $extract = Start-Process -FilePath wsl.exe -ArgumentList @(
            '--distribution', $DistroName,
            '--user', 'root',
            '--exec', '/bin/tar', '-xf', '-', '-C', $Destination
        ) -RedirectStandardInput $archivePath -NoNewWindow -Wait -PassThru
        if ($extract.ExitCode -ne 0) {
            throw "Asset extraction in WSL exited with code $($extract.ExitCode)."
        }
    } finally {
        Remove-Item -LiteralPath $archivePath -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-WslWithUtf8Stdin {
    param(
        [Parameter(Mandatory)][string]$Content,
        [Parameter(Mandatory)][string[]]$Arguments
    )

    $inputName = 'aicq-workspace-stdin-{0}.txt' -f [Guid]::NewGuid().ToString('N')
    $inputPath = Join-Path ([IO.Path]::GetTempPath()) $inputName
    try {
        $utf8 = New-Object Text.UTF8Encoding($false)
        [IO.File]::WriteAllText($inputPath, $Content + "`n", $utf8)
        $process = Start-Process -FilePath wsl.exe -ArgumentList $Arguments `
            -RedirectStandardInput $inputPath -NoNewWindow -Wait -PassThru
        if ($process.ExitCode -ne 0) {
            throw "WSL stdin operation exited with code $($process.ExitCode)."
        }
    } finally {
        Remove-Item -LiteralPath $inputPath -Force -ErrorAction SilentlyContinue
    }
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

Write-WorkspaceStage -Name 'preflight'
Write-Host '[computer] Preflight checks'
if ($env:OS -ne 'Windows_NT') { throw 'Agent computer provisioning requires Windows.' }
if (($Recreate -and $Resume) -or ($RebuildSystem -and ($Recreate -or $Resume))) {
    throw 'Recreate, Resume, and RebuildSystem are mutually exclusive.'
}
foreach ($command in @('wsl.exe', 'tar.exe')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required command is unavailable: $command"
    }
}

$version = (& wsl.exe --version 2>&1 | Out-String) -replace "`0", ''
if ($version -notmatch '2\.\d+\.\d+') {
    Write-Host '[computer] WSL 2 is not ready; requesting Windows to install the prerequisite.'
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    $isAdministrator = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if ($isAdministrator) {
        & wsl.exe --install --no-distribution
        $installCode = $LASTEXITCODE
    } else {
        Write-Host '[computer] Waiting for the user to approve the Windows UAC prompt.'
        $elevated = Start-Process -FilePath wsl.exe -Verb RunAs -ArgumentList @('--install', '--no-distribution') -Wait -PassThru
        $installCode = $elevated.ExitCode
    }
    if ($installCode -ne 0 -and $installCode -ne 3010) {
        throw "WSL prerequisite installation failed with code $installCode"
    }
    Write-Host '[computer] Windows must restart before Agent computer provisioning can continue.'
    exit 3010
}
$help = (& wsl.exe --help | Out-String) -replace "`0", ''
foreach ($flag in @('--location', '--name', '--vhd-size', '--manage', '--set-sparse')) {
    if (-not $help.Contains($flag)) { throw "This WSL build does not support $flag." }
}

Assert-SafeInstallLocation
$pathRoot = [IO.Path]::GetPathRoot($InstallLocation)
if ([string]::IsNullOrWhiteSpace($pathRoot) -or $pathRoot.StartsWith('\\')) {
    throw "Agent computer install location must use a local Windows drive: $InstallLocation"
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
$FreshCleanupAuthorized = $false
if ($existing -contains $DistroName) {
    if ($Recreate) {
        Write-WorkspaceStage -Name 'recovering_partial_install'
        Assert-SafeRepairableDistro
        $FreshCleanupAuthorized = $true
        Write-Host "[computer] Removing the safely identified partial Agent computer distro $DistroName"
        Stop-DistroAndWait -Name $DistroName
        Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--unregister', $DistroName)
    } else {
        $UpgradeExisting = $true
        if ($Resume) {
            Assert-SafeRepairableDistro
            Write-Host "[computer] Resuming the owned partial build of $DistroName in place; completed appliance work will be reused"
        } else {
            Write-Host "[computer] Replacing the managed system container; /home/agent will be preserved and legacy Agent files will be migrated"
        }

        $installedDiskGiB = Get-InstalledDiskGiB
        if ($DiskGiB -lt $installedDiskGiB) {
            throw 'Agent computer disk shrinking is not supported; uninstall and rebuild instead.'
        }
        if ($DiskGiB -gt $installedDiskGiB) {
            Write-WorkspaceStage -Name 'expanding_disk'
            Write-Host "[computer] Expanding sparse VHD to ${DiskGiB}GB"
            Stop-DistroAndWait -Name $DistroName
            Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--manage', $DistroName, '--resize', "${DiskGiB}GB") -MaxAttempts 60 -RetryDelaySeconds 2
        }
    }
}
if ($Resume -and -not ($existing -contains $DistroName)) {
    throw "Cannot resume because $DistroName is not registered."
}

if ((Test-Path $InstallLocation) -and -not $UpgradeExisting) {
    if (-not $Recreate) { throw "$InstallLocation exists; pass -Recreate to remove it." }
    Assert-SafeInstallLocation
    if (-not $FreshCleanupAuthorized -and -not (Test-ProvisioningMarker)) {
        throw 'The Agent computer install directory exists without a matching provisioning marker; refusing automatic deletion.'
    }
    Remove-InstallDirectoryWithRetry
}
New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null

if (-not $UpgradeExisting) {
    Set-ProvisioningMarker -Phase 'installing_distro'
    Write-WorkspaceStage -Name 'installing_distro'
    Write-Host '[computer] Installing a fresh Ubuntu 24.04 WSL2 appliance'
    Install-FreshDistro
} else {
    Write-WorkspaceStage -Name 'preparing_upgrade'
    Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'aicqws', '--exec', '/usr/bin/env', 'XDG_RUNTIME_DIR=/run/user/1000', 'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus', '/usr/bin/systemctl', '--user', 'stop', 'aicq-workspace-broker.service')
    Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'root', '--exec', '/bin/bash', '-c', 'mkdir -p /var/lib/aicq-workspace/commands && find /var/lib/aicq-workspace/commands -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +')
}

if (-not $UpgradeExisting) { Set-ProvisioningMarker -Phase 'configuring_appliance' }
Write-WorkspaceStage -Name 'configuring_appliance'
Write-Host '[computer] Starting first boot as root'
Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'root', '--exec', '/bin/true')
Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'root', '--exec', '/bin/mkdir', '-p', '/tmp/aicq-workspace-stage')

Write-Host '[computer] Transferring appliance assets through binary-safe WSL stdin'
Copy-ApplianceAssetsToDistro -Source $Assets -Destination '/tmp/aicq-workspace-stage'
$manifestUpdater = @'
import json
import pathlib
import sys

path = pathlib.Path('/tmp/aicq-workspace-stage/opt/aicq-workspace/protocol-manifest.json')
manifest = json.loads(path.read_text(encoding='utf-8'))
manifest['limits']['cpus'] = int(sys.argv[1])
manifest['limits']['memory_bytes'] = int(sys.argv[2]) * 1024 * 1024 * 1024
path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
'@
Invoke-WslWithUtf8Stdin -Content $manifestUpdater -Arguments @(
    '--distribution', $DistroName,
    '--user', 'root',
    '--exec', '/usr/bin/python3', '-', $Cpus, $MemoryGiB
)
Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'root', '--exec', '/bin/bash', '/tmp/aicq-workspace-stage/bootstrap.sh')

$resourceConfig = @{
    cpus = $Cpus
    memory_gib = $MemoryGiB
    disk_gib = $DiskGiB
} | ConvertTo-Json -Compress
Invoke-WslWithUtf8Stdin -Content $resourceConfig -Arguments @(
    '--distribution', $DistroName,
    '--user', 'root',
    '--exec', '/bin/dd', 'of=/etc/aicq-workspace-config.json', 'status=none'
)
Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'root', '--exec', '/bin/chmod', '0644', '/etc/aicq-workspace-config.json')

Write-Host '[computer] Restarting WSL so wsl.conf and systemd take effect'
Write-WorkspaceStage -Name 'restarting_distro'
Stop-DistroAndWait -Name $DistroName
& wsl.exe --distribution $DistroName --user root --exec /bin/systemctl is-system-running --wait | Out-Host
Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'root', '--exec', '/bin/systemctl', 'restart', 'aicq-workspace-firewall.service')
Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--distribution', $DistroName, '--user', 'aicqws', '--exec', '/usr/bin/env', 'XDG_RUNTIME_DIR=/run/user/1000', 'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus', '/usr/bin/systemctl', '--user', 'restart', 'aicq-workspace-broker.service')
Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--manage', $DistroName, '--set-default-user', 'aicqws')

Write-Host '[computer] Building and creating the default container through the provisioning-only entry point'
if ((-not $UpgradeExisting) -or $Resume) { Set-ProvisioningMarker -Phase 'building_container' }
Write-WorkspaceStage -Name 'building_container'
$containerProvisionEnvironment = @(
    'XDG_RUNTIME_DIR=/run/user/1000',
    'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus'
)
if ($Resume) {
    $containerProvisionEnvironment += 'AICQ_WORKSPACE_REUSE_VALID_IMAGE=1'
}
if ($RebuildSystem) {
    $containerProvisionEnvironment += 'AICQ_WORKSPACE_REBUILD_IMAGE=1'
}
$containerProvisionArguments = @('--distribution', $DistroName, '--user', 'aicqws', '--exec', '/usr/bin/env')
$containerProvisionArguments += $containerProvisionEnvironment
$containerProvisionArguments += '/opt/aicq-workspace/provision-container.sh'
Invoke-NativeChecked -FilePath wsl.exe -Arguments $containerProvisionArguments
Invoke-NativeChecked -FilePath wsl.exe -Arguments @(
    '--distribution', $DistroName,
    '--user', 'root',
    '--exec', '/bin/systemctl', 'restart', 'aicq-workspace-firewall.service'
)

# A new distro can remain internally held by WSL for a short period after
# installation. Sparse conversion is deliberately deferred until all initial
# setup is complete, then retried while only the dedicated distro is stopped.
if ((-not $UpgradeExisting) -or $Resume) { Set-ProvisioningMarker -Phase 'configuring_sparse_vhd' }
Write-WorkspaceStage -Name 'configuring_sparse_vhd'
Write-Host '[computer] Stopping the Agent computer before enabling sparse VHD mode'
Stop-WslVmForVhdManagement
Invoke-NativeChecked -FilePath wsl.exe -Arguments @('--manage', $DistroName, '--set-sparse', 'true', '--allow-unsafe') -MaxAttempts 60 -RetryDelaySeconds 2

if (-not $SkipVerification) {
    Write-WorkspaceStage -Name 'verifying'
    Write-Host '[computer] Running internal verification'
    & $VerifyScript
    if ($LASTEXITCODE -ne 0) { throw 'Agent computer verification failed.' }
}

@{
    distro_name = $DistroName
    protocol_version = 4
    broker_version = '0.5.0'
    install_location = [IO.Path]::GetFullPath($InstallLocation)
    resources = @{ cpus = $Cpus; memory_gib = $MemoryGiB; disk_gib = $DiskGiB }
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ManagedMarker -Encoding utf8
Remove-Item -LiteralPath $ProvisioningMarker -Force -ErrorAction SilentlyContinue

Write-WorkspaceStage -Name 'completed'
Write-Host '[computer] Provisioning complete.'
