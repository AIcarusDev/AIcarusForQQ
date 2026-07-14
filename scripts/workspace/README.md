# Linux workspace

This directory provisions the isolated Linux appliance used by the model-facing
`workspace` namespace. The namespace starts folded and is opened through the
normal namespace lifecycle; there is no interactive user shell or Web setting.

The install root is read from the machine-local `config/config_user.yaml`:

```yaml
workspace:
  provisioning:
    install_root: "E:\\Aic_forQ\\wsl"
```

An empty value uses `%LOCALAPPDATA%\AICQ\Workspace`. This setting is file-only
in phase one and is not exposed through Web settings. `-InstallRoot` can be
used as an explicit command-line override.

Install or upgrade the appliance from Windows PowerShell 5.1 or PowerShell 7 in
a session that can manage WSL:

```powershell
.\scripts\workspace\provision-workspace.ps1
```

An in-place upgrade preserves `/workspace`, rebuilds the container, and clears
ephemeral command history. Use `-Recreate` only for an explicit destructive
rebuild of the whole WSL appliance.

Provisioning only terminates, unregisters, or restarts `AICQ-Workspace`; it
does not stop Docker Desktop or any other WSL distribution.

The script performs all capability, disk, and network preflight checks before
unregistering an existing `AICQ-Workspace`. Assets are streamed through tar and
stdin; the Windows repository is never mounted into the appliance.

Run the more expensive apt/git/pip/compiler probes explicitly:

```powershell
.\scripts\workspace\verify-workspace.ps1 -Full
```

The work container and `/workspace` persist across Core and WSL restarts. Core
shutdown only closes in-flight bridge processes. Destructive rebuild remains an
explicit `-Recreate` administrator action.
