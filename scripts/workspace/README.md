# Linux workspace

This directory provisions the isolated Linux appliance used by the
model-facing `workspace` namespace. The namespace is visible only when the
user enables it and starts folded when enabled. Agent-facing runtime code can
start an existing stopped container, but cannot install WSL, build images, or
create containers.

The canonical configuration is stored in `config/config_user.yaml`:

```yaml
workspace:
  enabled: false
  install_root: "E:\\Aic_forQ\\wsl"
  resources:
    cpus: 4
    memory_gib: 8
    disk_gib: 64
```

The default install root is the ignored project path `data/workspace`. The
settings page exposes the enable switch, install root, resource limits, build,
apply, and upgrade controls. The maintenance page exposes confirmed restart,
clear, and uninstall actions. These controls work in WebUI-only mode and run
through a detached job worker whose state and logs live under
`data/workspace-control`.

The preferred path is the explicit WebUI build/apply button. The same
user-owned provisioning entry point can be invoked manually for diagnostics:

```powershell
.\scripts\workspace\provision-workspace.ps1
```

An in-place apply preserves `/workspace`, rebuilds the container, updates the
appliance, and expands the sparse VHD when requested. Disk shrinking and path
migration are intentionally unsupported; fully uninstall and rebuild instead.

Provisioning only terminates, unregisters, or restarts `AICQ-Workspace`; it
does not stop Docker Desktop or any other WSL distribution. Assets are streamed
through tar/stdin, so the Windows repository is never mounted into the
appliance.

Run the more expensive apt/git/pip/compiler probes explicitly:

```powershell
.\scripts\workspace\verify-workspace.ps1 -Full
```

The broker's `ensure_default` method only validates and starts existing
artifacts. Image build and container creation live exclusively in the
provisioning-only `provision-container.sh` entry point.
