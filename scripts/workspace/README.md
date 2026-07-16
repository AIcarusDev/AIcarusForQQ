# Agent Linux computer

This directory provisions the isolated Linux computer owned and used by the
Agent. The model-facing namespace is `computer`. Runtime code may start the
existing computer, but installation, system replacement, and destructive
maintenance remain explicit user actions.

## Linux identity and filesystem contract

Inside the computer, the Agent is an ordinary Linux user with the following
stable identity:

- user and group: `agent` (`uid=1000`, `gid=1000`)
- home and default working directory: `/home/agent`
- shell: Bash
- administrator access: passwordless `sudo`

The persistent host-backed data root is mounted as the whole Agent home. A new
home receives the normal files from `/etc/skel` (for example `.profile` and
`.bashrc`); existing files are never overwritten by home initialization.
Relative file paths are resolved from `/home/agent`, and file export is limited
to that home.

The WSL distribution and its `aicqws` service account are internal appliance
implementation details. They are not the Agent's Linux identity and are not
part of the model-facing computer contract.

## Configuration and controls

The canonical configuration remains stored in the internal `workspace` block
of `config/config_user.yaml`:

```yaml
workspace:
  enabled: false
  install_root: "E:\\Aic_forQ\\core\\data\\computer"
  resources:
    cpus: 4
    memory_gib: 8
    disk_gib: 64
```

The settings page exposes installation, system update, and resource controls.
The maintenance page exposes confirmed system rebuild, restart, Agent-home
erase, and complete uninstall actions. Detached job state and logs live under
the internal `data/workspace-control` directory.

The same user-owned provisioning entry point can be invoked manually:

```powershell
.\scripts\workspace\provision-workspace.ps1
```

## Persistence and lifecycle

The lifecycle deliberately separates ordinary adjustments from system
replacement:

- **Restart** stops and starts the existing container. The home and current
  writable system layer remain intact.
- **Apply resources** updates CPU, memory, process, preview, firewall, and disk
  settings in place. It does not remove or recreate the container. Disk may be
  expanded but not shrunk.
- **Update system** builds/uses the current managed image and replaces the
  container while retaining the complete Agent home.
- **Rebuild system** forces a clean image build and replaces the system
  container while retaining the complete Agent home.
- **Erase Agent home** removes the Agent's files, then reinitializes only the
  standard home files.
- **Uninstall computer** removes the complete appliance and its persistent
  data.

Because system update/rebuild replaces the container, packages or files added
with `sudo` outside `/home/agent` are intentionally system-layer state and are
not promised across that operation. Files in the Agent home are the durable
personal state.

During the protocol-3 upgrade, files from the former persistent `/workspace`
layout are copied into `/home/agent` without overwriting any destination file.
The former data directory is deleted only after the new computer starts and
passes identity, home, and sudo checks.

## Isolation and verification

Provisioning publishes only container TCP port `6080`, bound to a random WSL
loopback port. The no-argument `computer.preview` tool exposes the fixed
preview mapping; the Agent cannot choose a host address or create containers.

Provisioning only operates on `AICQ-Workspace`; it does not stop Docker Desktop
or another WSL distribution. Assets are streamed through tar/stdin, so the
Windows repository is never mounted into the appliance.

Run the more expensive apt/git/pip/compiler probes explicitly:

```powershell
.\scripts\workspace\verify-workspace.ps1 -Full
```

The broker's `ensure_default` method validates and starts existing artifacts.
Image build and container creation remain exclusive to the explicit
provisioning path.
