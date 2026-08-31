#!/usr/bin/env bash
set -euo pipefail

# This entry point is invoked only by the Windows user-owned provisioning
# worker. The broker and all model-facing RPCs deliberately cannot reach it.

manifest=/opt/aicq-workspace/protocol-manifest.json
image_context=/opt/aicq-workspace/image
home_root=/var/lib/aicq-workspace/home
legacy_workspace_root=/var/lib/aicq-workspace/workspace
home_layout_marker=/var/lib/aicq-workspace/.home-layout-v3
command_root=/var/lib/aicq-workspace/commands
container_command_root=/run/aicq-workspace/commands
podman_bin=${AICQ_WORKSPACE_PODMAN_BIN:-/usr/bin/podman}
reuse_valid_image=${AICQ_WORKSPACE_REUSE_VALID_IMAGE:-0}
rebuild_image=${AICQ_WORKSPACE_REBUILD_IMAGE:-0}

mapfile -t values < <(/usr/bin/python3 - "$manifest" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
limits = manifest["limits"]
for value in (
    manifest["protocol_version"],
    manifest["container_name"],
    manifest["image_name"],
    manifest["base_image_digest"],
    limits["cpus"],
    limits["memory_bytes"],
    limits["pids"],
    manifest["network_isolation"]["network"],
):
    print(value)
PY
)

protocol=${values[0]}
container=${values[1]}
image=${values[2]}
digest=${values[3]}
cpus=${values[4]}
memory=${values[5]}
pids=${values[6]}
isolated_network=${values[7]}

test -f /run/aicq-workspace/firewall.ready
test -d "${XDG_RUNTIME_DIR:?XDG_RUNTIME_DIR must name the managed Podman runtime directory}"
if [[ "$(stat -c '%U:%G:%a' "$XDG_RUNTIME_DIR")" != "aicqws:aicqws:700" ]]; then
  echo "[computer] Invalid Podman runtime directory ownership or mode: $XDG_RUNTIME_DIR" >&2
  exit 1
fi
install -d -m 0700 "$home_root" "$command_root"

legacy_home_pending=0
if [[ -d "$legacy_workspace_root" && ! -e "$home_layout_marker" ]]; then
  echo "[computer] Copying existing Agent files from /workspace into /home/agent"
  cp -a --update=none "$legacy_workspace_root/." "$home_root/"
  legacy_home_pending=1
fi

build_status=1
if [[ "$reuse_valid_image" == 1 ]] && "$podman_bin" image exists "$image"; then
  installed_protocol=$("$podman_bin" image inspect --format '{{ index .Labels "io.aicq.workspace.protocol" }}' "$image")
  installed_digest=$("$podman_bin" image inspect --format '{{ index .Labels "io.aicq.workspace.base-digest" }}' "$image")
  if [[ "$installed_protocol" == "$protocol" && "$installed_digest" == "$digest" ]]; then
    echo "[computer] Reusing the completed protocol $protocol image from the interrupted build"
    build_status=0
  fi
fi
if (( build_status != 0 )); then
  for build_attempt in 1 2 3; do
    build_args=(
      --pull=missing
      --label "io.aicq.workspace.protocol=$protocol"
      --label "io.aicq.workspace.base-digest=$digest"
      --tag "$image"
    )
    if [[ "$rebuild_image" == 1 ]]; then
      build_args+=(--no-cache)
    fi
    if "$podman_bin" build "${build_args[@]}" "$image_context"; then
      build_status=0
      break
    else
      build_status=$?
    fi
    if (( build_attempt < 3 )); then
      delay=$((build_attempt * 3))
      echo "[computer][retry] Container image build exited with code $build_status; retrying the uncommitted failed layer in ${delay}s (attempt $((build_attempt + 1))/3)"
      sleep "$delay"
    fi
  done
fi
if (( build_status != 0 )); then
  echo "[computer] Container image build failed after 3 attempts" >&2
  exit "$build_status"
fi
if ! "$podman_bin" image exists "$image"; then
  echo "[computer] Container image build returned success but did not create $image" >&2
  exit 1
fi

if "$podman_bin" container exists "$container"; then
  "$podman_bin" rm -f "$container"
fi

"$podman_bin" create \
  --pull=never \
  --name "$container" \
  --hostname agent-computer \
  --add-host agent-computer:127.0.0.1 \
  --label "io.aicq.workspace.protocol=$protocol" \
  --label "io.aicq.workspace.base-digest=$digest" \
  --userns keep-id:uid=1000,gid=1000 \
  --user agent:agent \
  --workdir /home/agent \
  --cpus "$cpus" \
  --memory "$memory" \
  --pids-limit "$pids" \
  --network "$isolated_network" \
  --volume "$home_root:/home/agent:rw" \
  --volume "$command_root:$container_command_root:rw" \
  --stop-timeout 10 \
  "$image"

"$podman_bin" start "$container"
sleep 0.2
if [[ "$($podman_bin inspect --format '{{.State.Running}}' "$container")" != "true" ]]; then
  "$podman_bin" logs "$container" >&2 || true
  echo "[computer] Agent computer stopped during startup" >&2
  exit 1
fi
# Rootless Buildah strips setuid bits while committing the image on this
# appliance. Restore sudo in the newly created system layer before the Agent
# can use the container; this layer persists across ordinary stop/start.
"$podman_bin" exec --user 0 "$container" /bin/chmod 4755 /usr/bin/sudo

rm -f "${XDG_CONFIG_HOME:-$HOME/.config}/aicq-workspace/preview-port"

"$podman_bin" exec --user agent --workdir /home/agent "$container" \
  /bin/bash -c 'test "$(id -un)" = agent && test "$HOME" = /home/agent && sudo -n true'

if (( legacy_home_pending )); then
  rm -rf -- "$legacy_workspace_root"
  touch "$home_layout_marker"
  echo "[computer] Existing Agent files now live in /home/agent"
fi

# Provisioning starts the container only to validate the immutable image and
# persistent home contract. Runtime starts belong to the delegated broker
# service so every container process inherits its enforced systemd cgroup.
"$podman_bin" stop --time 10 "$container"
