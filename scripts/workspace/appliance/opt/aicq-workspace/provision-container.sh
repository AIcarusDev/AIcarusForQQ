#!/usr/bin/env bash
set -euo pipefail

# This entry point is invoked only by the Windows user-owned provisioning
# worker. The broker and all model-facing RPCs deliberately cannot reach it.

manifest=/opt/aicq-workspace/protocol-manifest.json
image_context=/opt/aicq-workspace/image
workspace_root=/var/lib/aicq-workspace/workspace
command_root=/var/lib/aicq-workspace/commands
container_command_root=/run/aicq-workspace/commands

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

test -f /run/aicq-workspace/firewall.ready
install -d -m 0700 "$workspace_root" "$command_root"

/usr/bin/podman build \
  --pull=missing \
  --label "io.aicq.workspace.protocol=$protocol" \
  --label "io.aicq.workspace.base-digest=$digest" \
  --tag "$image" \
  "$image_context"

if /usr/bin/podman container exists "$container"; then
  /usr/bin/podman rm -f "$container"
fi

/usr/bin/podman create \
  --name "$container" \
  --hostname workspace \
  --label "io.aicq.workspace.protocol=$protocol" \
  --label "io.aicq.workspace.base-digest=$digest" \
  --user 0:0 \
  --workdir /workspace \
  --cpus "$cpus" \
  --memory "$memory" \
  --pids-limit "$pids" \
  --network pasta \
  --volume "$workspace_root:/workspace:rw" \
  --volume "$command_root:$container_command_root:rw" \
  --stop-timeout 10 \
  "$image"

/usr/bin/podman start "$container"
