#!/usr/bin/env bash
set -euo pipefail

# This entry point is invoked only by the Windows user-owned provisioning
# worker. The broker and all model-facing RPCs deliberately cannot reach it.

manifest=/opt/aicq-workspace/protocol-manifest.json
image_context=/opt/aicq-workspace/image
workspace_root=/var/lib/aicq-workspace/workspace
command_root=/var/lib/aicq-workspace/commands
container_command_root=/run/aicq-workspace/commands
podman_bin=${AICQ_WORKSPACE_PODMAN_BIN:-/usr/bin/podman}
reuse_valid_image=${AICQ_WORKSPACE_REUSE_VALID_IMAGE:-0}

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

build_status=1
if [[ "$reuse_valid_image" == 1 ]] && "$podman_bin" image exists "$image"; then
  installed_protocol=$("$podman_bin" image inspect --format '{{ index .Labels "io.aicq.workspace.protocol" }}' "$image")
  installed_digest=$("$podman_bin" image inspect --format '{{ index .Labels "io.aicq.workspace.base-digest" }}' "$image")
  if [[ "$installed_protocol" == "$protocol" && "$installed_digest" == "$digest" ]]; then
    echo "[workspace] Reusing the completed protocol $protocol image from the interrupted build"
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
    if "$podman_bin" build "${build_args[@]}" "$image_context"; then
      build_status=0
      break
    else
      build_status=$?
    fi
    if (( build_attempt < 3 )); then
      delay=$((build_attempt * 3))
      echo "[workspace][retry] Container image build exited with code $build_status; retrying the uncommitted failed layer in ${delay}s (attempt $((build_attempt + 1))/3)"
      sleep "$delay"
    fi
  done
fi
if (( build_status != 0 )); then
  echo "[workspace] Container image build failed after 3 attempts" >&2
  exit "$build_status"
fi

if "$podman_bin" container exists "$container"; then
  "$podman_bin" rm -f "$container"
fi

"$podman_bin" create \
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

"$podman_bin" start "$container"
