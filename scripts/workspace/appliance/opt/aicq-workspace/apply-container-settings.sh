#!/usr/bin/env bash
set -euo pipefail

manifest=/opt/aicq-workspace/protocol-manifest.json
podman_bin=${AICQ_WORKSPACE_PODMAN_BIN:-/usr/bin/podman}

mapfile -t values < <(/usr/bin/python3 - "$manifest" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
limits = manifest["limits"]
for value in (
    manifest["container_name"],
    limits["cpus"],
    limits["memory_bytes"],
    limits["pids"],
):
    print(value)
PY
)

container=${values[0]}
cpus=${values[1]}
memory=${values[2]}
pids=${values[3]}

if ! "$podman_bin" container exists "$container"; then
  echo "[computer] Computer container does not exist" >&2
  exit 1
fi

running=$($podman_bin inspect --format '{{.State.Running}}' "$container")
if [[ "$running" != "true" ]]; then
  "$podman_bin" start "$container"
fi
"$podman_bin" update \
  --cpus "$cpus" \
  --memory "$memory" \
  --pids-limit "$pids" \
  "$container"

preview_endpoint=$($podman_bin port "$container" 6080/tcp)
if [[ ! "$preview_endpoint" =~ ^127\.0\.0\.1:([0-9]{1,5})$ ]]; then
  echo "[computer] Invalid loopback preview mapping: $preview_endpoint" >&2
  exit 1
fi
preview_host_port=${BASH_REMATCH[1]}
if (( preview_host_port < 1 || preview_host_port > 65535 )); then
  echo "[computer] Preview host port is outside the valid TCP range" >&2
  exit 1
fi

preview_config_dir=${XDG_CONFIG_HOME:-$HOME/.config}/aicq-workspace
install -d -m 0700 "$preview_config_dir"
printf '%s\n' "$preview_host_port" | install -m 0600 /dev/stdin "$preview_config_dir/preview-port"
