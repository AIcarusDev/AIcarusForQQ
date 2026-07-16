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
    manifest["web_projection"]["network"],
):
    print(value)
PY
)

container=${values[0]}
cpus=${values[1]}
memory=${values[2]}
pids=${values[3]}
projection_network=${values[4]}

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

mapfile -t create_command < <("$podman_bin" inspect --format '{{range .Config.CreateCommand}}{{println .}}{{end}}' "$container")
projection_network_ready=0
for ((index = 0; index < ${#create_command[@]}; index++)); do
  if [[ "${create_command[$index]}" == --publish || "${create_command[$index]}" == -p ]]; then
    echo "[computer] Agent computer still uses explicit port publishing; update the computer system first" >&2
    exit 1
  fi
  if [[ "${create_command[$index]}" == --network && "${create_command[$((index + 1))]:-}" == "$projection_network" ]]; then
    projection_network_ready=1
  fi
done
if (( projection_network_ready != 1 )); then
  echo "[computer] Agent computer Web projection network is outdated; update the computer system first" >&2
  exit 1
fi
rm -f "${XDG_CONFIG_HOME:-$HOME/.config}/aicq-workspace/preview-port"
